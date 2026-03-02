import argparse
import copy
import getpass
import hashlib
import hmac
import json
import os
import re
import sys
from pathlib import Path

RECOMMEND_LIMIT = 20


# Match run.py startup environment to avoid TensorFlow-side noise.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

sys.path.append(str(Path(__file__).parent.parent))


def _build_parser():
    """
    Build parser.
    
    This function implements the build parser step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run EraEx For You recommendation pipeline from CLI and show "
            "profile-fit / bot-validation metrics."
        )
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default="",
        help="User id used to fetch For You recommendations.",
    )
    parser.add_argument(
        "--username",
        type=str,
        default="",
        help="Account username for CLI login (alternative to --user-id).",
    )
    parser.add_argument(
        "--password",
        type=str,
        default="",
        help="Account password for CLI login (alternative to --user-id).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast/profile-seed mode instead of deep adaptive ranking.",
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use Flask test-client route (/api/recommend) instead of direct engine call.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON payload instead of formatted rows.",
    )
    parser.add_argument(
        "--show-ranking-score",
        action="store_true",
        help="Also print internal ranking score (hidden by default).",
    )
    parser.add_argument(
        "--bot-k",
        type=int,
        default=15,
        help="Top-K rows used for live bot validation (default: 15).",
    )
    return parser


def _safe_float(value, default=0.0):
    """
    Safely convert float.
    
    This function implements the safe float step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp01(value):
    """
    Execute clamp01.
    
    This function implements the clamp01 step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    return max(0.0, min(1.0, _safe_float(value)))


def _tokenize(text):
    """
    Execute tokenize.
    
    This function implements the tokenize step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    return set(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def _jaccard(a, b):
    """
    Execute jaccard.
    
    This function implements the jaccard step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    if not a or not b:
        return 0.0
    union = a.union(b)
    if not union:
        return 0.0
    return len(a.intersection(b)) / float(len(union))


def _ordered_unique_song_ids(*lists):
    """
    Execute ordered unique song ids.
    
    This function implements the ordered unique song ids step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    out = []
    seen = set()
    for values in lists:
        for raw in list(values or []):
            sid = str(raw or "").strip()
            if not sid or sid in seen:
                continue
            seen.add(sid)
            out.append(sid)
    return out


class _ApiRunner:
    def __init__(self):
        """
        Initialize the instance state.
        
        This method implements the init step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        from src.web_api.web_app import app

        self._client = app.test_client()

    def fetch(self, user_id, limit, fast):
        """
        Execute fetch.
        
        This method implements the fetch step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        resp = self._client.get(
            "/api/recommend",
            query_string={
                "user_id": str(user_id or "").strip(),
                "n": int(limit),
                "fast": "1" if bool(fast) else "0",
                "covers": "0",
            },
        )
        try:
            payload = resp.get_json()
        except Exception:
            payload = {"error": f"Non-JSON response (status={resp.status_code})"}
        if resp.status_code != 200:
            return {
                "user_id": str(user_id or "").strip(),
                "recommendations": [],
                "error": payload,
                "status_code": resp.status_code,
            }
        return payload if isinstance(payload, dict) else {"user_id": user_id, "recommendations": []}


class _DirectRunner:
    def __init__(self):
        """
        Initialize the instance state.
        
        This method implements the init step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        self._handler = None
        self._init_error = ""

    def _get_handler(self):
        """
        Get handler.
        
        This method implements the get handler step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if self._handler is None:
            try:
                from src.recommendation.recommendation_engine import ColdStartHandler

                self._handler = ColdStartHandler()
            except Exception as exc:
                self._init_error = str(exc)
                return None
        return self._handler

    @staticmethod
    def _history_counts(signals):
        """
        Execute history counts.
        
        This method implements the history counts step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        liked = list(signals.get("liked") or [])
        played = list(signals.get("played") or [])
        disliked = list(signals.get("disliked") or [])
        playlist_tracks = list(signals.get("playlist_tracks") or [])
        skip_summary = signals.get("skip_summary") if isinstance(signals.get("skip_summary"), dict) else {}
        return {
            "likes": len(liked),
            "plays": len(played),
            "dislikes": len(disliked),
            "playlist_tracks": len(playlist_tracks),
            "skip_next": int(sum((skip_summary or {}).get("next_counts", {}).values())),
            "skip_prev": int(sum((skip_summary or {}).get("prev_counts", {}).values())),
            "skip_early": int(sum((skip_summary or {}).get("early_next_counts", {}).values())),
        }

    def fetch(self, user_id, limit, fast):
        """
        Execute fetch.
        
        This method implements the fetch step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        signals = _load_user_signals(user_id)
        liked = list(signals.get("liked") or [])
        played = list(signals.get("played") or [])
        disliked = list(signals.get("disliked") or [])
        playlist_tracks = list(signals.get("playlist_tracks") or [])
        skip_summary = signals.get("skip_summary") if isinstance(signals.get("skip_summary"), dict) else {}
        history_counts = self._history_counts(signals)
        has_personal_history = bool(liked or played or playlist_tracks)

        handler = self._get_handler()
        if handler is None:
            return {
                "user_id": str(user_id or "").strip(),
                "recommendation_mode": "error",
                "history_counts": history_counts,
                "recommendations": [],
                "error": f"Recommendation engine unavailable: {self._init_error or 'initialization failed'}",
            }

        n = max(1, int(limit))
        if has_personal_history:
            try:
                rows = handler.recommend(
                    liked_ids=liked,
                    played_ids=played,
                    k=n,
                    disliked_ids=disliked,
                    playlist_track_ids=playlist_tracks,
                    skip_feedback=skip_summary,
                )
            except TypeError:
                rows = handler.recommend(liked, played, n=n)
            mode = "profile_seed_fast" if bool(fast) else "adaptive"
        else:
            rows = handler.get_trending(n=n)
            mode = "cold_start_fast" if bool(fast) else "cold_start"
        return {
            "user_id": str(user_id or "").strip(),
            "recommendation_mode": mode,
            "history_counts": history_counts,
            "engine_quota": dict(getattr(handler, "last_quota_stats", {}) or {}),
            "recommendations": list(rows or []),
        }


def _load_user_signals(user_id):
    """
    Load user signals.
    
    This function implements the load user signals step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    from src.user_profiles.user_profile_store import UserProfile

    store = UserProfile()
    uid = str(user_id or "").strip()
    liked = list(store.get_likes(uid) or [])
    played = list(store.get_plays(uid, limit=120) or [])
    disliked = list(store.get_dislikes(uid) or [])
    playlist_tracks = list(store.get_all_playlist_track_ids(uid, limit=240) or [])
    skip_summary = store.get_skip_summary(uid, limit=600)
    profile = store.get_profile(uid) or {}
    return {
        "user_id": uid,
        "liked": liked,
        "played": played,
        "disliked": disliked,
        "playlist_tracks": playlist_tracks,
        "skip_summary": skip_summary if isinstance(skip_summary, dict) else {},
        "profile": profile if isinstance(profile, dict) else {},
    }


def _verify_password_hash(stored_hash, password):
    # Supports Werkzeug-style scrypt / pbkdf2 hashes without requiring Flask/Werkzeug runtime.
    """
    Execute verify password hash.
    
    This function implements the verify password hash step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    raw = str(stored_hash or "").strip()
    plain = str(password or "")
    if not raw:
        return False
    try:
        from werkzeug.security import check_password_hash as _werkzeug_check_password_hash
    except Exception:
        _werkzeug_check_password_hash = None
    if _werkzeug_check_password_hash is not None:
        try:
            return bool(_werkzeug_check_password_hash(raw, plain))
        except Exception:
            pass
    if "$" not in raw:
        return hmac.compare_digest(raw, plain)
    try:
        method_part, salt, digest_hex = raw.split("$", 2)
    except Exception:
        return False
    method_part = str(method_part or "").strip().lower()
    salt = str(salt or "")
    digest_hex = str(digest_hex or "").strip().lower()
    if not method_part or not salt or not digest_hex:
        return False
    if len(digest_hex) % 2 != 0:
        return False
    try:
        expected = bytes.fromhex(digest_hex)
    except Exception:
        return False
    if not expected:
        return False

    try:
        if method_part.startswith("scrypt"):
            # Example: scrypt:32768:8:1$salt$hex_digest
            parts = method_part.split(":")
            n = int(parts[1]) if len(parts) > 1 else 32768
            r = int(parts[2]) if len(parts) > 2 else 8
            p = int(parts[3]) if len(parts) > 3 else 1
            # OpenSSL-backed hashlib.scrypt can fail with "memory limit exceeded"
            # unless maxmem is explicitly set above the default cap.
            min_maxmem = max(64 * 1024 * 1024, 128 * n * r * max(1, p))
            derived = hashlib.scrypt(
                plain.encode("utf-8"),
                salt=salt.encode("utf-8"),
                n=n,
                r=r,
                p=p,
                dklen=len(expected),
                maxmem=min_maxmem,
            )
            return hmac.compare_digest(derived, expected)

        if method_part.startswith("pbkdf2"):
            # Example: pbkdf2:sha256:600000$salt$hex_digest
            parts = method_part.split(":")
            algo = str(parts[1] if len(parts) > 1 else "sha256").strip() or "sha256"
            iterations = int(parts[2]) if len(parts) > 2 else 260000
            derived = hashlib.pbkdf2_hmac(
                algo,
                plain.encode("utf-8"),
                salt.encode("utf-8"),
                iterations,
                dklen=len(expected),
            )
            return hmac.compare_digest(derived, expected)
    except Exception:
        return False
    return False


def _resolve_user_id_from_args(args):
    """
    Resolve user id from args.
    
    This function implements the resolve user id from args step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    explicit_user_id = str(args.user_id or "").strip()
    username = str(args.username or "").strip()
    password = str(args.password or "")

    if explicit_user_id:
        return explicit_user_id

    if not username:
        try:
            username = input("Username: ").strip()
        except Exception:
            username = ""
    if username:
        if not password:
            try:
                password = getpass.getpass("Password: ")
            except Exception:
                password = ""
        from src.user_profiles.user_profile_store import UserProfile

        store = UserProfile()
        account = store.get_account_by_username(username)
        if not isinstance(account, dict):
            raise SystemExit("Login failed: username not found.")
        stored_hash = str(account.get("password_hash") or "")
        if not _verify_password_hash(stored_hash, password):
            raise SystemExit("Login failed: invalid password.")
        return str(account.get("user_id") or "").strip()

    raise SystemExit("Login failed: username required.")


def _init_recommender_handler():
    """
    Initialize recommender handler.
    
    This function implements the init recommender handler step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    try:
        from src.recommendation.recommendation_engine import ColdStartHandler
    except Exception:
        return None, "Could not import recommendation engine."
    try:
        return ColdStartHandler(), ""
    except Exception as exc:
        return None, f"Could not initialize recommendation engine: {exc}"


def _row_track_id(row):
    """
    Execute row track id.
    
    This function implements the row track id step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    return str(
        row.get("id")
        or row.get("track_id")
        or row.get("song_id")
        or row.get("video_id")
        or ""
    ).strip()


def _row_meta_text(row, meta):
    """
    Execute row meta text.
    
    This function implements the row meta text step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    title = str(row.get("title") or (meta or {}).get("title") or "").strip()
    artist = str(row.get("artist") or (meta or {}).get("artist_name") or "").strip()
    album = str(row.get("album") or (meta or {}).get("album") or "").strip()
    desc = str(row.get("description") or (meta or {}).get("description") or "").strip()
    return " ".join([title, artist, album, desc]).strip()


def _compute_row_profile_similarity(handler, sid, seed_ids, feature_weights):
    """
    Compute row profile similarity.
    
    This function implements the compute row profile similarity step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    if handler is None or not sid or not seed_ids:
        return 0.0
    best = 0.0
    for seed in seed_ids:
        sim = handler._combined_similarity(sid, seed, feature_weights)
        if sim > best:
            best = sim
    return _clamp01(best)


def _compute_live_bot_validation(rows, handler, active_bots, top_k=15):
    """
    Compute live bot validation.
    
    This function implements the compute live bot validation step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    if handler is None:
        return {
            "enabled": False,
            "error": "Recommendation engine unavailable for live bot validation.",
            "macro_mrr": 0.0,
            "bots": {},
        }
    top_k = max(1, int(top_k or 1))
    bots = list(active_bots or [])
    if not bots:
        return {
            "enabled": False,
            "error": "No active profile bots inferred from this user's history yet.",
            "macro_mrr": 0.0,
            "bots": {},
        }

    bot_rows = {}
    for bot in bots:
        profile_tokens = set(bot.get("tokens", set()))
        label = str(bot.get("label", bot.get("bot", "bot")))
        confidence = _clamp01(bot.get("confidence", 0.0))
        scores = []
        for row in (rows or [])[:top_k]:
            sid = _row_track_id(row)
            if not sid:
                scores.append(0.0)
                continue
            meta = handler.sim_manager.get_track_info(sid) or {}
            cand_tokens = handler._meta_bot_tokens(meta)
            overlap = handler._bot_overlap_score(cand_tokens, profile_tokens)
            scores.append(_clamp01(confidence * overlap))

        hit_threshold = 0.08
        first_hit_rank = 0
        hit_count = 0
        total_match = 0.0
        for idx, score in enumerate(scores, start=1):
            score = _clamp01(score)
            total_match += score
            if score >= hit_threshold:
                hit_count += 1
                if first_hit_rank == 0:
                    first_hit_rank = idx
        mrr = (1.0 / float(first_hit_rank)) if first_hit_rank > 0 else 0.0
        hit_rate = (hit_count / float(max(1, len(scores)))) if scores else 0.0
        avg_match = (total_match / float(max(1, len(scores)))) if scores else 0.0
        bot_rows[label] = {
            "confidence": round(confidence, 6),
            "mrr": round(mrr, 6),
            "hit_rate": round(hit_rate, 6),
            "avg_match": round(avg_match, 6),
            "top_k": int(top_k),
            "threshold": hit_threshold,
        }

    macro_mrr = (
        sum(_safe_float(v.get("mrr", 0.0)) for v in bot_rows.values()) / float(len(bot_rows))
        if bot_rows
        else 0.0
    )
    return {
        "enabled": True,
        "error": "",
        "macro_mrr": round(macro_mrr, 6),
        "bots": bot_rows,
    }


def _attach_recommendation_metrics(payload, user_signals, bot_k=15):
    """
    Attach recommendation metrics.
    
    This function implements the attach recommendation metrics step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    if not isinstance(payload, dict):
        payload = {"recommendations": []}

    rows = payload.get("recommendations")
    rows = rows if isinstance(rows, list) else []
    liked = list(user_signals.get("liked") or [])
    played = list(user_signals.get("played") or [])
    disliked = {
        str(song_id or "").strip()
        for song_id in list(user_signals.get("disliked") or [])
        if str(song_id or "").strip()
    }
    playlist_tracks = list(user_signals.get("playlist_tracks") or [])
    profile = user_signals.get("profile") if isinstance(user_signals.get("profile"), dict) else {}

    handler, handler_error = _init_recommender_handler()
    if handler is None:
        payload["recommendation_metrics"] = {
            "profile_intent": {
                "active_bots": [],
                "bot_intent_confidence": 0.0,
                "handler_ready": False,
                "note": handler_error or "Recommendation engine unavailable.",
            },
            "result_quality": {
                "result_count": len(rows),
                "avg_profile_fit": 0.0,
                "top1_profile_fit": 0.0,
                "avg_profile_similarity": 0.0,
                "avg_token_overlap": 0.0,
                "avg_bot_match": 0.0,
                "long_tail_share": 0.0,
                "unique_artist_ratio": 0.0,
            },
        }
        payload["bot_validation"] = _compute_live_bot_validation(
            rows=rows,
            handler=None,
            active_bots=[],
            top_k=bot_k,
        )
        return payload

    active_bots = handler._resolve_active_profile_bots(
        liked_history=liked,
        playlist_history=playlist_tracks,
        played_history=played,
    )
    active_bots_compact = [
        {
            "bot": str(bot.get("bot", "")),
            "label": str(bot.get("label", bot.get("bot", ""))),
            "confidence": round(_clamp01(bot.get("confidence", 0.0)), 6),
        }
        for bot in active_bots
        if isinstance(bot, dict)
    ]

    ordered_seeds = _ordered_unique_song_ids(liked, playlist_tracks, played)
    canonical_seeds = []
    seen_seed_ids = set()
    for raw_sid in ordered_seeds:
        sid = handler._canonical_song_id(raw_sid)
        sid = str(sid or "").strip()
        if not sid or sid in seen_seed_ids or sid in disliked:
            continue
        seen_seed_ids.add(sid)
        canonical_seeds.append(sid)
    seed_ids = canonical_seeds[:12]
    feature_seed = (liked[:3] + playlist_tracks[:3] + played[:2])[:8]
    if len(feature_seed) >= 2:
        feature_weights = handler.calculate_dynamic_weights(list(reversed(feature_seed)))
    else:
        feature_weights = handler.calculate_dynamic_weights(seed_ids)

    rec_ids = [handler._canonical_song_id(_row_track_id(row)) for row in rows if isinstance(row, dict)]
    rec_ids = [str(sid or "").strip() for sid in rec_ids if str(sid or "").strip()]
    if rec_ids and seed_ids:
        try:
            handler.sim_manager.precompute_embeddings(rec_ids + seed_ids)
        except Exception:
            pass

    seed_token_bag = set()
    seed_artist_set = set()
    seed_affinity_artist_set = handler._seed_affinity_artist_set(
        liked_history=liked,
        playlist_history=playlist_tracks,
        played_history=played,
    )
    for sid in seed_ids:
        meta = handler.sim_manager.get_track_info(sid) or {}
        seed_token_bag.update(handler._meta_bot_tokens(meta))
        artist_norm = str(meta.get("artist_name", "") or "").strip().lower()
        if artist_norm:
            seed_artist_set.add(artist_norm)

    total_fit = 0.0
    total_sim = 0.0
    total_overlap = 0.0
    total_bot = 0.0
    total_long_tail = 0.0
    matched_seed_artist_count = 0
    unique_artist_set = set()

    enriched_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        sid = handler._canonical_song_id(_row_track_id(row))
        sid = str(sid or "").strip()
        meta = handler.sim_manager.get_track_info(sid) if sid else {}
        meta = meta if isinstance(meta, dict) else {}
        profile_similarity = _compute_row_profile_similarity(
            handler=handler,
            sid=sid,
            seed_ids=seed_ids,
            feature_weights=feature_weights,
        )

        cand_tokens = handler._meta_bot_tokens(meta) if meta else set()
        token_overlap = _jaccard(seed_token_bag, cand_tokens) if seed_token_bag else 0.0

        bot_profile = str(row.get("bot_profile", "") or "").strip()
        bot_match = _clamp01(row.get("bot_match_score", 0.0))
        if (not bot_profile) or bot_match <= 0.0:
            inferred_label, inferred_score = handler._candidate_bot_profile_match(sid, active_bots)
            if inferred_score > bot_match:
                bot_match = _clamp01(inferred_score)
                bot_profile = str(inferred_label or "")

        long_tail = _clamp01(handler._long_tail_strength(sid))
        mainstream = _clamp01(handler._mainstream_strength(sid))
        if seed_ids:
            profile_fit = _clamp01(
                0.58 * profile_similarity
                + 0.18 * token_overlap
                + 0.16 * bot_match
                + 0.08 * long_tail
            )
        else:
            profile_fit = _clamp01(
                0.52 * token_overlap
                + 0.28 * bot_match
                + 0.20 * long_tail
            )

        title = str(row.get("title") or meta.get("title") or "").strip()
        artist = str(row.get("artist") or meta.get("artist_name") or "").strip()
        artist_norm = artist.lower()
        if artist_norm:
            unique_artist_set.add(artist_norm)
            if artist_norm in seed_affinity_artist_set:
                matched_seed_artist_count += 1

        updated = dict(row)
        updated["profile_similarity"] = round(profile_similarity, 6)
        updated["token_overlap"] = round(token_overlap, 6)
        updated["bot_match_score"] = round(bot_match, 6)
        updated["bot_profile"] = bot_profile
        updated["long_tail_strength"] = round(long_tail, 6)
        updated["mainstream_strength"] = round(mainstream, 6)
        updated["profile_fit"] = round(profile_fit, 6)
        updated["match_type"] = (
            "profile_seed_match"
            if profile_similarity >= 0.40
            else ("bot_aligned" if bot_match >= 0.18 else "explore")
        )
        updated["profile_prompt_similarity"] = round(
            _jaccard(_tokenize(_row_meta_text(updated, meta)), seed_token_bag),
            6,
        ) if seed_token_bag else 0.0
        enriched_rows.append(updated)

        total_fit += profile_fit
        total_sim += profile_similarity
        total_overlap += token_overlap
        total_bot += bot_match
        total_long_tail += long_tail

    payload["recommendations"] = enriched_rows
    result_count = len(enriched_rows)
    avg_fit = total_fit / float(max(1, result_count))
    avg_sim = total_sim / float(max(1, result_count))
    avg_overlap = total_overlap / float(max(1, result_count))
    avg_bot = total_bot / float(max(1, result_count))
    avg_long_tail = total_long_tail / float(max(1, result_count))
    top1_fit = _safe_float(enriched_rows[0].get("profile_fit", 0.0), 0.0) if enriched_rows else 0.0
    long_tail_share_threshold = 0.001
    long_tail_count = sum(
        1
        for row in enriched_rows
        if _safe_float(row.get("long_tail_strength", 0.0), 0.0) >= long_tail_share_threshold
    )
    long_tail_share = (
        long_tail_count
        / float(max(1, result_count))
    )
    unique_artist_ratio = len(unique_artist_set) / float(max(1, result_count))
    seed_artist_coverage = matched_seed_artist_count / float(max(1, result_count))
    artist_counts = {}
    for row in enriched_rows:
        if not isinstance(row, dict):
            continue
        artist = str(row.get("artist", "") or "").strip().lower()
        if not artist:
            continue
        artist_counts[artist] = int(artist_counts.get(artist, 0)) + 1
    max_artist_count = max(artist_counts.values()) if artist_counts else 0
    hard_top20_target = bool(result_count >= 20 and len(seed_ids) > 0)
    engine_quota = payload.get("engine_quota") if isinstance(payload.get("engine_quota"), dict) else {}
    desired_seed_target = int(
        engine_quota.get("seed_target_desired", 6 if hard_top20_target else 0)
    )
    desired_long_tail_target = int(
        engine_quota.get("long_tail_target_desired", 4 if hard_top20_target else 0)
    )
    effective_seed_target = int(
        engine_quota.get(
            "seed_target_effective",
            min(desired_seed_target, len(seed_affinity_artist_set), result_count),
        )
    )
    effective_long_tail_target = int(
        engine_quota.get(
            "long_tail_target_effective",
            min(desired_long_tail_target, int(long_tail_count), result_count),
        )
    )
    constraints = {
        "enabled": bool(engine_quota.get("enabled", hard_top20_target)),
        "seed_target": int(effective_seed_target),
        "seed_target_desired": int(desired_seed_target),
        "seed_actual": int(matched_seed_artist_count),
        "long_tail_target": int(effective_long_tail_target),
        "long_tail_target_desired": int(desired_long_tail_target),
        "long_tail_actual": int(long_tail_count),
        "long_tail_threshold": float(long_tail_share_threshold),
        "max_artist_target": int(engine_quota.get("max_per_artist_target", 1 if hard_top20_target else 0)),
        "max_artist_actual": int(max_artist_count),
        "seed_artist_pool": int(engine_quota.get("seed_pool_affinity", len(seed_affinity_artist_set))),
        "seed_artist_exact_pool": int(engine_quota.get("seed_pool_exact", len(seed_artist_set))),
    }
    constraints["met"] = bool(
        (not constraints["enabled"])
        or (
            int(constraints["seed_actual"]) >= int(constraints["seed_target"])
            and int(constraints["long_tail_actual"]) >= int(constraints["long_tail_target"])
            and int(constraints["max_artist_actual"]) <= int(constraints["max_artist_target"])
        )
    )

    payload["recommendation_metrics"] = {
        "profile_intent": {
            "active_bots": active_bots_compact,
            "bot_intent_confidence": float(
                round(_safe_float(active_bots_compact[0]["confidence"], 0.0), 6)
            )
            if active_bots_compact
            else 0.0,
            "handler_ready": True,
            "seed_count": len(seed_ids),
            "seed_token_count": len(seed_token_bag),
        },
        "result_quality": {
            "result_count": int(result_count),
            "avg_profile_fit": float(round(avg_fit, 6)),
            "top1_profile_fit": float(round(top1_fit, 6)),
            "avg_profile_similarity": float(round(avg_sim, 6)),
            "avg_token_overlap": float(round(avg_overlap, 6)),
            "avg_bot_match": float(round(avg_bot, 6)),
            "avg_long_tail_strength": float(round(avg_long_tail, 6)),
            "long_tail_share": float(round(long_tail_share, 6)),
            "seed_artist_coverage": float(round(seed_artist_coverage, 6)),
            "unique_artist_ratio": float(round(unique_artist_ratio, 6)),
            "constraint_check": constraints,
        },
    }

    history_counts = payload.get("history_counts")
    if not isinstance(history_counts, dict):
        history_counts = {}
    payload["history_counts"] = {
        "likes": int(history_counts.get("likes", len(liked))),
        "plays": int(history_counts.get("plays", len(played))),
        "dislikes": int(history_counts.get("dislikes", len(disliked))),
        "playlist_tracks": int(history_counts.get("playlist_tracks", len(playlist_tracks))),
        "skip_next": int(history_counts.get("skip_next", profile.get("skip_next_count", 0))),
        "skip_prev": int(history_counts.get("skip_prev", profile.get("skip_prev_count", 0))),
        "skip_early": int(history_counts.get("skip_early", profile.get("skip_early_count", 0))),
    }

    payload["bot_validation"] = _compute_live_bot_validation(
        rows=enriched_rows,
        handler=handler,
        active_bots=active_bots,
        top_k=bot_k,
    )
    return payload


def _print_formatted(payload, show_ranking_score=False, bot_k=15):
    """
    Execute print formatted.
    
    This function implements the print formatted step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    user_id = str(payload.get("user_id", "") or "")
    mode = str(payload.get("recommendation_mode", "") or "")
    rows = payload.get("recommendations")
    rows = rows if isinstance(rows, list) else []
    history_counts = payload.get("history_counts", {})
    if not isinstance(history_counts, dict):
        history_counts = {}
    metrics = payload.get("recommendation_metrics", {})
    result_quality = metrics.get("result_quality", {}) if isinstance(metrics, dict) else {}

    print(f"User: {user_id}")
    print(f"Mode: {mode or 'unknown'}")
    print(f"Results: {len(rows)}")
    print(
        "Signals: "
        f"likes={int(history_counts.get('likes', 0) or 0)} "
        f"plays={int(history_counts.get('plays', 0) or 0)} "
        f"playlists={int(history_counts.get('playlist_tracks', 0) or 0)} "
        f"dislikes={int(history_counts.get('dislikes', 0) or 0)}"
    )
    if result_quality:
        print(
            "Profile-fit: "
            f"avg={100.0 * _safe_float(result_quality.get('avg_profile_fit', 0.0)):.1f}% "
            f"top1={100.0 * _safe_float(result_quality.get('top1_profile_fit', 0.0)):.1f}% "
            f"sim={100.0 * _safe_float(result_quality.get('avg_profile_similarity', 0.0)):.1f}% "
            f"bot={100.0 * _safe_float(result_quality.get('avg_bot_match', 0.0)):.1f}% "
            f"long_tail={100.0 * _safe_float(result_quality.get('long_tail_share', 0.0)):.1f}%"
        )

    main_metrics = []
    if result_quality:
        main_metrics.append(
            f"Avg Profile-Fit: {100.0 * _safe_float(result_quality.get('avg_profile_fit', 0.0)):.2f}%"
        )
        main_metrics.append(
            f"Avg Similarity: {100.0 * _safe_float(result_quality.get('avg_profile_similarity', 0.0)):.2f}%"
        )
        main_metrics.append(
            f"Top1 Profile-Fit: {100.0 * _safe_float(result_quality.get('top1_profile_fit', 0.0)):.2f}%"
        )
        main_metrics.append(
            f"Avg Bot Match: {100.0 * _safe_float(result_quality.get('avg_bot_match', 0.0)):.2f}%"
        )
        main_metrics.append(
            f"Long-tail Share: {100.0 * _safe_float(result_quality.get('long_tail_share', 0.0)):.2f}%"
        )
    bot_validation = payload.get("bot_validation")
    if isinstance(bot_validation, dict) and bool(bot_validation.get("enabled")):
        main_metrics.append(f"Macro MRR: {_safe_float(bot_validation.get('macro_mrr', 0.0)):.4f}")
    if main_metrics:
        print("Main Metrics: " + " | ".join(main_metrics))
    constraint_row = result_quality.get("constraint_check", {}) if isinstance(result_quality, dict) else {}
    if isinstance(constraint_row, dict) and bool(constraint_row.get("enabled")):
        met = bool(constraint_row.get("met"))
        status = "PASS" if met else "PARTIAL"
        seed_target = int(constraint_row.get("seed_target", 0))
        seed_target_desired = int(constraint_row.get("seed_target_desired", seed_target))
        long_target = int(constraint_row.get("long_tail_target", 0))
        long_target_desired = int(constraint_row.get("long_tail_target_desired", long_target))
        seed_target_display = (
            f"{seed_target}/{seed_target_desired}" if seed_target != seed_target_desired else str(seed_target)
        )
        long_target_display = (
            f"{long_target}/{long_target_desired}" if long_target != long_target_desired else str(long_target)
        )
        print(
            "Top-20 Constraints: "
            f"{status} | "
            f"seed {int(constraint_row.get('seed_actual', 0))}/{seed_target_display} "
            f"(pool={int(constraint_row.get('seed_artist_pool', 0))}, exact={int(constraint_row.get('seed_artist_exact_pool', 0))}) | "
            f"long_tail {int(constraint_row.get('long_tail_actual', 0))}/{long_target_display} "
            f"(thr={100.0 * _safe_float(constraint_row.get('long_tail_threshold', 0.0)):.1f}%) | "
            f"max_per_artist {int(constraint_row.get('max_artist_actual', 0))}/{int(constraint_row.get('max_artist_target', 0))}"
        )

    profile_intent = metrics.get("profile_intent", {}) if isinstance(metrics, dict) else {}
    active_bots = profile_intent.get("active_bots", []) if isinstance(profile_intent, dict) else []
    if isinstance(active_bots, list) and active_bots:
        top = active_bots[0] if isinstance(active_bots[0], dict) else {}
        print(
            "Live bot router: "
            f"{str(top.get('label', top.get('bot', 'unknown')))} "
            f"({100.0 * _safe_float(top.get('confidence', 0.0)):.1f}% confidence)"
        )

    print("-" * 80)
    if not rows:
        print("No recommendations returned.")
    else:
        for idx, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            title = str(row.get("title", "Unknown") or "Unknown")
            artist = str(row.get("artist", "Unknown") or "Unknown")
            year = row.get("year", "")
            sid = _row_track_id(row)
            profile_fit = 100.0 * _safe_float(row.get("profile_fit", 0.0))
            sim = 100.0 * _safe_float(row.get("profile_similarity", 0.0))
            overlap = 100.0 * _safe_float(row.get("token_overlap", 0.0))
            bot = 100.0 * _safe_float(row.get("bot_match_score", 0.0))
            long_tail = 100.0 * _safe_float(row.get("long_tail_strength", 0.0))
            print(f"{idx}. {title} - {artist}")
            print(
                f"   id={sid} year={year} "
                f"profile_fit={profile_fit:.1f}% sim={sim:.1f}% "
                f"token_overlap={overlap:.1f}% bot_match={bot:.1f}% "
                f"long_tail={long_tail:.1f}%"
            )
            if show_ranking_score:
                print(f"   ranking_score={_safe_float(row.get('score', 0.0)):.6f}")

    print("-" * 80)
    print("Validation Metrics")
    if result_quality:
        print(
            f"seed_artist_coverage: {100.0 * _safe_float(result_quality.get('seed_artist_coverage', 0.0)):.2f}%"
        )
        print(
            f"unique_artist_ratio: {100.0 * _safe_float(result_quality.get('unique_artist_ratio', 0.0)):.2f}%"
        )
        print(
            f"avg_long_tail_strength: {100.0 * _safe_float(result_quality.get('avg_long_tail_strength', 0.0)):.2f}%"
        )
    note = str(profile_intent.get("note", "") or "")
    if note:
        print(f"Note: {note}")

    print("-" * 80)
    print("Automated Bot Validation")
    if not isinstance(bot_validation, dict) or not bot_validation.get("enabled"):
        err = ""
        if isinstance(bot_validation, dict):
            err = str(bot_validation.get("error", "") or "")
        print(err or "Bot validation unavailable.")
    else:
        print(
            f"Macro MRR: {_safe_float(bot_validation.get('macro_mrr', 0.0)):.4f} "
            f"(top_k={int(bot_k)})"
        )
        bots = bot_validation.get("bots", {})
        if isinstance(bots, dict):
            for name, score_row in bots.items():
                if not isinstance(score_row, dict):
                    continue
                print(
                    f"{name}: "
                    f"MRR={_safe_float(score_row.get('mrr', 0.0)):.4f} "
                    f"HitRate={100.0 * _safe_float(score_row.get('hit_rate', 0.0)):.2f}% "
                    f"AvgMatch={100.0 * _safe_float(score_row.get('avg_match', 0.0)):.2f}%"
                )


def _strip_internal_tokens_for_json(payload):
    """
    Execute strip internal tokens for json.
    
    This function implements the strip internal tokens for json step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    if not isinstance(payload, dict):
        return payload
    out = copy.deepcopy(payload)
    metrics = out.get("recommendation_metrics")
    if isinstance(metrics, dict):
        profile_intent = metrics.get("profile_intent")
        if isinstance(profile_intent, dict):
            active = profile_intent.get("active_bots")
            if isinstance(active, list):
                cleaned = []
                for row in active:
                    if not isinstance(row, dict):
                        continue
                    item = dict(row)
                    item.pop("tokens", None)
                    cleaned.append(item)
                profile_intent["active_bots"] = cleaned
    return out


def main():
    """
    Run the command entry point.
    
    This function implements the main step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    args = _build_parser().parse_args()
    user_id = _resolve_user_id_from_args(args)
    if not user_id:
        raise SystemExit("Could not resolve a valid user id.")

    if bool(args.use_api):
        runner = _ApiRunner()
    else:
        runner = _DirectRunner()
    payload = runner.fetch(
        user_id=user_id,
        limit=RECOMMEND_LIMIT,
        fast=bool(args.fast),
    )
    if not isinstance(payload, dict):
        payload = {"user_id": user_id, "recommendations": []}
    user_signals = _load_user_signals(user_id)
    payload = _attach_recommendation_metrics(
        payload=payload,
        user_signals=user_signals,
        bot_k=max(1, int(args.bot_k)),
    )
    output_payload = _strip_internal_tokens_for_json(payload)

    if args.json:
        print(json.dumps(output_payload, indent=2, ensure_ascii=False))
        return

    warning = output_payload.get("warning") if isinstance(output_payload, dict) else None
    error = output_payload.get("error") if isinstance(output_payload, dict) else None
    if warning:
        print(f"[WARN] {warning}")
    if error:
        print(f"[ERROR] {error}")
    _print_formatted(
        output_payload if isinstance(output_payload, dict) else {"user_id": user_id, "recommendations": []},
        show_ranking_score=bool(args.show_ranking_score),
        bot_k=max(1, int(args.bot_k)),
    )


if __name__ == "__main__":
    main()