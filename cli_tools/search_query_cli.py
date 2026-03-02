import argparse
import copy
import difflib
import json
import os
import re
import sys
from pathlib import Path


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
            "Run EraEx search pipeline from CLI (no frontend) and show "
            "query-fit / automated bot metrics."
        )
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="",
        help="Free-text search query (optional if using --artist/--song)",
    )
    parser.add_argument(
        "--artist",
        type=str,
        default="",
        help="Artist name to include in search query",
    )
    parser.add_argument(
        "--song",
        "--title",
        dest="song",
        type=str,
        default="",
        help="Song title to include in search query",
    )
    parser.add_argument("--limit", type=int, default=10, help="Number of results to return")
    parser.add_argument("--offset", type=int, default=0, help="Result offset for paging")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON payload instead of formatted rows",
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use Flask test client route (/search) instead of direct pipeline call",
    )
    parser.add_argument(
        "--show-ranking-score",
        action="store_true",
        help="Also print internal ranking score (hidden by default).",
    )
    parser.add_argument(
        "--run-bot-validation",
        action="store_true",
        help=(
            "Run recommendation bot simulation (MRR-based) and print scores. "
            "This is optional and can take longer."
        ),
    )
    parser.add_argument(
        "--bot-steps",
        type=int,
        default=5,
        help="Simulation steps per bot for --run-bot-validation (default: 5).",
    )
    parser.add_argument(
        "--bot-k",
        type=int,
        default=10,
        help="Top-K recommendations generated per bot step (default: 10).",
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


def _string_similarity(query, candidate):
    """
    Execute string similarity.
    
    This function implements the string similarity step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    q = str(query or "").strip().lower()
    c = str(candidate or "").strip().lower()
    if not q or not c:
        return 0.0
    return float(difflib.SequenceMatcher(None, q, c).ratio())


class _DirectRunner:
    def __init__(self):
        """
        Initialize the instance state.
        
        This method implements the init step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        self._pipeline = None
        self._init_error = ""

    def _get_pipeline(self):
        """
        Get pipeline.
        
        This method implements the get pipeline step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if self._pipeline is None:
            try:
                from src.search.search_pipeline import UnifiedSearchPipeline

                self._pipeline = UnifiedSearchPipeline()
            except Exception as exc:
                self._init_error = str(exc)
                return None
        return self._pipeline

    def fetch(self, query, limit, offset):
        """
        Execute fetch.
        
        This method implements the fetch step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        pipeline = self._get_pipeline()
        if pipeline is None:
            hint = "Install FAISS and search dependencies, then rebuild indexes."
            return {
                "query": query,
                "results": [],
                "error": f"Search pipeline unavailable: {self._init_error or 'initialization failed'}",
                "warning": hint,
            }
        if pipeline.faiss_index is None:
            return {
                "query": query,
                "results": [],
                "warning": "FAISS index not found. Build data/indexes first.",
            }
        return pipeline.search(query, limit=limit, offset=offset)


class _ApiRunner:
    def __init__(self):
        """
        Initialize the instance state.
        
        This method implements the init step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        from src.web_api.web_app import app

        self._client = app.test_client()

    def fetch(self, query, limit, offset):
        """
        Execute fetch.
        
        This method implements the fetch step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        resp = self._client.get(
            "/search",
            query_string={"q": str(query), "limit": int(limit), "offset": int(offset)},
        )
        try:
            payload = resp.get_json()
        except Exception:
            payload = {"error": f"Non-JSON response (status={resp.status_code})"}
        if resp.status_code != 200:
            return {
                "query": query,
                "results": [],
                "error": payload,
                "status_code": resp.status_code,
            }
        return payload if isinstance(payload, dict) else {"query": query, "results": []}


def _metric_min_max(rows, key):
    """
    Execute metric min max.
    
    This function implements the metric min max step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    values = []
    for row in rows:
        if isinstance(row, dict):
            values.append(_safe_float(row.get(key, 0.0), default=0.0))
    if not values:
        return (0.0, 1.0)
    return (min(values), max(values))


def _normalize_with_min_max(value, low, high):
    """
    Normalize with min max.
    
    This function implements the normalize with min max step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    value = _safe_float(value, default=0.0)
    denom = max(float(high) - float(low), 1e-9)
    return _clamp01((value - float(low)) / denom)


def _attach_query_metrics(payload):
    """
    Attach query metrics.
    
    This function implements the attach query metrics step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    if not isinstance(payload, dict):
        return {"query": "", "results": []}

    existing_query_metrics = payload.get("query_metrics")
    existing_query_metrics = (
        dict(existing_query_metrics)
        if isinstance(existing_query_metrics, dict)
        else {}
    )
    query = str(payload.get("query", "") or "")
    results = payload.get("results")
    results = results if isinstance(results, list) else []
    semantic_low, semantic_high = _metric_min_max(results, "semantic_score")
    query_tokens = _tokenize(query)

    enriched = []
    for row in results:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title", "") or "")
        artist = str(row.get("artist", "") or "")
        description = str(row.get("description", "") or "")
        album = str(row.get("album", "") or "")

        candidate_text = " ".join([title, artist, album, description]).strip()
        candidate_tokens = _tokenize(candidate_text)
        token_overlap = _jaccard(query_tokens, candidate_tokens)
        prompt_similarity = _string_similarity(query, candidate_text)

        overlap_score = _clamp01(row.get("overlap_score", 0.0))
        facet_score = _clamp01(row.get("facet_score", 0.0))
        description_score = _clamp01(row.get("description_score", 0.0))
        title_score = _clamp01(row.get("title_score", 0.0))
        semantic_similarity = (
            _clamp01(row.get("semantic_similarity", 0.0))
            if row.get("semantic_similarity") is not None
            else _normalize_with_min_max(
                row.get("semantic_score", 0.0),
                semantic_low,
                semantic_high,
            )
        )
        lexical_score = (
            token_overlap + prompt_similarity + max(overlap_score, title_score, description_score)
        ) / 3.0
        if row.get("query_fit_score") is not None:
            query_fit = _clamp01(row.get("query_fit_score", 0.0))
        else:
            query_fit = (
                0.46 * semantic_similarity
                + 0.24 * token_overlap
                + 0.12 * prompt_similarity
                + 0.10 * facet_score
                + 0.08 * description_score
            )
            query_fit = _clamp01(query_fit)

        updated = dict(row)
        updated["query_fit"] = round(query_fit, 6)
        updated["semantic_similarity"] = round(semantic_similarity, 6)
        updated["token_overlap"] = round(token_overlap, 6)
        updated["prompt_similarity"] = round(prompt_similarity, 6)
        updated["lexical_score"] = round(_clamp01(lexical_score), 6)
        enriched.append(updated)

    payload["results"] = enriched

    if not enriched:
        payload["query_metrics"] = {
            "result_count": 0,
            "avg_query_fit": 0.0,
            "top1_query_fit": 0.0,
            "avg_semantic_similarity": 0.0,
            "avg_token_overlap": 0.0,
            "avg_prompt_similarity": 0.0,
            "genre_hit_rate_at_k": 0.0,
            "facet_hit_rate_at_k": 0.0,
            "avg_facet_alignment": 0.0,
            "avg_mixed_intent_coverage": 0.0,
            "mixed_intent_hit_rate_at_k": 0.0,
        }
        for key, value in existing_query_metrics.items():
            if key not in payload["query_metrics"]:
                payload["query_metrics"][key] = value
        return payload

    avg_query_fit = sum(_safe_float(r.get("query_fit")) for r in enriched) / float(len(enriched))
    avg_sem = sum(_safe_float(r.get("semantic_similarity")) for r in enriched) / float(len(enriched))
    avg_tok = sum(_safe_float(r.get("token_overlap")) for r in enriched) / float(len(enriched))
    avg_prompt = sum(_safe_float(r.get("prompt_similarity")) for r in enriched) / float(len(enriched))
    genre_hit_threshold = 0.10
    facet_hit_threshold = 0.15
    genre_hits = 0
    facet_hits = 0
    total_facet_alignment = 0.0
    total_facet_coverage = 0.0
    total_mixed_coverage = 0.0
    mixed_hits = 0
    mixed_hit_threshold = 0.25
    mixed_hard_focus_weight = 0.20
    mixed_hard_cov_weight = 0.80
    use_hard_facet_cov = any(
        _clamp01((row or {}).get("hard_facet_token_coverage", 0.0)) > 0.0
        for row in enriched
        if isinstance(row, dict)
    )
    for row in enriched:
        genre_score = _clamp01(row.get("genre_score", 0.0))
        vibe_score = _clamp01(row.get("vibe_score", 0.0))
        mood_score = _clamp01(row.get("mood_score", 0.0))
        facet_token_coverage = _clamp01(row.get("facet_token_coverage", genre_score))
        hard_facet_token_coverage = _clamp01(
            row.get("hard_facet_token_coverage", facet_token_coverage)
        )
        facet_alignment = _clamp01(
            row.get("facet_alignment_score", row.get("facet_score", 0.0))
        )
        soft_cov = _clamp01(row.get("soft_title_facet_coverage", row.get("soft_facet_token_coverage", 0.0)))
        soft_focus = _clamp01(row.get("soft_focus_match", 0.0))
        soft_signal = _clamp01(row.get("mixed_soft_signal", 0.0))
        hard_focus = _clamp01(row.get("hard_focus_match", 0.0))
        mixed_cov = _clamp01(
            row.get(
                "mixed_intent_coverage",
                min(
                    max(
                        mixed_hard_focus_weight * hard_focus,
                        mixed_hard_cov_weight * hard_facet_token_coverage,
                    ),
                    max(soft_cov, soft_focus, soft_signal),
                ),
            )
        )
        total_facet_alignment += facet_alignment
        total_facet_coverage += facet_token_coverage
        total_mixed_coverage += mixed_cov
        effective_genre_cov = hard_facet_token_coverage if use_hard_facet_cov else facet_token_coverage
        if effective_genre_cov >= genre_hit_threshold:
            genre_hits += 1
        if max(facet_alignment, facet_token_coverage, vibe_score, mood_score) >= facet_hit_threshold:
            facet_hits += 1
        if mixed_cov >= mixed_hit_threshold:
            mixed_hits += 1
    payload["query_metrics"] = {
        "result_count": len(enriched),
        "avg_query_fit": round(avg_query_fit, 6),
        "top1_query_fit": round(_safe_float(enriched[0].get("query_fit")), 6),
        "avg_semantic_similarity": round(avg_sem, 6),
        "avg_token_overlap": round(avg_tok, 6),
        "avg_prompt_similarity": round(avg_prompt, 6),
        "genre_hit_rate_at_k": round(genre_hits / float(len(enriched)), 6),
        "facet_hit_rate_at_k": round(facet_hits / float(len(enriched)), 6),
        "avg_facet_alignment": round(total_facet_alignment / float(len(enriched)), 6),
        "avg_facet_token_coverage": round(total_facet_coverage / float(len(enriched)), 6),
        "avg_mixed_intent_coverage": round(total_mixed_coverage / float(len(enriched)), 6),
        "mixed_intent_hit_rate_at_k": round(mixed_hits / float(len(enriched)), 6),
    }
    for key, value in existing_query_metrics.items():
        if key not in payload["query_metrics"]:
            payload["query_metrics"][key] = value
    return payload


def _run_bot_validation(steps=5, k=10):
    """
    Run bot validation.
    
    This function implements the run bot validation step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    try:
        from src.recommendation.recommendation_engine import ColdStartHandler
    except Exception as exc:
        return {
            "enabled": False,
            "error": f"Could not import recommendation engine: {exc}",
            "bots": {},
            "macro_mrr": 0.0,
        }

    try:
        handler = ColdStartHandler()
    except Exception as exc:
        return {
            "enabled": False,
            "error": f"Could not initialize ColdStartHandler: {exc}",
            "bots": {},
            "macro_mrr": 0.0,
        }

    if not getattr(handler, "metadata", None):
        return {
            "enabled": False,
            "error": "Recommendation metadata not loaded.",
            "bots": {},
            "macro_mrr": 0.0,
        }

    bots = [
        {"name": "2015 Fan", "type": "era", "target": 2015, "tol": 1},
        {"name": "Pop Fan", "type": "mood", "target": "pop"},
        {"name": "Rock Fan", "type": "mood", "target": "rock"},
    ]
    steps = max(1, int(steps))
    k = max(1, int(k))
    bot_scores = {}

    for bot in bots:
        history = []
        mrr_scores = []
        hits = 0
        for _ in range(steps):
            recs = handler.recommend(liked_ids=history, played_ids=[], k=k)
            relevant_indices = []
            for i, item in enumerate(recs):
                track_id = str(item.get("id", "") or "")
                if not track_id:
                    continue
                meta = handler.sim_manager.get_track_info(track_id)
                relevant = False
                if bot["type"] == "era":
                    year = meta.get("year", 0)
                    try:
                        if abs(int(year) - int(bot["target"])) <= int(bot["tol"]):
                            relevant = True
                    except Exception:
                        relevant = False
                elif bot["type"] == "mood":
                    tags = meta.get("deezer_tags", [])
                    if isinstance(tags, str):
                        try:
                            tags = json.loads(tags.replace("'", '"'))
                        except Exception:
                            tags = []
                    relevant = any(
                        str(bot["target"]).lower() in str(tag).lower()
                        for tag in (tags or [])
                    )
                if relevant:
                    relevant_indices.append(i)

            if relevant_indices:
                rank = relevant_indices[0] + 1
                mrr_scores.append(1.0 / float(rank))
                hits += 1
                history.append(str(recs[relevant_indices[0]].get("id", "") or ""))
            else:
                mrr_scores.append(0.0)
                if recs:
                    history.append(str(recs[0].get("id", "") or ""))

        avg_mrr = (sum(mrr_scores) / float(len(mrr_scores))) if mrr_scores else 0.0
        bot_scores[bot["name"]] = {
            "mrr": round(avg_mrr, 6),
            "hit_rate": round(hits / float(steps), 6),
            "steps": steps,
            "k": k,
        }

    macro_mrr = (
        sum(_safe_float(v.get("mrr", 0.0)) for v in bot_scores.values()) / float(len(bot_scores))
        if bot_scores
        else 0.0
    )
    return {
        "enabled": True,
        "error": "",
        "steps": steps,
        "k": k,
        "bots": bot_scores,
        "macro_mrr": round(macro_mrr, 6),
    }


def _hide_fallback_metric(payload):
    """
    Execute hide fallback metric.
    
    This function implements the hide fallback metric step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    if not isinstance(payload, dict):
        return payload
    cleaned = copy.deepcopy(payload)
    cleaned.pop("fallback_lexical_used", None)
    search_metrics = cleaned.get("search_metrics")
    if isinstance(search_metrics, dict):
        query_intent = search_metrics.get("query_intent")
        if isinstance(query_intent, dict):
            query_intent.pop("fallback_lexical_used", None)
    return cleaned


def _print_formatted(payload, show_ranking_score=False):
    """
    Execute print formatted.
    
    This function implements the print formatted step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    query = str(payload.get("query", "") or "")
    results = payload.get("results") if isinstance(payload, dict) else []
    results = results if isinstance(results, list) else []

    print(f"Query: {query}")
    print(f"Results: {len(results)}")
    fetched_batches = payload.get("fetched_batches")
    scanned_results = payload.get("scanned_results")
    if fetched_batches is not None and scanned_results is not None:
        print(f"Scanned: {scanned_results} candidates across {fetched_batches} batch(es)")
    query_metrics = payload.get("query_metrics", {})
    if isinstance(query_metrics, dict) and query_metrics.get("result_count", 0):
        print(
            "Query-fit: "
            f"avg={100.0 * _safe_float(query_metrics.get('avg_query_fit', 0.0)):.1f}% "
            f"top1={100.0 * _safe_float(query_metrics.get('top1_query_fit', 0.0)):.1f}% "
            f"semantic={100.0 * _safe_float(query_metrics.get('avg_semantic_similarity', 0.0)):.1f}% "
            f"token_overlap={100.0 * _safe_float(query_metrics.get('avg_token_overlap', 0.0)):.1f}% "
            f"genre_hit@k={100.0 * _safe_float(query_metrics.get('genre_hit_rate_at_k', 0.0)):.1f}% "
            f"facet_cov={100.0 * _safe_float(query_metrics.get('avg_facet_token_coverage', 0.0)):.1f}% "
            f"mixed_hit@k={100.0 * _safe_float(query_metrics.get('mixed_intent_hit_rate_at_k', 0.0)):.1f}%"
        )
    main_metrics = []
    top1_qf = (
        _safe_float(query_metrics.get("top1_query_fit", 0.0), 0.0)
        if isinstance(query_metrics, dict)
        else 0.0
    )
    if top1_qf > 0.0:
        main_metrics.append(f"Top1 Query-Fit: {100.0 * top1_qf:.2f}%")
    bot_validation_main = payload.get("bot_validation")
    if isinstance(bot_validation_main, dict) and bool(bot_validation_main.get("enabled")):
        main_metrics.append(
            f"Macro MRR: {_safe_float(bot_validation_main.get('macro_mrr', 0.0)):.4f}"
        )
    if main_metrics:
        print("Main Metrics: " + " | ".join(main_metrics))
    search_metrics = payload.get("search_metrics")
    if isinstance(search_metrics, dict):
        query_intent = search_metrics.get("query_intent", {})
        if isinstance(query_intent, dict):
            active_bots = query_intent.get("active_bots", [])
            if isinstance(active_bots, list) and active_bots:
                top_bot = active_bots[0] if isinstance(active_bots[0], dict) else {}
                print(
                    "Live bot router: "
                    f"{str(top_bot.get('label', top_bot.get('bot', 'unknown')))} "
                    f"({100.0 * _safe_float(top_bot.get('confidence', 0.0)):.1f}% confidence)"
                )
    print("-" * 80)

    if not results:
        print("No results returned.")
    else:
        for idx, row in enumerate(results, start=1):
            if not isinstance(row, dict):
                continue
            title = str(row.get("title", "Unknown") or "Unknown")
            artist = str(row.get("artist", "Unknown") or "Unknown")
            year = row.get("year", "")
            track_id = row.get("id", row.get("track_id", ""))
            query_fit = 100.0 * _safe_float(row.get("query_fit", 0.0))
            semantic = 100.0 * _safe_float(row.get("semantic_similarity", 0.0))
            overlap = 100.0 * _safe_float(row.get("token_overlap", 0.0))
            prompt_sim = 100.0 * _safe_float(row.get("prompt_similarity", 0.0))
            print(f"{idx}. {title} - {artist}")
            print(
                f"   id={track_id} year={year} "
                f"query_fit={query_fit:.1f}% semantic={semantic:.1f}% "
                f"token_overlap={overlap:.1f}% prompt_similarity={prompt_sim:.1f}%"
            )
            bot_profile = str(row.get("bot_profile", "") or "").strip()
            if bot_profile:
                bot_match = 100.0 * _safe_float(row.get("bot_profile_score", 0.0))
                print(f"   bot_profile={bot_profile} bot_match={bot_match:.1f}%")
            if show_ranking_score:
                print(f"   ranking_score={_safe_float(row.get('score', 0.0)):.6f}")

    bot_validation = payload.get("bot_validation")
    if isinstance(bot_validation, dict):
        print("-" * 80)
        print("Automated Bot Validation")
        if not bot_validation.get("enabled"):
            print(str(bot_validation.get("error", "Bot validation unavailable.")))
        else:
            print(
                f"Macro MRR: {_safe_float(bot_validation.get('macro_mrr', 0.0)):.4f} "
                f"(steps={int(bot_validation.get('steps', 0) or 0)}, "
                f"k={int(bot_validation.get('k', 0) or 0)})"
            )
            bots = bot_validation.get("bots", {})
            if isinstance(bots, dict):
                for name, score_row in bots.items():
                    if not isinstance(score_row, dict):
                        continue
                    print(
                        f"{name}: "
                        f"MRR={_safe_float(score_row.get('mrr', 0.0)):.4f} "
                        f"HitRate={100.0 * _safe_float(score_row.get('hit_rate', 0.0)):.2f}%"
                    )


def _build_effective_query(free_text="", artist="", song=""):
    """
    Build effective query.
    
    This function implements the build effective query step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    free_text = str(free_text or "").strip()
    artist = str(artist or "").strip()
    song = str(song or "").strip()
    if free_text:
        return free_text
    parts = [song, artist]
    return " ".join([p for p in parts if p]).strip()


def _fetch_results(runner, query, limit, offset):
    """
    Execute fetch results.
    
    This function implements the fetch results step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    payload = runner.fetch(query=query, limit=limit, offset=offset)
    if not isinstance(payload, dict):
        payload = {"query": query, "results": []}
    return payload


def main():
    """
    Run the command entry point.
    
    This function implements the main step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    args = _build_parser().parse_args()

    query = _build_effective_query(
        free_text=args.query,
        artist=args.artist,
        song=args.song,
    )
    if not query:
        raise SystemExit("Provide a query or use --artist/--song.")

    limit = max(1, int(args.limit))
    offset = max(0, int(args.offset))
    runner = _ApiRunner() if bool(args.use_api) else _DirectRunner()

    payload = _fetch_results(
        runner=runner,
        query=query,
        limit=limit,
        offset=offset,
    )
    payload = _attach_query_metrics(payload)

    if bool(args.run_bot_validation):
        payload["bot_validation"] = _run_bot_validation(
            steps=max(1, int(args.bot_steps)),
            k=max(1, int(args.bot_k)),
        )

    display_payload = _hide_fallback_metric(payload)

    if args.json:
        print(json.dumps(display_payload, indent=2, ensure_ascii=False))
        return

    warning = payload.get("warning") if isinstance(payload, dict) else None
    error = payload.get("error") if isinstance(payload, dict) else None
    if warning:
        print(f"[WARN] {warning}")
    if error:
        print(f"[ERROR] {error}")

    _print_formatted(
        display_payload if isinstance(display_payload, dict) else {"query": query, "results": []},
        show_ranking_score=bool(args.show_ranking_score),
    )


if __name__ == "__main__":
    main()