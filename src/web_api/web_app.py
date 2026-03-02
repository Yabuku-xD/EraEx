import base64
import heapq
import hashlib
import json
import logging
import math
import os
import re
import secrets
import socket
import sys
import threading
import time
from datetime import timedelta
import urllib.request
import urllib.parse
from pathlib import Path
from flask import Flask, jsonify, render_template, request, has_request_context, session
from werkzeug.security import generate_password_hash, check_password_hash
from config import settings
from src.core.lazy_loading import LazyLoader, preload_in_background
from src.core.lyrics_provider_engine import LyricsProviderEngine
from src.core.media_metadata import (
    SpotifyCoverResolver,
    VideoResolver,
    YouTubeMetadataEnricher,
    is_youtube_id,
    thumbnail_candidates,
)
from src.user_profiles.user_profile_store import UserProfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
app = Flask(
    __name__,
    template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), "../templates")),
    static_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), "../static")),
)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.secret_key = str(os.getenv("FLASK_SECRET_KEY", "eraex-dev-secret-change-me") or "").strip()
app.permanent_session_lifetime = timedelta(days=45)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = str(
    os.getenv("SESSION_COOKIE_SECURE", "0") or "0"
).strip().lower() in {"1", "true", "yes", "y", "on"}


@app.after_request
def _set_referrer_policy_header(response):
    """
    Set referrer policy header.
    
    This function implements the set referrer policy header step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return response


user_profile = UserProfile()
video_resolver = VideoResolver()
spotify_cover_resolver = SpotifyCoverResolver()
metadata_enricher = YouTubeMetadataEnricher()
lyrics_provider_engine = LyricsProviderEngine()
_metadata_cache = {}
_metadata_by_video_id_cache = {}
_metadata_trending_cache = []
_metadata_artist_index_cache = {}
_metadata_year_index_cache = {}
_metadata_long_tail_cache = []
_metadata_popularity_pct_cache = {}
_reco_cache_store = None
_reco_cache_lock = threading.Lock()
_RECO_CACHE_PATH = settings.CACHE_DIR / "recommendation_cache.json"
_RECO_CACHE_TTL_SEC = max(60, int(os.getenv("RECO_CACHE_TTL_SEC", "1800") or "1800"))
_RECO_CACHE_MAX_USERS = max(20, int(os.getenv("RECO_CACHE_MAX_USERS", "500") or "500"))
_MAX_RECO_COVER_ENRICH_PER_REQUEST = max(
    1, int(os.getenv("RECO_COVER_ENRICH_PER_REQUEST", "12") or "12")
)
_lyrics_cache = {}
_LYRICS_CACHE_TTL_SEC = 60 * 60 * 24
_LYRICS_CACHE_MAX = 5000
_LYRICS_CACHE_ENABLED = str(os.getenv("LYRICS_CACHE_ENABLED", "0") or "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
_lyrics_provider_timeout_streak = 0
_lyrics_provider_backoff_until = 0.0
_LYRICS_PROVIDER_TIMEOUT_BACKOFF_SEC = 90
_LYRICA_BASE_URL = str(os.getenv("LYRICA_BASE_URL", "") or "").strip().rstrip("/")
_LYRICA_ENABLED = bool(_LYRICA_BASE_URL)
_LYRICA_TIMEOUT_SEC = max(1.0, float(os.getenv("LYRICA_TIMEOUT_SEC", "20.0") or 20.0))
_LYRICA_SEQUENCE = str(os.getenv("LYRICA_SEQUENCE", "") or "").strip()
# Lyrica docs sequence (current): 2=LrcLib, 3=YouTube Music. Default to LrcLib for timestamped lyrics.
_LYRICA_TIMESTAMP_SEQUENCE = str(os.getenv("LYRICA_TIMESTAMP_SEQUENCE", "2") or "2").strip()
_LYRICA_PLAIN_SEQUENCE = str(os.getenv("LYRICA_PLAIN_SEQUENCE", "3,1,5,6") or "3,1,5,6").strip()
_LYRICA_EXTRA_BASE_URLS = [
    str(v).strip().rstrip("/")
    for v in str(os.getenv("LYRICA_EXTRA_BASE_URLS", "") or "").split(",")
    if str(v).strip()
]
_LYRICS_LOOKUP_BUDGET_SEC = max(2.0, float(os.getenv("LYRICS_LOOKUP_BUDGET_SEC", "26.0") or 26.0))
_CLICK_DEBUG_ENABLED = str(os.getenv("CLICK_DEBUG_LOGS", "1") or "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
_SPOTIFY_PLAYBACK_ENABLED = bool(spotify_cover_resolver.enabled) and str(
    os.getenv("SPOTIFY_PLAYBACK_ENABLED", "1") or "1"
).strip().lower() in {"1", "true", "yes", "y", "on"}
_SPOTIFY_REDIRECT_URI = str(
    os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:5000/spotify/callback") or ""
).strip()
_SPOTIFY_AUTH_SCOPES = str(
    os.getenv(
        "SPOTIFY_AUTH_SCOPES",
        "streaming user-read-email user-read-private user-modify-playback-state user-read-playback-state",
    )
    or ""
).strip()
_SPOTIFY_AUTH_TIMEOUT_SEC = max(
    2.0, float(os.getenv("SPOTIFY_AUTH_TIMEOUT_SEC", "8.0") or 8.0)
)
_SHARED_USER_ID = str(os.getenv("SHARED_USER_ID", "") or "").strip()
_SHARED_USER_ID_FORCE = str(os.getenv("SHARED_USER_ID_FORCE", "0") or "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
_spotify_oauth_pending_state = ""
_spotify_oauth_pending_created_at = 0.0
_spotify_user_token = {
    "access_token": "",
    "refresh_token": "",
    "expires_at": 0.0,
    "scope": "",
    "token_type": "",
}


def _short(value, max_len=120):
    """
    Execute short.
    
    This function implements the short step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _http_json(url, *, method="GET", headers=None, data=None, timeout_sec=8.0):
    """
    Execute http json.
    
    This function implements the http json step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    req = urllib.request.Request(
        url,
        data=data,
        method=str(method or "GET").upper(),
        headers=headers or {},
    )
    with urllib.request.urlopen(req, timeout=float(timeout_sec)) as response:
        body = response.read().decode("utf-8", errors="ignore")
        if not body:
            return {}
        return json.loads(body)


def _spotify_playback_available():
    """
    Execute spotify playback available.
    
    This function implements the spotify playback available step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    redirect_uri = _spotify_redirect_uri()
    return bool(
        _SPOTIFY_PLAYBACK_ENABLED
        and spotify_cover_resolver.enabled
        and redirect_uri
    )


def _spotify_redirect_uri():
    """
    Execute spotify redirect uri.
    
    This function implements the spotify redirect uri step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    if _SPOTIFY_REDIRECT_URI:
        return _SPOTIFY_REDIRECT_URI
    if has_request_context():
        try:
            root = str(request.url_root or "").strip().rstrip("/")
            if root:
                return f"{root}/spotify/callback"
        except Exception:
            pass
    return "http://127.0.0.1:5000/spotify/callback"


def _spotify_token_is_valid():
    """
    Execute spotify token is valid.
    
    This function implements the spotify token is valid step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    token = str(_spotify_user_token.get("access_token") or "").strip()
    exp = float(_spotify_user_token.get("expires_at") or 0.0)
    return bool(token) and time.time() < (exp - 30.0)


def _spotify_store_token_payload(payload):
    """
    Execute spotify store token payload.
    
    This function implements the spotify store token payload step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    if not isinstance(payload, dict):
        return False
    access_token = str(payload.get("access_token") or "").strip()
    if not access_token:
        return False
    expires_in = max(60, int(payload.get("expires_in") or 3600))
    _spotify_user_token["access_token"] = access_token
    refresh_token = str(payload.get("refresh_token") or "").strip()
    if refresh_token:
        _spotify_user_token["refresh_token"] = refresh_token
    _spotify_user_token["expires_at"] = time.time() + expires_in
    _spotify_user_token["scope"] = str(payload.get("scope") or _spotify_user_token.get("scope") or "")
    _spotify_user_token["token_type"] = str(payload.get("token_type") or "Bearer")
    return True


def _spotify_exchange_code_for_token(code):
    """
    Execute spotify exchange code for token.
    
    This function implements the spotify exchange code for token step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    if not _spotify_playback_available():
        return False
    client_id = spotify_cover_resolver.client_id
    client_secret = spotify_cover_resolver.client_secret
    raw = f"{client_id}:{client_secret}".encode("utf-8")
    basic = base64.b64encode(raw).decode("ascii")
    redirect_uri = _spotify_redirect_uri()
    body = urllib.parse.urlencode(
        {
            "grant_type": "authorization_code",
            "code": str(code or ""),
            "redirect_uri": redirect_uri,
        }
    ).encode("utf-8")
    try:
        payload = _http_json(
            "https://accounts.spotify.com/api/token",
            method="POST",
            headers={
                "Authorization": f"Basic {basic}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data=body,
            timeout_sec=_SPOTIFY_AUTH_TIMEOUT_SEC,
        )
    except Exception:
        return False
    return _spotify_store_token_payload(payload)


def _spotify_refresh_user_access_token():
    """
    Execute spotify refresh user access token.
    
    This function implements the spotify refresh user access token step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    if not _spotify_playback_available():
        return ""
    if _spotify_token_is_valid():
        return str(_spotify_user_token.get("access_token") or "")
    refresh_token = str(_spotify_user_token.get("refresh_token") or "").strip()
    if not refresh_token:
        return ""
    client_id = spotify_cover_resolver.client_id
    client_secret = spotify_cover_resolver.client_secret
    raw = f"{client_id}:{client_secret}".encode("utf-8")
    basic = base64.b64encode(raw).decode("ascii")
    body = urllib.parse.urlencode(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
    ).encode("utf-8")
    try:
        payload = _http_json(
            "https://accounts.spotify.com/api/token",
            method="POST",
            headers={
                "Authorization": f"Basic {basic}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data=body,
            timeout_sec=_SPOTIFY_AUTH_TIMEOUT_SEC,
        )
    except Exception:
        return ""
    if not _spotify_store_token_payload(payload):
        return ""
    return str(_spotify_user_token.get("access_token") or "")


def _spotify_user_token_payload():
    """
    Execute spotify user token payload.
    
    This function implements the spotify user token payload step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    access_token = _spotify_refresh_user_access_token()
    if not access_token:
        return None
    return {
        "access_token": access_token,
        "expires_at": int(float(_spotify_user_token.get("expires_at") or 0.0)),
        "scope": str(_spotify_user_token.get("scope") or ""),
        "token_type": str(_spotify_user_token.get("token_type") or "Bearer"),
    }


def _debug_log(event_name, **fields):
    """
    Execute debug log.
    
    This function implements the debug log step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    if not _CLICK_DEBUG_ENABLED:
        return
    try:
        parts = []
        for key, value in fields.items():
            if isinstance(value, (list, tuple)):
                rendered = "[" + ", ".join(_short(v, 24) for v in list(value)[:6]) + "]"
            elif isinstance(value, dict):
                rendered = "{" + ", ".join(
                    f"{k}={_short(v, 24)}" for k, v in list(value.items())[:6]
                ) + "}"
            else:
                rendered = _short(value, 120)
            parts.append(f"{key}={rendered}")
        logging.info("[click-debug] %s | %s", event_name, " | ".join(parts))
    except Exception:
        logging.info("[click-debug] %s", event_name)


def _is_missing_or_fallback_cover(value):
    """
    Return whether missing or fallback cover.
    
    This function implements the is missing or fallback cover step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    text = str(value or "").strip().lower()
    if not text:
        return True
    return "cover-fallback.svg" in text or text.endswith("/no-image") or "/placeholder" in text


def _is_youtube_thumb_url(value):
    """
    Return whether youtube thumb url.
    
    This function implements the is youtube thumb url step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    text = str(value or "").strip().lower()
    return "i.ytimg.com/vi/" in text


def _dedupe_urls(values):
    """
    Deduplicate urls.
    
    This function implements the dedupe urls step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    seen = set()
    out = []
    for value in values or []:
        url = str(value or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out


def _spotify_cover_lookup(title="", artist="", track_id=""):
    """
    Execute spotify cover lookup.
    
    This function implements the spotify cover lookup step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    if not spotify_cover_resolver.enabled:
        return None
    try:
        return spotify_cover_resolver.resolve_cover(
            title=str(title or ""),
            artist=str(artist or ""),
            track_id=str(track_id or ""),
        )
    except Exception:
        return None


def _merge_cover_candidates(existing_candidates, *groups):
    """
    Merge cover candidates.
    
    This function implements the merge cover candidates step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    merged = []
    for group in groups:
        if isinstance(group, list):
            merged.extend(group)
        elif group:
            merged.append(group)
    if isinstance(existing_candidates, list):
        merged.extend(existing_candidates)
    return _dedupe_urls(merged)


def _enrich_missing_recommendation_covers(rows):
    """
    Execute enrich missing recommendation covers.
    
    This function implements the enrich missing recommendation covers step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    if not isinstance(rows, list) or not rows:
        return rows
    if not spotify_cover_resolver.enabled:
        return rows

    enriched_count = 0
    for row in rows:
        if enriched_count >= _MAX_RECO_COVER_ENRICH_PER_REQUEST:
            break
        if not isinstance(row, dict):
            continue

        cover_url = row.get("cover_url")
        existing_candidates = row.get("cover_candidates") or row.get("thumbnail_candidates") or []
        has_valid_candidate = False
        has_non_youtube_candidate = False
        if isinstance(existing_candidates, list):
            for candidate in existing_candidates:
                if not _is_missing_or_fallback_cover(candidate):
                    has_valid_candidate = True
                    if not _is_youtube_thumb_url(candidate):
                        has_non_youtube_candidate = True

        thumb_url = row.get("thumbnail")
        primary_missing = _is_missing_or_fallback_cover(cover_url) or _is_missing_or_fallback_cover(
            thumb_url
        )

        if (not primary_missing) and has_valid_candidate and has_non_youtube_candidate:
            continue

        spotify_cover = _spotify_cover_lookup(
            title=str(row.get("title", "") or ""),
            artist=str(row.get("artist", "") or row.get("artist_name", "") or ""),
            track_id=str(row.get("track_id", "") or row.get("id", "") or ""),
        )
        if not isinstance(spotify_cover, dict):
            continue
        resolved_thumb = str(spotify_cover.get("thumbnail", "") or "")
        resolved_candidates = (
            spotify_cover.get("thumbnail_candidates")
            if isinstance(spotify_cover.get("thumbnail_candidates"), list)
            else []
        )
        if not resolved_thumb and not resolved_candidates:
            continue
        if resolved_thumb:
            row["cover_url"] = resolved_thumb
            row["thumbnail"] = resolved_thumb

        merged_candidates = _merge_cover_candidates(
            existing_candidates,
            resolved_candidates,
            resolved_thumb,
        )
        if merged_candidates:
            row["cover_candidates"] = merged_candidates
        if resolved_thumb or merged_candidates:
            enriched_count += 1

    return rows


def _lookup_track_meta_by_song_id(song_id):
    """
    Execute lookup track meta by song id.
    
    This function implements the lookup track meta by song id step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    sid = str(song_id or "").strip()
    if not sid:
        return "", {}
    metadata = _metadata_cache or _load_metadata() or {}
    if sid in metadata and isinstance(metadata.get(sid), dict):
        return sid, metadata.get(sid) or {}
    if is_youtube_id(sid):
        mapped_track_id = str((_metadata_by_video_id_cache or {}).get(sid) or "").strip()
        if mapped_track_id and isinstance(metadata.get(mapped_track_id), dict):
            return mapped_track_id, metadata.get(mapped_track_id) or {}
    return sid, {}


def _meta_cover_candidates(meta, video_id):
    """
    Execute meta cover candidates.
    
    This function implements the meta cover candidates step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    candidates = []
    raw_candidates = (meta or {}).get("thumbnail_candidates") or (meta or {}).get("cover_candidates") or []
    if isinstance(raw_candidates, list):
        candidates.extend(raw_candidates)
    for key in ("cover_url", "thumbnail", "album_cover"):
        value = (meta or {}).get(key)
        if value:
            candidates.append(value)
    if is_youtube_id(video_id):
        candidates.extend(thumbnail_candidates(video_id, None))
    return _dedupe_urls(candidates)


def _serialize_profile_track(song_id, source="profile_list"):
    """
    Execute serialize profile track.
    
    This function implements the serialize profile track step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    resolved_track_id, meta = _lookup_track_meta_by_song_id(song_id)
    meta = meta if isinstance(meta, dict) else {}

    meta_video_id = str(meta.get("video_id") or "").strip()
    if is_youtube_id(meta_video_id):
        video_id = meta_video_id
    elif is_youtube_id(song_id):
        video_id = str(song_id)
    else:
        video_id = ""

    title = str(meta.get("title") or song_id or "Unknown")
    artist = str(meta.get("artist_name") or meta.get("artist") or "Unknown")
    description = str(meta.get("description") or "")
    cover_candidates = _meta_cover_candidates(meta, video_id)
    primary_cover = (
        str(meta.get("cover_url") or "").strip()
        or str(meta.get("thumbnail") or "").strip()
        or str(meta.get("album_cover") or "").strip()
        or (cover_candidates[0] if cover_candidates else "")
    )

    return {
        "id": str(resolved_track_id or song_id or ""),
        "track_id": str(resolved_track_id or song_id or ""),
        "video_id": str(video_id or ""),
        "title": title,
        "artist": artist,
        "artist_name": artist,
        "description": description,
        "instrumental": meta.get("instrumental"),
        "instrumental_confidence": float(meta.get("instrumental_confidence", 0.0) or 0.0),
        "thumbnail": str(meta.get("thumbnail") or primary_cover or ""),
        "album_cover": str(meta.get("album_cover") or ""),
        "cover_url": str(primary_cover or ""),
        "cover_candidates": cover_candidates,
        "source": str(source or "profile_list"),
    }


def _profile_rows_from_song_ids(song_ids, source="profile_list", limit=50, enrich_covers=False):
    """
    Execute profile rows from song ids.
    
    This function implements the profile rows from song ids step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    rows = []
    seen = set()
    max_rows = max(1, int(limit or 50))
    for raw_song_id in list(song_ids or []):
        song_id = str(raw_song_id or "").strip()
        if not song_id or song_id in seen:
            continue
        seen.add(song_id)
        rows.append(_serialize_profile_track(song_id, source=source))
        if len(rows) >= max_rows:
            break
    if enrich_covers and rows:
        rows = _enrich_missing_recommendation_covers(rows)
    return rows


def _ordered_unique_song_ids(*groups):
    """
    Execute ordered unique song ids.
    
    This function implements the ordered unique song ids step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    seen = set()
    out = []
    for group in groups:
        for raw_id in list(group or []):
            sid = str(raw_id or "").strip()
            if not sid or sid in seen:
                continue
            seen.add(sid)
            out.append(sid)
    return out


def _recommendation_history_counts(liked, played, disliked, playlist_tracks, skip_summary):
    """
    Execute recommendation history counts.
    
    This function implements the recommendation history counts step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    return {
        "likes": len(liked or []),
        "plays": len(played or []),
        "dislikes": len(disliked or []),
        "playlist_tracks": len(playlist_tracks or []),
        "skip_next": int(sum((skip_summary or {}).get("next_counts", {}).values())),
        "skip_prev": int(sum((skip_summary or {}).get("prev_counts", {}).values())),
        "skip_early": int(sum((skip_summary or {}).get("early_next_counts", {}).values())),
    }


def _recommendation_signal_fingerprint(
    liked,
    played,
    disliked,
    playlist_tracks,
    skip_summary,
):
    """
    Execute recommendation signal fingerprint.
    
    This function implements the recommendation signal fingerprint step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    payload = {
        "liked": [str(song_id or "").strip() for song_id in list(liked or [])[:120]],
        "played": [str(song_id or "").strip() for song_id in list(played or [])[:120]],
        "disliked": [str(song_id or "").strip() for song_id in list(disliked or [])[:120]],
        "playlist": [str(song_id or "").strip() for song_id in list(playlist_tracks or [])[:180]],
        "skip_next": sorted(
            (str(song_id or "").strip(), int(count or 0))
            for song_id, count in dict((skip_summary or {}).get("next_counts") or {}).items()
            if str(song_id or "").strip() and int(count or 0) > 0
        )[:100],
        "skip_prev": sorted(
            (str(song_id or "").strip(), int(count or 0))
            for song_id, count in dict((skip_summary or {}).get("prev_counts") or {}).items()
            if str(song_id or "").strip() and int(count or 0) > 0
        )[:100],
        "skip_early": sorted(
            (str(song_id or "").strip(), int(count or 0))
            for song_id, count in dict((skip_summary or {}).get("early_next_counts") or {}).items()
            if str(song_id or "").strip() and int(count or 0) > 0
        )[:100],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _recommendation_cache_load_locked():
    """
    Execute recommendation cache load locked.
    
    This function implements the recommendation cache load locked step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    global _reco_cache_store
    if isinstance(_reco_cache_store, dict):
        users = _reco_cache_store.get("users")
        if isinstance(users, dict):
            return _reco_cache_store
    payload = {"users": {}}
    try:
        if _RECO_CACHE_PATH.exists():
            loaded = json.loads(_RECO_CACHE_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                users = loaded.get("users")
                if isinstance(users, dict):
                    payload = {"users": users}
    except Exception:
        payload = {"users": {}}
    _reco_cache_store = payload
    return _reco_cache_store


def _recommendation_cache_prune_locked(now_ts):
    """
    Execute recommendation cache prune locked.
    
    This function implements the recommendation cache prune locked step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    store = _recommendation_cache_load_locked()
    users = store.get("users")
    if not isinstance(users, dict):
        store["users"] = {}
        users = store["users"]

    stale_users = []
    for user_id, per_user in list(users.items()):
        if not isinstance(per_user, dict):
            stale_users.append(user_id)
            continue
        keep_modes = {}
        for mode_key in ("fast", "deep"):
            entry = per_user.get(mode_key)
            if not isinstance(entry, dict):
                continue
            ts = float(entry.get("ts", 0.0) or 0.0)
            recs = entry.get("recommendations")
            if not isinstance(recs, list) or not recs:
                continue
            if now_ts - ts > float(_RECO_CACHE_TTL_SEC):
                continue
            keep_modes[mode_key] = entry
        if keep_modes:
            users[user_id] = keep_modes
        else:
            stale_users.append(user_id)

    for user_id in stale_users:
        users.pop(user_id, None)

    if len(users) > int(_RECO_CACHE_MAX_USERS):
        ranked_users = sorted(
            users.items(),
            key=lambda item: max(
                float((item[1].get("fast") or {}).get("ts", 0.0) or 0.0),
                float((item[1].get("deep") or {}).get("ts", 0.0) or 0.0),
            ),
            reverse=True,
        )
        keep_ids = {str(uid) for uid, _entry in ranked_users[: int(_RECO_CACHE_MAX_USERS)]}
        for uid in list(users.keys()):
            if uid not in keep_ids:
                users.pop(uid, None)


def _recommendation_cache_save_locked():
    """
    Execute recommendation cache save locked.
    
    This function implements the recommendation cache save locked step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    store = _recommendation_cache_load_locked()
    try:
        settings.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = _RECO_CACHE_PATH.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(store, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        tmp_path.replace(_RECO_CACHE_PATH)
    except Exception:
        pass


def _recommendation_cache_get(
    user_id,
    cache_kind,
    n,
    signal_fingerprint,
    allow_cross_mode_fallback=True,
):
    """
    Execute recommendation cache get.
    
    This function implements the recommendation cache get step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    uid = str(user_id or "").strip()
    if not uid:
        return None
    requested_n = max(1, int(n or 1))
    preferred = "deep" if str(cache_kind or "").strip().lower() == "deep" else "fast"
    order = [preferred]
    if preferred == "deep" and bool(allow_cross_mode_fallback):
        order.append("fast")
    now_ts = float(time.time())
    with _reco_cache_lock:
        _recommendation_cache_prune_locked(now_ts)
        store = _recommendation_cache_load_locked()
        per_user = (store.get("users") or {}).get(uid)
        if not isinstance(per_user, dict):
            return None
        for mode_key in order:
            entry = per_user.get(mode_key)
            if not isinstance(entry, dict):
                continue
            entry_fp = str(entry.get("signal_fingerprint") or "").strip()
            if signal_fingerprint and entry_fp and entry_fp != str(signal_fingerprint):
                continue
            recs = entry.get("recommendations")
            if not isinstance(recs, list) or not recs:
                continue
            # Avoid locking pagination into a short cached list. If caller asks for
            # more than this cache entry has, treat as miss so a larger pool can be
            # regenerated and re-cached.
            if len(recs) < requested_n:
                continue
            limited = recs[:requested_n]
            return {
                "user_id": uid,
                "recommendation_mode": str(entry.get("recommendation_mode") or ""),
                "history_counts": dict(entry.get("history_counts") or {}),
                "covers_enriched": False,
                "ranker_ready": bool(entry.get("ranker_ready")),
                "recommendations": limited,
                "engine_quota": dict(entry.get("engine_quota") or {}),
                "cache_hit": True,
                "cache_kind": mode_key,
            }
    return None


def _recommendation_cache_set(
    user_id,
    cache_kind,
    signal_fingerprint,
    recommendation_mode,
    history_counts,
    ranker_ready,
    recommendations,
    engine_quota=None,
):
    """
    Execute recommendation cache set.
    
    This function implements the recommendation cache set step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    uid = str(user_id or "").strip()
    mode_key = "deep" if str(cache_kind or "").strip().lower() == "deep" else "fast"
    recs = list(recommendations or [])
    if not uid or not recs:
        return
    now_ts = float(time.time())
    entry = {
        "ts": now_ts,
        "signal_fingerprint": str(signal_fingerprint or ""),
        "recommendation_mode": str(recommendation_mode or ""),
        "history_counts": dict(history_counts or {}),
        "ranker_ready": bool(ranker_ready),
        "recommendations": recs[: max(24, min(420, int(len(recs) or 0)))],
        "engine_quota": dict(engine_quota or {}),
    }
    with _reco_cache_lock:
        _recommendation_cache_prune_locked(now_ts)
        store = _recommendation_cache_load_locked()
        users = store.setdefault("users", {})
        per_user = users.setdefault(uid, {})
        per_user[mode_key] = entry
        users[uid] = per_user
        _recommendation_cache_prune_locked(now_ts)
        _recommendation_cache_save_locked()


def _meta_year_value(meta):
    """
    Execute meta year value.
    
    This function implements the meta year value step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    item = meta if isinstance(meta, dict) else {}
    for key in ("year", "release_year", "album_year"):
        raw = item.get(key)
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        if text.isdigit():
            year = int(text)
            if 1900 <= year <= 2100:
                return year
            continue
        match = re.search(r"(19|20)\d{2}", text)
        if match:
            year = int(match.group(0))
            if 1900 <= year <= 2100:
                return year
    return 0


def _ensure_metadata_profile_indexes():
    """
    Execute ensure metadata profile indexes.
    
    This function implements the ensure metadata profile indexes step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    global _metadata_artist_index_cache, _metadata_year_index_cache, _metadata_long_tail_cache
    global _metadata_popularity_pct_cache
    if (
        _metadata_artist_index_cache
        and _metadata_year_index_cache
        and _metadata_long_tail_cache
        and _metadata_popularity_pct_cache
    ):
        return

    metadata = _metadata_cache or _load_metadata() or {}
    artist_index = {}
    year_index = {}
    popularity_rows = []
    for raw_song_id, meta in (metadata or {}).items():
        sid = str(raw_song_id or "").strip()
        if not sid:
            continue
        item = meta if isinstance(meta, dict) else {}
        artist = str(item.get("artist_name") or item.get("artist") or "").strip().lower()
        if artist:
            artist_index.setdefault(artist, []).append(sid)
        year = _meta_year_value(item)
        if year > 0:
            year_index.setdefault(int(year), []).append(sid)
        popularity_rows.append((_metadata_popularity_score(item), sid))

    score_by_id = {sid: float(score) for score, sid in popularity_rows}
    for key, ids in artist_index.items():
        artist_index[key] = sorted(ids, key=lambda song_id: score_by_id.get(song_id, 0.0), reverse=True)
    for key, ids in year_index.items():
        year_index[key] = sorted(ids, key=lambda song_id: score_by_id.get(song_id, 0.0), reverse=True)

    ranked = sorted(popularity_rows, key=lambda row: row[0], reverse=True)
    popularity_pct = {}
    if ranked:
        denom = float(max(1, len(ranked) - 1))
        for idx, (_score, sid) in enumerate(ranked):
            popularity_pct[str(sid)] = float(idx) / denom
        head_cut = max(150, int(len(ranked) * 0.25))
        tail = [sid for _score, sid in ranked[head_cut:]]
        if not tail:
            tail = [sid for _score, sid in ranked]
    else:
        tail = []

    _metadata_artist_index_cache = artist_index
    _metadata_year_index_cache = year_index
    _metadata_long_tail_cache = tail
    _metadata_popularity_pct_cache = popularity_pct


def _fast_profile_seed_recommendations(
    liked_ids,
    playlist_track_ids,
    played_ids,
    disliked_ids=None,
    *,
    n=24,
    enrich_covers=False,
):
    """
    Execute fast profile seed recommendations.
    
    This function implements the fast profile seed recommendations step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    global _metadata_popularity_pct_cache
    target = max(1, int(n or 1))
    disliked = {
        str(song_id or "").strip()
        for song_id in list(disliked_ids or [])
        if str(song_id or "").strip()
    }
    # Prioritize explicit signals first, then implicit history.
    seed_ids = _ordered_unique_song_ids(
        liked_ids,
        playlist_track_ids,
        played_ids,
    )
    if disliked:
        seed_ids = [sid for sid in seed_ids if sid not in disliked]
    if not seed_ids:
        return []
    _ensure_metadata_profile_indexes()

    artist_cap_per_seed = max(8, min(26, target // 2))
    year_cap_per_bucket = max(4, min(14, target // 4))
    long_tail_cap = max(target * 3, 180)
    expanded = []
    seen_expanded = set()

    def _push(song_id):
        """
        Execute push.
        
        This function implements the push step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        sid = str(song_id or "").strip()
        if not sid or sid in seen_expanded or sid in disliked:
            return False
        seen_expanded.add(sid)
        expanded.append(sid)
        return len(expanded) >= max(target * 6, 240)

    # Keep original explicit signals first.
    for sid in seed_ids:
        _push(sid)

    metadata = _metadata_cache or _load_metadata() or {}
    for seed_id in seed_ids[:18]:
        _resolved_seed_id, seed_meta = _lookup_track_meta_by_song_id(seed_id)
        seed_meta = seed_meta if isinstance(seed_meta, dict) else {}
        artist_key = str(seed_meta.get("artist_name") or seed_meta.get("artist") or "").strip().lower()
        if artist_key:
            for neighbor_id in (_metadata_artist_index_cache.get(artist_key) or [])[:artist_cap_per_seed]:
                if _push(neighbor_id):
                    break

        year = _meta_year_value(seed_meta)
        if year > 0:
            for yr in range(year - 2, year + 3):
                for neighbor_id in (_metadata_year_index_cache.get(yr) or [])[:year_cap_per_bucket]:
                    if _push(neighbor_id):
                        break
                if len(expanded) >= max(target * 6, 240):
                    break

        # Use same-album cover matches when metadata has explicit album cover identifiers.
        album_cover = str(seed_meta.get("album_cover") or "").strip()
        if album_cover:
            matched = 0
            for candidate_id, candidate_meta in metadata.items():
                if matched >= 12:
                    break
                candidate_sid = str(candidate_id or "").strip()
                if not candidate_sid:
                    continue
                candidate_album = str((candidate_meta or {}).get("album_cover") or "").strip()
                if not candidate_album or candidate_album != album_cover:
                    continue
                if _push(candidate_sid):
                    break
                matched += 1

        if len(expanded) >= max(target * 6, 240):
            break

    if len(expanded) < max(target * 2, 80):
        for sid in (_metadata_long_tail_cache or [])[:long_tail_cap]:
            if _push(sid):
                break

    pool_limit = max(target * 5, min(900, target * 7))
    rows = _profile_rows_from_song_ids(
        expanded,
        source="profile_seed_fast",
        limit=pool_limit,
        enrich_covers=enrich_covers,
    )
    if len(rows) <= target:
        return rows

    seed_artists = set()
    seed_years = []
    for sid in seed_ids[:24]:
        _resolved_seed_id, seed_meta = _lookup_track_meta_by_song_id(sid)
        item = seed_meta if isinstance(seed_meta, dict) else {}
        artist_key = str(item.get("artist_name") or item.get("artist") or "").strip().lower()
        if artist_key:
            seed_artists.add(artist_key)
        year = _meta_year_value(item)
        if year > 0:
            seed_years.append(int(year))

    def _score_profile_row(row):
        """
        Score profile row.
        
        This function implements the score profile row step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        rid = str(row.get("track_id") or row.get("id") or "").strip()
        if not rid:
            return -1e9
        _resolved_row_id, row_meta = _lookup_track_meta_by_song_id(rid)
        item = row_meta if isinstance(row_meta, dict) else {}
        artist_key = str(item.get("artist_name") or item.get("artist") or "").strip().lower()
        year = _meta_year_value(item)
        popularity_pct = float(_metadata_popularity_pct_cache.get(rid, 0.5))
        if rid not in _metadata_popularity_pct_cache and _resolved_row_id:
            popularity_pct = float(_metadata_popularity_pct_cache.get(str(_resolved_row_id), 0.5))

        affinity = 0.0
        if artist_key and artist_key in seed_artists:
            affinity += 1.35
        if seed_years and year > 0:
            min_year_dist = min(abs(int(year) - int(seed_year)) for seed_year in seed_years)
            affinity += max(0.0, 1.0 - 0.18 * float(min_year_dist))

        # Favor long-tail tracks while avoiding overfitting to the global chart head.
        long_tail_bonus = 0.24 + 0.86 * max(0.0, popularity_pct - 0.42)
        mainstream_penalty = 0.62 * max(0.0, (0.24 - popularity_pct) / 0.24)
        return float(affinity + long_tail_bonus - mainstream_penalty)

    rows = sorted(rows, key=_score_profile_row, reverse=True)

    # Keep a small artist cap to avoid near-duplicates in the visible list.
    max_per_artist = 3
    artist_counts = {}
    selected = []
    selected_ids = set()
    for row in rows:
        rid = str(row.get("track_id") or row.get("id") or "").strip()
        if rid in selected_ids:
            continue
        artist = str(row.get("artist") or row.get("artist_name") or "").strip().lower()
        if artist and int(artist_counts.get(artist, 0)) >= max_per_artist:
            continue
        selected.append(row)
        selected_ids.add(rid)
        if artist:
            artist_counts[artist] = int(artist_counts.get(artist, 0)) + 1
        if len(selected) >= target:
            return selected

    if len(selected) < target:
        for row in rows:
            rid = str(row.get("track_id") or row.get("id") or "").strip()
            if not rid or rid in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(rid)
            if len(selected) >= target:
                break
    return selected[:target]


def _metadata_popularity_score(meta):
    """
    Execute metadata popularity score.
    
    This function implements the metadata popularity score step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    item = meta if isinstance(meta, dict) else {}
    views = float(item.get("views", 0) or 0)
    playcount = float(item.get("deezer_playcount", 0) or 0)
    rank = float(item.get("deezer_rank", item.get("rank", 0)) or 0)
    if views > 0 or playcount > 0:
        return float(math.log1p(views * 0.7 + playcount * 0.3))
    if rank > 0:
        return float(1.0 / math.log1p(rank))
    return 0.0


def _metadata_trending_rows(n=24, enrich_covers=False):
    """
    Execute metadata trending rows.
    
    This function implements the metadata trending rows step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    global _metadata_trending_cache
    target = max(1, int(n or 1))
    if not _metadata_trending_cache:
        metadata = _metadata_cache or _load_metadata() or {}
        top_cap = max(1200, target * 40)
        heap = []
        for sid, meta in metadata.items():
            score = _metadata_popularity_score(meta)
            if score <= 0.0:
                continue
            item = (score, str(sid or "").strip())
            if not item[1]:
                continue
            if len(heap) < top_cap:
                heapq.heappush(heap, item)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, item)
        ranked = sorted(heap, key=lambda row: row[0], reverse=True)
        _metadata_trending_cache = [sid for _score, sid in ranked]
    return _profile_rows_from_song_ids(
        _metadata_trending_cache,
        source="metadata_trending_fast",
        limit=target,
        enrich_covers=enrich_covers,
    )


def _playlist_payload_for_user(user_id, include_tracks=True, track_limit=120, enrich_covers=False):
    """
    Execute playlist payload for user.
    
    This function implements the playlist payload for user step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    playlists = user_profile.get_playlists(user_id)
    payload = []
    for playlist in playlists:
        playlist_id = str(playlist.get("playlist_id") or "")
        track_ids = user_profile.get_playlist_track_ids(user_id, playlist_id, limit=track_limit)
        row = {
            "playlist_id": playlist_id,
            "name": str(playlist.get("name") or ""),
            "cover_image": str(playlist.get("cover_image") or ""),
            "created_at": float(playlist.get("created_at") or 0.0),
            "updated_at": float(playlist.get("updated_at") or 0.0),
            "track_count": len(track_ids),
        }
        if include_tracks:
            row["tracks"] = _profile_rows_from_song_ids(
                track_ids,
                source=f"playlist:{playlist_id}",
                limit=track_limit,
                enrich_covers=enrich_covers,
            )
        payload.append(row)
    return payload


def _normalize_playlist_cover_image(value, max_chars=2_500_000):
    """
    Normalize playlist cover image.
    
    This function implements the normalize playlist cover image step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    raw = str(value or "").strip()
    if not raw:
        return ""
    if len(raw) > int(max_chars or 0):
        raise ValueError("cover image too large")
    if raw.startswith("data:image/"):
        # Keep uploads constrained to standard web-safe image data URLs.
        if ";base64," not in raw:
            raise ValueError("invalid cover image format")
        return raw
    # Allow user-hosted HTTPS URLs as a fallback input type.
    if raw.startswith("https://") or raw.startswith("http://"):
        return raw
    raise ValueError("invalid cover image")


def _norm_lyrics_text(value):
    """
    Execute norm lyrics text.
    
    This function implements the norm lyrics text step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*[\(\[\-–—].*?(official|lyrics?|audio|video).*?[\)\]]?\s*$", "", text, flags=re.I)
    text = re.sub(r"\s*[\(\[]?\s*(feat\.?|featuring|ft\.)\s+[^)\]]+[\)\]]?\s*$", "", text, flags=re.I)
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _lyrics_cache_key(title, artist):
    """
    Execute lyrics cache key.
    
    This function implements the lyrics cache key step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    return (_norm_lyrics_text(title), _norm_lyrics_text(artist))


def _get_cached_lyrics(title, artist):
    """
    Get cached lyrics.
    
    This function implements the get cached lyrics step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    if not _LYRICS_CACHE_ENABLED:
        return None
    key = _lyrics_cache_key(title, artist)
    hit = _lyrics_cache.get(key)
    if not hit:
        return None
    if (time.time() - float(hit.get("ts", 0))) > _LYRICS_CACHE_TTL_SEC:
        _lyrics_cache.pop(key, None)
        return None
    return hit.get("payload")


def _set_cached_lyrics(title, artist, payload):
    """
    Set cached lyrics.
    
    This function implements the set cached lyrics step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    if not _LYRICS_CACHE_ENABLED:
        return
    key = _lyrics_cache_key(title, artist)
    if len(_lyrics_cache) >= _LYRICS_CACHE_MAX:
        # Drop an arbitrary oldest-ish item (insertion order in py3.7+ dict).
        try:
            oldest_key = next(iter(_lyrics_cache.keys()))
            _lyrics_cache.pop(oldest_key, None)
        except Exception:
            _lyrics_cache.clear()
    _lyrics_cache[key] = {"ts": time.time(), "payload": payload}


def _is_timeout_exc(exc):
    """
    Return whether timeout exc.
    
    This function implements the is timeout exc step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, socket.timeout):
        return True
    text = str(exc or "").lower()
    return "timed out" in text or "timeout" in text


def _boolish(value):
    """
    Execute boolish.
    
    This function implements the boolish step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _current_session_user_id():
    """
    Execute current session user id.
    
    This function implements the current session user id step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    return str(session.get("auth_user_id") or "").strip()


def _set_auth_session(user_id):
    """
    Set auth session.
    
    This function implements the set auth session step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    uid = str(user_id or "").strip()
    if not uid:
        return False
    session["auth_user_id"] = uid
    session.permanent = True
    return True


def _clear_auth_session():
    """
    Clear auth session.
    
    This function implements the clear auth session step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    session.pop("auth_user_id", None)


def _auth_response_payload():
    """
    Execute auth response payload.
    
    This function implements the auth response payload step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    uid = _current_session_user_id()
    if not uid:
        return {
            "authenticated": False,
            "user": {
                "user_id": "",
                "username": "",
                "avatar_image": "",
                "banner_image": "",
            },
        }
    account = user_profile.get_account_by_user_id(uid)
    if not account:
        _clear_auth_session()
        return {
            "authenticated": False,
            "user": {
                "user_id": "",
                "username": "",
                "avatar_image": "",
                "banner_image": "",
            },
        }
    return {
        "authenticated": True,
        "user": {
            "user_id": str(account.get("user_id") or uid),
            "username": str(account.get("username") or ""),
            "avatar_image": str(account.get("avatar_image") or ""),
            "banner_image": str(account.get("banner_image") or ""),
        },
    }


def _resolve_request_user_id(explicit_user_id="", *, required=False):
    """
    Resolve request user id.
    
    This function implements the resolve request user id step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    explicit = str(explicit_user_id or "").strip()
    session_uid = _current_session_user_id()
    if session_uid:
        if explicit and explicit != session_uid:
            raise PermissionError("user_id does not match authenticated session")
        return session_uid
    if required and not explicit:
        raise ValueError("user_id required")
    return explicit


def _validate_auth_username(username):
    """
    Validate auth username.
    
    This function implements the validate auth username step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    text = str(username or "").strip()
    if len(text) < 3:
        raise ValueError("username must be at least 3 characters")
    if len(text) > 32:
        raise ValueError("username must be at most 32 characters")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", text):
        raise ValueError("username can only use letters, numbers, _, -, and .")
    return text


def _validate_auth_password(password):
    """
    Validate auth password.
    
    This function implements the validate auth password step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    text = str(password or "")
    if len(text) < 6:
        raise ValueError("password must be at least 6 characters")
    if len(text) > 160:
        raise ValueError("password must be at most 160 characters")
    return text


def _normalize_account_media_image(value, label="image", max_chars=2_500_000):
    """
    Normalize account media image.
    
    This function implements the normalize account media image step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    field = str(label or "image").strip().lower()
    raw = str(value or "").strip()
    if not raw:
        return ""
    if len(raw) > int(max_chars or 0):
        raise ValueError(f"{field} too large")
    if raw.startswith("data:image/"):
        if ";base64," not in raw:
            raise ValueError(f"invalid {field} format")
        return raw
    if raw.startswith("https://") or raw.startswith("http://"):
        return raw
    raise ValueError(f"invalid {field}")


def _normalize_account_avatar_image(value, max_chars=2_500_000):
    """
    Normalize account avatar image.
    
    This function implements the normalize account avatar image step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    return _normalize_account_media_image(value, label="avatar image", max_chars=max_chars)


def _normalize_account_banner_image(value, max_chars=3_500_000):
    """
    Normalize account banner image.
    
    This function implements the normalize account banner image step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    return _normalize_account_media_image(value, label="banner image", max_chars=max_chars)


def _lyrics_provider_available():
    """
    Execute lyrics provider available.
    
    This function implements the lyrics provider available step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    return time.time() >= _lyrics_provider_backoff_until


def _lyrica_candidate_base_urls():
    """
    Execute lyrica candidate base urls.
    
    This function implements the lyrica candidate base urls step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    urls = []
    if _LYRICA_BASE_URL:
        urls.append(_LYRICA_BASE_URL)
    urls.extend(_LYRICA_EXTRA_BASE_URLS)
    # Common defaults for Lyrica Flask app. Helps when .env points to a wrong local port.
    urls.extend(
        [
            "http://127.0.0.1:9999",
            "http://localhost:9999",
            "http://127.0.0.1:8080",
            "http://localhost:8080",
            "http://127.0.0.1:8000",
            "http://localhost:8000",
        ]
    )
    deduped = _dedupe_urls(urls)
    explicit_allowed = set(_dedupe_urls([_LYRICA_BASE_URL, *_LYRICA_EXTRA_BASE_URLS]))
    # Avoid probing this app itself by default (common source of confusion when EraEx runs on :5000),
    # but still allow explicit user-configured Lyrica URLs on :5000.
    out = []
    for url in deduped:
        if re.search(r":5000(?:/|$)", url) and url not in explicit_allowed:
            continue
        out.append(url)
    return out


def _record_lyrics_provider_timeout():
    """
    Execute record lyrics provider timeout.
    
    This function implements the record lyrics provider timeout step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    global _lyrics_provider_timeout_streak, _lyrics_provider_backoff_until
    _lyrics_provider_timeout_streak += 1
    if _lyrics_provider_timeout_streak >= 3:
        _lyrics_provider_backoff_until = time.time() + _LYRICS_PROVIDER_TIMEOUT_BACKOFF_SEC


def _record_lyrics_provider_success():
    """
    Execute record lyrics provider success.
    
    This function implements the record lyrics provider success step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    global _lyrics_provider_timeout_streak, _lyrics_provider_backoff_until
    _lyrics_provider_timeout_streak = 0
    _lyrics_provider_backoff_until = 0.0


def _lrclib_get(track_name="", artist_name=""):
    """
    Execute lrclib get.
    
    This function implements the lrclib get step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    q_track = str(track_name or "").strip()
    q_artist = str(artist_name or "").strip()
    if not q_track:
        return None
    if not _lyrics_provider_available():
        return None
    url = (
        "https://lrclib.net/api/get?"
        f"track_name={urllib.parse.quote(q_track)}&artist_name={urllib.parse.quote(q_artist)}"
    )
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "EraEx Music App",
            "Accept": "application/json",
            "Connection": "close",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=4) as response:
            data = json.loads(response.read().decode())
        _record_lyrics_provider_success()
    except Exception as exc:
        if _is_timeout_exc(exc):
            _record_lyrics_provider_timeout()
        raise
    return data if isinstance(data, dict) else None


def _lyrica_ms_to_lrc_tag(ms_value):
    """
    Execute lyrica ms to lrc tag.
    
    This function implements the lyrica ms to lrc tag step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    try:
        total_ms = int(float(ms_value or 0))
    except Exception:
        total_ms = 0
    if total_ms < 0:
        total_ms = 0
    minutes = total_ms // 60000
    seconds = (total_ms % 60000) // 1000
    millis = total_ms % 1000
    return f"[{minutes:02d}:{seconds:02d}.{millis:03d}]"


def _lyrica_timed_to_lrc(lines):
    """
    Execute lyrica timed to lrc.
    
    This function implements the lyrica timed to lrc step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    if not isinstance(lines, list):
        return ""
    out_lines = []
    seen = set()
    for item in lines:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "") or "").strip()
        if not text:
            continue
        start_time = item.get("start_time")
        if start_time is None:
            start_time = item.get("start")
        if start_time is None:
            continue
        tag = _lyrica_ms_to_lrc_tag(start_time)
        key = (tag, text)
        if key in seen:
            continue
        seen.add(key)
        out_lines.append(f"{tag}{text}")
    return "\n".join(out_lines).strip()


def _clean_plain_lyrics_text(value, title="", artist=""):
    """
    Execute clean plain lyrics text.
    
    This function implements the clean plain lyrics text step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    raw = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    if not raw.strip():
        return ""

    safe_title = re.escape(str(title or "").strip())
    safe_artist = re.escape(str(artist or "").strip())
    boilerplate_patterns = [
        # French headers commonly returned by scraped/plain providers.
        rf"^\s*Paroles de (?:la )?chanson\s+.*?(?:\s+par\s+.*)?\s*$",
        rf"^\s*Paroles\s*:\s*.*$",
        # English boilerplate headers.
        rf"^\s*Lyrics (?:to|of)\s+.*$",
        rf"^\s*Song lyrics\s+.*$",
        # Generic title/artist headers (only remove if it looks like a header line).
        rf"^\s*{safe_title}\s*(?:[-–—|:]\s*{safe_artist})?\s*$" if safe_title else r"^$",
        rf"^\s*{safe_artist}\s*(?:[-–—|:]\s*{safe_title})?\s*$" if (safe_title and safe_artist) else r"^$",
    ]
    compiled = [re.compile(p, flags=re.I) for p in boilerplate_patterns if p and p != r"^$"]

    cleaned_lines = []
    for idx, line in enumerate(raw.split("\n")):
        text = str(line or "").strip()
        if not text:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        # Only strip boilerplate aggressively near the top.
        if idx < 6 and any(p.match(text) for p in compiled):
            continue
        cleaned_lines.append(text)

    # Collapse excessive blank lines.
    out = []
    prev_blank = False
    for line in cleaned_lines:
        is_blank = (line == "")
        if is_blank and prev_blank:
            continue
        out.append(line)
        prev_blank = is_blank
    return "\n".join(out).strip()


def _lyrica_lookup_payload(title="", artist="", timestamps=True, plain_only=False):
    """
    Execute lyrica lookup payload.
    
    This function implements the lyrica lookup payload step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    if not _LYRICA_ENABLED:
        return None
    song = str(title or "").strip()
    artist = str(artist or "").strip()
    if not song:
        return None
    params = {
        "song": song,
        "artist": artist,
        "timestamps": "true" if timestamps and not plain_only else "false",
    }
    use_timestamp_mode = bool(timestamps and not plain_only)
    if use_timestamp_mode:
        lyrica_sequence = _LYRICA_TIMESTAMP_SEQUENCE or _LYRICA_SEQUENCE
    else:
        lyrica_sequence = _LYRICA_PLAIN_SEQUENCE or _LYRICA_SEQUENCE
    if lyrica_sequence:
        params["pass"] = "true"
        params["sequence"] = lyrica_sequence
    for base_url in _lyrica_candidate_base_urls():
        url = f"{base_url}/lyrics/?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "EraEx Music App",
                "Connection": "close",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=_LYRICA_TIMEOUT_SEC) as response:
                payload = json.loads(response.read().decode())
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue
        if str(payload.get("status", "")).lower() != "success":
            continue
        data = payload.get("data")
        if not isinstance(data, dict):
            continue

        plain_lyrics = _clean_plain_lyrics_text(
            data.get("lyrics", ""),
            title=song,
            artist=artist,
        )
        if plain_lyrics:
            return {
                "synced": False,
                "lyrics": plain_lyrics,
                "provider": f"lyrica:{data.get('source', 'unknown')}",
                "provider_url": base_url,
            }

        if not plain_only:
            timed_lines = data.get("timed_lyrics")
            if not isinstance(timed_lines, list):
                timed_lines = data.get("timedLyrics")
            lrc_text = _lyrica_timed_to_lrc(timed_lines)
            if lrc_text and (_boolish(data.get("hasTimestamps")) or timed_lines):
                return {
                    "synced": True,
                    "lyrics": lrc_text,
                    "provider": f"lyrica:{data.get('source', 'unknown')}",
                    "provider_url": base_url,
                }
    return None


def _lrclib_search(track_name="", artist_name=""):
    """
    Execute lrclib search.
    
    This function implements the lrclib search step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    q_track = str(track_name or "").strip()
    q_artist = str(artist_name or "").strip()
    if not q_track and not q_artist:
        return []
    if not _lyrics_provider_available():
        return []
    url = (
        "https://lrclib.net/api/search?"
        f"track_name={urllib.parse.quote(q_track)}&artist_name={urllib.parse.quote(q_artist)}"
    )
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "EraEx Music App",
            "Accept": "application/json",
            "Connection": "close",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
        _record_lyrics_provider_success()
    except Exception as exc:
        if _is_timeout_exc(exc):
            _record_lyrics_provider_timeout()
        raise
    return data if isinstance(data, list) else []


def _score_lyrics_match(item, title, artist):
    """
    Score lyrics match.
    
    This function implements the score lyrics match step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    item_track = _norm_lyrics_text(item.get("trackName") or item.get("name") or "")
    item_artist = _norm_lyrics_text(item.get("artistName") or "")
    q_track = _norm_lyrics_text(title)
    q_artist = _norm_lyrics_text(artist)
    score = 0
    if q_track and item_track == q_track:
        score += 6
    elif q_track and q_track in item_track:
        score += 4
    elif q_track and item_track in q_track:
        score += 2
    if q_artist and item_artist == q_artist:
        score += 5
    elif q_artist and q_artist in item_artist:
        score += 3
    elif q_artist and item_artist in q_artist:
        score += 1
    if item.get("syncedLyrics"):
        score += 2
    elif item.get("plainLyrics"):
        score += 1
    return score


def _lookup_lyrics_payload(title, artist):
    """
    Execute lookup lyrics payload.
    
    This function implements the lookup lyrics payload step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    title = str(title or "").strip()
    artist = str(artist or "").strip()
    if not title and not artist:
        return {"synced": False, "lyrics": ""}

    cached = _get_cached_lyrics(title, artist)
    if cached is not None:
        return cached
    deadline = time.time() + _LYRICS_LOOKUP_BUDGET_SEC

    # Prefer Lyrica (YouTube Music/LrcLib capable, timestamped support) when configured.
    lyrica_payload = None
    if time.time() < deadline:
        lyrica_payload = _lyrica_lookup_payload(title=title, artist=artist)
    if lyrica_payload:
        _set_cached_lyrics(title, artist, lyrica_payload)
        return lyrica_payload

    # Fast exact lookup first; often returns quickly and avoids expensive search results.
    for q_title in [title, _norm_lyrics_text(title)]:
        if time.time() >= deadline:
            break
        q_title = str(q_title or "").strip()
        if not q_title:
            continue
        try:
            exact = _lrclib_get(track_name=q_title, artist_name=artist)
        except Exception:
            exact = None
        if isinstance(exact, dict) and (exact.get("syncedLyrics") or exact.get("plainLyrics")):
            payload = {
                "synced": bool(exact.get("syncedLyrics")),
                "lyrics": exact.get("syncedLyrics") or exact.get("plainLyrics") or "",
                "provider": "lrclib:get",
            }
            _set_cached_lyrics(title, artist, payload)
            return payload

    norm_title = _norm_lyrics_text(title)
    queries = []
    seen_queries = set()
    for q_title, q_artist in [
        (title, artist),
        (norm_title, artist),
        (title, ""),
        (norm_title, ""),
    ]:
        key = (str(q_title or "").strip(), str(q_artist or "").strip())
        if key in seen_queries:
            continue
        seen_queries.add(key)
        if not key[0] and not key[1]:
            continue
        queries.append(key)

    best_match = None
    best_score = -1
    for q_title, q_artist in queries:
        if time.time() >= deadline:
            break
        try:
            results = _lrclib_search(track_name=q_title, artist_name=q_artist)
        except Exception:
            continue
        for item in results:
            if not isinstance(item, dict):
                continue
            if not (item.get("syncedLyrics") or item.get("plainLyrics")):
                continue
            score = _score_lyrics_match(item, title, artist)
            if score > best_score:
                best_score = score
                best_match = item
        if best_match and best_score >= 8:
            break

    payload = {
        "synced": bool(best_match and best_match.get("syncedLyrics")),
        "lyrics": (
            (best_match.get("syncedLyrics") or best_match.get("plainLyrics") or "")
            if best_match
            else ""
        ),
        "provider": ("lrclib:search" if best_match else ""),
    }
    # Cache only successful lyric fetches; avoid poisoning cache with timeout-driven empty payloads.
    if payload.get("lyrics"):
        _set_cached_lyrics(title, artist, payload)
    return payload


# Load metadata.
def _load_metadata():
    """
    Load metadata.
    
    This function implements the load metadata step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    global _metadata_cache, _metadata_by_video_id_cache
    meta_path = settings.INDEX_DIR / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            _metadata_cache = json.load(f)
        # Secondary lookup speeds profile/history reconstruction when only a YouTube ID was stored.
        by_video_id = {}
        for track_id, meta in (_metadata_cache or {}).items():
            if not isinstance(meta, dict):
                continue
            video_id = str(meta.get("video_id") or "").strip()
            if is_youtube_id(video_id) and video_id not in by_video_id:
                by_video_id[video_id] = str(track_id)
        _metadata_by_video_id_cache = by_video_id
    return _metadata_cache


# Internal helper to init search pipeline.
def _init_search_pipeline():
    """
    Initialize search pipeline.
    
    This function implements the init search pipeline step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    from src.search.search_pipeline import UnifiedSearchPipeline

    return UnifiedSearchPipeline()


# Internal helper to init recommender.
def _init_recommender():
    """
    Initialize recommender.
    
    This function implements the init recommender step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    from src.recommendation.recommendation_engine import ColdStartHandler

    metadata = _metadata_cache or _load_metadata()
    return ColdStartHandler(metadata_path=None, metadata=metadata)


search_pipeline = LazyLoader(lambda: _init_search_pipeline(), name="search")
recommender = LazyLoader(lambda: _init_recommender(), name="recommender")
_load_metadata()
preload_in_background(search_pipeline, recommender)


# Check this operation.
@app.route("/health", methods=["GET"])
def health():
    """
    Execute health.
    
    This function implements the health step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    return jsonify(
        {
            "status": "healthy",
            "yt_dlp_enabled": bool(video_resolver.enabled),
            "spotify_cover_enabled": bool(spotify_cover_resolver.enabled),
            "spotify_playback_enabled": bool(_spotify_playback_available()),
            "spotify_playback_authenticated": bool(_spotify_token_is_valid()),
            "resolve_video_playback_policy": "yt_dlp_only",
            "resolve_video_covers_only_policy": "spotify_only",
            "track_meta_enabled": bool(metadata_enricher.enabled),
            **lyrics_provider_engine.health(),
            "lyrica_enabled": bool(_LYRICA_ENABLED),
            "lyrics_provider_policy": "lrclib_direct",
            "lyrica_sequence": (_LYRICA_PLAIN_SEQUENCE or _LYRICA_SEQUENCE or ""),
            "lyrica_base_url": _LYRICA_BASE_URL,
            "lyrics_cache_enabled": bool(_LYRICS_CACHE_ENABLED),
        }
    )


# Render this operation.
@app.route("/", methods=["GET"])
def index():
    """
    Execute index.
    
    This function implements the index step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    return render_template(
        "index.html",
        shared_user_id=_SHARED_USER_ID,
        shared_user_id_force=bool(_SHARED_USER_ID_FORCE),
    )


@app.route("/api/auth/session", methods=["GET"])
def auth_session_status():
    """
    Execute auth session status.
    
    This function implements the auth session status step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    return jsonify(_auth_response_payload())


@app.route("/api/auth/register", methods=["POST"])
def auth_register():
    """
    Execute auth register.
    
    This function implements the auth register step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    data = request.get_json(silent=True) or {}
    username_raw = data.get("username")
    password_raw = data.get("password")
    avatar_raw = data.get("avatar_image")
    banner_raw = data.get("banner_image")
    try:
        username = _validate_auth_username(username_raw)
        password = _validate_auth_password(password_raw)
        avatar_image = _normalize_account_avatar_image(avatar_raw)
        banner_image = _normalize_account_banner_image(banner_raw)
    except ValueError as exc:
        return (jsonify({"error": str(exc)}), 400)

    password_hash = generate_password_hash(password)
    try:
        account = user_profile.create_account(
            username=username,
            password_hash=password_hash,
            avatar_image=avatar_image,
            banner_image=banner_image,
        )
    except ValueError as exc:
        msg = str(exc or "Could not create account")
        if "already exists" in msg.lower():
            return (jsonify({"error": msg}), 409)
        return (jsonify({"error": msg}), 400)
    except Exception:
        return (jsonify({"error": "Could not create account"}), 500)

    _set_auth_session(account.get("user_id"))
    payload = _auth_response_payload()
    payload["status"] = "registered"
    return jsonify(payload)


@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    """
    Execute auth login.
    
    This function implements the auth login step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    data = request.get_json(silent=True) or {}
    username_raw = data.get("username")
    password_raw = data.get("password")
    try:
        username = _validate_auth_username(username_raw)
        password = _validate_auth_password(password_raw)
    except ValueError as exc:
        return (jsonify({"error": str(exc)}), 400)

    account = user_profile.get_account_by_username(username)
    if not account:
        return (jsonify({"error": "invalid username or password"}), 401)

    password_hash = str(account.get("password_hash") or "")
    if not password_hash or not check_password_hash(password_hash, password):
        return (jsonify({"error": "invalid username or password"}), 401)

    _set_auth_session(account.get("user_id"))
    payload = _auth_response_payload()
    payload["status"] = "logged_in"
    return jsonify(payload)


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    """
    Execute auth logout.
    
    This function implements the auth logout step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    _clear_auth_session()
    return jsonify({"status": "logged_out"})


@app.route("/api/auth/avatar", methods=["POST"])
def auth_set_avatar():
    """
    Execute auth set avatar.
    
    This function implements the auth set avatar step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    uid = _current_session_user_id()
    if not uid:
        return (jsonify({"error": "authentication required"}), 401)
    data = request.get_json(silent=True) or {}
    try:
        avatar_image = _normalize_account_avatar_image(data.get("avatar_image"))
        updated = user_profile.set_account_avatar(uid, avatar_image)
        if not updated:
            return (jsonify({"error": "account not found"}), 404)
    except ValueError as exc:
        return (jsonify({"error": str(exc)}), 400)
    payload = _auth_response_payload()
    payload["status"] = "avatar_updated"
    return jsonify(payload)


@app.route("/api/auth/banner", methods=["POST"])
def auth_set_banner():
    """
    Execute auth set banner.
    
    This function implements the auth set banner step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    uid = _current_session_user_id()
    if not uid:
        return (jsonify({"error": "authentication required"}), 401)
    data = request.get_json(silent=True) or {}
    try:
        banner_image = _normalize_account_banner_image(data.get("banner_image"))
        updated = user_profile.set_account_banner(uid, banner_image)
        if not updated:
            return (jsonify({"error": "account not found"}), 404)
    except ValueError as exc:
        return (jsonify({"error": str(exc)}), 400)
    payload = _auth_response_payload()
    payload["status"] = "banner_updated"
    return jsonify(payload)


@app.route("/api/spotify/status", methods=["GET"])
def spotify_status():
    """
    Execute spotify status.
    
    This function implements the spotify status step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    return jsonify(
        {
            "enabled": bool(_spotify_playback_available()),
            "authenticated": bool(_spotify_token_is_valid()),
            "client_id_configured": bool(spotify_cover_resolver.client_id),
            "redirect_uri": _spotify_redirect_uri(),
            "scopes": _SPOTIFY_AUTH_SCOPES,
        }
    )


@app.route("/api/spotify/login", methods=["GET"])
def spotify_login():
    """
    Execute spotify login.
    
    This function implements the spotify login step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    global _spotify_oauth_pending_state, _spotify_oauth_pending_created_at
    if not _spotify_playback_available():
        return (jsonify({"error": "spotify playback not configured"}), 400)
    redirect_uri = _spotify_redirect_uri()
    _spotify_oauth_pending_state = secrets.token_urlsafe(18)
    _spotify_oauth_pending_created_at = time.time()
    params = {
        "response_type": "code",
        "client_id": spotify_cover_resolver.client_id,
        "scope": _SPOTIFY_AUTH_SCOPES,
        "redirect_uri": redirect_uri,
        "state": _spotify_oauth_pending_state,
        "show_dialog": "false",
    }
    authorize_url = "https://accounts.spotify.com/authorize?" + urllib.parse.urlencode(params)
    return jsonify({"authorize_url": authorize_url})


@app.route("/spotify/callback", methods=["GET"])
def spotify_callback():
    """
    Execute spotify callback.
    
    This function implements the spotify callback step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    global _spotify_oauth_pending_state, _spotify_oauth_pending_created_at
    error = str(request.args.get("error", "") or "").strip()
    code = str(request.args.get("code", "") or "").strip()
    state_val = str(request.args.get("state", "") or "").strip()
    if error:
        return render_template(
            "index.html"
        )
    if (
        not code
        or not _spotify_oauth_pending_state
        or not state_val
        or state_val != _spotify_oauth_pending_state
        or (time.time() - float(_spotify_oauth_pending_created_at or 0.0)) > 600.0
    ):
        return render_template("index.html")
    ok = _spotify_exchange_code_for_token(code)
    _spotify_oauth_pending_state = ""
    _spotify_oauth_pending_created_at = 0.0
    # Redirect back to app shell. Frontend checks /api/spotify/status.
    target = "/?spotify_auth=" + ("ok" if ok else "failed")
    return (
        "<!doctype html><html><body><script>"
        "try{window.opener&&window.opener.postMessage({type:'spotify-auth',status:'%s'}, window.location.origin);}catch(e){}"
        "window.location.replace('%s');"
        "</script></body></html>"
    ) % (("ok" if ok else "failed"), target)


@app.route("/api/spotify/access_token", methods=["GET"])
def spotify_access_token():
    """
    Execute spotify access token.
    
    This function implements the spotify access token step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    if not _spotify_playback_available():
        return (jsonify({"error": "spotify playback not configured"}), 400)
    payload = _spotify_user_token_payload()
    if not payload:
        return (jsonify({"error": "spotify auth required"}), 401)
    return jsonify(payload)


@app.route("/api/spotify/resolve_track", methods=["GET"])
def spotify_resolve_track():
    """
    Execute spotify resolve track.
    
    This function implements the spotify resolve track step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    if not spotify_cover_resolver.enabled:
        return (jsonify({"error": "spotify not configured"}), 400)
    title = str(request.args.get("title", "") or "").strip()
    artist = str(request.args.get("artist", "") or "").strip()
    track_id = str(request.args.get("track_id", "") or "").strip()
    if not any([title, artist, track_id]):
        return (jsonify({"error": "title, artist, or track_id required"}), 400)
    resolved = _spotify_cover_lookup(title=title, artist=artist, track_id=track_id)
    if not isinstance(resolved, dict):
        return (jsonify({"error": "spotify track not found"}), 404)
    payload = {
        "provider": "spotify",
        "spotify_uri": str(resolved.get("spotify_uri") or ""),
        "spotify_track_id": str(resolved.get("spotify_track_id") or ""),
        "spotify_title": str(resolved.get("spotify_title") or title),
        "spotify_artist": str(resolved.get("spotify_artist") or artist),
        "thumbnail": str(resolved.get("thumbnail") or ""),
        "thumbnail_candidates": (
            resolved.get("thumbnail_candidates")
            if isinstance(resolved.get("thumbnail_candidates"), list)
            else []
        ),
    }
    if not payload["spotify_uri"]:
        return (jsonify({"error": "spotify uri unavailable"}), 404)
    return jsonify(payload)


# Search route.
@app.route("/search", methods=["GET"])
def search_route():
    """
    Execute search route.
    
    This function implements the search route step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    query = request.args.get("q", "")
    limit = int(request.args.get("limit", 20))
    offset = int(request.args.get("offset", 0))
    if not query:
        return (jsonify({"error": "No query provided"}), 400)
    pipeline = search_pipeline.instance
    if pipeline is None:
        return (jsonify({"error": "Search pipeline not loaded"}), 503)
    return jsonify(pipeline.search(query, limit=limit, offset=offset))


# Handle sonic route.
@app.route("/sonic", methods=["GET"])
def sonic_route():
    """
    Execute sonic route.
    
    This function implements the sonic route step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    query = request.args.get("q", "")
    limit = int(request.args.get("limit", 20))
    offset = int(request.args.get("offset", 0))
    if not query:
        return (jsonify({"error": "No query provided"}), 400)
    pipeline = search_pipeline.instance
    if pipeline is None:
        return (jsonify({"error": "Search pipeline not loaded"}), 503)
    return jsonify(pipeline.search(query, limit=limit, offset=offset))


# Handle like song.
@app.route("/api/like", methods=["POST"])
def like_song():
    """
    Execute like song.
    
    This function implements the like song step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    data = request.get_json() or {}
    try:
        user_id = _resolve_request_user_id(data.get("user_id"), required=True)
    except PermissionError as exc:
        return (jsonify({"error": str(exc)}), 403)
    except ValueError as exc:
        return (jsonify({"error": str(exc)}), 400)
    song_id = data.get("song_id")
    if not user_id or not song_id:
        return (jsonify({"error": "user_id and song_id required"}), 400)
    user_profile.add_like(user_id, song_id)
    return jsonify({"status": "liked"})


# Handle unlike song.
@app.route("/api/unlike", methods=["POST"])
def unlike_song():
    """
    Execute unlike song.
    
    This function implements the unlike song step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    data = request.get_json() or {}
    try:
        user_id = _resolve_request_user_id(data.get("user_id"), required=True)
    except PermissionError as exc:
        return (jsonify({"error": str(exc)}), 403)
    except ValueError as exc:
        return (jsonify({"error": str(exc)}), 400)
    song_id = data.get("song_id")
    if not user_id or not song_id:
        return (jsonify({"error": "user_id and song_id required"}), 400)
    user_profile.remove_like(user_id, song_id)
    return jsonify({"status": "unliked"})


@app.route("/api/dislike", methods=["POST"])
def dislike_song():
    """
    Execute dislike song.
    
    This function implements the dislike song step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    data = request.get_json() or {}
    try:
        user_id = _resolve_request_user_id(data.get("user_id"), required=True)
    except PermissionError as exc:
        return (jsonify({"error": str(exc)}), 403)
    except ValueError as exc:
        return (jsonify({"error": str(exc)}), 400)
    song_id = data.get("song_id")
    if not user_id or not song_id:
        return (jsonify({"error": "user_id and song_id required"}), 400)
    user_profile.add_dislike(user_id, song_id)
    return jsonify({"status": "disliked"})


@app.route("/api/undislike", methods=["POST"])
def undislike_song():
    """
    Execute undislike song.
    
    This function implements the undislike song step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    data = request.get_json() or {}
    try:
        user_id = _resolve_request_user_id(data.get("user_id"), required=True)
    except PermissionError as exc:
        return (jsonify({"error": str(exc)}), 403)
    except ValueError as exc:
        return (jsonify({"error": str(exc)}), 400)
    song_id = data.get("song_id")
    if not user_id or not song_id:
        return (jsonify({"error": "user_id and song_id required"}), 400)
    user_profile.remove_dislike(user_id, song_id)
    return jsonify({"status": "undisliked"})


@app.route("/api/skip", methods=["POST"])
def record_skip():
    """
    Execute record skip.
    
    This function implements the record skip step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    data = request.get_json() or {}
    try:
        user_id = _resolve_request_user_id(data.get("user_id"), required=True)
    except PermissionError as exc:
        return (jsonify({"error": str(exc)}), 403)
    except ValueError as exc:
        return (jsonify({"error": str(exc)}), 400)
    song_id = data.get("song_id")
    event = str(data.get("event") or "next").strip().lower()
    position_sec = data.get("position_sec")
    duration_sec = data.get("duration_sec")
    if not user_id or not song_id:
        return (jsonify({"error": "user_id and song_id required"}), 400)
    try:
        user_profile.add_skip_event(
            user_id=user_id,
            song_id=song_id,
            event=event,
            position_sec=position_sec,
            duration_sec=duration_sec,
        )
    except ValueError as exc:
        return (jsonify({"error": str(exc)}), 400)
    return jsonify({"status": "recorded"})


# Handle record play.
@app.route("/api/play", methods=["POST"])
def record_play():
    """
    Execute record play.
    
    This function implements the record play step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    started = time.perf_counter()
    data = request.get_json() or {}
    try:
        user_id = _resolve_request_user_id(data.get("user_id"), required=True)
    except PermissionError as exc:
        return (jsonify({"error": str(exc)}), 403)
    except ValueError as exc:
        return (jsonify({"error": str(exc)}), 400)
    song_id = data.get("song_id")
    source = str(data.get("source", "") or "")
    query = str(data.get("query", "") or "")
    _debug_log(
        "api.play.request",
        user_id=user_id,
        song_id=song_id,
        source=source,
        query=query,
    )
    if not user_id or not song_id:
        _debug_log(
            "api.play.error",
            reason="missing user_id/song_id",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
        return (jsonify({"error": "user_id and song_id required"}), 400)
    user_profile.add_play(user_id, song_id)
    _debug_log(
        "api.play.ok",
        user_id=user_id,
        song_id=song_id,
        source=source,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )
    return jsonify({"status": "recorded"})


# Receive frontend player debug events (best-effort; CLI tracing).
@app.route("/api/player_debug", methods=["POST"])
def player_debug():
    """
    Execute player debug.
    
    This function implements the player debug step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    started = time.perf_counter()
    data = request.get_json(silent=True) or {}
    event = str(data.get("event", "") or "").strip()
    if not event:
        return (jsonify({"error": "event required"}), 400)
    track = data.get("track") if isinstance(data.get("track"), dict) else {}
    _debug_log(
        f"player.{event}",
        title=(track or {}).get("title"),
        artist=(track or {}).get("artist"),
        video_id=(track or {}).get("videoId") or data.get("video_id"),
        code=data.get("code"),
        candidate_index=data.get("candidate_index"),
        candidates=data.get("candidates") if isinstance(data.get("candidates"), list) else [],
        note=data.get("note"),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )
    return ("", 204)


# Get profile.
@app.route("/api/profile", methods=["GET"])
def get_profile():
    """
    Get profile.
    
    This function implements the get profile step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    try:
        user_id = _resolve_request_user_id(request.args.get("user_id"), required=True)
    except PermissionError as exc:
        return (jsonify({"error": str(exc)}), 403)
    except ValueError as exc:
        return (jsonify({"error": str(exc)}), 400)
    return jsonify(user_profile.get_profile(user_id))


@app.route("/api/liked", methods=["GET"])
def get_liked_tracks():
    """
    Get liked tracks.
    
    This function implements the get liked tracks step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    try:
        user_id = _resolve_request_user_id(request.args.get("user_id"), required=True)
    except PermissionError as exc:
        return (jsonify({"error": str(exc)}), 403)
    except ValueError as exc:
        return (jsonify({"error": str(exc)}), 400)
    requested_n = int(request.args.get("n", 100))
    enrich_covers = _boolish(request.args.get("covers", "1"))
    n = max(1, min(requested_n, 300))
    liked_ids = user_profile.get_likes(user_id)
    rows = _profile_rows_from_song_ids(
        liked_ids,
        source="liked",
        limit=n,
        enrich_covers=enrich_covers,
    )
    return jsonify({"user_id": user_id, "liked": rows})


@app.route("/api/history", methods=["GET"])
def get_history_tracks():
    """
    Get history tracks.
    
    This function implements the get history tracks step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    try:
        user_id = _resolve_request_user_id(request.args.get("user_id"), required=True)
    except PermissionError as exc:
        return (jsonify({"error": str(exc)}), 403)
    except ValueError as exc:
        return (jsonify({"error": str(exc)}), 400)
    requested_n = int(request.args.get("n", 100))
    enrich_covers = _boolish(request.args.get("covers", "1"))
    n = max(1, min(requested_n, 300))
    history_ids = user_profile.get_plays(user_id, limit=n)
    rows = _profile_rows_from_song_ids(
        history_ids,
        source="history",
        limit=n,
        enrich_covers=enrich_covers,
    )
    return jsonify({"user_id": user_id, "history": rows})


@app.route("/api/playlists", methods=["GET"])
def get_playlists():
    """
    Get playlists.
    
    This function implements the get playlists step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    try:
        user_id = _resolve_request_user_id(request.args.get("user_id"), required=True)
    except PermissionError as exc:
        return (jsonify({"error": str(exc)}), 403)
    except ValueError as exc:
        return (jsonify({"error": str(exc)}), 400)
    include_tracks = _boolish(request.args.get("tracks", "1"))
    enrich_covers = _boolish(request.args.get("covers", "0"))
    requested_limit = int(request.args.get("track_limit", 120))
    track_limit = max(1, min(requested_limit, 300))
    playlists = _playlist_payload_for_user(
        user_id,
        include_tracks=include_tracks,
        track_limit=track_limit,
        enrich_covers=enrich_covers,
    )
    return jsonify({"user_id": user_id, "playlists": playlists})


@app.route("/api/playlists/create", methods=["POST"])
def create_playlist():
    """
    Create playlist.
    
    This function implements the create playlist step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    data = request.get_json() or {}
    try:
        user_id = _resolve_request_user_id(data.get("user_id"), required=True)
    except PermissionError as exc:
        return (jsonify({"error": str(exc)}), 403)
    except ValueError as exc:
        return (jsonify({"error": str(exc)}), 400)
    name = str(data.get("name") or "").strip()
    cover_image = data.get("cover_image")
    if not user_id or not name:
        return (jsonify({"error": "user_id and name required"}), 400)
    try:
        playlist_id = user_profile.create_playlist(
            user_id,
            name,
            cover_image=_normalize_playlist_cover_image(cover_image),
        )
    except ValueError as exc:
        return (jsonify({"error": str(exc)}), 400)
    return jsonify({"status": "created", "playlist_id": playlist_id})


@app.route("/api/playlists/rename", methods=["POST"])
def rename_playlist():
    """
    Execute rename playlist.
    
    This function implements the rename playlist step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    data = request.get_json() or {}
    try:
        user_id = _resolve_request_user_id(data.get("user_id"), required=True)
    except PermissionError as exc:
        return (jsonify({"error": str(exc)}), 403)
    except ValueError as exc:
        return (jsonify({"error": str(exc)}), 400)
    playlist_id = str(data.get("playlist_id") or "").strip()
    name = str(data.get("name") or "").strip()
    if not user_id or not playlist_id or not name:
        return (jsonify({"error": "user_id, playlist_id and name required"}), 400)
    try:
        updated = user_profile.rename_playlist(user_id, playlist_id, name)
        if not updated:
            return (jsonify({"error": "playlist not found"}), 404)
    except ValueError as exc:
        return (jsonify({"error": str(exc)}), 400)
    return jsonify({"status": "renamed"})


@app.route("/api/playlists/add_track", methods=["POST"])
def add_playlist_track():
    """
    Add playlist track.
    
    This function implements the add playlist track step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    data = request.get_json() or {}
    try:
        user_id = _resolve_request_user_id(data.get("user_id"), required=True)
    except PermissionError as exc:
        return (jsonify({"error": str(exc)}), 403)
    except ValueError as exc:
        return (jsonify({"error": str(exc)}), 400)
    playlist_id = str(data.get("playlist_id") or "").strip()
    song_id = str(data.get("song_id") or "").strip()
    if not user_id or not playlist_id or not song_id:
        return (jsonify({"error": "user_id, playlist_id and song_id required"}), 400)
    try:
        added = user_profile.add_playlist_track(user_id, playlist_id, song_id)
    except ValueError as exc:
        return (jsonify({"error": str(exc)}), 400)
    return jsonify({"status": "added" if added else "duplicate", "added": bool(added)})


@app.route("/api/playlists/set_cover", methods=["POST"])
def set_playlist_cover():
    """
    Set playlist cover.
    
    This function implements the set playlist cover step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    data = request.get_json() or {}
    try:
        user_id = _resolve_request_user_id(data.get("user_id"), required=True)
    except PermissionError as exc:
        return (jsonify({"error": str(exc)}), 403)
    except ValueError as exc:
        return (jsonify({"error": str(exc)}), 400)
    playlist_id = str(data.get("playlist_id") or "").strip()
    cover_image = data.get("cover_image")
    if not user_id or not playlist_id:
        return (jsonify({"error": "user_id and playlist_id required"}), 400)
    try:
        normalized_cover = _normalize_playlist_cover_image(cover_image)
        updated = user_profile.set_playlist_cover(user_id, playlist_id, normalized_cover)
        if not updated:
            return (jsonify({"error": "playlist not found"}), 404)
    except ValueError as exc:
        return (jsonify({"error": str(exc)}), 400)
    return jsonify({"status": "updated"})


@app.route("/api/playlists/remove_track", methods=["POST"])
def remove_playlist_track():
    """
    Remove playlist track.
    
    This function implements the remove playlist track step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    data = request.get_json() or {}
    try:
        user_id = _resolve_request_user_id(data.get("user_id"), required=True)
    except PermissionError as exc:
        return (jsonify({"error": str(exc)}), 403)
    except ValueError as exc:
        return (jsonify({"error": str(exc)}), 400)
    playlist_id = str(data.get("playlist_id") or "").strip()
    song_id = str(data.get("song_id") or "").strip()
    if not user_id or not playlist_id or not song_id:
        return (jsonify({"error": "user_id, playlist_id and song_id required"}), 400)
    user_profile.remove_playlist_track(user_id, playlist_id, song_id)
    return jsonify({"status": "removed"})


@app.route("/api/playlists/clear", methods=["POST"])
def clear_playlist():
    """
    Clear playlist.
    
    This function implements the clear playlist step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    data = request.get_json() or {}
    try:
        user_id = _resolve_request_user_id(data.get("user_id"), required=True)
    except PermissionError as exc:
        return (jsonify({"error": str(exc)}), 403)
    except ValueError as exc:
        return (jsonify({"error": str(exc)}), 400)
    playlist_id = str(data.get("playlist_id") or "").strip()
    if not user_id or not playlist_id:
        return (jsonify({"error": "user_id and playlist_id required"}), 400)
    user_profile.clear_playlist(user_id, playlist_id)
    return jsonify({"status": "cleared"})


@app.route("/api/playlists/delete", methods=["POST"])
def delete_playlist():
    """
    Delete playlist.
    
    This function implements the delete playlist step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    data = request.get_json() or {}
    try:
        user_id = _resolve_request_user_id(data.get("user_id"), required=True)
    except PermissionError as exc:
        return (jsonify({"error": str(exc)}), 403)
    except ValueError as exc:
        return (jsonify({"error": str(exc)}), 400)
    playlist_id = str(data.get("playlist_id") or "").strip()
    if not user_id or not playlist_id:
        return (jsonify({"error": "user_id and playlist_id required"}), 400)
    user_profile.delete_playlist(user_id, playlist_id)
    return jsonify({"status": "deleted"})


# Resolve video.
@app.route("/api/resolve_video", methods=["GET"])
def resolve_video():
    """
    Resolve video.
    
    This function implements the resolve video step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    started = time.perf_counter()
    title = request.args.get("title", "")
    artist = request.args.get("artist", "")
    track_id = request.args.get("track_id", "")
    requested_video_id = str(request.args.get("video_id", "") or "").strip()
    covers_only = _boolish(request.args.get("covers_only", "0"))
    _debug_log(
        "api.resolve_video.request",
        title=title,
        artist=artist,
        track_id=track_id,
        video_id=requested_video_id,
        covers_only=covers_only,
    )
    if not title and (not artist) and (not track_id):
        _debug_log(
            "api.resolve_video.error",
            reason="missing title/artist/track_id",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
        return (jsonify({"error": "title, artist, or track_id required"}), 400)

    # Cover-only path uses Spotify album art (runtime cover provider) and does not mutate playback IDs.
    if covers_only:
        resolved_cover = _spotify_cover_lookup(title=title, artist=artist, track_id=track_id)
        if not resolved_cover:
            _debug_log(
                "api.resolve_video.cover_miss",
                title=title,
                artist=artist,
                track_id=track_id,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )
            return (jsonify({"error": "no fallback cover found"}), 404)
        payload = dict(resolved_cover)
        payload["provider"] = "spotify_cover"
        _debug_log(
            "api.resolve_video.cover_ok",
            provider=payload.get("provider"),
            video_id=payload.get("video_id"),
            video_candidates=payload.get("video_candidates", []),
            thumb=payload.get("thumbnail"),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
        return jsonify(payload)

    if not video_resolver.enabled:
        _debug_log(
            "api.resolve_video.error",
            reason="yt-dlp not available",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
        return (jsonify({"error": "yt-dlp not available"}), 503)

    # Playback/source resolution path: keep original yt-dlp-first behavior (do not block on Spotify).
    # If the client already has a trusted metadata `video_id`, use it first to avoid song mismatch.
    if is_youtube_id(requested_video_id):
        direct_candidates = thumbnail_candidates(requested_video_id, None)
        payload = {
            "video_id": requested_video_id,
            "video_candidates": [requested_video_id],
            "thumbnail": (direct_candidates[0] if direct_candidates else ""),
            "thumbnail_candidates": direct_candidates,
            "provider": "yt_dlp:video_id",
        }
        # Also fetch title-search candidates as fallbacks for embed-restricted videos.
        try:
            resolved_fallback = video_resolver.resolve(
                title=title, artist=artist, track_id=track_id
            )
        except Exception:
            resolved_fallback = None
        if isinstance(resolved_fallback, dict):
            fallback_candidates = []
            if isinstance(resolved_fallback.get("video_candidates"), list):
                fallback_candidates.extend(resolved_fallback.get("video_candidates") or [])
            if resolved_fallback.get("video_id"):
                fallback_candidates.append(resolved_fallback.get("video_id"))
            merged_video_candidates = []
            seen_video_ids = set()
            for vid in [requested_video_id, *fallback_candidates]:
                vid = str(vid or "").strip()
                if not is_youtube_id(vid) or vid in seen_video_ids:
                    continue
                seen_video_ids.add(vid)
                merged_video_candidates.append(vid)
            if merged_video_candidates:
                payload["video_candidates"] = merged_video_candidates
        _debug_log(
            "api.resolve_video.direct_ok",
            provider=payload.get("provider"),
            requested_video_id=requested_video_id,
            resolved_video_id=payload.get("video_id"),
            video_candidates=payload.get("video_candidates", []),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
        return jsonify(payload)

    resolved = video_resolver.resolve(title=title, artist=artist, track_id=track_id)
    if not resolved:
        _debug_log(
            "api.resolve_video.miss",
            title=title,
            artist=artist,
            track_id=track_id,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
        return (jsonify({"error": "no fallback video found"}), 404)
    payload = dict(resolved)
    if not payload.get("thumbnail_candidates"):
        payload["thumbnail_candidates"] = thumbnail_candidates(
            payload.get("video_id", ""), payload.get("thumbnail")
        )
    payload["provider"] = "yt_dlp"
    _debug_log(
        "api.resolve_video.search_ok",
        provider=payload.get("provider"),
        title=title,
        artist=artist,
        track_id=track_id,
        resolved_video_id=payload.get("video_id"),
        video_candidates=payload.get("video_candidates", []),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )
    return jsonify(payload)


# Handle track enrich.
@app.route("/api/track_enrich", methods=["GET"])
def track_enrich():
    """
    Execute track enrich.
    
    This function implements the track enrich step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    started = time.perf_counter()
    track_id = request.args.get("track_id", "")
    title = request.args.get("title", "")
    artist = request.args.get("artist", "")
    _debug_log("api.track_enrich.request", track_id=track_id, title=title, artist=artist)
    if not track_id and (not title):
        _debug_log(
            "api.track_enrich.error",
            reason="missing track_id/title",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
        return (jsonify({"error": "track_id or title required"}), 400)
    enriched = metadata_enricher.enrich(track_id=track_id, title=title, artist=artist)
    _debug_log(
        "api.track_enrich.ok",
        track_id=track_id or (enriched or {}).get("track_id"),
        title=(enriched or {}).get("title") or title,
        artist=(enriched or {}).get("artist") or artist,
        video_id=(enriched or {}).get("video_id"),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )
    return jsonify(enriched)


# Get recommendations.
@app.route("/api/recommend", methods=["GET"])
def get_recommendations():
    """
    Get recommendations.
    
    This function implements the get recommendations step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    started = time.perf_counter()
    try:
        user_id = _resolve_request_user_id(request.args.get("user_id"), required=True)
    except PermissionError as exc:
        return (jsonify({"error": str(exc)}), 403)
    except ValueError as exc:
        return (jsonify({"error": str(exc)}), 400)
    requested_n = int(request.args.get("n", 10))
    enrich_covers = _boolish(request.args.get("covers", "1"))
    fast_mode = _boolish(request.args.get("fast", "0"))
    reco_response_max = int(getattr(settings, "RECO_RESPONSE_MAX", 360) or 360)
    n = max(1, min(requested_n, reco_response_max))
    liked = user_profile.get_likes(user_id)
    played = user_profile.get_plays(user_id)
    disliked = user_profile.get_dislikes(user_id)
    playlist_tracks = user_profile.get_all_playlist_track_ids(user_id, limit=240)
    skip_summary = user_profile.get_skip_summary(user_id, limit=600)
    history_counts = _recommendation_history_counts(
        liked, played, disliked, playlist_tracks, skip_summary
    )
    signal_fingerprint = _recommendation_signal_fingerprint(
        liked, played, disliked, playlist_tracks, skip_summary
    )
    has_personal_history = bool(liked or played or playlist_tracks)
    mode = "adaptive" if has_personal_history else "cold_start"
    _debug_log(
        "api.recommend.request",
        user_id=user_id,
        requested_n=requested_n,
        n=n,
        covers=enrich_covers,
        fast_mode=fast_mode,
        mode=mode,
        liked_count=int(history_counts.get("likes", 0)),
        played_count=int(history_counts.get("plays", 0)),
        disliked_count=int(history_counts.get("dislikes", 0)),
        playlist_track_count=int(history_counts.get("playlist_tracks", 0)),
        skip_next_count=int(history_counts.get("skip_next", 0)),
        skip_prev_count=int(history_counts.get("skip_prev", 0)),
        skip_early_count=int(history_counts.get("skip_early", 0)),
    )
    if fast_mode:
        cached_fast = _recommendation_cache_get(
            user_id=user_id,
            cache_kind="fast",
            n=n,
            signal_fingerprint=signal_fingerprint,
        )
        if cached_fast:
            cached_fast["covers_enriched"] = bool(enrich_covers)
            if enrich_covers:
                cached_fast["recommendations"] = _enrich_missing_recommendation_covers(
                    list(cached_fast.get("recommendations") or [])
                )
            return jsonify(cached_fast)

        fast_rows = _fast_profile_seed_recommendations(
            liked_ids=liked,
            playlist_track_ids=playlist_tracks,
            played_ids=played,
            disliked_ids=disliked,
            n=n,
            enrich_covers=enrich_covers,
        )
        if fast_rows:
            _recommendation_cache_set(
                user_id=user_id,
                cache_kind="fast",
                signal_fingerprint=signal_fingerprint,
                recommendation_mode="profile_seed_fast",
                history_counts=history_counts,
                ranker_ready=bool(recommender.is_loaded),
                recommendations=fast_rows,
            )
            return jsonify(
                {
                    "user_id": user_id,
                    "recommendation_mode": "profile_seed_fast",
                    "history_counts": history_counts,
                    "covers_enriched": bool(enrich_covers),
                    "ranker_ready": bool(recommender.is_loaded),
                    "recommendations": fast_rows,
                }
            )
        trending_rows = _metadata_trending_rows(n=n, enrich_covers=enrich_covers)
        _recommendation_cache_set(
            user_id=user_id,
            cache_kind="fast",
            signal_fingerprint=signal_fingerprint,
            recommendation_mode="cold_start_fast",
            history_counts=history_counts,
            ranker_ready=bool(recommender.is_loaded),
            recommendations=trending_rows,
        )
        return jsonify(
            {
                "user_id": user_id,
                "recommendation_mode": "cold_start_fast",
                "history_counts": history_counts,
                "covers_enriched": bool(enrich_covers),
                "ranker_ready": bool(recommender.is_loaded),
                "recommendations": trending_rows,
            }
        )

    ranker = None
    ranker_error = ""
    if recommender.is_loaded:
        try:
            ranker = recommender.instance
        except Exception as exc:
            ranker_error = f"{type(exc).__name__}: {exc}"
            logging.exception("Recommender load failed")
    else:
        # Keep API responsive while background preload is warming up.
        ranker_error = "warming_up"

    if ranker is None:
        cached_fallback = _recommendation_cache_get(
            user_id=user_id,
            cache_kind="deep" if has_personal_history else "fast",
            n=n,
            signal_fingerprint=signal_fingerprint,
            allow_cross_mode_fallback=not has_personal_history,
        )
        if cached_fallback:
            cached_fallback["covers_enriched"] = bool(enrich_covers)
            if enrich_covers:
                cached_fallback["recommendations"] = _enrich_missing_recommendation_covers(
                    list(cached_fallback.get("recommendations") or [])
                )
            return jsonify(cached_fallback)

        if has_personal_history:
            seed_rows = _fast_profile_seed_recommendations(
                liked_ids=liked,
                playlist_track_ids=playlist_tracks,
                played_ids=played,
                disliked_ids=disliked,
                n=n,
                enrich_covers=enrich_covers,
            )
            if seed_rows:
                mode = "personalized_seed"
                _recommendation_cache_set(
                    user_id=user_id,
                    cache_kind="deep",
                    signal_fingerprint=signal_fingerprint,
                    recommendation_mode=mode,
                    history_counts=history_counts,
                    ranker_ready=False,
                    recommendations=seed_rows,
                )
                _debug_log(
                    "api.recommend.personal_seed_ok",
                    user_id=user_id,
                    requested_n=requested_n,
                    n=n,
                    returned=len(seed_rows),
                    ranker_error=ranker_error,
                    elapsed_ms=int((time.perf_counter() - started) * 1000),
                )
                return jsonify(
                    {
                        "user_id": user_id,
                        "recommendation_mode": mode,
                        "history_counts": history_counts,
                        "covers_enriched": bool(enrich_covers),
                        "ranker_ready": False,
                        "recommendations": seed_rows,
                    }
                )
            return jsonify(
                {
                    "user_id": user_id,
                    "recommendation_mode": "warming_up",
                    "history_counts": history_counts,
                    "covers_enriched": bool(enrich_covers),
                    "ranker_ready": False,
                    "recommendations": [],
                }
            )

        trending_rows = _metadata_trending_rows(n=n, enrich_covers=enrich_covers)
        mode = "cold_start"
        _recommendation_cache_set(
            user_id=user_id,
            cache_kind="fast",
            signal_fingerprint=signal_fingerprint,
            recommendation_mode=mode,
            history_counts=history_counts,
            ranker_ready=False,
            recommendations=trending_rows,
        )
        _debug_log(
            "api.recommend.cold_start_ok",
            user_id=user_id,
            requested_n=requested_n,
            n=n,
            returned=len(trending_rows),
            ranker_error=ranker_error,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
        return jsonify(
            {
                "user_id": user_id,
                "recommendation_mode": mode,
                "history_counts": history_counts,
                "covers_enriched": bool(enrich_covers),
                "ranker_ready": False,
                "recommendations": trending_rows,
            }
        )

    try:
        results = ranker.recommend(
            liked,
            played,
            k=n,
            disliked_ids=disliked,
            playlist_track_ids=playlist_tracks,
            skip_feedback=skip_summary,
        )
    except TypeError:
        results = ranker.recommend(liked, played, n=n)
    except Exception as exc:
        logging.exception("Recommender request failed")
        cached_after_error = _recommendation_cache_get(
            user_id=user_id,
            cache_kind="deep" if has_personal_history else "fast",
            n=n,
            signal_fingerprint=signal_fingerprint,
            allow_cross_mode_fallback=not has_personal_history,
        )
        if cached_after_error:
            cached_after_error["covers_enriched"] = bool(enrich_covers)
            if enrich_covers:
                cached_after_error["recommendations"] = _enrich_missing_recommendation_covers(
                    list(cached_after_error.get("recommendations") or [])
                )
            return jsonify(cached_after_error)
        if has_personal_history:
            seed_rows = _fast_profile_seed_recommendations(
                liked_ids=liked,
                playlist_track_ids=playlist_tracks,
                played_ids=played,
                disliked_ids=disliked,
                n=n,
                enrich_covers=enrich_covers,
            )
            if seed_rows:
                mode = "personalized_seed"
                _recommendation_cache_set(
                    user_id=user_id,
                    cache_kind="deep",
                    signal_fingerprint=signal_fingerprint,
                    recommendation_mode=mode,
                    history_counts=history_counts,
                    ranker_ready=False,
                    recommendations=seed_rows,
                )
                _debug_log(
                    "api.recommend.personal_seed_after_error",
                    user_id=user_id,
                    requested_n=requested_n,
                    n=n,
                    returned=len(seed_rows),
                    ranker_error=f"{type(exc).__name__}: {exc}",
                    elapsed_ms=int((time.perf_counter() - started) * 1000),
                )
                return jsonify(
                    {
                        "user_id": user_id,
                        "recommendation_mode": mode,
                        "history_counts": history_counts,
                        "covers_enriched": bool(enrich_covers),
                        "ranker_ready": False,
                        "recommendations": seed_rows,
                    }
                )
            return jsonify(
                {
                    "user_id": user_id,
                    "recommendation_mode": "warming_up",
                    "history_counts": history_counts,
                    "covers_enriched": bool(enrich_covers),
                    "ranker_ready": False,
                    "recommendations": [],
                }
            )
        results = _metadata_trending_rows(n=n, enrich_covers=enrich_covers)
        mode = "cold_start"
        _recommendation_cache_set(
            user_id=user_id,
            cache_kind="fast",
            signal_fingerprint=signal_fingerprint,
            recommendation_mode=mode,
            history_counts=history_counts,
            ranker_ready=False,
            recommendations=results,
        )
        return jsonify(
            {
                "user_id": user_id,
                "recommendation_mode": mode,
                "history_counts": history_counts,
                "covers_enriched": bool(enrich_covers),
                "ranker_ready": False,
                "recommendations": results,
            }
        )
    engine_quota = {}
    try:
        engine_quota = dict(getattr(ranker, "last_quota_stats", {}) or {})
    except Exception:
        engine_quota = {}
    cacheable_results = list(results or [])
    _recommendation_cache_set(
        user_id=user_id,
        cache_kind="deep",
        signal_fingerprint=signal_fingerprint,
        recommendation_mode=mode,
        history_counts=history_counts,
        ranker_ready=True,
        recommendations=cacheable_results,
        engine_quota=engine_quota,
    )
    _recommendation_cache_set(
        user_id=user_id,
        cache_kind="fast",
        signal_fingerprint=signal_fingerprint,
        recommendation_mode=mode,
        history_counts=history_counts,
        ranker_ready=True,
        recommendations=cacheable_results,
        engine_quota=engine_quota,
    )
    if enrich_covers:
        results = _enrich_missing_recommendation_covers(results)
    _debug_log(
        "api.recommend.ok",
        user_id=user_id,
        mode=mode,
        requested_n=requested_n,
        n=n,
        covers=enrich_covers,
        returned=len(results or []),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )
    return jsonify(
        {
            "user_id": user_id,
            "recommendation_mode": mode,
            "history_counts": history_counts,
            "covers_enriched": bool(enrich_covers),
            "engine_quota": engine_quota,
            "recommendations": results,
        }
    )


# Get trending.
@app.route("/api/trending", methods=["GET"])
def get_trending():
    """
    Get trending.
    
    This function implements the get trending step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    n = int(request.args.get("n", 10))
    enrich_covers = _boolish(request.args.get("covers", "1"))
    rows = []
    if recommender.is_loaded:
        try:
            ranker = recommender.instance
            if ranker is not None:
                rows = ranker.get_trending(n)
        except Exception:
            logging.exception("Trending fallback: recommender failed")
            rows = []
    if not rows:
        rows = _metadata_trending_rows(n=n, enrich_covers=enrich_covers)
    elif enrich_covers:
        rows = _enrich_missing_recommendation_covers(rows)
    return jsonify({"trending": rows})


# Get lyrics via Lyrica (optional) + LRCLIB fallback
@app.route("/api/lyrics", methods=["GET"])
def get_lyrics():
    """
    Get lyrics.
    
    This function implements the get lyrics step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    started = time.perf_counter()
    title = request.args.get("title", "")
    artist = request.args.get("artist", "")
    fast_mode = _boolish(request.args.get("fast", "0"))
    plain_only = _boolish(request.args.get("plain_only", "0"))
    _debug_log(
        "api.lyrics.request",
        title=title,
        artist=artist,
        fast=fast_mode,
        plain_only=plain_only,
    )
    if not title and not artist:
        _debug_log(
            "api.lyrics.error",
            reason="missing title/artist",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
        return (jsonify({"error": "title or artist required"}), 400)

    try:
        result = lyrics_provider_engine.lookup(
            title=title,
            artist=artist,
            fast_mode=fast_mode,
            plain_only=plain_only,
        )
        _debug_log(
            "api.lyrics.ok",
            title=title,
            artist=artist,
            provider=(result or {}).get("provider"),
            synced=(result or {}).get("synced"),
            lyrics_len=len(str((result or {}).get("lyrics", "") or "")),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
        return jsonify(result)
    except Exception as e:
        if lyrics_provider_engine.is_timeout_exception(e):
            logging.warning("Lyrics provider timeout")
        else:
            logging.error(f"Error fetching lyrics: {e}")
        _debug_log(
            "api.lyrics.fail",
            title=title,
            artist=artist,
            error=type(e).__name__,
            message=str(e),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
        return jsonify({"synced": False, "lyrics": ""})


if __name__ == "__main__":
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app.run(debug=False, use_reloader=False, port=5000, host="0.0.0.0")