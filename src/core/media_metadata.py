import base64
import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Tuple

from config import settings

try:
    import yt_dlp
except ImportError:
    yt_dlp = None


_INSTRUMENTAL_HINTS = {
    "instrumental",
    "karaoke",
    "type beat",
    "no vocals",
    "without vocals",
    "lofi beat",
    "lo-fi beat",
    "piano version",
    "orchestral version",
    "backing track",
    "bgm",
    "music only",
    "clean beat",
}
_VOCAL_HINTS = {
    "lyrics",
    "lyric video",
    "official video",
    "official music video",
    "feat.",
    "featuring",
    "ft.",
    "vocals",
    "acoustic",
    "live",
    "cover",
}


# Handle is youtube id.
def is_youtube_id(value):
    """
    Return whether youtube id.
    
    This function implements the is youtube id step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{11}", str(value or "")))


# Handle thumb from video id.
def thumb_from_video_id(video_id):
    """
    Execute thumb from video id.
    
    This function implements the thumb from video id step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    if is_youtube_id(video_id):
        return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
    return ""


# Handle thumbnail candidates.
def thumbnail_candidates(video_id, primary_thumbnail=None):
    """
    Execute thumbnail candidates.
    
    This function implements the thumbnail candidates step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    candidates = []
    if primary_thumbnail:
        candidates.append(primary_thumbnail)
    if is_youtube_id(video_id):
        base = f"https://i.ytimg.com/vi/{video_id}"
        candidates.extend(
            [
                f"{base}/maxresdefault.jpg",
                f"{base}/sddefault.jpg",
                f"{base}/hqdefault.jpg",
                f"{base}/mqdefault.jpg",
                f"{base}/default.jpg",
            ]
        )
    deduped = []
    seen = set()
    for url in candidates:
        if url and url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


# Handle as optional bool.
def as_optional_bool(value) -> Optional[bool]:
    """
    Execute as optional bool.
    
    This function implements the as optional bool step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
    return None


# Clean description.
def clean_description(raw_value: str, max_chars: int = 280) -> str:
    """
    Execute clean description.
    
    This function implements the clean description step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    text = str(raw_value or "")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return f"{clipped}..."


# Parse list-like metadata fields (list or JSON-like string) into strings.
def parse_metadata_list(value, lowercase: bool = False):
    """
    Parse metadata list.
    
    This function implements the parse metadata list step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    items = []
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        text = value.strip()
        if text:
            try:
                parsed = json.loads(text.replace("'", '"'))
                if isinstance(parsed, list):
                    items = parsed
            except Exception:
                items = []
    out = []
    seen = set()
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        if lowercase:
            text = text.lower()
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


# Convert score-like values to unit floats [0, 1], accepting 0-100 inputs.
def safe_unit_float(value, default: float = 0.0) -> float:
    """
    Safely convert unit float.
    
    This function implements the safe unit float step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    try:
        num = float(value)
    except Exception:
        num = float(default)
    if num > 1.0:
        num = num / 100.0
    return float(min(1.0, max(0.0, num)))


# Bucket a unit score into low/medium/high.
def bucket_unit_score(value: float, low: float = 0.35, high: float = 0.67) -> str:
    """
    Execute bucket unit score.
    
    This function implements the bucket unit score step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    v = safe_unit_float(value, default=0.0)
    if v < low:
        return "low"
    if v > high:
        return "high"
    return "medium"


# Derive vibe tags from audio feature scores.
def derive_audio_vibe_tokens(
    tempo: float,
    energy: float,
    brightness: float,
    mood: float,
    valence: float,
):
    """
    Execute derive audio vibe tokens.
    
    This function implements the derive audio vibe tokens step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    tempo = safe_unit_float(tempo, default=0.0)
    energy = safe_unit_float(energy, default=0.0)
    brightness = safe_unit_float(brightness, default=0.0)
    mood = safe_unit_float(mood, default=0.0)
    valence = safe_unit_float(valence, default=0.0)
    vibes = []
    if mood >= 0.68:
        vibes.append("moody")
    if valence >= 0.66:
        vibes.append("uplifting")
    elif valence <= 0.36:
        vibes.append("melancholic")
    if energy >= 0.66:
        vibes.append("energetic")
    elif energy <= 0.38:
        vibes.append("chill")
    if tempo >= 0.70:
        vibes.append("fast")
    elif tempo <= 0.38:
        vibes.append("slow")
    if brightness <= 0.35:
        vibes.append("dark")
    elif brightness >= 0.68:
        vibes.append("bright")
    deduped = []
    seen = set()
    for vibe in vibes:
        if vibe and vibe not in seen:
            seen.add(vibe)
            deduped.append(vibe)
    return deduped


# Build track embedding text.
def build_track_embedding_text(
    title: str = "",
    artist: str = "",
    tags: Optional[Iterable[str]] = None,
    description: str = "",
    instrumental: Optional[bool] = None,
    max_description_chars: int = 220,
) -> str:
    """
    Build track embedding text.
    
    This function implements the build track embedding text step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    safe_title = str(title or "Unknown").strip() or "Unknown"
    safe_artist = str(artist or "Unknown").strip() or "Unknown"
    normalized_tags = []
    for value in tags or []:
        tag_text = str(value or "").strip()
        if tag_text:
            normalized_tags.append(tag_text)
    tag_text = " ".join(normalized_tags)
    safe_description = clean_description(description, max_chars=max_description_chars)
    if instrumental is True:
        vocal_type = "instrumental"
    elif instrumental is False:
        vocal_type = "non-instrumental"
    else:
        vocal_type = "unknown-vocals"
    parts = [f"{safe_title} by {safe_artist}"]
    if tag_text:
        parts.append(f"Tags: {tag_text}")
    if safe_description:
        parts.append(f"Description: {safe_description}")
    parts.append(f"Vocal type: {vocal_type}")
    return ". ".join(parts)


# Build context-first embedding text (used by search indexing + recommendation runtime).
def build_track_embedding_text_context_first(
    meta: Optional[dict] = None,
    *,
    title: str = "",
    artist: str = "",
    max_description_chars: int = 280,
) -> str:
    """
    Build track embedding text context first.
    
    This function implements the build track embedding text context first step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    meta = meta or {}
    safe_title = str(
        title or meta.get("title") or meta.get("name") or "Unknown"
    ).strip() or "Unknown"
    safe_artist = str(
        artist or meta.get("artist_name") or meta.get("artist") or "Unknown"
    ).strip() or "Unknown"

    tags = parse_metadata_list(meta.get("deezer_tags", []), lowercase=True)
    genres = parse_metadata_list(meta.get("genres", []), lowercase=True)
    vibe_tags = parse_metadata_list(meta.get("vibe_tags", []), lowercase=True)

    tempo = safe_unit_float(meta.get("tempo", 0.0), default=0.0)
    energy = safe_unit_float(meta.get("energy", 0.0), default=0.0)
    brightness = safe_unit_float(meta.get("brightness", 0.0), default=0.0)
    mood = safe_unit_float(meta.get("mood", 0.0), default=0.0)
    valence = safe_unit_float(meta.get("valence", 0.0), default=0.0)

    audio_vibes = derive_audio_vibe_tokens(tempo, energy, brightness, mood, valence)
    if audio_vibes:
        vibe_tags = list(dict.fromkeys(vibe_tags + audio_vibes))

    description = clean_description(
        str(meta.get("description", "") or ""), max_chars=max_description_chars
    )

    instrumental = as_optional_bool(meta.get("instrumental"))
    if instrumental is True:
        vocal_type = "instrumental"
    elif instrumental is False:
        vocal_type = "non-instrumental"
    else:
        vocal_type = "unknown-vocals"

    parts = []
    if genres:
        parts.append(f"Genre profile: {' '.join(genres[:10])}")
    if tags:
        parts.append(f"Tag profile: {' '.join(tags[:16])}")
    if vibe_tags:
        parts.append(f"Vibe profile: {' '.join(vibe_tags[:16])}")
    if description:
        parts.append(f"Description: {description}")

    parts.append(
        "Audio feature scores: "
        f"tempo {tempo:.3f}, energy {energy:.3f}, brightness {brightness:.3f}, "
        f"mood {mood:.3f}, valence {valence:.3f}."
    )
    parts.append(
        "Audio feature buckets: "
        f"tempo {bucket_unit_score(tempo)}, energy {bucket_unit_score(energy)}, "
        f"brightness {bucket_unit_score(brightness)}, mood {bucket_unit_score(mood)}, "
        f"valence {bucket_unit_score(valence)}."
    )
    parts.append(f"Vocal type: {vocal_type}")
    parts.append(f"Identity reference: artist {safe_artist}. title {safe_title}.")
    return ". ".join([p.strip() for p in parts if p and str(p).strip()])


# Infer instrumental from text.
def infer_instrumental_from_text(
    title: str = "",
    description: str = "",
    tags: Optional[Iterable[str]] = None,
    categories: Optional[Iterable[str]] = None,
) -> Tuple[bool, float]:
    """
    Execute infer instrumental from text.
    
    This function implements the infer instrumental from text step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    tags = tags or []
    categories = categories or []
    parts = [
        str(title or ""),
        str(description or ""),
        " ".join(str(v or "") for v in tags),
        " ".join(str(v or "") for v in categories),
    ]
    text = " ".join(parts).lower()
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return False, 0.0

    score = 0
    if "non instrumental" in text or "not instrumental" in text:
        score -= 3

    for hint in _INSTRUMENTAL_HINTS:
        if hint in text:
            score += 2
    for hint in _VOCAL_HINTS:
        if hint in text:
            score -= 2

    instrumental = score >= 2
    confidence = min(abs(score) / 8.0, 1.0)
    return instrumental, float(confidence)


class VideoResolver:
    # Initialize class state.
    def __init__(self):
        """
        Initialize the instance state.
        
        This method implements the init step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        self._ydlp_module = yt_dlp
        self._ydlp_bin = shutil.which("yt-dlp") or shutil.which("yt-dlp.exe")
        self.enabled = self._ydlp_module is not None or self._ydlp_bin is not None
        self.extractor_args_raw = str(
            os.getenv(
                "YTDLP_EXTRACTOR_ARGS",
                "youtube:player_client=default,-android_sdkless",
            )
            or ""
        ).strip()
        self.extractor_args = self._parse_extractor_args(self.extractor_args_raw)
        self.search_results_limit = max(5, int(os.getenv("YTDLP_SEARCH_RESULTS", "15") or 15))
        self.candidate_limit = max(
            5, int(os.getenv("YTDLP_CANDIDATE_LIMIT", "10") or 10)
        )

    # Build query.
    @staticmethod
    def _build_query(title="", artist="", track_id=""):
        """
        Build query.
        
        This method implements the build query step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        q_parts = [artist or "", title or ""]
        query = " ".join([p.strip() for p in q_parts if p and str(p).strip()])
        if not query:
            query = str(track_id or "").strip()
        return query

    # Parse yt-dlp extractor args CLI string into Python API dict shape.
    @staticmethod
    def _parse_extractor_args(raw):
        """
        Parse extractor args.
        
        This method implements the parse extractor args step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        text = str(raw or "").strip()
        if not text:
            return None
        # Example: "youtube:player_client=default,-android_sdkless;skip=..."
        if ":" not in text:
            return None
        extractor, spec = text.split(":", 1)
        extractor = extractor.strip()
        spec = spec.strip()
        if not extractor or not spec:
            return None
        parsed = {}
        for chunk in spec.split(";"):
            chunk = str(chunk or "").strip()
            if not chunk or "=" not in chunk:
                continue
            key, value = chunk.split("=", 1)
            key = key.strip()
            if not key:
                continue
            items = [v.strip() for v in str(value or "").split(",") if v.strip()]
            if not items:
                continue
            parsed[key] = items
        if not parsed:
            return None
        return {extractor: parsed}

    # Resolve with module.
    @staticmethod
    def _norm_search_text(value):
        """
        Execute norm search text.
        
        This method implements the norm search text step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        text = str(value or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-z0-9 ]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    # Tokenize normalized title text for fuzzy overlap scoring.
    @staticmethod
    def _title_tokens(value):
        """
        Execute title tokens.
        
        This method implements the title tokens step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        text = VideoResolver._norm_search_text(value)
        if not text:
            return []
        # Remove extremely common noise terms that appear in many uploads.
        stop = {
            "official",
            "audio",
            "video",
            "lyrics",
            "lyric",
            "topic",
            "hd",
            "hq",
        }
        return [tok for tok in text.split(" ") if tok and tok not in stop]

    # Score yt-dlp search entries by title+artist fit to reduce duplicate-title mismatches.
    def _score_search_entry(self, entry, title="", artist=""):
        """
        Score search entry.
        
        This method implements the score search entry step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if not isinstance(entry, dict):
            return -999.0
        q_title = self._norm_search_text(title)
        q_artist = self._norm_search_text(artist)
        e_title = self._norm_search_text(entry.get("title") or "")
        e_uploader = self._norm_search_text(
            entry.get("uploader") or entry.get("channel") or entry.get("channel_name") or ""
        )
        score = 0.0
        title_phrase_match = False
        if q_title and e_title:
            if q_title == e_title:
                score += 12.0
                title_phrase_match = True
            elif q_title in e_title:
                score += 9.0
                title_phrase_match = True
            elif e_title in q_title:
                score += 2.0
            # Token-overlap scoring catches punctuation/case variants and "on"/"the" formatting.
            q_title_tokens = set(self._title_tokens(q_title))
            e_title_tokens = set(self._title_tokens(e_title))
            if q_title_tokens and e_title_tokens:
                overlap = len(q_title_tokens & e_title_tokens) / max(
                    1, len(q_title_tokens)
                )
                if overlap >= 0.99:
                    score += 6.0
                elif overlap >= 0.75:
                    score += 4.0
                elif overlap >= 0.5:
                    score += 1.5
                else:
                    # Strongly penalize same-artist but wrong-title results.
                    score -= 8.0
            elif q_title and e_title and not title_phrase_match:
                score -= 8.0
        if q_artist:
            if q_artist and e_title and q_artist in e_title:
                score += 6.0
            if q_artist and e_uploader and q_artist == e_uploader:
                score += 8.0
            elif q_artist and e_uploader and q_artist in e_uploader:
                score += 5.0
        # Tune fallback preferences: avoid live/performance uploads, prefer playable lyric uploads
        # over embed-blocked official/live variants when exact metadata video fails.
        title_text = str(entry.get("title") or "").lower()
        if "official audio" in title_text:
            score += 0.5
        elif "official video" in title_text:
            score += 0.25
        if "lyrics" in title_text or "lyric" in title_text:
            score += 2.0
        if "live" in title_text or "concert" in title_text:
            score -= 8.0
        if "session" in title_text or "acoustic" in title_text:
            score -= 4.0
        # Strongly demote transformation uploads for playback fallback quality.
        transform_noise = (
            "slowed",
            "sped up",
            "speed up",
            "reverb",
            "8d",
            "remix",
            "karaoke",
            "instrumental",
        )
        if any(token in title_text for token in transform_noise):
            score -= 5.0
        # If the artist matches strongly but the title does not, suppress the result.
        if q_title and e_title and not title_phrase_match:
            q_title_tokens = set(self._title_tokens(q_title))
            e_title_tokens = set(self._title_tokens(e_title))
            if q_title_tokens and e_title_tokens and not (q_title_tokens & e_title_tokens):
                score -= 4.0
        return score

    def _resolve_with_module(self, query, title, artist):
        """
        Resolve with module.
        
        This method implements the resolve with module step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "skip_download": True,
            "noplaylist": True,
            "socket_timeout": 8,
            "retries": 0,
        }
        if self.extractor_args:
            ydl_opts["extractor_args"] = self.extractor_args
        search_exprs = [
            f"ytsearch{self.search_results_limit}:{query} official audio",
            f"ytsearch{self.search_results_limit}:{query} lyrics",
            f"ytsearch{self.search_results_limit}:{query}",
        ]
        scored_entries = []
        seen_ids = set()
        for search_expr in search_exprs:
            try:
                with self._ydlp_module.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(search_expr, download=False)
            except Exception:
                continue
            entries = info.get("entries") if isinstance(info, dict) else None
            entries = entries or []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                video_id = entry.get("id")
                if not is_youtube_id(video_id):
                    continue
                video_id = str(video_id)
                if video_id in seen_ids:
                    continue
                seen_ids.add(video_id)
                entry = dict(entry)
                entry["_video_id"] = video_id
                entry["_score"] = self._score_search_entry(entry, title=title, artist=artist)
                scored_entries.append(entry)
        if scored_entries:
            scored_entries.sort(key=lambda e: float(e.get("_score", -999.0)), reverse=True)
            # Keep top unique candidate IDs in ranked order.
            candidate_ids = []
            seen_ids = set()
            for entry in scored_entries:
                video_id = str(entry.get("_video_id") or "")
                if not video_id or video_id in seen_ids:
                    continue
                seen_ids.add(video_id)
                candidate_ids.append(video_id)
                if len(candidate_ids) >= self.candidate_limit:
                    break
            first_entry = next((e for e in scored_entries if str(e.get("_video_id") or "") == candidate_ids[0]), None) if candidate_ids else None
        else:
            candidate_ids = []
            first_entry = None
        if first_entry and candidate_ids:
            video_id = candidate_ids[0]
            return {
                "video_id": video_id,
                "video_candidates": candidate_ids,
                "thumbnail": first_entry.get("thumbnail") or thumb_from_video_id(video_id),
                "thumbnail_candidates": thumbnail_candidates(
                    video_id, first_entry.get("thumbnail") or thumb_from_video_id(video_id)
                ),
                "title": first_entry.get("title") or title,
            }
        return None

    # Resolve with binary.
    def _resolve_with_binary(self, query, title, artist):
        """
        Resolve with binary.
        
        This method implements the resolve with binary step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if not self._ydlp_bin:
            return None
        search_exprs = [
            f"ytsearch{self.search_results_limit}:{query} official audio",
            f"ytsearch{self.search_results_limit}:{query} lyrics",
            f"ytsearch{self.search_results_limit}:{query}",
        ]
        scored_entries = []
        seen_ids = set()
        for search_expr in search_exprs:
            cmd = [
                self._ydlp_bin,
                "--dump-single-json",
                "--flat-playlist",
                "--skip-download",
                "--no-warnings",
                "--quiet",
                "--no-playlist",
            ]
            if self.extractor_args_raw:
                cmd.extend(["--extractor-args", self.extractor_args_raw])
            cmd.append(search_expr)
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=20, check=False
                )
            except Exception:
                continue
            if proc.returncode != 0 or not proc.stdout.strip():
                continue
            try:
                payload = json.loads(proc.stdout)
            except Exception:
                continue
            entries = payload.get("entries") if isinstance(payload, dict) else None
            entries = entries or []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                video_id = entry.get("id")
                if not is_youtube_id(video_id):
                    continue
                video_id = str(video_id)
                if video_id in seen_ids:
                    continue
                seen_ids.add(video_id)
                entry = dict(entry)
                entry["_video_id"] = video_id
                entry["_score"] = self._score_search_entry(entry, title=title, artist=artist)
                scored_entries.append(entry)
        if scored_entries:
            scored_entries.sort(key=lambda e: float(e.get("_score", -999.0)), reverse=True)
            candidate_ids = []
            seen_ids = set()
            for entry in scored_entries:
                video_id = str(entry.get("_video_id") or "")
                if not video_id or video_id in seen_ids:
                    continue
                seen_ids.add(video_id)
                candidate_ids.append(video_id)
                if len(candidate_ids) >= self.candidate_limit:
                    break
            first_entry = next((e for e in scored_entries if str(e.get("_video_id") or "") == candidate_ids[0]), None) if candidate_ids else None
        else:
            candidate_ids = []
            first_entry = None
        if first_entry and candidate_ids:
            video_id = candidate_ids[0]
            return {
                "video_id": video_id,
                "video_candidates": candidate_ids,
                "thumbnail": first_entry.get("thumbnail") or thumb_from_video_id(video_id),
                "thumbnail_candidates": thumbnail_candidates(
                    video_id, first_entry.get("thumbnail") or thumb_from_video_id(video_id)
                ),
                "title": first_entry.get("title") or title,
            }
        return None

    # Resolve this operation.
    @lru_cache(maxsize=5000)
    def resolve(self, title="", artist="", track_id=""):
        """
        Execute resolve.
        
        This method implements the resolve step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if not self.enabled:
            return None
        query = self._build_query(title=title, artist=artist, track_id=track_id)
        if not query:
            return None
        if self._ydlp_module is not None:
            resolved = self._resolve_with_module(query, title, artist)
            if resolved:
                return resolved
        return self._resolve_with_binary(query, title, artist)


class SpotifyCoverResolver:
    # Resolve album cover art via Spotify Web API (client credentials flow).
    def __init__(self):
        """
        Initialize the instance state.
        
        This method implements the init step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        self.client_id = str(os.getenv("SPOTIFY_CLIENT_ID", "") or "").strip()
        self.client_secret = str(os.getenv("SPOTIFY_CLIENT_SECRET", "") or "").strip()
        self.enabled = bool(self.client_id and self.client_secret)
        self.timeout_sec = max(
            1.0, float(os.getenv("SPOTIFY_TIMEOUT_SEC", "4.5") or 4.5)
        )
        self._access_token = ""
        self._access_token_expires_at = 0.0

    # Normalize matching text.
    @staticmethod
    def _norm(value):
        """
        Execute norm.
        
        This method implements the norm step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        text = str(value or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(
            r"\s*[\(\[\-–—].*?(official|lyrics?|audio|video).*?[\)\]]?\s*$",
            "",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\s*[\(\[]?\s*(feat\.?|featuring|ft\.)\s+[^)\]]+[\)\]]?\s*$",
            "",
            text,
            flags=re.I,
        )
        text = re.sub(r"[^a-z0-9 ]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    # Score track candidate against request.
    def _score_track(self, item, title, artist):
        """
        Score track.
        
        This method implements the score track step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        q_title = self._norm(title)
        q_artist = self._norm(artist)
        item_title = self._norm(item.get("name") or "")
        artists = item.get("artists") if isinstance(item, dict) else []
        item_artists = " ".join(
            [self._norm(a.get("name")) for a in (artists or []) if isinstance(a, dict)]
        ).strip()
        score = 0
        if q_title and item_title == q_title:
            score += 8
        elif q_title and q_title in item_title:
            score += 5
        elif q_title and item_title in q_title:
            score += 2
        if q_artist and item_artists:
            if q_artist == item_artists:
                score += 7
            elif q_artist in item_artists:
                score += 5
            elif any(
                q_artist == self._norm(a.get("name"))
                for a in (artists or [])
                if isinstance(a, dict)
            ):
                score += 6
        popularity = int(item.get("popularity") or 0)
        score += min(popularity, 100) / 100.0
        return float(score)

    # Get Spotify access token with in-memory expiry cache.
    def _get_access_token(self):
        """
        Get access token.
        
        This method implements the get access token step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if not self.enabled:
            return ""
        now = time.time()
        if self._access_token and now < (self._access_token_expires_at - 30):
            return self._access_token
        raw = f"{self.client_id}:{self.client_secret}".encode("utf-8")
        basic = base64.b64encode(raw).decode("ascii")
        data = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode(
            "utf-8"
        )
        req = urllib.request.Request(
            "https://accounts.spotify.com/api/token",
            data=data,
            method="POST",
            headers={
                "Authorization": f"Basic {basic}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8", errors="ignore"))
        token = str(payload.get("access_token", "") or "").strip()
        expires_in = max(60, int(payload.get("expires_in") or 3600))
        if token:
            self._access_token = token
            self._access_token_expires_at = now + expires_in
        return token

    # Build Spotify search query.
    @staticmethod
    def _build_query(title="", artist="", track_id=""):
        """
        Build query.
        
        This method implements the build query step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        safe_title = str(title or "").strip()
        safe_artist = str(artist or "").strip()
        if safe_title and safe_artist:
            return f'track:"{safe_title}" artist:"{safe_artist}"'
        if safe_title:
            return safe_title
        if safe_artist:
            return safe_artist
        return str(track_id or "").strip()

    # Extract deduped album image URLs (largest first).
    @staticmethod
    def _album_image_candidates(track_item):
        """
        Execute album image candidates.
        
        This method implements the album image candidates step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        album = track_item.get("album") if isinstance(track_item, dict) else {}
        images = album.get("images") if isinstance(album, dict) else []
        if not isinstance(images, list):
            return []
        urls = []
        seen = set()
        for img in images:
            url = str(img.get("url") if isinstance(img, dict) else "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            urls.append(url)
        return urls

    # Search Spotify and return cover candidates.
    @lru_cache(maxsize=4000)
    def resolve_cover(self, title="", artist="", track_id=""):
        """
        Resolve cover.
        
        This method implements the resolve cover step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if not self.enabled:
            return None
        query = self._build_query(title=title, artist=artist, track_id=track_id)
        if not query:
            return None
        token = self._get_access_token()
        if not token:
            return None

        url = (
            "https://api.spotify.com/v1/search?"
            + urllib.parse.urlencode(
                {"q": query, "type": "track", "limit": 5, "market": "US"}
            )
        )
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as response:
                payload = json.loads(response.read().decode("utf-8", errors="ignore"))
        except Exception:
            return None

        tracks = payload.get("tracks") if isinstance(payload, dict) else {}
        items = tracks.get("items") if isinstance(tracks, dict) else []
        if not isinstance(items, list) or not items:
            return None

        best_item = None
        best_score = -1.0
        for item in items:
            if not isinstance(item, dict):
                continue
            score = self._score_track(item, title, artist)
            if score > best_score:
                best_score = score
                best_item = item

        if not isinstance(best_item, dict):
            return None
        cover_candidates = self._album_image_candidates(best_item)
        if not cover_candidates:
            return None

        artists = best_item.get("artists") if isinstance(best_item, dict) else []
        first_artist = ""
        if isinstance(artists, list):
            for a in artists:
                if isinstance(a, dict) and str(a.get("name", "")).strip():
                    first_artist = str(a.get("name") or "").strip()
                    break
        return {
            "thumbnail": cover_candidates[0],
            "thumbnail_candidates": cover_candidates,
            "spotify_track_id": str(best_item.get("id", "") or ""),
            "spotify_uri": str(best_item.get("uri", "") or ""),
            "spotify_title": str(best_item.get("name", "") or title or ""),
            "spotify_artist": first_artist or str(artist or ""),
        }

class YouTubeMetadataEnricher:
    # Initialize class state.
    def __init__(self, db_path=None):
        """
        Initialize the instance state.
        
        This method implements the init step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        self.db_path = Path(db_path or settings.API_CACHE_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ydlp_module = yt_dlp
        self._ydlp_bin = shutil.which("yt-dlp") or shutil.which("yt-dlp.exe")
        self.enabled = self._ydlp_module is not None or self._ydlp_bin is not None
        self._init_db()

    # Internal helper to init db.
    def _init_db(self):
        """
        Initialize db.
        
        This method implements the init db step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ytdlp_track_meta (
                track_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()

    # Read cached.
    def _read_cached(self, track_id: str):
        """
        Execute read cached.
        
        This method implements the read cached step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        conn = sqlite3.connect(str(self.db_path))
        row = conn.execute(
            "SELECT payload_json FROM ytdlp_track_meta WHERE track_id=?",
            (str(track_id),),
        ).fetchone()
        conn.close()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except Exception:
            return None

    # Write cached.
    def _write_cached(self, track_id: str, payload):
        """
        Execute write cached.
        
        This method implements the write cached step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """
            INSERT OR REPLACE INTO ytdlp_track_meta (track_id, payload_json, updated_at)
            VALUES (?, ?, ?)
            """,
            (str(track_id), json.dumps(payload, ensure_ascii=False), time.time()),
        )
        conn.commit()
        conn.close()

    # Internal helper to normalize payload.
    @staticmethod
    def _normalize_payload(track_id: str, title: str, artist: str, info):
        """
        Normalize payload.
        
        This method implements the normalize payload step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        description_raw = str(info.get("description", "") if isinstance(info, dict) else "")
        description = clean_description(
            description_raw,
            max_chars=getattr(settings, "TRACK_DESCRIPTION_MAX_CHARS", 280),
        )
        tags = info.get("tags", []) if isinstance(info, dict) else []
        categories = info.get("categories", []) if isinstance(info, dict) else []
        detected_title = str((info.get("title") if isinstance(info, dict) else "") or title or "")
        instrumental, confidence = infer_instrumental_from_text(
            title=detected_title,
            description=description_raw,
            tags=tags if isinstance(tags, list) else [],
            categories=categories if isinstance(categories, list) else [],
        )
        return {
            "id": str(track_id or ""),
            "title": detected_title,
            "artist": str(artist or ""),
            "description": description,
            "instrumental": bool(instrumental),
            "instrumental_confidence": float(confidence),
            "source": "yt-dlp",
        }

    # Extract with module.
    def _extract_with_module(self, track_id: str):
        """
        Extract with module.
        
        This method implements the extract with module step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if not self._ydlp_module or not is_youtube_id(track_id):
            return None
        url = f"https://www.youtube.com/watch?v={track_id}"
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "socket_timeout": int(getattr(settings, "YTDLP_SOCKET_TIMEOUT_SEC", 8)),
            "retries": 0,
        }
        try:
            with self._ydlp_module.YoutubeDL(ydl_opts) as ydl:
                return ydl.extract_info(url, download=False)
        except Exception:
            return None

    # Extract with binary.
    def _extract_with_binary(self, track_id: str):
        """
        Extract with binary.
        
        This method implements the extract with binary step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if not self._ydlp_bin or not is_youtube_id(track_id):
            return None
        cmd = [
            self._ydlp_bin,
            "--dump-single-json",
            "--skip-download",
            "--no-warnings",
            "--quiet",
            f"https://www.youtube.com/watch?v={track_id}",
        ]
        timeout = int(getattr(settings, "YTDLP_METADATA_TIMEOUT_SEC", 12))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        except Exception:
            return None
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        try:
            return json.loads(proc.stdout)
        except Exception:
            return None

    # Enrich this operation.
    def enrich(self, track_id: str, title: str = "", artist: str = "", force: bool = False):
        """
        Execute enrich.
        
        This method implements the enrich step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        normalized_track_id = str(track_id or "").strip()
        if not normalized_track_id:
            instrumental, confidence = infer_instrumental_from_text(title=title)
            return {
                "id": "",
                "title": str(title or ""),
                "artist": str(artist or ""),
                "description": "",
                "instrumental": bool(instrumental),
                "instrumental_confidence": float(confidence),
                "source": "heuristic",
            }
        if not force:
            cached = self._read_cached(normalized_track_id)
            if cached is not None:
                return cached

        info = None
        if self.enabled:
            if self._ydlp_module is not None:
                info = self._extract_with_module(normalized_track_id)
            if info is None:
                info = self._extract_with_binary(normalized_track_id)

        if info is None:
            instrumental, confidence = infer_instrumental_from_text(title=title)
            payload = {
                "id": normalized_track_id,
                "title": str(title or ""),
                "artist": str(artist or ""),
                "description": "",
                "instrumental": bool(instrumental),
                "instrumental_confidence": float(confidence),
                "source": "heuristic",
            }
        else:
            payload = self._normalize_payload(normalized_track_id, title, artist, info)

        self._write_cached(normalized_track_id, payload)
        return payload