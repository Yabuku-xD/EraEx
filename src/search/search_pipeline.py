import json
import pickle
import re
import time
from collections import Counter, OrderedDict
from pathlib import Path

import faiss
import numpy as np

from config import settings
from src.core.text_embeddings import embedding_handler
from src.core.media_metadata import (
    as_optional_bool,
    build_track_embedding_text,
    thumb_from_video_id,
    thumbnail_candidates,
)


class UnifiedSearchPipeline:

    MOOD_TOKENS = {
        "sad",
        "happy",
        "chill",
        "melancholy",
        "calm",
        "romantic",
        "dark",
        "dreamy",
        "nostalgic",
        "upbeat",
        "focus",
        "relax",
        "midnight",
        "late",
        "night",
        "moody",
        "vibey",
        "vibe",
    }
    LIKE_PATTERNS = (
        r"\bsongs?\s+like\s+(.+)$",
        r"\btracks?\s+like\s+(.+)$",
        r"\bmusic\s+like\s+(.+)$",
        r"\bsimilar\s+to\s+(.+)$",
        r"\blike\s+(.+)$",
    )
    ANCHOR_STOP_TOKENS = {
        "song",
        "songs",
        "track",
        "tracks",
        "music",
        "artist",
        "a",
        "an",
        "the",
        "me",
        "give",
        "show",
        "please",
    }
    STRICT_ARTIST_BLOCK_TOKENS = {
        "song",
        "songs",
        "track",
        "tracks",
        "music",
        "playlist",
        "like",
        "similar",
        "recommend",
        "recommendation",
        "mood",
        "vibe",
    }
    INSTRUMENTAL_HINTS = {
        "instrumental",
        "no vocals",
        "without vocals",
        "karaoke",
        "backing track",
        "bgm",
        "music only",
        "lofi beat",
        "lo fi beat",
        "lo-fi beat",
    }
    VOCAL_HINTS = {
        "non instrumental",
        "not instrumental",
        "with vocals",
        "with singing",
        "lyrics",
        "lyric",
        "vocal",
        "vocals",
        "singer",
        "singing",
    }
    QUERY_STOP_TOKENS = {
        "a",
        "an",
        "and",
        "any",
        "best",
        "for",
        "from",
        "give",
        "i",
        "in",
        "is",
        "like",
        "me",
        "music",
        "my",
        "of",
        "playlist",
        "please",
        "recommend",
        "recommendation",
        "show",
        "song",
        "songs",
        "some",
        "that",
        "the",
        "to",
        "track",
        "tracks",
        "with",
        "by",
    }
    FACET_HINT_TOKENS = {
        "ambient",
        "calm",
        "chill",
        "classical",
        "dark",
        "dance",
        "dreamy",
        "drill",
        "edm",
        "electronic",
        "energetic",
        "focus",
        "drive",
        "funk",
        "happy",
        "hip",
        "hop",
        "house",
        "jazz",
        "late",
        "lofi",
        "melancholy",
        "metal",
        "midnight",
        "moody",
        "night",
        "drive",
        "nostalgic",
        "party",
        "pop",
        "rnb",
        "rap",
        "relax",
        "rock",
        "romantic",
        "sad",
        "soul",
        "study",
        "trap",
        "upbeat",
        "vibe",
        "vibey",
    }
    FACET_SOFT_TOKENS = {
        "calm",
        "chill",
        "dark",
        "dreamy",
        "focus",
        "happy",
        "late",
        "melancholy",
        "midnight",
        "moody",
        "night",
        "nostalgic",
        "relax",
        "romantic",
        "sad",
        "upbeat",
        "vibe",
        "vibey",
    }
    HARD_FOCUS_EXCLUDE_TOKENS = {
        "ambient",
        "classical",
        "dance",
        "drill",
        "edm",
        "electronic",
        "funk",
        "hip",
        "hop",
        "house",
        "jazz",
        "lofi",
        "metal",
        "pop",
        "rap",
        "rnb",
        "rock",
        "soul",
        "study",
        "trap",
    }
    FACET_TOKEN_VARIANTS = {
        "rnb": {"rnb", "soul", "neosoul"},
        "soul": {"soul", "rnb", "neosoul"},
        "trap": {"trap", "rap", "hiphop", "drill"},
        "rap": {"rap", "hiphop", "trap"},
        "hip": {"hip", "hop", "hiphop", "rap", "trap"},
        "hop": {"hop", "hip", "hiphop", "rap", "trap"},
        "lofi": {"lofi", "chill", "ambient"},
        "edm": {"edm", "electronic", "dance", "house", "techno"},
        "electronic": {"electronic", "edm", "house", "techno", "synth"},
        "rock": {"rock", "metal", "punk", "grunge", "indie"},
        "metal": {"metal", "rock", "punk"},
        "midnight": {"midnight", "night", "late"},
        "night": {"night", "midnight", "late"},
        "late": {"late", "night", "midnight"},
    }
    AUDIO_PHRASE_PROFILES = {
        "late night": {
            "tempo": 0.34,
            "energy": 0.32,
            "brightness": 0.28,
            "mood": 0.68,
            "valence": 0.36,
        },
        "night drive": {
            "tempo": 0.56,
            "energy": 0.54,
            "brightness": 0.44,
            "mood": 0.60,
            "valence": 0.42,
        },
        "no vocals": {
            "tempo": 0.52,
            "energy": 0.48,
            "brightness": 0.45,
            "mood": 0.56,
            "valence": 0.52,
        },
        "without vocals": {
            "tempo": 0.52,
            "energy": 0.48,
            "brightness": 0.45,
            "mood": 0.56,
            "valence": 0.52,
        },
    }
    AUDIO_TOKEN_PROFILES = {
        "chill": {
            "tempo": 0.34,
            "energy": 0.26,
            "brightness": 0.42,
            "mood": 0.68,
            "valence": 0.56,
        },
        "calm": {
            "tempo": 0.30,
            "energy": 0.22,
            "brightness": 0.40,
            "mood": 0.66,
            "valence": 0.54,
        },
        "dark": {
            "tempo": 0.44,
            "energy": 0.52,
            "brightness": 0.22,
            "mood": 0.30,
            "valence": 0.22,
        },
        "dreamy": {
            "tempo": 0.40,
            "energy": 0.35,
            "brightness": 0.50,
            "mood": 0.72,
            "valence": 0.58,
        },
        "energetic": {
            "tempo": 0.78,
            "energy": 0.86,
            "brightness": 0.72,
            "mood": 0.66,
            "valence": 0.74,
        },
        "focus": {
            "tempo": 0.42,
            "energy": 0.40,
            "brightness": 0.48,
            "mood": 0.66,
            "valence": 0.50,
        },
        "happy": {
            "tempo": 0.70,
            "energy": 0.74,
            "brightness": 0.68,
            "mood": 0.74,
            "valence": 0.86,
        },
        "melancholy": {
            "tempo": 0.40,
            "energy": 0.34,
            "brightness": 0.34,
            "mood": 0.48,
            "valence": 0.24,
        },
        "midnight": {
            "tempo": 0.48,
            "energy": 0.34,
            "brightness": 0.24,
            "mood": 0.62,
            "valence": 0.30,
        },
        "night": {
            "tempo": 0.50,
            "energy": 0.40,
            "brightness": 0.30,
            "mood": 0.60,
            "valence": 0.34,
        },
        "moody": {
            "tempo": 0.46,
            "energy": 0.44,
            "brightness": 0.36,
            "mood": 0.64,
            "valence": 0.34,
        },
        "nostalgic": {
            "tempo": 0.50,
            "energy": 0.48,
            "brightness": 0.46,
            "mood": 0.66,
            "valence": 0.44,
        },
        "party": {
            "tempo": 0.84,
            "energy": 0.90,
            "brightness": 0.76,
            "mood": 0.76,
            "valence": 0.78,
        },
        "relax": {
            "tempo": 0.32,
            "energy": 0.24,
            "brightness": 0.42,
            "mood": 0.70,
            "valence": 0.52,
        },
        "romantic": {
            "tempo": 0.46,
            "energy": 0.40,
            "brightness": 0.44,
            "mood": 0.72,
            "valence": 0.60,
        },
        "drive": {
            "tempo": 0.62,
            "energy": 0.58,
            "brightness": 0.46,
            "mood": 0.56,
            "valence": 0.46,
        },
        "rnb": {
            "tempo": 0.56,
            "energy": 0.48,
            "brightness": 0.40,
            "mood": 0.66,
            "valence": 0.50,
        },
        "rap": {
            "tempo": 0.70,
            "energy": 0.72,
            "brightness": 0.40,
            "mood": 0.46,
            "valence": 0.40,
        },
        "sad": {
            "tempo": 0.36,
            "energy": 0.32,
            "brightness": 0.30,
            "mood": 0.48,
            "valence": 0.16,
        },
        "trap": {
            "tempo": 0.68,
            "energy": 0.74,
            "brightness": 0.36,
            "mood": 0.44,
            "valence": 0.34,
        },
        "upbeat": {
            "tempo": 0.74,
            "energy": 0.80,
            "brightness": 0.70,
            "mood": 0.74,
            "valence": 0.80,
        },
    }
    BOT_QUERY_PROFILES = {
        "rnb_bot": {
            "label": "R&B Bot",
            "tokens": {
                "rnb",
                "soul",
                "neosoul",
                "smooth",
                "romantic",
                "slow",
                "jam",
                "late",
                "night",
            },
            "phrases": {"r and b", "r&b", "late night", "slow jam"},
        },
        "hiphop_bot": {
            "label": "Hip-Hop Bot",
            "tokens": {
                "hip",
                "hop",
                "rap",
                "trap",
                "drill",
                "bars",
                "freestyle",
                "street",
            },
            "phrases": {"hip hop", "old school rap"},
        },
        "rock_bot": {
            "label": "Rock Bot",
            "tokens": {
                "rock",
                "metal",
                "punk",
                "grunge",
                "guitar",
                "band",
                "alt",
                "indie",
            },
            "phrases": {"alternative rock", "hard rock"},
        },
        "electronic_bot": {
            "label": "Electronic Bot",
            "tokens": {
                "edm",
                "electronic",
                "house",
                "techno",
                "dance",
                "club",
                "synth",
                "bass",
            },
            "phrases": {"night drive", "dance floor"},
        },
        "chill_bot": {
            "label": "Chill Bot",
            "tokens": {
                "chill",
                "calm",
                "focus",
                "study",
                "lofi",
                "ambient",
                "relax",
                "dreamy",
            },
            "phrases": {"lo fi", "late night", "study music"},
        },
    }

    # Initialize class state.
    def __init__(self):
        """
        Initialize the instance state.
        
        This method implements the init step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        self.faiss_index = None
        self.index = None
        self.id_map = []
        self.metadata = {}
        self._artist_to_track_ids = {}
        self._artist_token_to_artists = {}
        self._facet_title_artist_token_to_track_ids = {}
        self._soft_title_artist_token_to_track_ids = {}
        self._result_cache = OrderedDict()
        self._query_vec_cache = OrderedDict()
        self._faiss_cache = OrderedDict()
        self._track_feature_cache = OrderedDict()
        self._track_vec_cache = OrderedDict()
        self._title_artist_token_df = Counter()
        self._max_result_cache = int(getattr(settings, "SEARCH_CACHE_RESULTS_MAX", 256))
        self._max_query_vec_cache = int(getattr(settings, "SEARCH_CACHE_QUERY_VECS_MAX", 384))
        self._max_faiss_cache = int(getattr(settings, "SEARCH_CACHE_FAISS_MAX", 384))
        self._max_track_feature_cache = int(getattr(settings, "SEARCH_CACHE_TRACK_FEATURES_MAX", 80000))
        self._max_track_vec_cache = int(getattr(settings, "SEARCH_CACHE_TRACK_VECS_MAX", 20000))
        self._anchor_cache = {}
        self._search_hparams_cache = None
        self._search_hparams_mtime = None
        self._search_hparams_last_check = 0.0
        self.load_resources()

    # Load resources.
    def load_resources(self):
        """
        Load resources.
        
        This method implements the load resources step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        index_candidates = [
            settings.INDEX_DIR / "faiss_index.bin",
            settings.INDEX_DIR / "faiss.index",
        ]
        for index_path in index_candidates:
            if index_path.exists():
                self.faiss_index = faiss.read_index(str(index_path))
                self.index = self.faiss_index
                break
        map_json_path = settings.INDEX_DIR / "id_map.json"
        map_pickle_path = settings.INDEX_DIR / "track_ids.pkl"
        if map_json_path.exists():
            with open(map_json_path, "r", encoding="utf-8") as f:
                self.id_map = json.load(f)
        elif map_pickle_path.exists():
            with open(map_pickle_path, "rb") as f:
                self.id_map = pickle.load(f)
        meta_path = settings.INDEX_DIR / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        self._build_artist_lookup()
        self._clear_runtime_caches()

    # Internal helper to clear runtime caches.
    def _clear_runtime_caches(self):
        """
        Clear runtime caches.
        
        This method implements the clear runtime caches step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        self._result_cache.clear()
        self._query_vec_cache.clear()
        self._faiss_cache.clear()
        self._track_feature_cache.clear()
        self._track_vec_cache.clear()
        self._anchor_cache.clear()

    # Internal helper to cache get.
    @staticmethod
    def _cache_get(cache: OrderedDict, key):
        """
        Execute cache get.
        
        This method implements the cache get step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if key not in cache:
            return None
        value = cache.pop(key)
        cache[key] = value
        return value

    # Internal helper to cache set.
    @staticmethod
    def _cache_set(cache: OrderedDict, key, value, max_size: int):
        """
        Execute cache set.
        
        This method implements the cache set step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if key in cache:
            cache.pop(key, None)
        cache[key] = value
        while len(cache) > max(1, int(max_size)):
            cache.popitem(last=False)

    # Internal helper to dedupe ranked rows while preserving order.
    def _dedupe_ranked_rows(self, rows):
        """
        Deduplicate ranked rows.
        
        This method implements the dedupe ranked rows step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        deduped = []
        seen_ids = set()
        seen_title_artist = set()
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            track_id = str(row.get("id", "") or "").strip()
            if track_id and track_id in seen_ids:
                continue
            title_norm = self._normalize(row.get("title", ""))
            artist_norm = self._normalize(row.get("artist", ""))
            title_artist_key = (title_norm, artist_norm)
            if title_norm and artist_norm and title_artist_key in seen_title_artist:
                continue
            if track_id:
                seen_ids.add(track_id)
            if title_norm and artist_norm:
                seen_title_artist.add(title_artist_key)
            deduped.append(row)
        return deduped

    # Internal helper to enforce mixed-intent coverage quota in top-k rows.
    def _apply_mixed_intent_topk_quota(self, rows, *, top_k, min_hits, min_coverage):
        """
        Apply mixed intent topk quota.
        
        This method implements the apply mixed intent topk quota step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        ranked = list(rows or [])
        if not ranked:
            return ranked
        k = max(0, min(int(top_k or 0), len(ranked)))
        if k <= 0:
            return ranked
        target_hits = max(0, min(int(min_hits or 0), k))
        if target_hits <= 0:
            return ranked
        coverage_thr = self._clamp01(float(min_coverage or 0.0))

        def _cov(row):
            """
            Execute cov.
            
            This function implements the cov step for this module.
            It is used to keep the broader workflow readable and easier to maintain.
            """
            return self._clamp01(self._safe_float((row or {}).get("mixed_intent_coverage", 0.0), 0.0))

        def _soft(row):
            """
            Execute soft.
            
            This function implements the soft step for this module.
            It is used to keep the broader workflow readable and easier to maintain.
            """
            return self._clamp01(self._safe_float((row or {}).get("mixed_soft_signal", 0.0), 0.0))

        def _score(row):
            """
            Execute score.
            
            This function implements the score step for this module.
            It is used to keep the broader workflow readable and easier to maintain.
            """
            return float(self._safe_float((row or {}).get("score", 0.0), 0.0))

        def _current_hits():
            """
            Execute current hits.
            
            This function implements the current hits step for this module.
            It is used to keep the broader workflow readable and easier to maintain.
            """
            return [i for i in range(k) if _cov(ranked[i]) >= coverage_thr]

        top_hit_indices = _current_hits()
        if len(top_hit_indices) >= target_hits:
            return ranked

        # Repair by swapping in the best qualifying rows from outside top-k.
        max_swaps = max(0, target_hits - len(top_hit_indices))
        swaps = 0
        while swaps < max_swaps:
            top_hit_indices = _current_hits()
            if len(top_hit_indices) >= target_hits:
                break

            pool = [
                idx for idx in range(k, len(ranked))
                if _cov(ranked[idx]) >= coverage_thr
            ]
            if not pool:
                break
            pool.sort(
                key=lambda idx: (_cov(ranked[idx]), _soft(ranked[idx]), _score(ranked[idx])),
                reverse=True,
            )
            candidate_idx = pool[0]

            victims = [idx for idx in range(k) if _cov(ranked[idx]) < coverage_thr]
            if not victims:
                break
            victims.sort(
                key=lambda idx: (
                    _cov(ranked[idx]),
                    _soft(ranked[idx]),
                    _score(ranked[idx]),
                )
            )
            victim_idx = victims[0]
            ranked[victim_idx], ranked[candidate_idx] = ranked[candidate_idx], ranked[victim_idx]
            swaps += 1

        # Preserve mixed-intent ordering quality inside top-k after swaps.
        top_block = list(ranked[:k])
        top_block.sort(
            key=lambda row: (_cov(row), _soft(row), _score(row)),
            reverse=True,
        )
        ranked[:k] = top_block
        return ranked

    # Internal helper to load optional runtime search hyperparameter overrides (hot-reload).
    def _load_search_hparams_overrides(self):
        """
        Load search hparams overrides.
        
        This method implements the load search hparams overrides step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if not bool(getattr(settings, "SEARCH_DYNAMIC_WEIGHTS", True)):
            return {}
        path = getattr(settings, "SEARCH_HPARAMS_PATH", None)
        if not path:
            return {}
        try:
            path = Path(path)
        except Exception:
            return {}
        now = time.time()
        reload_sec = float(getattr(settings, "SEARCH_HPARAMS_RELOAD_SEC", 5.0))
        if (now - float(self._search_hparams_last_check or 0.0)) < max(0.25, reload_sec):
            return self._search_hparams_cache or {}
        self._search_hparams_last_check = now
        if not path.exists():
            self._search_hparams_cache = {}
            self._search_hparams_mtime = None
            return {}
        try:
            mtime = float(path.stat().st_mtime)
            if self._search_hparams_mtime == mtime and isinstance(self._search_hparams_cache, dict):
                return self._search_hparams_cache
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                loaded = {}
            self._search_hparams_cache = loaded
            self._search_hparams_mtime = mtime
            return loaded
        except Exception:
            return self._search_hparams_cache or {}

    # Internal helper to merge numeric overrides into a weight dict.
    @staticmethod
    def _merge_numeric_dict(base_dict, override_dict):
        """
        Merge numeric dict.
        
        This method implements the merge numeric dict step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        out = dict(base_dict or {})
        if not isinstance(override_dict, dict):
            return out
        for key, value in override_dict.items():
            if key not in out:
                continue
            try:
                out[key] = float(value)
            except Exception:
                continue
        return out

    # Internal helper to apply multiplicative adjustments to numeric weights.
    @staticmethod
    def _apply_weight_multipliers(weight_dict, multipliers):
        """
        Apply weight multipliers.
        
        This method implements the apply weight multipliers step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        out = dict(weight_dict or {})
        if not isinstance(multipliers, dict):
            return out
        for key, factor in multipliers.items():
            if key not in out:
                continue
            try:
                out[key] = float(out[key]) * float(factor)
            except Exception:
                continue
        return out

    # Internal helper to normalize positive weights while preserving total magnitude.
    @staticmethod
    def _normalize_positive_weights(weight_dict, keys):
        """
        Normalize positive weights.
        
        This method implements the normalize positive weights step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        out = dict(weight_dict or {})
        keys = [k for k in keys if k in out]
        if not keys:
            return out
        original_total = sum(max(float(out.get(k, 0.0)), 0.0) for k in keys)
        if original_total <= 0:
            return out
        clipped = {k: max(float(out.get(k, 0.0)), 0.0) for k in keys}
        clipped_total = sum(clipped.values())
        if clipped_total <= 0:
            return out
        scale = original_total / clipped_total
        for key in keys:
            out[key] = clipped[key] * scale
        return out

    # Internal helper to resolve dynamic query-adaptive search weights.
    def _resolve_dynamic_search_weights(
        self,
        *,
        query_tokens,
        like_mode,
        strict_artist_mode,
        facet_heavy_query,
        query_audio_target,
        query_year,
        instrumental_intent,
        base_weights,
        like_weights,
        non_like_weights,
        facet_weights,
        instrumental_weights,
    ):
        """
        Resolve dynamic search weights.
        
        This method implements the resolve dynamic search weights step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if not bool(getattr(settings, "SEARCH_DYNAMIC_WEIGHTS", True)):
            return {
                "base": base_weights,
                "like": like_weights,
                "non_like": non_like_weights,
                "facet": facet_weights,
                "instrumental": instrumental_weights,
            }

        qlen = len(query_tokens or set())
        has_audio_intent = query_audio_target is not None
        has_year_intent = bool(query_year)
        has_instrumental_intent = instrumental_intent is not None

        tuned_like = dict(like_weights or {})
        tuned_non_like = dict(non_like_weights or {})
        tuned_facet = dict(facet_weights or {})
        tuned_instr = dict(instrumental_weights or {})
        tuned_base = dict(base_weights or {})

        # Query-adaptive multipliers (runtime heuristic tuning).
        if like_mode:
            tuned_like = self._apply_weight_multipliers(
                tuned_like,
                {
                    "facet": 1.12,
                    "artist": 1.18,
                    "audio": 1.10 if has_audio_intent else 1.0,
                    "popularity": 0.90,
                },
            )
        if strict_artist_mode:
            tuned_non_like = self._apply_weight_multipliers(
                tuned_non_like,
                {
                    "semantic": 0.86,
                    "tag_overlap": 0.92,
                    "facet": 0.85,
                    "description": 0.85,
                    "title_penalty": 0.70,
                    "popularity": 0.90,
                },
            )
        if facet_heavy_query and (not like_mode):
            tuned_non_like = self._apply_weight_multipliers(
                tuned_non_like,
                {
                    "facet": 1.12,
                    "description": 1.10,
                    "audio": 1.08 if has_audio_intent else 1.0,
                    "title_penalty": 1.10,
                },
            )
        if has_audio_intent:
            tuned_non_like = self._apply_weight_multipliers(tuned_non_like, {"audio": 1.25})
            tuned_like = self._apply_weight_multipliers(tuned_like, {"audio": 1.22})
        if has_year_intent:
            tuned_non_like = self._apply_weight_multipliers(tuned_non_like, {"year_bonus": 1.35})
        if has_instrumental_intent:
            tuned_instr = self._apply_weight_multipliers(
                tuned_instr,
                {"match_boost": 1.20, "mismatch_penalty": 1.20},
            )
        if qlen >= 6 and (not strict_artist_mode):
            tuned_non_like = self._apply_weight_multipliers(
                tuned_non_like,
                {"semantic": 0.92, "description": 1.08, "facet": 1.06},
            )
        elif qlen <= 2 and (not strict_artist_mode):
            tuned_non_like = self._apply_weight_multipliers(
                tuned_non_like,
                {"semantic": 1.08, "title_penalty": 0.92},
            )

        # Optional external overrides for hyperparameter tuning without code edits.
        overrides = self._load_search_hparams_overrides()
        if isinstance(overrides, dict):
            tuned_base = self._merge_numeric_dict(tuned_base, overrides.get("SEARCH_WEIGHTS"))
            tuned_like = self._merge_numeric_dict(tuned_like, overrides.get("SEARCH_LIKE_WEIGHTS"))
            tuned_non_like = self._merge_numeric_dict(
                tuned_non_like, overrides.get("SEARCH_NON_LIKE_WEIGHTS")
            )
            tuned_facet = self._merge_numeric_dict(
                tuned_facet, overrides.get("SEARCH_FACET_WEIGHTS")
            )
            tuned_instr = self._merge_numeric_dict(
                tuned_instr, overrides.get("SEARCH_INSTRUMENTAL_WEIGHTS")
            )
            profiles = overrides.get("profiles") if isinstance(overrides.get("profiles"), dict) else {}
            active_profile_names = []
            if like_mode:
                active_profile_names.append("like_mode")
            if strict_artist_mode:
                active_profile_names.append("strict_artist")
            if facet_heavy_query:
                active_profile_names.append("facet_heavy")
            if has_audio_intent:
                active_profile_names.append("audio_intent")
            if has_year_intent:
                active_profile_names.append("year_intent")
            if has_instrumental_intent:
                active_profile_names.append("instrumental_intent")
            for name in active_profile_names:
                profile = profiles.get(name)
                if not isinstance(profile, dict):
                    continue
                tuned_like = self._merge_numeric_dict(tuned_like, profile.get("SEARCH_LIKE_WEIGHTS"))
                tuned_non_like = self._merge_numeric_dict(
                    tuned_non_like, profile.get("SEARCH_NON_LIKE_WEIGHTS")
                )
                tuned_facet = self._merge_numeric_dict(
                    tuned_facet, profile.get("SEARCH_FACET_WEIGHTS")
                )
                tuned_instr = self._merge_numeric_dict(
                    tuned_instr, profile.get("SEARCH_INSTRUMENTAL_WEIGHTS")
                )

        # Keep weight groups numerically stable.
        tuned_like = self._normalize_positive_weights(
            tuned_like,
            ["semantic", "overlap", "popularity", "facet", "artist", "audio"],
        )
        tuned_non_like = self._normalize_positive_weights(
            tuned_non_like,
            ["semantic", "tag_overlap", "popularity", "facet", "description", "audio", "year_bonus"],
        )
        tuned_facet = self._normalize_positive_weights(tuned_facet, ["genre", "vibe", "mood"])

        return {
            "base": tuned_base,
            "like": tuned_like,
            "non_like": tuned_non_like,
            "facet": tuned_facet,
            "instrumental": tuned_instr,
        }

    # Build artist lookup.
    def _build_artist_lookup(self):
        """
        Build artist lookup.
        
        This method implements the build artist lookup step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        artist_to_track_ids = {}
        artist_token_to_artists = {}
        title_artist_token_df = Counter()
        facet_title_artist_token_to_track_ids = {}
        soft_title_artist_token_to_track_ids = {}
        for track_id, meta in self.metadata.items():
            if not isinstance(meta, dict):
                continue
            artist_norm = self._normalize(meta.get("artist_name", ""))
            if not artist_norm:
                continue
            artist_to_track_ids.setdefault(artist_norm, []).append(str(track_id))
            for token in set(artist_norm.split()):
                if not token:
                    continue
                artist_token_to_artists.setdefault(token, set()).add(artist_norm)
            title_artist_text = f"{meta.get('title', '')} {meta.get('artist_name', '')}"
            for token in set(self._query_tokens(title_artist_text)):
                if token:
                    title_artist_token_df[token] += 1
                    if token in self.FACET_HINT_TOKENS:
                        facet_title_artist_token_to_track_ids.setdefault(token, []).append(
                            str(track_id)
                        )
                    if token in self.FACET_SOFT_TOKENS:
                        soft_title_artist_token_to_track_ids.setdefault(token, []).append(str(track_id))
        self._artist_to_track_ids = artist_to_track_ids
        self._artist_token_to_artists = artist_token_to_artists
        self._title_artist_token_df = title_artist_token_df
        self._facet_title_artist_token_to_track_ids = facet_title_artist_token_to_track_ids
        self._soft_title_artist_token_to_track_ids = soft_title_artist_token_to_track_ids

    # Internal helper to safe year.
    @staticmethod
    def _safe_year(meta):
        """
        Safely convert year.
        
        This method implements the safe year step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        year = meta.get("year", 0)
        try:
            return int(year)
        except Exception:
            return 0

    # Internal helper to normalize.
    @staticmethod
    def _normalize(value):
        """
        Execute normalize.
        
        This method implements the normalize step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        cleaned = re.sub("[^a-z0-9 ]+", " ", str(value or "").lower())
        return " ".join(cleaned.split())

    # Internal helper to as list.
    @staticmethod
    def _as_list(value):
        """
        Execute as list.
        
        This method implements the as list step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            v = value.strip()
            if not v:
                return []
            try:
                parsed = json.loads(v.replace("'", '"'))
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        return []

    # Internal helper to query tokens.
    @staticmethod
    def _query_tokens(query):
        """
        Execute query tokens.
        
        This method implements the query tokens step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        return set(re.findall("[a-z0-9]+", query.lower()))

    # Internal helper to keep high-signal query tokens.
    def _content_query_tokens(self, query):
        """
        Execute content query tokens.
        
        This method implements the content query tokens step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        raw_tokens = self._query_tokens(query)
        return {token for token in raw_tokens if token not in self.QUERY_STOP_TOKENS}

    # Set overlap.
    @staticmethod
    def _set_overlap(left, right):
        """
        Set overlap.
        
        This method implements the set overlap step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if not left or not right:
            return 0.0
        return len(left.intersection(right)) / min(len(left), len(right))

    # Balanced overlap (F1-style) to avoid saturation on one-token matches.
    @staticmethod
    def _balanced_overlap(left, right):
        """
        Execute balanced overlap.
        
        This method implements the balanced overlap step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if not left or not right:
            return 0.0
        left_set = set(left)
        right_set = set(right)
        if not left_set or not right_set:
            return 0.0
        inter = len(left_set.intersection(right_set))
        if inter <= 0:
            return 0.0
        precision = inter / float(len(left_set))
        recall = inter / float(len(right_set))
        denom = precision + recall
        if denom <= 0:
            return 0.0
        return 2.0 * precision * recall / denom

    # Internal helper to parse float safely.
    @staticmethod
    def _safe_float(value, default=0.0):
        """
        Safely convert float.
        
        This method implements the safe float step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        try:
            return float(value)
        except Exception:
            return float(default)

    # Internal helper to clamp audio values to [0, 1].
    def _audio_value(self, value):
        """
        Execute audio value.
        
        This method implements the audio value step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        return min(1.0, max(0.0, self._safe_float(value, 0.0)))

    # Internal helper to compute mean audio similarity.
    @staticmethod
    def _mean_audio_similarity(candidate_audio, target_audio):
        """
        Execute mean audio similarity.
        
        This method implements the mean audio similarity step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if not candidate_audio or not target_audio:
            return 0.0
        keys = [k for k in target_audio.keys() if k in candidate_audio]
        if not keys:
            return 0.0
        score = 0.0
        for key in keys:
            score += max(0.0, 1.0 - abs(float(candidate_audio[key]) - float(target_audio[key])))
        return score / float(len(keys))

    # Internal helper to build query audio targets.
    def _query_audio_targets(self, query):
        """
        Execute query audio targets.
        
        This method implements the query audio targets step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        normalized = self._normalize(query)
        if not normalized:
            return None, 0
        aggregate = {}
        hit_count = 0

        for phrase, profile in self.AUDIO_PHRASE_PROFILES.items():
            if phrase in normalized:
                for key, value in profile.items():
                    aggregate[key] = aggregate.get(key, 0.0) + float(value)
                hit_count += 1

        for token in self._content_query_tokens(normalized):
            profile = self.AUDIO_TOKEN_PROFILES.get(token)
            if not profile:
                continue
            for key, value in profile.items():
                aggregate[key] = aggregate.get(key, 0.0) + float(value)
            hit_count += 1

        if hit_count <= 0:
            return None, 0
        target = {key: min(1.0, max(0.0, value / float(hit_count))) for key, value in aggregate.items()}
        return target, hit_count

    # Internal helper to detect facet-heavy text queries.
    def _is_facet_heavy_query(self, query_tokens, query_year, query_audio_hits, like_mode):
        """
        Return whether facet heavy query.
        
        This method implements the is facet heavy query step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if like_mode:
            return False
        if not query_tokens:
            return False
        facet_hits = len(query_tokens.intersection(self.FACET_HINT_TOKENS))
        coverage = facet_hits / max(1.0, float(len(query_tokens)))
        if query_audio_hits > 0:
            return True
        if query_year and (facet_hits > 0):
            return True
        if facet_hits >= 2:
            return True
        return coverage >= 0.34

    @staticmethod
    def _clamp01(value):
        """
        Execute clamp01.
        
        This method implements the clamp01 step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return 0.0

    # Resolve active lightweight query-bots from user query text.
    def _resolve_active_query_bots(self, query, query_tokens):
        """
        Resolve active query bots.
        
        This method implements the resolve active query bots step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        normalized_query = self._normalize(query)
        tokens = set(query_tokens or set())
        if not normalized_query and not tokens:
            return []

        active = []
        for bot_key, profile in self.BOT_QUERY_PROFILES.items():
            profile_tokens = set(profile.get("tokens", set()))
            profile_phrases = set(profile.get("phrases", set()))
            if not profile_tokens and not profile_phrases:
                continue

            token_hits = len(profile_tokens.intersection(tokens)) if profile_tokens else 0
            phrase_hits = 0
            for phrase in profile_phrases:
                phrase_norm = self._normalize(phrase)
                if phrase_norm and phrase_norm in normalized_query:
                    phrase_hits += 1

            token_ratio = (
                float(token_hits) / float(max(1, min(len(tokens), len(profile_tokens))))
                if profile_tokens and tokens
                else 0.0
            )
            phrase_ratio = (
                float(phrase_hits) / float(len(profile_phrases))
                if profile_phrases
                else 0.0
            )
            confidence = self._clamp01(0.72 * token_ratio + 0.28 * phrase_ratio)
            if confidence < 0.12:
                continue
            active.append(
                {
                    "bot": str(bot_key),
                    "label": str(profile.get("label", bot_key)),
                    "confidence": float(confidence),
                    "tokens": profile_tokens,
                    "phrases": profile_phrases,
                }
            )
        active.sort(key=lambda row: float(row.get("confidence", 0.0)), reverse=True)
        return active[:3]

    # Compute candidate compatibility with the active query-bots.
    def _candidate_bot_profile_match(self, meta, features, active_query_bots):
        """
        Execute candidate bot profile match.
        
        This method implements the candidate bot profile match step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if not active_query_bots:
            return "", 0.0
        meta_obj = meta if isinstance(meta, dict) else {}
        feat = features if isinstance(features, dict) else {}
        candidate_tokens = set(feat.get("tag_tokens", set()))
        candidate_tokens.update(feat.get("genre_tokens", set()))
        candidate_tokens.update(feat.get("mood_tokens", set()))
        if self._safe_float(feat.get("description_trust", 0.0), 0.0) >= 0.5:
            candidate_tokens.update(feat.get("description_tokens", set()))
        candidate_tokens.update(feat.get("title_tokens", set()))
        candidate_tokens.update(self._query_tokens(meta_obj.get("artist_name", "")))

        include_desc_text = self._safe_float(feat.get("description_trust", 0.0), 0.0) >= 0.5
        candidate_text = " ".join(
            [
                str(meta_obj.get("title", "") or ""),
                str(meta_obj.get("artist_name", "") or ""),
                str(meta_obj.get("description", "") or "") if include_desc_text else "",
            ]
        ).strip()
        candidate_norm = self._normalize(candidate_text)

        best_label = ""
        best_score = 0.0
        for bot in active_query_bots:
            profile_tokens = set(bot.get("tokens", set()))
            profile_phrases = set(bot.get("phrases", set()))
            if not profile_tokens and not profile_phrases:
                continue

            token_overlap = self._set_overlap(candidate_tokens, profile_tokens)
            phrase_hits = 0
            if candidate_norm:
                for phrase in profile_phrases:
                    phrase_norm = self._normalize(phrase)
                    if phrase_norm and phrase_norm in candidate_norm:
                        phrase_hits += 1
            phrase_overlap = (
                float(phrase_hits) / float(len(profile_phrases))
                if profile_phrases
                else 0.0
            )
            base_match = self._clamp01(0.82 * token_overlap + 0.18 * phrase_overlap)
            weighted = self._clamp01(float(bot.get("confidence", 0.0)) * base_match)
            if weighted > best_score:
                best_score = weighted
                best_label = str(bot.get("label", bot.get("bot", "")))
        return best_label, best_score

    # Internal helper to compute facet alignment strength for facet-heavy queries.
    def _facet_alignment_score(
        self,
        *,
        facet_score,
        genre_score,
        vibe_score,
        mood_score,
        overlap_score,
        description_score,
        audio_query_score,
        bot_profile_score,
    ):
        """
        Execute facet alignment score.
        
        This method implements the facet alignment score step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        core_facet = max(
            self._clamp01(facet_score),
            self._clamp01(genre_score),
            self._clamp01(vibe_score),
            self._clamp01(mood_score),
        )
        lexical_context = max(
            self._clamp01(overlap_score),
            self._clamp01(description_score),
        )
        score = (
            0.56 * core_facet
            + 0.20 * lexical_context
            + 0.16 * self._clamp01(audio_query_score)
            + 0.08 * self._clamp01(bot_profile_score)
        )
        return self._clamp01(score)

    # Internal helper to expand facet tokens with lightweight semantic variants.
    def _expand_facet_tokens(self, tokens):
        """
        Execute expand facet tokens.
        
        This method implements the expand facet tokens step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        token_set = set(tokens or set())
        if not token_set:
            return set()
        expanded = set(token_set)
        for token in list(token_set):
            expanded.update(self.FACET_TOKEN_VARIANTS.get(str(token), set()))
        return expanded

    # Internal helper to estimate title+artist token document frequency ratio.
    def _title_artist_token_df_ratio(self, token):
        """
        Execute title artist token df ratio.
        
        This method implements the title artist token df ratio step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        tok = str(token or "").strip().lower()
        if not tok:
            return 1.0
        denom = max(1.0, float(len(self.id_map or [])))
        return float(self._title_artist_token_df.get(tok, 0)) / denom

    # Internal helper to choose the rarest soft facet token as focus.
    def _select_soft_focus_token(self, soft_tokens):
        """
        Execute select soft focus token.
        
        This method implements the select soft focus token step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        token_set = set(soft_tokens or set())
        if not token_set:
            return ""
        ranked = sorted(
            token_set,
            key=lambda tok: (
                self._title_artist_token_df.get(str(tok), 10**9),
                len(str(tok)),
                str(tok),
            ),
        )
        return str(ranked[0]) if ranked else ""

    # Internal helper to choose a rare contextual hard token as focus.
    def _select_hard_focus_token(self, hard_tokens):
        """
        Execute select hard focus token.
        
        This method implements the select hard focus token step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        token_set = {
            str(tok)
            for tok in set(hard_tokens or set())
            if str(tok) and str(tok) not in self.HARD_FOCUS_EXCLUDE_TOKENS
        }
        if not token_set:
            return ""
        ranked = sorted(
            token_set,
            key=lambda tok: (
                self._title_artist_token_df.get(tok, 10**9),
                len(tok),
                tok,
            ),
        )
        return str(ranked[0]) if ranked else ""

    # Internal helper to top token set.
    @staticmethod
    def _top_token_set(counter, limit=20, min_ratio=0.2):
        """
        Execute top token set.
        
        This method implements the top token set step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if not counter:
            return set()
        top = counter.most_common(limit)
        peak = float(top[0][1])
        if peak <= 0:
            return set()
        return {token for token, value in top if float(value) >= peak * float(min_ratio)}

    # Internal helper to overlap score.
    def _overlap_score(self, query_tokens, meta):
        """
        Execute overlap score.
        
        This method implements the overlap score step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        tags = self._as_list(meta.get("deezer_tags", []))
        tag_tokens = set()
        for tag in tags:
            tag_tokens.update(re.findall("[a-z0-9]+", str(tag).lower()))
        return self._set_overlap(query_tokens, tag_tokens)

    # Internal helper to popularity score.
    @staticmethod
    def _popularity_score(meta):
        """
        Execute popularity score.
        
        This method implements the popularity score step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        views = float(meta.get("views", 0) or 0)
        playcount = float(meta.get("deezer_playcount", 0) or 0)
        if views > 0 or playcount > 0:
            return float(np.log1p(views * 0.7 + playcount * 0.3))
        rank = float(meta.get("deezer_rank", meta.get("rank", 0)) or 0)
        if rank > 0:
            return float(1.0 / np.log1p(rank))
        return 0.0

    # Internal helper to meta tag tokens.
    def _meta_tag_tokens(self, meta):
        """
        Execute meta tag tokens.
        
        This method implements the meta tag tokens step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        tags = self._as_list(meta.get("deezer_tags", []))
        if not tags:
            tags = self._as_list(meta.get("tags", []))
        tokens = set()
        for value in tags:
            tokens.update(re.findall("[a-z0-9]+", str(value).lower()))
        return tokens

    # Internal helper to meta genre tokens.
    def _meta_genre_tokens(self, meta):
        """
        Execute meta genre tokens.
        
        This method implements the meta genre tokens step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        genres = self._as_list(meta.get("genres", []))
        if not genres:
            genres = self._as_list(meta.get("genre", []))
        if not genres:
            genres = self._as_list(meta.get("deezer_tags", []))
        tokens = set()
        for value in genres:
            tokens.update(re.findall("[a-z0-9]+", str(value).lower()))
        return tokens

    # Internal helper to meta mood tokens.
    def _meta_mood_tokens(self, meta):
        """
        Execute meta mood tokens.
        
        This method implements the meta mood tokens step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        return self._meta_tag_tokens(meta).intersection(self.MOOD_TOKENS)

    # Internal helper to meta description tokens.
    def _meta_description_tokens(self, meta):
        """
        Execute meta description tokens.
        
        This method implements the meta description tokens step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        description = str(meta.get("description", "") or "").lower()
        if not description:
            return set()
        return set(re.findall("[a-z0-9]+", description))

    # Internal helper to estimate whether description text is trusted.
    @staticmethod
    def _description_trust(meta):
        """
        Execute description trust.
        
        This method implements the description trust step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        source = str((meta or {}).get("description_source", "") or "").strip().lower()
        if not source:
            return 0.0
        trusted_sources = {"deezer", "spotify", "yt", "youtube", "manual", "editorial", "verified"}
        if source in trusted_sources:
            return 1.0
        if "generated" in source or "synthetic" in source or "auto" in source:
            return 0.0
        return 0.35

    # Internal helper to cached track features.
    def _get_track_features(self, track_id, meta):
        """
        Get track features.
        
        This method implements the get track features step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        key = str(track_id)
        cached = self._cache_get(self._track_feature_cache, key)
        if cached is not None:
            return cached
        features = {
            "year": self._safe_year(meta),
            "popularity": self._popularity_score(meta),
            "tag_tokens": self._meta_tag_tokens(meta),
            "genre_tokens": self._meta_genre_tokens(meta),
            "mood_tokens": self._meta_mood_tokens(meta),
            "description_tokens": self._meta_description_tokens(meta),
            "description_trust": self._description_trust(meta),
            "title_tokens": self._query_tokens(meta.get("title", "")),
            "audio": {
                "tempo": self._audio_value(meta.get("tempo", 0.0)),
                "energy": self._audio_value(meta.get("energy", 0.0)),
                "brightness": self._audio_value(meta.get("brightness", 0.0)),
                "mood": self._audio_value(meta.get("mood", 0.0)),
                "valence": self._audio_value(meta.get("valence", 0.0)),
            },
        }
        self._cache_set(
            self._track_feature_cache,
            key,
            features,
            self._max_track_feature_cache,
        )
        return features

    # Internal helper to get cached query vector.
    def _get_query_vector(self, cache_key, text):
        """
        Get query vector.
        
        This method implements the get query vector step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        cached = self._cache_get(self._query_vec_cache, cache_key)
        if cached is not None:
            return cached.copy()
        vec = embedding_handler.encode([text], normalize_embeddings=True)
        vec = np.asarray(vec, dtype=np.float32)
        self._cache_set(self._query_vec_cache, cache_key, vec, self._max_query_vec_cache)
        return vec.copy()

    # Backfill semantic scores for candidate ids missing FAISS score (strict title-by-artist path).
    def _backfill_candidate_semantic_scores(self, *, query_vec, candidate_track_ids, semantic_score_by_track):
        """
        Execute backfill candidate semantic scores.
        
        This method implements the backfill candidate semantic scores step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if query_vec is None:
            return semantic_score_by_track
        if not candidate_track_ids:
            return semantic_score_by_track
        score_map = semantic_score_by_track if isinstance(semantic_score_by_track, dict) else {}
        missing = []
        for tid in candidate_track_ids:
            track_id = str(tid or "")
            if not track_id:
                continue
            if float(score_map.get(track_id, 0.0) or 0.0) > 0.0:
                continue
            missing.append(track_id)
        if not missing:
            return score_map

        cap = int(getattr(settings, "SEARCH_STRICT_ARTIST_BACKFILL_MAX", 420))
        cap = max(1, cap)
        missing = missing[:cap]

        q = np.asarray(query_vec, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        qv = q[0]

        vec_by_track = {}
        to_encode_ids = []
        to_encode_texts = []
        for track_id in missing:
            cached = self._cache_get(self._track_vec_cache, track_id)
            if cached is not None:
                vec_by_track[track_id] = np.asarray(cached, dtype=np.float32).reshape(1, -1)[0]
                continue
            meta_obj = self.metadata.get(track_id, {})
            tags = self._as_list(meta_obj.get("deezer_tags", []))
            instrumental = as_optional_bool(meta_obj.get("instrumental"))
            text = build_track_embedding_text(
                title=str(meta_obj.get("title", "") or ""),
                artist=str(meta_obj.get("artist_name", "") or ""),
                tags=tags,
                description=str(meta_obj.get("description", "") or ""),
                instrumental=instrumental,
            )
            to_encode_ids.append(track_id)
            to_encode_texts.append(text)

        if to_encode_texts:
            try:
                batch_vecs = embedding_handler.encode(to_encode_texts, normalize_embeddings=True)
                batch_vecs = np.asarray(batch_vecs, dtype=np.float32)
                if batch_vecs.ndim == 1:
                    batch_vecs = batch_vecs.reshape(1, -1)
                for idx, track_id in enumerate(to_encode_ids):
                    vec = batch_vecs[idx : idx + 1]
                    self._cache_set(
                        self._track_vec_cache,
                        track_id,
                        vec,
                        self._max_track_vec_cache,
                    )
                    vec_by_track[track_id] = vec[0]
            except Exception:
                pass

        for track_id, tv in vec_by_track.items():
            try:
                sim = float(np.dot(qv, tv))
                score_map[track_id] = max(0.0, sim)
            except Exception:
                continue
        return score_map

    # Internal helper to get cached anchor seed vector.
    def _get_anchor_seed_vector(self, anchor_key, seed_texts):
        """
        Get anchor seed vector.
        
        This method implements the get anchor seed vector step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        cache_key = f"anchor-seed::{anchor_key}"
        cached = self._cache_get(self._query_vec_cache, cache_key)
        if cached is not None:
            return cached.copy(), cache_key
        seed_vectors = embedding_handler.encode(seed_texts[:24], normalize_embeddings=True)
        query_vec = np.asarray(seed_vectors, dtype=np.float32)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        if query_vec.shape[0] > 1:
            query_vec = np.mean(query_vec, axis=0, keepdims=True)
        norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
        query_vec = query_vec / np.maximum(norm, 1e-8)
        self._cache_set(self._query_vec_cache, cache_key, query_vec, self._max_query_vec_cache)
        return query_vec.copy(), cache_key

    # Internal helper to cached faiss search.
    def _faiss_search_cached(self, query_vec, query_vec_key, fetch_k):
        """
        Execute faiss search cached.
        
        This method implements the faiss search cached step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        cache_key = (str(query_vec_key), int(fetch_k))
        cached = self._cache_get(self._faiss_cache, cache_key)
        if cached is not None:
            distances, indices = cached
            return distances.copy(), indices.copy()
        distances, indices = self.faiss_index.search(query_vec, fetch_k)
        self._cache_set(
            self._faiss_cache,
            cache_key,
            (distances.copy(), indices.copy()),
            self._max_faiss_cache,
        )
        return distances, indices

    # Internal helper to convert FAISS hits into ordered track ids + semantic score map.
    def _faiss_hits_to_track_ids(self, distances, indices):
        """
        Execute faiss hits to track ids.
        
        This method implements the faiss hits to track ids step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        ordered_ids = []
        score_map = {}
        seen = set()
        if distances is None or indices is None:
            return ordered_ids, score_map
        for i in range(len(indices[0])):
            idx = int(indices[0][i])
            if idx < 0 or idx >= len(self.id_map):
                continue
            track_id = str(self.id_map[idx])
            score = max(float(distances[0][i]), 0.0)
            prev = score_map.get(track_id)
            if prev is None or score > prev:
                score_map[track_id] = score
            if track_id in seen:
                continue
            seen.add(track_id)
            ordered_ids.append(track_id)
        return ordered_ids, score_map

    # Internal helper to candidate ids from a semantic text query (FAISS).
    def _candidate_ids_from_semantic_text(self, text, fetch_k, cache_prefix="union"):
        """
        Execute candidate ids from semantic text.
        
        This method implements the candidate ids from semantic text step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        text_norm = self._normalize(text)
        if not text_norm or self.faiss_index is None:
            return [], {}
        query_vec_key = f"{cache_prefix}::{text_norm}"
        query_vec = self._get_query_vector(query_vec_key, text_norm)
        distances, indices = self._faiss_search_cached(
            query_vec=query_vec,
            query_vec_key=query_vec_key,
            fetch_k=max(1, min(int(fetch_k), max(1, len(self.id_map)))),
        )
        return self._faiss_hits_to_track_ids(distances, indices)

    # Internal helper to collect candidate ids from strict artist targets.
    def _candidate_ids_from_artist_targets(self, artist_norms, limit=600):
        """
        Execute candidate ids from artist targets.
        
        This method implements the candidate ids from artist targets step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if not artist_norms:
            return []
        limit = max(1, int(limit))
        out = []
        seen = set()
        for artist_norm in artist_norms:
            for track_id in self._artist_to_track_ids.get(artist_norm, []):
                tid = str(track_id)
                if tid in seen:
                    continue
                seen.add(tid)
                out.append(tid)
                if len(out) >= limit:
                    return out
        return out

    # Internal helper to collect candidates by title/artist facet tokens.
    def _candidate_ids_from_facet_title_artist_tokens(
        self,
        facet_tokens,
        *,
        limit=600,
        hard_filter_tokens=None,
        hard_filter_min=0.0,
    ):
        """
        Execute candidate ids from facet title artist tokens.
        
        This method implements the candidate ids from facet title artist tokens step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        token_set = {str(tok) for tok in set(facet_tokens or set()) if str(tok)}
        if not token_set:
            return []
        limit = max(1, int(limit))
        hard_tokens = {
            str(tok) for tok in set(hard_filter_tokens or set()) if str(tok)
        }
        hard_min = self._clamp01(hard_filter_min)
        out = []
        seen = set()
        ordered_tokens = sorted(
            token_set,
            key=lambda tok: (
                self._title_artist_token_df.get(tok, 10**9),
                len(tok),
                tok,
            ),
        )
        for token in ordered_tokens:
            for track_id in self._facet_title_artist_token_to_track_ids.get(token, []):
                tid = str(track_id)
                if not tid or tid in seen:
                    continue
                if hard_tokens:
                    meta = self.metadata.get(tid, {})
                    features = self._get_track_features(tid, meta)
                    candidate_tokens = (
                        set(features.get("genre_tokens", set()))
                        .union(features.get("tag_tokens", set()))
                        .union(features.get("title_tokens", set()))
                    )
                    if self._balanced_overlap(candidate_tokens, hard_tokens) < hard_min:
                        continue
                seen.add(tid)
                out.append(tid)
                if len(out) >= limit:
                    return out
        return out

    # Internal helper to collect anchor-artist candidates for "songs like X" queries.
    def _candidate_ids_from_anchor_profile(self, anchor_profile, limit=600):
        """
        Execute candidate ids from anchor profile.
        
        This method implements the candidate ids from anchor profile step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if not anchor_profile:
            return []
        artists = anchor_profile.get("artists", set()) or set()
        return self._candidate_ids_from_artist_targets(artists, limit=limit)

    # Internal helper to merge candidate id lists while preserving source order priority.
    @staticmethod
    def _merge_candidate_id_lists(candidate_lists, max_size):
        """
        Merge candidate id lists.
        
        This method implements the merge candidate id lists step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        merged = []
        seen = set()
        cap = max(1, int(max_size))
        for items in candidate_lists:
            for track_id in items or []:
                tid = str(track_id)
                if not tid or tid in seen:
                    continue
                seen.add(tid)
                merged.append(tid)
                if len(merged) >= cap:
                    return merged
        return merged

    # Internal helper to row-to-row similarity for MMR diversification.
    def _search_row_similarity(self, left_row, right_row):
        """
        Execute search row similarity.
        
        This method implements the search row similarity step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if not left_row or not right_row:
            return 0.0
        left_id = str(left_row.get("id", "") or "")
        right_id = str(right_row.get("id", "") or "")
        if not left_id or not right_id:
            return 0.0
        if left_id == right_id:
            return 1.0
        left_meta = self.metadata.get(left_id, {})
        right_meta = self.metadata.get(right_id, {})
        if not isinstance(left_meta, dict) or not isinstance(right_meta, dict):
            return 0.0
        left_features = self._get_track_features(left_id, left_meta)
        right_features = self._get_track_features(right_id, right_meta)
        same_artist = 1.0 if self._normalize(left_row.get("artist", "")) == self._normalize(right_row.get("artist", "")) else 0.0
        tag_sim = self._set_overlap(left_features["tag_tokens"], right_features["tag_tokens"])
        genre_sim = self._set_overlap(left_features["genre_tokens"], right_features["genre_tokens"])
        mood_sim = self._set_overlap(left_features["mood_tokens"], right_features["mood_tokens"])
        audio_sim = self._mean_audio_similarity(left_features["audio"], right_features["audio"])
        return float(
            0.28 * same_artist
            + 0.28 * tag_sim
            + 0.20 * genre_sim
            + 0.12 * mood_sim
            + 0.12 * audio_sim
        )

    # Internal helper to diversify top ranked rows using MMR.
    def _diversify_ranked_rows_mmr(self, ranked_rows, limit, offset, query_tokens):
        """
        Execute diversify ranked rows mmr.
        
        This method implements the diversify ranked rows mmr step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        rows = list(ranked_rows or [])
        if not rows:
            return rows
        if offset != 0:
            return rows
        if limit <= 1:
            return rows
        pool_size = int(getattr(settings, "SEARCH_MMR_DIVERSIFY_POOL", 80))
        if pool_size <= 1:
            return rows
        lambda_rel = float(getattr(settings, "SEARCH_MMR_LAMBDA", 0.78))
        lambda_rel = min(1.0, max(0.0, lambda_rel))
        query_tokens = set(query_tokens or [])
        # Do not diversify very short / exact-like queries.
        if len(query_tokens) <= int(getattr(settings, "SEARCH_MMR_MIN_QUERY_TOKENS", 2)):
            return rows

        head = rows[: max(1, min(pool_size, len(rows)))]
        tail = rows[len(head) :]
        if len(head) <= 2:
            return rows

        scores = [float(r.get("score", 0.0) or 0.0) for r in head]
        min_score = min(scores)
        max_score = max(scores)
        denom = max(max_score - min_score, 1e-8)

        selected = []
        remaining = list(range(len(head)))
        # Seed with the highest-relevance item (already sorted, but use score for safety).
        first_idx = max(remaining, key=lambda i: scores[i])
        selected.append(first_idx)
        remaining.remove(first_idx)

        while remaining and len(selected) < len(head):
            best_idx = None
            best_val = -1e18
            for idx in remaining:
                rel = (scores[idx] - min_score) / denom
                max_sim = 0.0
                for sidx in selected:
                    sim = self._search_row_similarity(head[idx], head[sidx])
                    if sim > max_sim:
                        max_sim = sim
                mmr = lambda_rel * rel - (1.0 - lambda_rel) * max_sim
                if mmr > best_val:
                    best_val = mmr
                    best_idx = idx
            if best_idx is None:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)

        diversified = [head[i] for i in selected]
        return diversified + tail

    # Internal helper to attach interpretable quality metrics to ranked rows.
    def _attach_ranked_query_metrics(
        self,
        ranked_rows,
        *,
        active_query_bots,
        like_mode,
        strict_artist_mode,
        facet_heavy_query,
        mixed_compositional_query=False,
        hard_facet_query=False,
        title_artist_query=False,
        fallback_lexical_used=False,
    ):
        """
        Attach ranked query metrics.
        
        This method implements the attach ranked query metrics step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        rows = list(ranked_rows or [])
        if not rows:
            return {
                "query_intent": {
                    "like_mode": bool(like_mode),
                    "strict_artist_mode": bool(strict_artist_mode),
                    "facet_heavy_query": bool(facet_heavy_query),
                    "title_artist_query": bool(title_artist_query),
                    "fallback_lexical_used": bool(fallback_lexical_used),
                    "active_bots": [],
                    "bot_intent_confidence": 0.0,
                },
                "result_quality": {
                    "result_count": 0,
                    "avg_query_fit": 0.0,
                    "top1_query_fit": 0.0,
                    "avg_semantic_similarity": 0.0,
                    "avg_bot_match": 0.0,
                    "genre_hit_rate_at_k": 0.0,
                    "facet_hit_rate_at_k": 0.0,
                    "avg_facet_alignment": 0.0,
                    "mixed_intent_hit_rate_at_k": 0.0,
                },
            }

        semantic_values = [self._safe_float(row.get("semantic_score", 0.0), 0.0) for row in rows]
        sem_max = max(semantic_values) if semantic_values else 1.0
        semantic_cold = sem_max <= 1e-12

        total_query_fit = 0.0
        total_semantic = 0.0
        total_bot = 0.0
        total_facet_alignment = 0.0
        total_facet_coverage = 0.0
        genre_hits = 0
        facet_hits = 0
        mixed_intent_hits = 0
        genre_hit_threshold = float(getattr(settings, "SEARCH_METRIC_GENRE_HIT_THRESHOLD", 0.10))
        facet_hit_threshold = float(getattr(settings, "SEARCH_METRIC_FACET_HIT_THRESHOLD", 0.15))
        mixed_intent_hit_threshold = float(
            getattr(settings, "SEARCH_METRIC_MIXED_INTENT_HIT_THRESHOLD", 0.25)
        )
        for row in rows:
            semantic_raw = self._safe_float(row.get("semantic_score", 0.0), 0.0)
            semantic_abs = self._clamp01(semantic_raw)
            title_score = self._clamp01(self._safe_float(row.get("title_score", 0.0), 0.0))
            overlap_score = self._clamp01(self._safe_float(row.get("overlap_score", 0.0), 0.0))
            description_score = self._clamp01(self._safe_float(row.get("description_score", 0.0), 0.0))
            genre_score = self._clamp01(self._safe_float(row.get("genre_score", 0.0), 0.0))
            vibe_score = self._clamp01(self._safe_float(row.get("vibe_score", 0.0), 0.0))
            mood_score = self._clamp01(self._safe_float(row.get("mood_score", 0.0), 0.0))
            facet_token_coverage = self._clamp01(
                self._safe_float(row.get("facet_token_coverage", genre_score), 0.0)
            )
            hard_facet_token_coverage = self._clamp01(
                self._safe_float(row.get("hard_facet_token_coverage", facet_token_coverage), 0.0)
            )
            artist_hint = self._clamp01(self._safe_float(row.get("artist_query_score", 0.0), 0.0))
            title_hint = self._clamp01(self._safe_float(row.get("title_hint_score", 0.0), 0.0))
            title_exact = self._clamp01(self._safe_float(row.get("title_exact_match", 0.0), 0.0))
            if title_artist_query and (not like_mode):
                lexical = self._clamp01(
                    0.52 * title_hint
                    + 0.24 * title_exact
                    + 0.14 * title_score
                    + 0.10 * description_score
                )
            else:
                lexical = self._clamp01(
                    0.30 * title_score
                    + 0.24 * overlap_score
                    + 0.24 * description_score
                    + 0.14 * title_hint
                    + 0.08 * artist_hint
                )
            facet = self._clamp01(self._safe_float(row.get("facet_score", 0.0), 0.0))
            bot_match = self._clamp01(self._safe_float(row.get("bot_profile_score", 0.0), 0.0))
            artist_exact = self._clamp01(self._safe_float(row.get("artist_exact_match", 0.0), 0.0))

            if like_mode:
                artist = self._clamp01(self._safe_float(row.get("artist_score", 0.0), 0.0))
                audio_anchor = self._clamp01(self._safe_float(row.get("audio_anchor_score", 0.0), 0.0))
                query_fit = (
                    0.38 * semantic_abs
                    + 0.19 * facet
                    + 0.15 * artist
                    + 0.14 * audio_anchor
                    + 0.08 * lexical
                    + 0.06 * bot_match
                )
            elif fallback_lexical_used or semantic_cold:
                query_fit = (
                    0.44 * lexical
                    + 0.24 * max(title_hint, artist_hint)
                    + 0.12 * title_hint
                    + 0.10 * artist_hint
                    + 0.06 * facet
                    + 0.04 * bot_match
                )
            elif title_artist_query and (not like_mode):
                raw_fit = (
                    0.42 * title_hint
                    + 0.18 * artist_hint
                    + 0.12 * title_exact
                    + 0.10 * semantic_abs
                    + 0.10 * lexical
                    + 0.08 * description_score
                )
                if title_exact >= 0.99 and artist_exact >= 0.99:
                    raw_fit += 0.04
                query_fit = 0.90 * raw_fit
            else:
                query_fit = (
                    0.46 * semantic_abs
                    + 0.18 * lexical
                    + 0.18 * facet
                    + 0.10 * self._clamp01(self._safe_float(row.get("audio_query_score", 0.0), 0.0))
                    + 0.08 * bot_match
                )
                if mixed_compositional_query and facet_heavy_query and hard_facet_query:
                    hard_focus = self._clamp01(self._safe_float(row.get("hard_focus_match", 0.0), 0.0))
                    soft_focus = self._clamp01(self._safe_float(row.get("soft_focus_match", 0.0), 0.0))
                    mixed_cov = self._clamp01(
                        self._safe_float(row.get("mixed_intent_coverage", 0.0), 0.0)
                    )
                    query_fit += (
                        float(getattr(settings, "SEARCH_MIXED_QF_HARD_FOCUS_BOOST", 0.05))
                        * hard_focus
                    )
                    query_fit += (
                        float(getattr(settings, "SEARCH_MIXED_QF_MIXED_COVERAGE_BOOST", 0.01))
                        * mixed_cov
                    )
                    if hard_focus <= 0.0:
                        query_fit -= float(
                            getattr(settings, "SEARCH_MIXED_QF_NO_HARD_PENALTY", 0.018)
                        )
                        query_fit -= float(
                            getattr(settings, "SEARCH_MIXED_QF_SOFT_WITHOUT_HARD_PENALTY", 0.025)
                        ) * soft_focus

            query_fit = self._clamp01(query_fit)
            row["semantic_similarity"] = float(round(semantic_abs, 6))
            row["lexical_similarity"] = float(round(lexical, 6))
            row["query_fit_score"] = float(round(query_fit, 6))
            facet_alignment = self._clamp01(
                self._safe_float(
                    row.get("facet_alignment_score", row.get("facet_score", 0.0)),
                    0.0,
                )
            )
            row["facet_alignment_score"] = float(round(facet_alignment, 6))

            total_query_fit += query_fit
            total_semantic += semantic_abs
            total_bot += bot_match
            total_facet_alignment += facet_alignment
            total_facet_coverage += facet_token_coverage
            effective_genre_cov = hard_facet_token_coverage if hard_facet_query else facet_token_coverage
            if effective_genre_cov >= genre_hit_threshold:
                genre_hits += 1
            if max(facet_alignment, facet_token_coverage, vibe_score, mood_score) >= facet_hit_threshold:
                facet_hits += 1
            mixed_hard_focus_weight = self._clamp01(
                float(getattr(settings, "SEARCH_MIXED_HARD_SIGNAL_FOCUS_WEIGHT", 0.20))
            )
            mixed_hard_cov_weight = self._clamp01(
                float(getattr(settings, "SEARCH_MIXED_HARD_SIGNAL_COVERAGE_WEIGHT", 0.80))
            )
            mixed_intent_cov = self._clamp01(
                self._safe_float(
                    row.get(
                        "mixed_intent_coverage",
                        min(
                            max(
                                mixed_hard_focus_weight
                                * self._clamp01(self._safe_float(row.get("hard_focus_match", 0.0), 0.0)),
                                mixed_hard_cov_weight * hard_facet_token_coverage,
                            ),
                            max(
                                self._clamp01(self._safe_float(row.get("mixed_soft_signal", 0.0), 0.0)),
                                self._clamp01(
                                    self._safe_float(row.get("soft_title_facet_coverage", 0.0), 0.0)
                                ),
                                self._clamp01(self._safe_float(row.get("soft_focus_match", 0.0), 0.0)),
                            ),
                        ),
                    ),
                    0.0,
                )
            )
            if mixed_intent_cov >= mixed_intent_hit_threshold:
                mixed_intent_hits += 1

        active_bots = []
        for bot in list(active_query_bots or []):
            if not isinstance(bot, dict):
                continue
            active_bots.append(
                {
                    "bot": str(bot.get("bot", "")),
                    "label": str(bot.get("label", bot.get("bot", ""))),
                    "confidence": float(round(self._clamp01(bot.get("confidence", 0.0)), 6)),
                }
            )

        avg_query_fit = total_query_fit / float(len(rows))
        avg_semantic = total_semantic / float(len(rows))
        avg_bot = total_bot / float(len(rows))
        avg_facet_alignment = total_facet_alignment / float(len(rows))
        avg_facet_coverage = total_facet_coverage / float(len(rows))
        top1_query_fit = self._safe_float(rows[0].get("query_fit_score", 0.0), 0.0)
        bot_intent_conf = (
            self._safe_float(active_bots[0].get("confidence", 0.0), 0.0) if active_bots else 0.0
        )
        return {
            "query_intent": {
                "like_mode": bool(like_mode),
                "strict_artist_mode": bool(strict_artist_mode),
                "facet_heavy_query": bool(facet_heavy_query),
                "title_artist_query": bool(title_artist_query),
                "fallback_lexical_used": bool(fallback_lexical_used),
                "active_bots": active_bots,
                "bot_intent_confidence": float(round(self._clamp01(bot_intent_conf), 6)),
            },
            "result_quality": {
                "result_count": int(len(rows)),
                "avg_query_fit": float(round(avg_query_fit, 6)),
                "top1_query_fit": float(round(top1_query_fit, 6)),
                "avg_semantic_similarity": float(round(avg_semantic, 6)),
                "avg_bot_match": float(round(avg_bot, 6)),
                "genre_hit_rate_at_k": float(round(genre_hits / float(len(rows)), 6)),
                "facet_hit_rate_at_k": float(round(facet_hits / float(len(rows)), 6)),
                "avg_facet_alignment": float(round(avg_facet_alignment, 6)),
                "avg_facet_token_coverage": float(round(avg_facet_coverage, 6)),
                "mixed_intent_hit_rate_at_k": float(
                    round(mixed_intent_hits / float(len(rows)), 6)
                ),
            },
        }

    # Internal helper to parse "title by artist" hints from free-text query.
    def _extract_title_artist_hints(self, query_norm):
        """
        Extract title artist hints.
        
        This method implements the extract title artist hints step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        text = self._normalize(query_norm)
        if not text or " by " not in text:
            return "", ""
        try:
            left, right = text.split(" by ", 1)
        except ValueError:
            return "", ""
        left_tokens = [t for t in self._content_query_tokens(left) if t]
        right_tokens = [t for t in self._content_query_tokens(right) if t]
        title_hint = " ".join(left_tokens).strip()
        artist_hint = " ".join(right_tokens).strip()
        return title_hint, artist_hint

    # Resolve artist targets from an explicit artist hint phrase.
    def _resolve_artist_hint_targets(self, artist_hint):
        """
        Resolve artist hint targets.
        
        This method implements the resolve artist hint targets step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        hint = self._normalize(artist_hint)
        if not hint:
            return set()
        hint_tokens = set(self._query_tokens(hint))
        if not hint_tokens:
            return set()
        if hint in self._artist_to_track_ids:
            return {hint}

        candidates = set()
        for token in hint_tokens:
            candidates.update(self._artist_token_to_artists.get(token, set()))
        if not candidates:
            return set()

        strong = set()
        for artist_norm in candidates:
            artist_tokens = set(self._query_tokens(artist_norm))
            overlap = self._set_overlap(artist_tokens, hint_tokens)
            if overlap >= 0.95 or hint in artist_norm or artist_norm in hint:
                strong.add(artist_norm)
        if not strong:
            return set()
        if len(strong) <= 20:
            return strong
        # Keep the closest token overlaps if many artists share a token.
        ranked = sorted(
            list(strong),
            key=lambda artist: (
                self._set_overlap(set(self._query_tokens(artist)), hint_tokens),
                -len(artist),
            ),
            reverse=True,
        )
        return set(ranked[:20])

    # Fallback lexical search used only when semantic ranking produces no rows.
    def _fallback_lexical_ranked_rows(
        self,
        *,
        query,
        query_tokens,
        strict_artist_targets,
        active_query_bots,
        bot_profile_weight,
    ):
        """
        Execute fallback lexical ranked rows.
        
        This method implements the fallback lexical ranked rows step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        query_norm = self._normalize(query)
        tokens = set(query_tokens or set())
        title_hint, artist_hint = self._extract_title_artist_hints(query_norm)
        title_hint_tokens = set(self._query_tokens(title_hint))
        artist_hint_tokens = set(self._query_tokens(artist_hint))
        title_artist_query = bool(title_hint_tokens and artist_hint_tokens)
        rows = []
        for track_id, meta in self.metadata.items():
            if not isinstance(meta, dict):
                continue
            features = self._get_track_features(track_id, meta)
            year = features["year"]
            if year and year not in settings.SEARCH_ERA_RANGE:
                continue
            artist_norm = self._normalize(meta.get("artist_name", ""))
            title_norm = self._normalize(meta.get("title", ""))
            if strict_artist_targets and artist_norm not in strict_artist_targets:
                continue

            token_overlap = self._set_overlap(
                features["tag_tokens"].union(features["genre_tokens"]).union(features["title_tokens"]),
                tokens,
            )
            description_overlap = self._set_overlap(features["description_tokens"], tokens)
            title_overlap = self._set_overlap(features["title_tokens"], tokens)
            artist_overlap = self._set_overlap(self._query_tokens(artist_norm), tokens)
            title_hint_overlap = (
                self._balanced_overlap(features["title_tokens"], title_hint_tokens)
                if title_hint_tokens
                else 0.0
            )
            artist_hint_overlap = (
                self._balanced_overlap(self._query_tokens(artist_norm), artist_hint_tokens)
                if artist_hint_tokens
                else 0.0
            )

            title_hit = 1.0 if (query_norm and query_norm in title_norm) else 0.0
            artist_hit = 1.0 if (query_norm and query_norm in artist_norm) else 0.0
            title_hint_hit = 1.0 if (title_hint and title_hint in title_norm) else 0.0
            artist_hint_hit = 1.0 if (artist_hint and artist_hint in artist_norm) else 0.0
            exact_pair = 1.0 if (title_hint_hit and artist_hint_hit) else 0.0
            if title_artist_query:
                if (
                    exact_pair <= 0.0
                    and title_hint_overlap < 0.22
                    and artist_hint_overlap < 0.22
                    and title_hit <= 0.0
                    and artist_hit <= 0.0
                ):
                    continue

            lexical_signal = max(
                token_overlap,
                title_overlap,
                artist_overlap,
                description_overlap * 0.85,
                title_hit,
                artist_hit,
                0.96 if (title_hint_hit and artist_hint_hit) else 0.0,
                0.72 if title_hint_hit else 0.0,
                0.62 if artist_hint_hit else 0.0,
            )
            if lexical_signal < 0.16:
                continue

            bot_label, bot_score = self._candidate_bot_profile_match(
                meta,
                features,
                active_query_bots,
            )
            popularity_raw = self._popularity_score(meta)

            meta_video_id = str(meta.get("video_id", "") or "").strip()
            cover = (
                meta.get("cover_url")
                or meta.get("album_cover")
                or meta.get("thumbnail")
                or thumb_from_video_id(meta_video_id)
            )
            rows.append(
                {
                    "id": str(track_id),
                    "video_id": meta_video_id,
                    "title": meta.get("title", "Unknown"),
                    "artist": meta.get("artist_name", "Unknown"),
                    "album": meta.get("album_title", "Unknown"),
                    "description": str(meta.get("description", "") or ""),
                    "instrumental": as_optional_bool(meta.get("instrumental")),
                    "instrumental_confidence": float(meta.get("instrumental_confidence", 0.0) or 0.0),
                    "cover": cover,
                    "thumbnail": cover,
                    "cover_candidates": thumbnail_candidates(meta_video_id, cover),
                    "year": year,
                    "release_date": f"{year}-01-01" if year else "",
                    "semantic_score": 0.0,
                    "overlap_score": float(token_overlap),
                    "popularity_raw": float(popularity_raw),
                    "popularity_score": 0.0,
                    "genre_score": float(self._set_overlap(features["genre_tokens"], tokens)),
                    "vibe_score": float(self._set_overlap(features["tag_tokens"], tokens)),
                    "mood_score": float(self._set_overlap(features["mood_tokens"], tokens)),
                    "artist_score": float(artist_overlap),
                    "artist_query_score": float(artist_hint_overlap),
                    "artist_exact_match": float(artist_hint_hit),
                    "facet_score": float(
                        max(
                            self._set_overlap(features["genre_tokens"], tokens),
                            self._set_overlap(features["tag_tokens"], tokens),
                            self._set_overlap(features["mood_tokens"], tokens),
                        )
                    ),
                    "description_score": float(description_overlap),
                    "title_score": float(title_overlap),
                    "title_hint_score": float(title_hint_overlap),
                    "title_exact_match": float(title_hint_hit),
                    "title_bias_penalty": 0.0,
                    "year_match_score": 0.0,
                    "instrumental_match_score": 0.0,
                    "audio_query_score": 0.0,
                    "audio_anchor_score": 0.0,
                    "bot_profile": str(bot_label or ""),
                    "bot_profile_score": float(bot_score),
                    "strict_artist_mode": bool(strict_artist_targets),
                    "_exact_pair": float(exact_pair),
                    "_artist_hint_hit": float(artist_hint_hit),
                    "_title_hint_hit": float(title_hint_hit),
                    "_lexical_signal": float(lexical_signal),
                    "score": 0.0,
                }
            )

        if not rows:
            return []
        pop_values = [float(row.get("popularity_raw", 0.0) or 0.0) for row in rows]
        pop_min = min(pop_values)
        pop_max = max(pop_values)
        pop_denom = max(pop_max - pop_min, 1e-9)
        for row in rows:
            pop_raw = float(row.get("popularity_raw", 0.0) or 0.0)
            pop_score = self._clamp01((pop_raw - pop_min) / pop_denom)
            row["popularity_score"] = float(pop_score)
            score = (
                2.60 * float(row.get("_exact_pair", 0.0))
                + 1.15 * float(row.get("artist_query_score", 0.0))
                + 0.82 * float(row.get("title_hint_score", 0.0))
                + 0.42 * float(row.get("title_score", 0.0))
                + 0.20 * float(row.get("overlap_score", 0.0))
                + 0.22 * float(row.get("description_score", 0.0))
                + 0.12 * pop_score
                + 0.24 * float(row.get("_artist_hint_hit", 0.0))
                + 0.08 * float(row.get("_title_hint_hit", 0.0))
                + float(bot_profile_weight) * float(row.get("bot_profile_score", 0.0))
            )
            row["score"] = float(score)
            row.pop("_exact_pair", None)
            row.pop("_artist_hint_hit", None)
            row.pop("_title_hint_hit", None)
            row.pop("_lexical_signal", None)
            row.pop("popularity_raw", None)

        rows.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
        return self._dedupe_ranked_rows(rows)

    # Reorder strict "title by artist" results: exact title+artist first, then artist-only fillers.
    def _reorder_title_artist_strict_rows(self, ranked_rows, *, title_hint):
        """
        Execute reorder title artist strict rows.
        
        This method implements the reorder title artist strict rows step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        rows = [dict(row) for row in list(ranked_rows or []) if isinstance(row, dict)]
        if not rows:
            return []
        title_hint_norm = self._normalize(title_hint)
        if not title_hint_norm:
            return rows

        stage_a = []
        stage_b = []
        for row in rows:
            title_signal = max(
                self._clamp01(row.get("title_exact_match", 0.0)),
                self._clamp01(row.get("title_hint_score", 0.0)),
            )
            artist_signal = max(
                self._clamp01(row.get("artist_exact_match", 0.0)),
                self._clamp01(row.get("artist_query_score", 0.0)),
            )
            if title_signal >= 0.45 and artist_signal >= 0.45:
                stage_a.append(row)
            else:
                stage_b.append(row)

        if not stage_a:
            return rows

        stage_a.sort(
            key=lambda row: (
                self._clamp01(row.get("title_exact_match", 0.0)),
                self._clamp01(row.get("title_hint_score", 0.0)),
                self._safe_float(row.get("semantic_score", 0.0), 0.0),
                self._safe_float(row.get("score", 0.0), 0.0),
            ),
            reverse=True,
        )
        stage_b.sort(
            key=lambda row: (
                self._safe_float(row.get("score", 0.0), 0.0),
                self._safe_float(row.get("semantic_score", 0.0), 0.0),
                self._clamp01(row.get("title_hint_score", 0.0)),
            ),
            reverse=True,
        )
        return stage_a + stage_b

    # Label why a row matched (helps debugging and CLI interpretation).
    def _resolve_match_type(self, row, *, like_mode, strict_title_artist_mode):
        """
        Resolve match type.
        
        This method implements the resolve match type step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if like_mode:
            return "personalized_like"
        row_obj = row if isinstance(row, dict) else {}
        title_exact = self._clamp01(row_obj.get("title_exact_match", 0.0))
        artist_exact = self._clamp01(row_obj.get("artist_exact_match", 0.0))
        title_hint = self._clamp01(row_obj.get("title_hint_score", 0.0))
        artist_hint = self._clamp01(row_obj.get("artist_query_score", 0.0))
        semantic = self._safe_float(row_obj.get("semantic_score", 0.0), 0.0)

        if strict_title_artist_mode:
            if title_exact >= 0.99 and artist_exact >= 0.99:
                return "exact_title_artist"
            if max(title_exact, title_hint) >= 0.45 and max(artist_exact, artist_hint) >= 0.45:
                return "title_artist_partial"
            if max(artist_exact, artist_hint) >= 0.45:
                return "artist_only_semantic" if semantic > 0.0 else "artist_only"
            return "semantic_fill" if semantic > 0.0 else "strict_fill"

        if title_exact >= 0.99:
            return "exact_title"
        if semantic > 0.0:
            return "semantic"
        if max(title_hint, artist_hint) > 0.0:
            return "lexical_hint"
        return "lexical"

    # Internal helper to extract query year.
    @staticmethod
    def _extract_query_year(tokens):
        """
        Extract query year.
        
        This method implements the extract query year step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        for token in tokens:
            if re.fullmatch(r"(19|20)\d{2}", token):
                try:
                    return int(token)
                except Exception:
                    return 0
        return 0

    # Internal helper to seed text from metadata.
    def _seed_text_from_meta(self, meta):
        """
        Execute seed text from meta.
        
        This method implements the seed text from meta step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        title = str(meta.get("title", "Unknown") or "Unknown").strip() or "Unknown"
        artist = str(meta.get("artist_name", "Unknown") or "Unknown").strip() or "Unknown"
        tags = [str(tag).strip() for tag in self._as_list(meta.get("deezer_tags", [])) if str(tag).strip()]
        description = str(meta.get("description", "") or "").strip()
        if len(description) > 220:
            description = f"{description[:220].rsplit(' ', 1)[0]}..."
        instrumental = as_optional_bool(meta.get("instrumental"))
        if instrumental is True:
            vocal_type = "instrumental"
        elif instrumental is False:
            vocal_type = "non-instrumental"
        else:
            vocal_type = "unknown-vocals"
        parts = [f"{title} by {artist}"]
        if tags:
            parts.append(f"Tags: {' '.join(tags[:8])}")
        if description:
            parts.append(f"Description: {description}")
        parts.append(f"Vocal type: {vocal_type}")
        return ". ".join(parts)

    # Extract like anchor.
    def _extract_like_anchor(self, query):
        """
        Extract like anchor.
        
        This method implements the extract like anchor step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        q = self._normalize(query)
        if not q:
            return ""
        for pattern in self.LIKE_PATTERNS:
            match = re.search(pattern, q)
            if not match:
                continue
            anchor = match.group(1).strip()
            anchor = re.split(
                r"(?:,|;|\bwith\b|\bfor\b|\bbut\b|\band\b|\bin\b)",
                anchor,
                maxsplit=1,
            )[0]
            tokens = [
                token
                for token in re.findall("[a-z0-9]+", anchor)
                if token not in self.ANCHOR_STOP_TOKENS
            ]
            return " ".join(tokens)
        return ""

    # Extract instrumental intent.
    def _extract_instrumental_intent(self, query):
        """
        Extract instrumental intent.
        
        This method implements the extract instrumental intent step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        normalized = self._normalize(query)
        if not normalized:
            return None
        for hint in self.VOCAL_HINTS:
            if hint in normalized:
                return False
        for hint in self.INSTRUMENTAL_HINTS:
            if hint in normalized:
                return True
        return None

    # Internal helper to artist match score.
    def _artist_match_score(self, artist_name, anchor_profile):
        """
        Execute artist match score.
        
        This method implements the artist match score step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        artist_norm = self._normalize(artist_name)
        if not artist_norm:
            return 0.0
        anchor_text = anchor_profile.get("anchor_text", "")
        anchor_artists = anchor_profile.get("artists", set())
        anchor_tokens = anchor_profile.get("anchor_tokens", set())
        if artist_norm in anchor_artists:
            return 1.0
        if anchor_text and (anchor_text in artist_norm or artist_norm in anchor_text):
            return 0.85
        artist_tokens = set(re.findall("[a-z0-9]+", artist_norm))
        return min(0.75, self._set_overlap(artist_tokens, anchor_tokens))

    # Internal helper to is anchor artist.
    def _is_anchor_artist(self, artist_name, anchor_profile):
        """
        Return whether anchor artist.
        
        This method implements the is anchor artist step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        artist_norm = self._normalize(artist_name)
        if not artist_norm:
            return False
        anchor_artists = anchor_profile.get("artists", set())
        anchor_text = anchor_profile.get("anchor_text", "")
        if artist_norm in anchor_artists:
            return True
        for anchor_artist in anchor_artists:
            if anchor_artist and (anchor_artist in artist_norm or artist_norm in anchor_artist):
                return True
        if anchor_text and len(anchor_text) >= 3:
            if anchor_text in artist_norm or artist_norm in anchor_text:
                return True
        return False

    # Internal helper to resolve strict artist-only query targets.
    def _resolve_strict_artist_targets(self, query, possible_like_mode=False):
        """
        Resolve strict artist targets.
        
        This method implements the resolve strict artist targets step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if possible_like_mode:
            return set()
        query_norm = self._normalize(query)
        if not query_norm:
            return set()
        raw_tokens = self._query_tokens(query_norm)
        if not raw_tokens:
            return set()
        if len(raw_tokens) > 5:
            return set()
        if raw_tokens.intersection(self.STRICT_ARTIST_BLOCK_TOKENS):
            return set()
        if raw_tokens.intersection(self.FACET_HINT_TOKENS):
            return set()
        if raw_tokens.intersection(self.MOOD_TOKENS):
            return set()
        if any(re.fullmatch(r"(19|20)\d{2}", token) for token in raw_tokens):
            return set()

        if query_norm in self._artist_to_track_ids:
            return {query_norm}

        token_sets = []
        for token in raw_tokens:
            artists_for_token = self._artist_token_to_artists.get(token)
            if not artists_for_token:
                return set()
            token_sets.append(set(artists_for_token))
        if not token_sets:
            return set()

        candidate_artists = set.intersection(*token_sets)
        if not candidate_artists:
            return set()

        strong = set()
        for artist_norm in candidate_artists:
            artist_tokens = set(re.findall("[a-z0-9]+", artist_norm))
            overlap = self._set_overlap(artist_tokens, raw_tokens)
            if overlap >= 0.95 or query_norm in artist_norm or artist_norm in query_norm:
                strong.add(artist_norm)
        if not strong:
            return set()

        if query_norm in strong:
            return {query_norm}

        if len(raw_tokens) == 1:
            return set()

        if len(strong) <= 3:
            return strong
        return set()

    # Internal helper to build profile from scored rows.
    def _build_profile_from_scored(self, scored, anchor_norm, anchor_tokens):
        """
        Build profile from scored.
        
        This method implements the build profile from scored step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if not scored:
            return {
                "anchor_text": anchor_norm,
                "anchor_tokens": anchor_tokens,
                "artists": set(),
                "genres": set(),
                "vibes": set(),
                "moods": set(),
                "semantic_texts": [],
            }
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:120]
        artist_counter = Counter()
        genre_counter = Counter()
        vibe_counter = Counter()
        mood_counter = Counter()
        semantic_texts = []
        audio_totals = {"tempo": 0.0, "energy": 0.0, "brightness": 0.0, "mood": 0.0, "valence": 0.0}
        audio_weight_sum = 0.0
        for score, meta, artist_norm in top:
            weight = float(max(score, 1e-6))
            if artist_norm:
                artist_counter[artist_norm] += weight
            for token in self._meta_genre_tokens(meta):
                genre_counter[token] += weight
            for token in self._meta_tag_tokens(meta):
                vibe_counter[token] += weight
            for token in self._meta_mood_tokens(meta):
                mood_counter[token] += weight
            semantic_texts.append(self._seed_text_from_meta(meta))
            audio = {
                "tempo": self._audio_value(meta.get("tempo", 0.0)),
                "energy": self._audio_value(meta.get("energy", 0.0)),
                "brightness": self._audio_value(meta.get("brightness", 0.0)),
                "mood": self._audio_value(meta.get("mood", 0.0)),
                "valence": self._audio_value(meta.get("valence", 0.0)),
            }
            for key, value in audio.items():
                audio_totals[key] += float(value) * weight
            audio_weight_sum += weight
        audio_profile = None
        if audio_weight_sum > 0:
            audio_profile = {
                key: min(1.0, max(0.0, value / audio_weight_sum))
                for key, value in audio_totals.items()
            }
        return {
            "anchor_text": anchor_norm,
            "anchor_tokens": anchor_tokens,
            "artists": self._top_token_set(artist_counter, limit=6, min_ratio=0.20),
            "genres": self._top_token_set(genre_counter, limit=24, min_ratio=0.16),
            "vibes": self._top_token_set(vibe_counter, limit=32, min_ratio=0.14),
            "moods": self._top_token_set(mood_counter, limit=18, min_ratio=0.14),
            "semantic_texts": semantic_texts[:32],
            "audio_profile": audio_profile,
        }

    # Internal helper to build profile from explicit artist matches.
    def _build_anchor_profile_from_artist(self, anchor_norm, anchor_tokens):
        """
        Build anchor profile from artist.
        
        This method implements the build anchor profile from artist step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if not anchor_norm:
            return None

        candidate_artists = set()
        if anchor_norm in self._artist_to_track_ids:
            candidate_artists.add(anchor_norm)
        for token in anchor_tokens:
            candidate_artists.update(self._artist_token_to_artists.get(token, set()))
        if not candidate_artists:
            return None

        scored = []
        for artist_norm in candidate_artists:
            score = 0.0
            if artist_norm == anchor_norm:
                score = 4.0
            elif anchor_norm in artist_norm or artist_norm in anchor_norm:
                score = 3.0
            else:
                overlap = self._set_overlap(set(artist_norm.split()), anchor_tokens)
                if overlap >= 0.66:
                    score = 2.0 + overlap
                elif overlap >= 0.40:
                    score = 1.0 + overlap
            if score <= 0:
                continue

            for track_id in self._artist_to_track_ids.get(artist_norm, []):
                meta = self.metadata.get(str(track_id), {})
                if not isinstance(meta, dict):
                    continue
                year = self._safe_year(meta)
                if year and year not in settings.SEARCH_ERA_RANGE:
                    continue
                pop_boost = min(self._popularity_score(meta) / 12.0, 0.75)
                scored.append((score + pop_boost, meta, artist_norm))

        if len(scored) < 3:
            return None
        return self._build_profile_from_scored(scored, anchor_norm, anchor_tokens)

    # Build anchor profile.
    def _build_anchor_profile(self, anchor_text):
        """
        Build anchor profile.
        
        This method implements the build anchor profile step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if self.faiss_index is None:
            return None
        anchor_norm = self._normalize(anchor_text)
        anchor_tokens = self._query_tokens(anchor_norm)
        if not anchor_tokens:
            return None

        # Prefer exact/near artist matching to avoid lexical drift (e.g., "party").
        direct_artist_profile = self._build_anchor_profile_from_artist(anchor_norm, anchor_tokens)
        if direct_artist_profile is not None:
            return direct_artist_profile

        query_vec = self._get_query_vector(f"text::{anchor_norm}", anchor_norm)
        fetch_k = max(120, min(320, len(self.id_map)))
        distances, indices = self._faiss_search_cached(
            query_vec=query_vec,
            query_vec_key=f"text::{anchor_norm}",
            fetch_k=fetch_k,
        )
        scored = []
        for i in range(len(indices[0])):
            idx = int(indices[0][i])
            if idx < 0 or idx >= len(self.id_map):
                continue
            track_id = str(self.id_map[idx])
            meta = self.metadata.get(track_id, {})
            if not meta:
                continue
            year = self._safe_year(meta)
            if year and year not in settings.SEARCH_ERA_RANGE:
                continue
            artist_norm = self._normalize(meta.get("artist_name", ""))
            title_norm = self._normalize(meta.get("title", ""))
            lexical = 0.0
            if artist_norm and (anchor_norm in artist_norm or artist_norm in anchor_norm):
                lexical = 0.35
            elif artist_norm and self._set_overlap(set(artist_norm.split()), anchor_tokens) > 0:
                lexical = 0.20
            elif title_norm and (anchor_norm in title_norm or title_norm in anchor_norm):
                lexical = 0.12
            score = max(float(distances[0][i]), 0.0) + lexical
            scored.append((score, meta, artist_norm))
        return self._build_profile_from_scored(scored, anchor_norm, anchor_tokens)

    # Get anchor profile.
    def _get_anchor_profile(self, anchor_text):
        """
        Get anchor profile.
        
        This method implements the get anchor profile step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        key = self._normalize(anchor_text)
        if not key:
            return None
        cached = self._anchor_cache.get(key)
        if cached is not None:
            return cached
        profile = self._build_anchor_profile(key)
        if profile is None:
            return None
        if len(self._anchor_cache) >= 32:
            oldest_key = next(iter(self._anchor_cache))
            self._anchor_cache.pop(oldest_key, None)
        self._anchor_cache[key] = profile
        return profile

    # Search this operation.
    def search(self, query, limit=20, offset=0):
        """
        Execute search.
        
        This method implements the search step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if self.faiss_index is None:
            return {"query": query, "results": []}

        limit = max(1, int(limit))
        offset = max(0, int(offset))
        query_norm = self._normalize(query)
        result_cache_key = (query_norm, limit, offset)
        cached_payload = self._cache_get(self._result_cache, result_cache_key)
        if cached_payload is not None:
            if isinstance(cached_payload, dict):
                cached_rows = cached_payload.get("results", [])
                return {
                    "query": query,
                    "results": [dict(row) for row in (cached_rows or []) if isinstance(row, dict)],
                    "query_metrics": dict(cached_payload.get("query_metrics", {}) or {}),
                    "search_metrics": dict(cached_payload.get("search_metrics", {}) or {}),
                    "fallback_lexical_used": bool(cached_payload.get("fallback_lexical_used", False)),
                }
            return {"query": query, "results": [dict(row) for row in (cached_payload or [])]}

        anchor_text = self._extract_like_anchor(query)
        possible_like_mode = bool(anchor_text)
        title_hint, artist_hint = self._extract_title_artist_hints(query)
        title_hint_tokens = set(self._query_tokens(title_hint))
        artist_hint_tokens = set(self._query_tokens(artist_hint))
        title_artist_query = bool(title_hint_tokens and artist_hint_tokens)
        strict_artist_targets = self._resolve_strict_artist_targets(
            query=query,
            possible_like_mode=possible_like_mode,
        )
        artist_hint_targets = self._resolve_artist_hint_targets(artist_hint)
        if artist_hint_targets:
            strict_artist_targets = set(artist_hint_targets)

        query_tokens = self._content_query_tokens(query)
        if not query_tokens:
            query_tokens = self._query_tokens(query)
        base_query_tokens = set(query_tokens)
        anchor_profile = self._get_anchor_profile(anchor_text) if anchor_text else None
        like_mode = bool(anchor_profile and anchor_profile.get("anchor_tokens"))
        strict_artist_mode = bool(strict_artist_targets) and (not like_mode)
        if like_mode:
            anchor_tokens = anchor_profile.get("anchor_tokens", set())
            if anchor_tokens:
                query_tokens = anchor_tokens
        active_query_bots = self._resolve_active_query_bots(query, base_query_tokens)
        query_facet_tokens = query_tokens.intersection(self.FACET_HINT_TOKENS)
        expanded_query_facet_tokens = self._expand_facet_tokens(query_facet_tokens)
        hard_query_facet_tokens = {
            token for token in query_facet_tokens if token not in self.FACET_SOFT_TOKENS
        }
        expanded_hard_query_facet_tokens = self._expand_facet_tokens(hard_query_facet_tokens)
        soft_query_facet_tokens = {
            token for token in query_facet_tokens if token in self.FACET_SOFT_TOKENS
        }
        expanded_soft_query_facet_tokens = self._expand_facet_tokens(soft_query_facet_tokens)
        query_mood_tokens = query_tokens.intersection(self.MOOD_TOKENS)
        instrumental_intent = self._extract_instrumental_intent(query)
        query_year = self._extract_query_year(query_tokens)
        query_audio_target, query_audio_hits = self._query_audio_targets(query)
        facet_heavy_query = self._is_facet_heavy_query(
            query_tokens=query_tokens,
            query_year=query_year,
            query_audio_hits=query_audio_hits,
            like_mode=like_mode,
        )
        facet_token_count = len(query_facet_tokens)
        facet_context_min_support = float(
            getattr(settings, "SEARCH_FACET_CONTEXT_MIN_SUPPORT", 0.18)
        )
        facet_alignment_min = float(getattr(settings, "SEARCH_FACET_ALIGNMENT_MIN", 0.20))
        facet_coverage_min = float(getattr(settings, "SEARCH_FACET_TOKEN_COVERAGE_MIN", 0.15))
        hard_facet_coverage_min = float(
            getattr(settings, "SEARCH_FACET_HARD_TOKEN_COVERAGE_MIN", 0.14)
        )
        soft_facet_coverage_min = float(
            getattr(settings, "SEARCH_FACET_SOFT_TOKEN_COVERAGE_MIN", 0.20)
        )
        facet_audio_min = float(getattr(settings, "SEARCH_FACET_AUDIO_MIN", 0.76))
        soft_facet_audio_bypass_min = float(
            getattr(settings, "SEARCH_FACET_SOFT_AUDIO_BYPASS_MIN", 0.92)
        )
        mixed_compositional_facet_query = bool(
            expanded_hard_query_facet_tokens and expanded_soft_query_facet_tokens
        )
        soft_facet_bypass_allowed = not (
            mixed_compositional_facet_query and len(soft_query_facet_tokens) >= 2
        )
        soft_focus_token = (
            self._select_soft_focus_token(soft_query_facet_tokens)
            if mixed_compositional_facet_query
            else ""
        )
        soft_focus_token_max_df = float(getattr(settings, "SEARCH_SOFT_FOCUS_MAX_DF", 0.02))
        hard_focus_token_max_df = float(getattr(settings, "SEARCH_HARD_FOCUS_MAX_DF", 0.06))
        enforce_soft_focus = bool(
            soft_focus_token
            and (
                self._title_artist_token_df_ratio(soft_focus_token)
                <= max(0.0, soft_focus_token_max_df)
            )
        )
        hard_focus_token = (
            self._select_hard_focus_token(hard_query_facet_tokens)
            if mixed_compositional_facet_query
            else ""
        )
        enforce_hard_focus = bool(
            hard_focus_token
            and (
                self._title_artist_token_df_ratio(hard_focus_token)
                <= max(0.0, hard_focus_token_max_df)
            )
        )
        # Keep hard focus optional for mixed compositional queries; forcing it can
        # collapse results into hard-keyword-only matches.
        if facet_token_count >= 3:
            facet_context_min_support += 0.03
            facet_alignment_min += 0.03
            facet_coverage_min += 0.03
            hard_facet_coverage_min += 0.03
            soft_facet_coverage_min += 0.02
            facet_audio_min += 0.02
            soft_facet_audio_bypass_min += 0.03
        if query_audio_target is not None:
            facet_context_min_support += 0.01
        facet_context_min_support = self._clamp01(facet_context_min_support)
        facet_alignment_min = self._clamp01(facet_alignment_min)
        facet_coverage_min = self._clamp01(facet_coverage_min)
        hard_facet_coverage_min = self._clamp01(hard_facet_coverage_min)
        soft_facet_coverage_min = self._clamp01(soft_facet_coverage_min)
        facet_audio_min = self._clamp01(facet_audio_min)
        soft_facet_audio_bypass_min = self._clamp01(soft_facet_audio_bypass_min)

        semantic_query = anchor_text if possible_like_mode else query
        if (not like_mode) and facet_heavy_query:
            semantic_parts = [query_norm]
            facet_terms = sorted(query_facet_tokens)
            if facet_terms:
                semantic_parts.append(f"mood genre vibe {' '.join(facet_terms[:8])}")
            if query_year:
                semantic_parts.append(f"year {query_year}")
            semantic_query = ". ".join(part for part in semantic_parts if part)
        semantic_key = self._normalize(semantic_query)

        if like_mode:
            seed_texts = anchor_profile.get("semantic_texts", [])
            if seed_texts:
                anchor_key = self._normalize(anchor_profile.get("anchor_text", anchor_text))
                query_vec, query_vec_key = self._get_anchor_seed_vector(anchor_key, seed_texts)
            else:
                query_vec_key = f"text::{semantic_key}"
                query_vec = self._get_query_vector(query_vec_key, semantic_query)
        else:
            query_vec_key = f"text::{semantic_key}"
            query_vec = self._get_query_vector(query_vec_key, semantic_query)

        base_multiplier = int(getattr(settings, "SEARCH_BASE_FETCH_MULTIPLIER", 4))
        fetch_k = max(limit * max(1, base_multiplier), limit)
        if like_mode or possible_like_mode:
            fetch_k = max(limit * 10, 220)
        if strict_artist_mode:
            strict_multiplier = int(getattr(settings, "SEARCH_STRICT_ARTIST_FETCH_MULTIPLIER", 120))
            strict_min = int(getattr(settings, "SEARCH_STRICT_ARTIST_FETCH_MIN", 2400))
            fetch_k = max(fetch_k, max(limit * max(1, strict_multiplier), strict_min))
        if facet_heavy_query and (not like_mode):
            facet_multiplier = int(getattr(settings, "SEARCH_FACET_FETCH_MULTIPLIER", 70))
            facet_min = int(getattr(settings, "SEARCH_FACET_FETCH_MIN", 1800))
            fetch_k = max(fetch_k, max(limit * max(1, facet_multiplier), facet_min))
        fetch_k = max(1, min(fetch_k, max(1, len(self.id_map))))
        distances, indices = self._faiss_search_cached(
            query_vec=query_vec,
            query_vec_key=query_vec_key,
            fetch_k=fetch_k,
        )
        base_candidate_ids, semantic_score_by_track = self._faiss_hits_to_track_ids(distances, indices)

        use_candidate_union = bool(getattr(settings, "SEARCH_USE_CANDIDATE_UNION", True))
        candidate_track_ids = list(base_candidate_ids)
        if use_candidate_union:
            union_sources = [base_candidate_ids]
            union_max = int(getattr(settings, "SEARCH_UNION_MAX_CANDIDATES", 5000))
            union_extra_fetch_k = int(
                getattr(settings, "SEARCH_UNION_EXTRA_FETCH_K", max(limit * 18, 900))
            )

            def _merge_semantic_scores(score_map):
                """
                Merge semantic scores.
                
                This function implements the merge semantic scores step for this module.
                It is used to keep the broader workflow readable and easier to maintain.
                """
                for tid, score in (score_map or {}).items():
                    prev = semantic_score_by_track.get(tid)
                    if prev is None or float(score) > float(prev):
                        semantic_score_by_track[tid] = float(score)

            # Alternate semantic retrieval on raw query when facet-expanded query is active.
            if semantic_key != query_norm and query_norm:
                raw_ids, raw_scores = self._candidate_ids_from_semantic_text(
                    query_norm,
                    fetch_k=min(union_extra_fetch_k, max(1, len(self.id_map))),
                    cache_prefix="union-raw",
                )
                union_sources.append(raw_ids)
                _merge_semantic_scores(raw_scores)

            # Facet-focused semantic retrieval to improve recall on vague / compositional queries.
            if facet_heavy_query and (not like_mode):
                facet_terms = sorted(query_tokens.intersection(self.FACET_HINT_TOKENS))
                facet_query_parts = []
                if facet_terms:
                    facet_query_parts.append(" ".join(facet_terms[:12]))
                if query_mood_tokens:
                    facet_query_parts.append("mood " + " ".join(sorted(query_mood_tokens)[:8]))
                if query_year:
                    facet_query_parts.append(f"year {query_year}")
                if query_audio_target is not None:
                    audio_clues = []
                    for key in ("tempo", "energy", "brightness", "mood", "valence"):
                        v = float(query_audio_target.get(key, 0.0))
                        if v >= 0.66:
                            audio_clues.append(f"{key} high")
                        elif v <= 0.34:
                            audio_clues.append(f"{key} low")
                    if audio_clues:
                        facet_query_parts.append("audio " + " ".join(audio_clues))
                facet_semantic_query = ". ".join([part for part in facet_query_parts if part]).strip()
                if facet_semantic_query:
                    facet_ids, facet_scores = self._candidate_ids_from_semantic_text(
                        facet_semantic_query,
                        fetch_k=min(union_extra_fetch_k, max(1, len(self.id_map))),
                        cache_prefix="union-facet",
                    )
                    union_sources.append(facet_ids)
                    _merge_semantic_scores(facet_scores)

            # Compositional facet recall channel:
            # pull candidates by rare query facet tokens in title/artist, then gate by hard facets.
            if facet_heavy_query and (not like_mode):
                if mixed_compositional_facet_query:
                    facet_title_artist_tokens = set()
                    if hard_focus_token:
                        facet_title_artist_tokens.add(hard_focus_token)
                    if soft_focus_token:
                        facet_title_artist_tokens.add(soft_focus_token)
                    mixed_hard_token_limit = max(
                        1,
                        int(getattr(settings, "SEARCH_UNION_MIXED_HARD_TOKEN_MAX", 2)),
                    )
                    mixed_soft_token_limit = max(
                        1,
                        int(getattr(settings, "SEARCH_UNION_MIXED_SOFT_TOKEN_MAX", 2)),
                    )
                    rare_hard_tokens = sorted(
                        expanded_hard_query_facet_tokens,
                        key=lambda tok: (
                            self._title_artist_token_df.get(str(tok), 10**9),
                            len(str(tok)),
                            str(tok),
                        ),
                    )
                    # For mixed compositional queries, keep soft recall anchored to
                    # the user's explicit soft token(s) first (e.g., "midnight"),
                    # instead of broad expanded variants (e.g., "night", "late").
                    soft_recall_token_pool = (
                        soft_query_facet_tokens
                        if soft_query_facet_tokens
                        else expanded_soft_query_facet_tokens
                    )
                    rare_soft_tokens = sorted(
                        soft_recall_token_pool,
                        key=lambda tok: (
                            self._title_artist_token_df.get(str(tok), 10**9),
                            len(str(tok)),
                            str(tok),
                        ),
                    )
                    facet_title_artist_tokens.update(rare_hard_tokens[:mixed_hard_token_limit])
                    facet_title_artist_tokens.update(rare_soft_tokens[:mixed_soft_token_limit])
                else:
                    facet_title_artist_tokens = set(expanded_soft_query_facet_tokens)
                    if enforce_hard_focus and hard_focus_token:
                        facet_title_artist_tokens.add(hard_focus_token)
                    elif enforce_soft_focus and soft_focus_token:
                        facet_title_artist_tokens.add(soft_focus_token)
                if facet_title_artist_tokens:
                    mixed_title_artist_filter_tokens = (
                        expanded_query_facet_tokens
                        if mixed_compositional_facet_query
                        else expanded_hard_query_facet_tokens
                    )
                    mixed_title_artist_filter_min = self._clamp01(
                        float(
                            getattr(
                                settings,
                                "SEARCH_UNION_FACET_TITLE_ARTIST_MIXED_MIN",
                                0.08,
                            )
                        )
                    )
                    title_artist_filter_min = (
                        mixed_title_artist_filter_min
                        if mixed_compositional_facet_query
                        else self._clamp01(
                            float(
                                getattr(
                                    settings,
                                    "SEARCH_UNION_FACET_TITLE_ARTIST_HARD_MIN",
                                    0.12,
                                )
                            )
                        )
                    )
                    title_artist_ids = self._candidate_ids_from_facet_title_artist_tokens(
                        facet_title_artist_tokens,
                        limit=int(
                            getattr(
                                settings,
                                "SEARCH_UNION_FACET_TITLE_ARTIST_CANDIDATES_MAX",
                                900,
                            )
                        ),
                        hard_filter_tokens=mixed_title_artist_filter_tokens,
                        hard_filter_min=title_artist_filter_min,
                    )
                    union_sources.append(title_artist_ids)

            # Explicit artist candidate channel for artist-only queries.
            if strict_artist_targets:
                union_sources.append(
                    self._candidate_ids_from_artist_targets(
                        strict_artist_targets,
                        limit=int(getattr(settings, "SEARCH_UNION_ARTIST_CANDIDATES_MAX", 800)),
                    )
                )

            # Artist-anchor candidate channel for "songs like X" queries.
            if like_mode and anchor_profile is not None:
                union_sources.append(
                    self._candidate_ids_from_anchor_profile(
                        anchor_profile,
                        limit=int(getattr(settings, "SEARCH_UNION_ANCHOR_CANDIDATES_MAX", 900)),
                    )
                )

            candidate_track_ids = self._merge_candidate_id_lists(union_sources, union_max)

        exclude_anchor_artist = bool(
            getattr(settings, "SEARCH_LIKE_EXCLUDE_ANCHOR_ARTIST", True)
        )

        base_weights = settings.SEARCH_WEIGHTS
        alpha = float(base_weights.get("alpha", 0.6))
        beta = float(base_weights.get("beta", 0.3))
        gamma = float(base_weights.get("gamma", 0.1))

        like_weights = getattr(settings, "SEARCH_LIKE_WEIGHTS", {})
        like_semantic = float(like_weights.get("semantic", 0.42))
        like_overlap = float(like_weights.get("overlap", 0.13))
        like_popularity = float(like_weights.get("popularity", 0.05))
        like_facet = float(like_weights.get("facet", 0.30))
        like_artist = float(like_weights.get("artist", 0.10))
        like_audio = float(like_weights.get("audio", 0.06))

        facet_weights = getattr(settings, "SEARCH_FACET_WEIGHTS", {})
        genre_weight = float(facet_weights.get("genre", 0.40))
        vibe_weight = float(facet_weights.get("vibe", 0.35))
        mood_weight = float(facet_weights.get("mood", 0.25))
        facet_total = max(genre_weight + vibe_weight + mood_weight, 1e-8)

        non_like_weights = getattr(settings, "SEARCH_NON_LIKE_WEIGHTS", {})
        non_like_semantic = float(non_like_weights.get("semantic", alpha))
        non_like_overlap = float(non_like_weights.get("tag_overlap", beta))
        non_like_popularity = float(non_like_weights.get("popularity", gamma))
        non_like_facet = float(non_like_weights.get("facet", 0.20))
        non_like_description = float(non_like_weights.get("description", 0.18))
        non_like_title_penalty = float(non_like_weights.get("title_penalty", 0.08))
        non_like_year_bonus = float(non_like_weights.get("year_bonus", 0.04))
        non_like_audio = float(non_like_weights.get("audio", 0.14))
        non_like_artist_query = float(non_like_weights.get("artist_query", 0.12))
        non_like_title_query = float(non_like_weights.get("title_query", 0.14))
        strict_title_artist_mode = bool(title_artist_query and strict_artist_mode and (not like_mode))
        bot_profile_weight = float(getattr(settings, "SEARCH_BOT_PROFILE_WEIGHT", 0.18) or 0.18)
        if like_mode:
            bot_profile_weight *= 0.55
        if strict_artist_mode:
            bot_profile_weight *= 0.35
        if not active_query_bots:
            bot_profile_weight = 0.0
        else:
            top_bot_conf = self._safe_float(active_query_bots[0].get("confidence", 0.0), 0.0)
            min_bot_conf = float(getattr(settings, "SEARCH_BOT_MIN_CONFIDENCE", 0.24))
            if top_bot_conf < min_bot_conf:
                bot_profile_weight = 0.0

        if facet_heavy_query and (not like_mode):
            semantic_scale = float(getattr(settings, "SEARCH_FACET_SEMANTIC_SCALE", 0.58))
            context_boost = float(getattr(settings, "SEARCH_FACET_CONTEXT_BOOST", 1.35))
            title_boost = float(getattr(settings, "SEARCH_FACET_TITLE_PENALTY_BOOST", 1.55))
            if len(query_tokens) <= 2:
                semantic_scale = min(semantic_scale, 0.50)
                context_boost = max(context_boost, 1.45)
                title_boost = max(title_boost, 1.65)
            non_like_semantic *= semantic_scale
            non_like_overlap *= context_boost
            non_like_facet *= context_boost
            non_like_description *= context_boost
            non_like_title_penalty *= title_boost

        instrumental_weights = getattr(settings, "SEARCH_INSTRUMENTAL_WEIGHTS", {})
        match_boost = float(instrumental_weights.get("match_boost", 0.12))
        mismatch_penalty = float(instrumental_weights.get("mismatch_penalty", 0.18))
        unknown_penalty = float(instrumental_weights.get("unknown_penalty", 0.04))
        confidence_floor = float(instrumental_weights.get("confidence_floor", 0.35))

        tuned_weight_groups = self._resolve_dynamic_search_weights(
            query_tokens=query_tokens,
            like_mode=like_mode,
            strict_artist_mode=strict_artist_mode,
            facet_heavy_query=facet_heavy_query,
            query_audio_target=query_audio_target,
            query_year=query_year,
            instrumental_intent=instrumental_intent,
            base_weights=base_weights,
            like_weights=like_weights,
            non_like_weights=non_like_weights,
            facet_weights=facet_weights,
            instrumental_weights=instrumental_weights,
        )
        base_weights = tuned_weight_groups["base"]
        like_weights = tuned_weight_groups["like"]
        non_like_weights = tuned_weight_groups["non_like"]
        facet_weights = tuned_weight_groups["facet"]
        instrumental_weights = tuned_weight_groups["instrumental"]

        alpha = float(base_weights.get("alpha", alpha))
        beta = float(base_weights.get("beta", beta))
        gamma = float(base_weights.get("gamma", gamma))
        like_semantic = float(like_weights.get("semantic", like_semantic))
        like_overlap = float(like_weights.get("overlap", like_overlap))
        like_popularity = float(like_weights.get("popularity", like_popularity))
        like_facet = float(like_weights.get("facet", like_facet))
        like_artist = float(like_weights.get("artist", like_artist))
        like_audio = float(like_weights.get("audio", like_audio))
        genre_weight = float(facet_weights.get("genre", genre_weight))
        vibe_weight = float(facet_weights.get("vibe", vibe_weight))
        mood_weight = float(facet_weights.get("mood", mood_weight))
        facet_total = max(genre_weight + vibe_weight + mood_weight, 1e-8)
        non_like_semantic = float(non_like_weights.get("semantic", non_like_semantic))
        non_like_overlap = float(non_like_weights.get("tag_overlap", non_like_overlap))
        non_like_popularity = float(non_like_weights.get("popularity", non_like_popularity))
        non_like_facet = float(non_like_weights.get("facet", non_like_facet))
        non_like_description = float(non_like_weights.get("description", non_like_description))
        non_like_title_penalty = float(non_like_weights.get("title_penalty", non_like_title_penalty))
        non_like_year_bonus = float(non_like_weights.get("year_bonus", non_like_year_bonus))
        non_like_audio = float(non_like_weights.get("audio", non_like_audio))
        non_like_artist_query = float(non_like_weights.get("artist_query", non_like_artist_query))
        non_like_title_query = float(non_like_weights.get("title_query", non_like_title_query))
        if title_artist_query:
            non_like_artist_query *= 2.25
            non_like_title_query *= 1.55
        if strict_artist_mode:
            non_like_popularity *= 0.60
        if strict_title_artist_mode:
            non_like_popularity *= 0.20
            non_like_artist_query *= 0.12
            non_like_title_query *= 2.70
            non_like_semantic *= 1.08
            semantic_score_by_track = self._backfill_candidate_semantic_scores(
                query_vec=query_vec,
                candidate_track_ids=candidate_track_ids,
                semantic_score_by_track=semantic_score_by_track,
            )
        match_boost = float(instrumental_weights.get("match_boost", match_boost))
        mismatch_penalty = float(instrumental_weights.get("mismatch_penalty", mismatch_penalty))
        unknown_penalty = float(instrumental_weights.get("unknown_penalty", unknown_penalty))
        confidence_floor = float(instrumental_weights.get("confidence_floor", confidence_floor))

        candidates = []
        for track_id in candidate_track_ids:
            semantic_score = float(semantic_score_by_track.get(str(track_id), 0.0))
            meta = self.metadata.get(track_id, {})
            features = self._get_track_features(track_id, meta)
            year = features["year"]
            if year and year not in settings.SEARCH_ERA_RANGE:
                continue
            if strict_artist_mode:
                candidate_artist = self._normalize(meta.get("artist_name", ""))
                if candidate_artist not in strict_artist_targets:
                    continue
            if (
                like_mode
                and exclude_anchor_artist
                and self._is_anchor_artist(meta.get("artist_name", ""), anchor_profile)
            ):
                continue
            overlap_tokens = (
                query_facet_tokens
                if (facet_heavy_query and (not like_mode) and query_facet_tokens)
                else query_tokens
            )
            overlap_score = self._set_overlap(features["tag_tokens"], overlap_tokens)
            popularity = features["popularity"]

            genre_score = 0.0
            vibe_score = 0.0
            mood_score = 0.0
            artist_score = 0.0
            facet_score = 0.0
            description_score = 0.0
            title_score = 0.0
            title_bias_penalty = 0.0
            year_match_score = 0.0
            instrumental_match_score = 0.0
            audio_query_score = 0.0
            audio_anchor_score = 0.0
            artist_query_score = 0.0
            title_hint_score = 0.0
            artist_exact_match = 0.0
            title_exact_match = 0.0
            facet_alignment = 0.0
            facet_token_coverage = 0.0
            hard_facet_token_coverage = 0.0
            soft_facet_token_coverage = 0.0
            soft_title_facet_coverage = 0.0
            soft_focus_match = 0.0
            hard_focus_match = 0.0
            mixed_soft_signal = 0.0
            mixed_intent_coverage = 0.0
            focus_gate_penalty = 0.0
            bot_profile_label = ""
            bot_profile_score = 0.0
            genre_tokens = features["genre_tokens"]
            vibe_tokens = features["tag_tokens"]
            mood_tokens = features["mood_tokens"]
            description_tokens = features["description_tokens"]
            description_trust = self._clamp01(features.get("description_trust", 0.0))
            title_tokens = features["title_tokens"]
            artist_tokens = self._query_tokens(meta.get("artist_name", ""))
            candidate_audio = features["audio"]
            bot_profile_label, bot_profile_score = self._candidate_bot_profile_match(
                meta,
                features,
                active_query_bots,
            )

            if like_mode:
                anchor_genres = anchor_profile.get("genres", set())
                anchor_vibes = anchor_profile.get("vibes", set())
                anchor_moods = anchor_profile.get("moods", set()) or query_mood_tokens
                anchor_audio = anchor_profile.get("audio_profile")

                genre_score = self._set_overlap(genre_tokens, anchor_genres)
                vibe_score = self._set_overlap(vibe_tokens, anchor_vibes)
                mood_score = self._set_overlap(mood_tokens, anchor_moods)
                artist_score = self._artist_match_score(meta.get("artist_name", ""), anchor_profile)
                facet_score = (
                    genre_weight * genre_score
                    + vibe_weight * vibe_score
                    + mood_weight * mood_score
                ) / facet_total
                audio_anchor_score = self._mean_audio_similarity(candidate_audio, anchor_audio)
            else:
                facet_query_tokens = (
                    query_facet_tokens
                    if (facet_heavy_query and query_facet_tokens)
                    else query_tokens
                )
                description_query_tokens = (
                    facet_query_tokens
                    if (facet_heavy_query and len(facet_query_tokens) >= 2)
                    else query_tokens
                )
                genre_score = self._set_overlap(genre_tokens, facet_query_tokens)
                vibe_score = self._set_overlap(vibe_tokens, facet_query_tokens)
                if query_mood_tokens:
                    mood_score = self._set_overlap(mood_tokens, query_mood_tokens)
                else:
                    mood_score = self._set_overlap(
                        mood_tokens,
                        facet_query_tokens.intersection(self.MOOD_TOKENS),
                    )
                facet_score = (
                    genre_weight * genre_score
                    + vibe_weight * vibe_score
                    + mood_weight * mood_score
                ) / facet_total
                description_score = self._set_overlap(description_tokens, description_query_tokens)
                if mixed_compositional_facet_query:
                    desc_trusted_scale = float(
                        getattr(settings, "SEARCH_MIXED_DESC_TRUSTED_SCALE", 1.0)
                    )
                    desc_untrusted_scale = float(
                        getattr(settings, "SEARCH_MIXED_DESC_UNTRUSTED_SCALE", 0.08)
                    )
                    description_score *= (
                        desc_trusted_scale if description_trust >= 0.5 else desc_untrusted_scale
                    )
                title_score = self._set_overlap(title_tokens, query_tokens)
                if query_audio_target is not None:
                    audio_query_score = self._mean_audio_similarity(candidate_audio, query_audio_target)
                candidate_facet_tokens = (
                    set(genre_tokens)
                    .union(vibe_tokens)
                    .union(mood_tokens)
                    .union(title_tokens)
                )
                if expanded_query_facet_tokens:
                    facet_token_coverage = self._balanced_overlap(
                        candidate_facet_tokens,
                        expanded_query_facet_tokens,
                    )
                if expanded_hard_query_facet_tokens:
                    hard_facet_token_coverage = self._balanced_overlap(
                        set(genre_tokens)
                        .union(vibe_tokens)
                        .union(title_tokens)
                        ,
                        expanded_hard_query_facet_tokens,
                    )
                if expanded_soft_query_facet_tokens:
                    soft_raw_cov = self._balanced_overlap(
                        set(title_tokens),
                        soft_query_facet_tokens,
                    ) if soft_query_facet_tokens else 0.0
                    soft_expanded_cov = self._balanced_overlap(
                        set(title_tokens),
                        expanded_soft_query_facet_tokens,
                    )
                    if mixed_compositional_facet_query:
                        soft_expanded_scale = self._clamp01(
                            float(getattr(settings, "SEARCH_MIXED_SOFT_EXPANDED_SCALE", 0.55))
                        )
                        soft_facet_token_coverage = self._clamp01(
                            max(soft_raw_cov, soft_expanded_scale * soft_expanded_cov)
                        )
                    else:
                        soft_facet_token_coverage = soft_expanded_cov
                    soft_title_facet_coverage = soft_facet_token_coverage
                if soft_focus_token:
                    soft_focus_match = (
                        1.0 if (soft_focus_token in title_tokens) else 0.0
                    )
                if hard_focus_token:
                    hard_focus_match = (
                        1.0
                        if (
                            hard_focus_token in title_tokens
                            or hard_focus_token in genre_tokens
                            or hard_focus_token in vibe_tokens
                        )
                        else 0.0
                    )
                if mixed_compositional_facet_query:
                    if soft_focus_match > 0.0 and hard_focus_match <= 0.0:
                        soft_focus_match *= self._clamp01(
                            float(
                                getattr(
                                    settings,
                                    "SEARCH_MIXED_SOFT_FOCUS_WITHOUT_HARD_SCALE",
                                    0.55,
                                )
                            )
                        )
                    soft_context_raw = self._balanced_overlap(
                        set(vibe_tokens).union(mood_tokens),
                        soft_query_facet_tokens,
                    ) if soft_query_facet_tokens else 0.0
                    soft_context_expanded = self._balanced_overlap(
                        set(vibe_tokens).union(mood_tokens),
                        expanded_soft_query_facet_tokens,
                    )
                    soft_context_expanded_scale = self._clamp01(
                        float(getattr(settings, "SEARCH_MIXED_SOFT_CONTEXT_EXPANDED_SCALE", 0.50))
                    )
                    soft_context_signal = self._clamp01(
                        max(soft_context_raw, soft_context_expanded_scale * soft_context_expanded)
                    )
                    mixed_hard_focus_weight = self._clamp01(
                        float(getattr(settings, "SEARCH_MIXED_HARD_SIGNAL_FOCUS_WEIGHT", 0.20))
                    )
                    mixed_hard_cov_weight = self._clamp01(
                        float(getattr(settings, "SEARCH_MIXED_HARD_SIGNAL_COVERAGE_WEIGHT", 0.80))
                    )
                    mixed_soft_signal = max(
                        soft_title_facet_coverage,
                        soft_focus_match,
                        self._clamp01(
                            float(getattr(settings, "SEARCH_MIXED_SOFT_CONTEXT_SCALE", 0.45))
                            * soft_context_signal
                        ),
                    )
                    mixed_hard_signal = max(
                        mixed_hard_focus_weight * self._clamp01(hard_focus_match),
                        mixed_hard_cov_weight * hard_facet_token_coverage,
                    )
                    mixed_intent_coverage = self._clamp01(
                        min(mixed_hard_signal, mixed_soft_signal)
                    )
                facet_alignment = self._facet_alignment_score(
                    facet_score=facet_score,
                    genre_score=genre_score,
                    vibe_score=vibe_score,
                    mood_score=mood_score,
                    overlap_score=overlap_score,
                    description_score=description_score,
                    audio_query_score=audio_query_score,
                    bot_profile_score=bot_profile_score,
                )
                if artist_hint_tokens:
                    candidate_artist_norm = self._normalize(meta.get("artist_name", ""))
                    artist_query_score = self._balanced_overlap(
                        self._query_tokens(candidate_artist_norm),
                        artist_hint_tokens,
                    )
                    if artist_hint:
                        artist_exact_match = 1.0 if artist_hint in candidate_artist_norm else 0.0
                if title_hint_tokens:
                    candidate_title_norm = self._normalize(meta.get("title", ""))
                    title_hint_score = self._balanced_overlap(
                        title_tokens,
                        title_hint_tokens,
                    )
                    if title_hint:
                        title_exact_match = 1.0 if title_hint in candidate_title_norm else 0.0
                context_support = max(description_score, facet_score, overlap_score, mood_score, audio_query_score)
                if title_artist_query:
                    context_support = max(
                        context_support,
                        0.62 * max(title_hint_score, title_exact_match) + 0.38 * artist_query_score,
                    )
                title_bias_penalty = max(0.0, title_score - context_support)
                if query_year:
                    year_match_score = 1.0 if year == query_year else 0.0
                if (
                    facet_heavy_query
                    and (not title_artist_query)
                    and (
                        context_support < facet_context_min_support
                        or facet_alignment < (0.70 * facet_alignment_min)
                    )
                ):
                    continue
                if (
                    facet_heavy_query
                    and (not title_artist_query)
                    and expanded_query_facet_tokens
                    and facet_token_coverage < facet_coverage_min
                    and context_support < (facet_context_min_support + 0.05)
                ):
                    continue
                if (
                    facet_heavy_query
                    and (not title_artist_query)
                    and expanded_hard_query_facet_tokens
                    and hard_facet_token_coverage < hard_facet_coverage_min
                    and audio_query_score < 0.62
                ):
                    if mixed_compositional_facet_query:
                        focus_gate_penalty += float(
                            getattr(settings, "SEARCH_MIXED_HARD_COVERAGE_PENALTY", 0.20)
                        )
                    else:
                        continue
                if (
                    facet_heavy_query
                    and (not title_artist_query)
                    and expanded_hard_query_facet_tokens
                    and expanded_soft_query_facet_tokens
                    and soft_title_facet_coverage < soft_facet_coverage_min
                ):
                    if (not soft_facet_bypass_allowed) or (
                        audio_query_score < soft_facet_audio_bypass_min
                    ):
                        if mixed_compositional_facet_query:
                            focus_gate_penalty += float(
                                getattr(settings, "SEARCH_MIXED_SOFT_COVERAGE_PENALTY", 0.45)
                            )
                        else:
                            continue
                if (
                    facet_heavy_query
                    and (not title_artist_query)
                    and enforce_soft_focus
                    and expanded_hard_query_facet_tokens
                    and expanded_soft_query_facet_tokens
                    and soft_focus_match <= 0.0
                ):
                    if (not soft_facet_bypass_allowed) or (
                        audio_query_score < soft_facet_audio_bypass_min
                    ):
                        if mixed_compositional_facet_query:
                            focus_gate_penalty += float(
                                getattr(settings, "SEARCH_MIXED_SOFT_FOCUS_PENALTY", 0.50)
                            )
                        else:
                            continue
                if (
                    facet_heavy_query
                    and (not title_artist_query)
                    and enforce_hard_focus
                    and expanded_hard_query_facet_tokens
                    and hard_focus_match <= 0.0
                    and audio_query_score < max(0.72, soft_facet_audio_bypass_min)
                ):
                    if mixed_compositional_facet_query:
                        focus_gate_penalty += float(
                            getattr(settings, "SEARCH_MIXED_HARD_FOCUS_PENALTY", 0.18)
                        )
                    else:
                        continue
                if (
                    facet_heavy_query
                    and (not title_artist_query)
                    and expanded_hard_query_facet_tokens
                    and expanded_soft_query_facet_tokens
                    and query_audio_target is not None
                    and audio_query_score < facet_audio_min
                ):
                    if mixed_compositional_facet_query:
                        focus_gate_penalty += float(
                            getattr(settings, "SEARCH_MIXED_AUDIO_GATE_PENALTY", 0.18)
                        )
                    else:
                        continue
                if (
                    mixed_compositional_facet_query
                    and mixed_intent_coverage
                    < self._clamp01(
                        float(getattr(settings, "SEARCH_MIXED_INTENT_COVERAGE_MIN", 0.12))
                    )
                ):
                    deficit = self._clamp01(
                        float(getattr(settings, "SEARCH_MIXED_INTENT_COVERAGE_MIN", 0.12))
                        - mixed_intent_coverage
                    )
                    focus_gate_penalty += float(
                        getattr(settings, "SEARCH_MIXED_INTENT_COVERAGE_PENALTY", 0.36)
                    ) * deficit
                if (
                    mixed_compositional_facet_query
                    and hard_focus_token
                    and soft_focus_match > 0.0
                    and hard_focus_match <= 0.0
                ):
                    focus_gate_penalty += float(
                        getattr(settings, "SEARCH_MIXED_SOFT_WITHOUT_HARD_PENALTY", 0.12)
                    )
                if (
                    mixed_compositional_facet_query
                    and hard_focus_match > 0.0
                ):
                    mixed_soft_min = self._clamp01(
                        float(getattr(settings, "SEARCH_MIXED_HARD_ONLY_SOFT_MIN", 0.18))
                    )
                    if mixed_soft_signal < mixed_soft_min:
                        focus_gate_penalty += float(
                            getattr(settings, "SEARCH_MIXED_HARD_ONLY_PENALTY", 0.24)
                        ) * (mixed_soft_min - mixed_soft_signal)
                if (
                    title_bias_penalty >= 0.55
                    and context_support <= 0.10
                    and (not title_artist_query)
                ):
                    continue
                if (
                    title_bias_penalty >= 0.55
                    and context_support <= 0.10
                    and title_artist_query
                    and max(title_hint_score, artist_query_score) < 0.45
                ):
                    continue

            if instrumental_intent is not None:
                candidate_instrumental = as_optional_bool(meta.get("instrumental"))
                if candidate_instrumental is None:
                    instrumental_match_score = -unknown_penalty
                else:
                    confidence = float(meta.get("instrumental_confidence", 0.0) or 0.0)
                    confidence = min(1.0, max(confidence, confidence_floor))
                    if candidate_instrumental == instrumental_intent:
                        instrumental_match_score = match_boost * confidence
                    else:
                        instrumental_match_score = -mismatch_penalty * confidence

            candidates.append(
                {
                    "track_id": track_id,
                    "meta": meta,
                    "year": year,
                    "semantic_score": semantic_score,
                    "overlap_score": overlap_score,
                    "raw_popularity": popularity,
                    "genre_score": genre_score,
                    "vibe_score": vibe_score,
                    "mood_score": mood_score,
                    "artist_score": artist_score,
                    "facet_score": facet_score,
                    "description_score": description_score,
                    "title_score": title_score,
                    "title_bias_penalty": title_bias_penalty,
                    "year_match_score": year_match_score,
                    "instrumental_match_score": instrumental_match_score,
                    "audio_query_score": audio_query_score,
                    "audio_anchor_score": audio_anchor_score,
                    "artist_query_score": artist_query_score,
                    "title_hint_score": title_hint_score,
                    "artist_exact_match": artist_exact_match,
                    "title_exact_match": title_exact_match,
                    "facet_alignment": facet_alignment,
                    "facet_token_coverage": facet_token_coverage,
                    "hard_facet_token_coverage": hard_facet_token_coverage,
                    "soft_facet_token_coverage": soft_facet_token_coverage,
                    "soft_title_facet_coverage": soft_title_facet_coverage,
                    "soft_focus_match": soft_focus_match,
                    "hard_focus_match": hard_focus_match,
                    "mixed_soft_signal": mixed_soft_signal,
                    "mixed_intent_coverage": mixed_intent_coverage,
                    "focus_gate_penalty": focus_gate_penalty,
                    "bot_profile_label": bot_profile_label,
                    "bot_profile_score": bot_profile_score,
                }
            )

        max_pop = max((c["raw_popularity"] for c in candidates), default=0.0)
        for candidate in candidates:
            candidate["popularity_score"] = (
                candidate["raw_popularity"] / max_pop if max_pop > 0 else 0.0
            )

        ranked = []
        for candidate in candidates:
            track_id = candidate["track_id"]
            meta = candidate["meta"]
            year = candidate["year"]
            semantic_score = candidate["semantic_score"]
            overlap_score = candidate["overlap_score"]
            popularity_score = candidate["popularity_score"]

            if like_mode:
                final_score = (
                    like_semantic * semantic_score
                    + like_overlap * overlap_score
                    + like_popularity * popularity_score
                    + like_facet * candidate["facet_score"]
                    + like_artist * candidate["artist_score"]
                    + like_audio * candidate["audio_anchor_score"]
                    + bot_profile_weight * candidate["bot_profile_score"]
                    + candidate["instrumental_match_score"]
                )
            else:
                final_score = (
                    non_like_semantic * semantic_score
                    + non_like_overlap * overlap_score
                    + non_like_popularity * popularity_score
                    + non_like_facet * candidate["facet_score"]
                    + non_like_description * candidate["description_score"]
                    + non_like_audio * candidate["audio_query_score"]
                    + non_like_year_bonus * candidate["year_match_score"]
                    + non_like_artist_query * candidate["artist_query_score"]
                    + non_like_title_query * candidate["title_hint_score"]
                    + bot_profile_weight * candidate["bot_profile_score"]
                    - non_like_title_penalty * candidate["title_bias_penalty"]
                    - float(candidate.get("focus_gate_penalty", 0.0))
                    + candidate["instrumental_match_score"]
                )
                if facet_heavy_query and (not strict_artist_mode):
                    facet_alignment_boost = float(
                        getattr(settings, "SEARCH_FACET_ALIGNMENT_BOOST", 0.22)
                    )
                    facet_alignment_penalty = float(
                        getattr(settings, "SEARCH_FACET_ALIGNMENT_PENALTY", 0.26)
                    )
                    facet_coverage_boost = float(
                        getattr(settings, "SEARCH_FACET_TOKEN_COVERAGE_BOOST", 0.28)
                    )
                    facet_hard_penalty = float(
                        getattr(settings, "SEARCH_FACET_HARD_TOKEN_PENALTY", 0.22)
                    )
                    facet_soft_boost = float(
                        getattr(settings, "SEARCH_FACET_SOFT_TOKEN_COVERAGE_BOOST", 0.16)
                    )
                    facet_soft_penalty = float(
                        getattr(settings, "SEARCH_FACET_SOFT_TOKEN_PENALTY", 0.14)
                    )
                    facet_audio_boost = float(
                        getattr(settings, "SEARCH_FACET_AUDIO_BOOST", 0.24)
                    )
                    facet_alignment = self._clamp01(candidate.get("facet_alignment", 0.0))
                    facet_token_coverage = self._clamp01(
                        candidate.get("facet_token_coverage", 0.0)
                    )
                    hard_facet_coverage = self._clamp01(
                        candidate.get("hard_facet_token_coverage", 0.0)
                    )
                    soft_facet_coverage = self._clamp01(
                        candidate.get("soft_facet_token_coverage", 0.0)
                    )
                    soft_title_facet_coverage = self._clamp01(
                        candidate.get("soft_title_facet_coverage", soft_facet_coverage)
                    )
                    soft_focus_match = self._clamp01(
                        candidate.get("soft_focus_match", 0.0)
                    )
                    soft_focus_boost = float(
                        getattr(settings, "SEARCH_SOFT_FOCUS_BOOST", 0.22)
                    )
                    soft_focus_penalty = float(
                        getattr(settings, "SEARCH_SOFT_FOCUS_PENALTY", 0.16)
                    )
                    hard_focus_boost = float(
                        getattr(settings, "SEARCH_HARD_FOCUS_BOOST", 0.24)
                    )
                    hard_focus_penalty = float(
                        getattr(settings, "SEARCH_HARD_FOCUS_PENALTY", 0.20)
                    )
                    mixed_hard_focus_boost = float(
                        getattr(settings, "SEARCH_MIXED_HARD_FOCUS_BOOST", 0.12)
                    )
                    mixed_hard_focus_miss_penalty = float(
                        getattr(settings, "SEARCH_MIXED_HARD_FOCUS_MISS_PENALTY", 0.06)
                    )
                    hard_focus_match = self._clamp01(
                        candidate.get("hard_focus_match", 0.0)
                    )
                    final_score += facet_alignment_boost * facet_alignment
                    final_score += facet_coverage_boost * facet_token_coverage
                    final_score += facet_soft_boost * soft_facet_coverage
                    final_score += 0.18 * soft_title_facet_coverage
                    final_score += soft_focus_boost * soft_focus_match
                    final_score += hard_focus_boost * hard_focus_match
                    if mixed_compositional_facet_query and hard_focus_token:
                        final_score += mixed_hard_focus_boost * hard_focus_match
                        if hard_focus_match <= 0.0:
                            final_score -= mixed_hard_focus_miss_penalty
                    final_score += float(
                        getattr(settings, "SEARCH_MIXED_INTENT_COVERAGE_BOOST", 0.32)
                    ) * self._clamp01(candidate.get("mixed_intent_coverage", 0.0))
                    final_score += facet_audio_boost * self._clamp01(
                        candidate.get("audio_query_score", 0.0)
                    )
                    if facet_alignment < facet_alignment_min:
                        final_score -= facet_alignment_penalty * (
                            facet_alignment_min - facet_alignment
                        )
                    if expanded_hard_query_facet_tokens and hard_facet_coverage < hard_facet_coverage_min:
                        final_score -= facet_hard_penalty * (
                            hard_facet_coverage_min - hard_facet_coverage
                        )
                    if (
                        expanded_hard_query_facet_tokens
                        and expanded_soft_query_facet_tokens
                        and soft_title_facet_coverage < soft_facet_coverage_min
                    ):
                        effective_soft_penalty = float(facet_soft_penalty)
                        if not soft_facet_bypass_allowed:
                            effective_soft_penalty *= 1.75
                        final_score -= effective_soft_penalty * (
                            soft_facet_coverage_min - soft_title_facet_coverage
                        )
                    if (
                        enforce_soft_focus
                        and expanded_hard_query_facet_tokens
                        and expanded_soft_query_facet_tokens
                        and soft_focus_match <= 0.0
                    ):
                        final_score -= soft_focus_penalty
                    if (
                        enforce_hard_focus
                        and expanded_hard_query_facet_tokens
                        and self._clamp01(candidate.get("hard_focus_match", 0.0)) <= 0.0
                    ):
                        final_score -= hard_focus_penalty

            meta_video_id = str(meta.get("video_id", "") or "").strip()
            cover = (
                meta.get("cover_url")
                or meta.get("album_cover")
                or meta.get("thumbnail")
                or thumb_from_video_id(meta_video_id)
            )
            cover_candidates = thumbnail_candidates(meta_video_id, cover)

            ranked.append(
                {
                    "id": track_id,
                    "video_id": meta_video_id,
                    "title": meta.get("title", "Unknown"),
                    "artist": meta.get("artist_name", "Unknown"),
                    "album": meta.get("album_title", "Unknown"),
                    "description": str(meta.get("description", "") or ""),
                    "instrumental": as_optional_bool(meta.get("instrumental")),
                    "instrumental_confidence": float(meta.get("instrumental_confidence", 0.0) or 0.0),
                    "cover": cover,
                    "thumbnail": cover,
                    "cover_candidates": cover_candidates,
                    "year": year,
                    "release_date": f"{year}-01-01" if year else "",
                    "semantic_score": semantic_score,
                    "overlap_score": overlap_score,
                    "popularity_score": popularity_score,
                    "genre_score": candidate["genre_score"],
                    "vibe_score": candidate["vibe_score"],
                    "mood_score": candidate["mood_score"],
                    "artist_score": candidate["artist_score"],
                    "facet_score": candidate["facet_score"],
                    "description_score": candidate["description_score"],
                    "title_score": candidate["title_score"],
                    "title_bias_penalty": candidate["title_bias_penalty"],
                    "year_match_score": candidate["year_match_score"],
                    "instrumental_match_score": candidate["instrumental_match_score"],
                    "audio_query_score": candidate["audio_query_score"],
                    "audio_anchor_score": candidate["audio_anchor_score"],
                    "artist_query_score": candidate["artist_query_score"],
                    "title_hint_score": candidate["title_hint_score"],
                    "artist_exact_match": candidate["artist_exact_match"],
                    "title_exact_match": candidate["title_exact_match"],
                    "facet_alignment_score": candidate.get("facet_alignment", 0.0),
                    "facet_token_coverage": candidate.get("facet_token_coverage", 0.0),
                    "hard_facet_token_coverage": candidate.get("hard_facet_token_coverage", 0.0),
                    "soft_facet_token_coverage": candidate.get("soft_facet_token_coverage", 0.0),
                    "soft_title_facet_coverage": candidate.get("soft_title_facet_coverage", 0.0),
                    "soft_focus_match": candidate.get("soft_focus_match", 0.0),
                    "hard_focus_match": candidate.get("hard_focus_match", 0.0),
                    "mixed_soft_signal": candidate.get("mixed_soft_signal", 0.0),
                    "mixed_intent_coverage": candidate.get("mixed_intent_coverage", 0.0),
                    "bot_profile": str(candidate["bot_profile_label"] or ""),
                    "bot_profile_score": float(candidate["bot_profile_score"]),
                    "strict_artist_mode": bool(strict_artist_mode),
                    "score": float(final_score),
                }
            )

        if mixed_compositional_facet_query and hard_focus_token:
            ranked.sort(
                key=lambda x: (
                    self._clamp01(x.get("mixed_intent_coverage", 0.0)),
                    self._clamp01(x.get("mixed_soft_signal", 0.0)),
                    x["score"],
                ),
                reverse=True,
            )
        else:
            ranked.sort(key=lambda x: x["score"], reverse=True)
        ranked = self._dedupe_ranked_rows(ranked)
        use_mmr_diversify = bool(getattr(settings, "SEARCH_USE_MMR_DIVERSIFY", True))
        if (
            use_mmr_diversify
            and (facet_heavy_query or like_mode)
            and (not strict_artist_mode)
            and (not mixed_compositional_facet_query)
        ):
            ranked = self._diversify_ranked_rows_mmr(
                ranked_rows=ranked,
                limit=limit,
                offset=offset,
                query_tokens=query_tokens,
            )
        fallback_lexical_used = False
        if not ranked:
            fallback_lexical_used = True
            ranked = self._fallback_lexical_ranked_rows(
                query=query,
                query_tokens=(
                    query_facet_tokens
                    if (facet_heavy_query and query_facet_tokens)
                    else (base_query_tokens if base_query_tokens else query_tokens)
                ),
                strict_artist_targets=strict_artist_targets,
                active_query_bots=active_query_bots,
                bot_profile_weight=bot_profile_weight,
            )
        if strict_title_artist_mode and ranked:
            ranked = self._reorder_title_artist_strict_rows(
                ranked,
                title_hint=title_hint,
            )
        if mixed_compositional_facet_query and ranked and offset == 0:
            if bool(getattr(settings, "SEARCH_MIXED_TOPK_QUOTA_ENABLED", True)):
                quota_k = max(1, int(getattr(settings, "SEARCH_MIXED_TOPK_QUOTA_K", 10)))
                quota_min_hits = max(
                    0,
                    int(getattr(settings, "SEARCH_MIXED_TOPK_QUOTA_MIN_HITS", 4)),
                )
                quota_min_cov = self._clamp01(
                    float(
                        getattr(
                            settings,
                            "SEARCH_MIXED_TOPK_QUOTA_MIN_COVERAGE",
                            getattr(settings, "SEARCH_METRIC_MIXED_INTENT_HIT_THRESHOLD", 0.25),
                        )
                    )
                )
                ranked = self._apply_mixed_intent_topk_quota(
                    ranked,
                    top_k=min(len(ranked), max(quota_k, limit)),
                    min_hits=quota_min_hits,
                    min_coverage=quota_min_cov,
                )
            # After quota repair, re-rank the top block by query-fit quality so
            # mixed-coverage gating doesn't dominate final ordering.
            resort_k = max(
                1,
                int(
                    getattr(
                        settings,
                        "SEARCH_MIXED_QUERYFIT_RESORT_K",
                        max(limit, int(getattr(settings, "SEARCH_MIXED_TOPK_QUOTA_K", 10))),
                    )
                ),
            )
            resort_k = min(len(ranked), resort_k)
            if resort_k > 1:
                mixed_pool = ranked[:resort_k]
                self._attach_ranked_query_metrics(
                    mixed_pool,
                    active_query_bots=active_query_bots,
                    like_mode=like_mode,
                    strict_artist_mode=strict_artist_mode,
                    facet_heavy_query=facet_heavy_query,
                    mixed_compositional_query=mixed_compositional_facet_query,
                    hard_facet_query=bool(expanded_hard_query_facet_tokens),
                    title_artist_query=title_artist_query,
                    fallback_lexical_used=fallback_lexical_used,
                )
                mixed_cov_floor = self._clamp01(
                    float(
                        getattr(
                            settings,
                            "SEARCH_MIXED_QUERYFIT_MIN_COVERAGE",
                            getattr(settings, "SEARCH_METRIC_MIXED_INTENT_HIT_THRESHOLD", 0.25),
                        )
                    )
                )
                mixed_sort_qf_weight = self._clamp01(
                    float(getattr(settings, "SEARCH_MIXED_QUERYFIT_SORT_QUERYFIT_WEIGHT", 0.72))
                )
                mixed_sort_cov_weight = self._clamp01(
                    float(getattr(settings, "SEARCH_MIXED_QUERYFIT_SORT_COVERAGE_WEIGHT", 0.18))
                )
                mixed_sort_hard_facet_weight = self._clamp01(
                    float(getattr(settings, "SEARCH_MIXED_QUERYFIT_SORT_HARD_FACET_WEIGHT", 0.08))
                )
                mixed_sort_hard_focus_weight = self._clamp01(
                    float(getattr(settings, "SEARCH_MIXED_QUERYFIT_SORT_HARD_FOCUS_WEIGHT", 0.02))
                )
                mixed_hard_focus_qf_min = self._clamp01(
                    float(getattr(settings, "SEARCH_MIXED_QUERYFIT_HARD_FOCUS_QF_MIN", 0.50))
                )

                def _mixed_sort_primary(row):
                    """
                    Execute mixed sort primary.
                    
                    This function implements the mixed sort primary step for this module.
                    It is used to keep the broader workflow readable and easier to maintain.
                    """
                    qf = self._clamp01(row.get("query_fit_score", 0.0))
                    mixed_cov = self._clamp01(row.get("mixed_intent_coverage", 0.0))
                    hard_cov = self._clamp01(row.get("hard_facet_token_coverage", 0.0))
                    hard_focus = self._clamp01(row.get("hard_focus_match", 0.0))
                    hard_focus_effective = hard_focus if qf >= mixed_hard_focus_qf_min else 0.0
                    return (
                        mixed_sort_qf_weight * qf
                        + mixed_sort_cov_weight * mixed_cov
                        + mixed_sort_hard_facet_weight * hard_cov
                        + mixed_sort_hard_focus_weight * hard_focus_effective
                    )

                mixed_pool.sort(
                    key=lambda row: (
                        _mixed_sort_primary(row),
                        self._clamp01(row.get("query_fit_score", 0.0)),
                        self._clamp01(row.get("mixed_intent_coverage", 0.0)),
                        self._clamp01(row.get("hard_facet_token_coverage", 0.0)),
                        self._clamp01(row.get("hard_focus_match", 0.0)),
                        1.0
                        if self._clamp01(row.get("mixed_intent_coverage", 0.0)) >= mixed_cov_floor
                        else 0.0,
                        self._clamp01(row.get("mixed_soft_signal", 0.0)),
                        float(self._safe_float(row.get("score", 0.0), 0.0)),
                    ),
                    reverse=True,
                )
                ranked[:resort_k] = mixed_pool
        if ranked:
            for row in ranked:
                if not isinstance(row, dict):
                    continue
                row["match_type"] = self._resolve_match_type(
                    row,
                    like_mode=like_mode,
                    strict_title_artist_mode=strict_title_artist_mode,
                )
        final_results = ranked[offset: offset + limit]
        search_metrics = self._attach_ranked_query_metrics(
            final_results,
            active_query_bots=active_query_bots,
            like_mode=like_mode,
            strict_artist_mode=strict_artist_mode,
            facet_heavy_query=facet_heavy_query,
            mixed_compositional_query=mixed_compositional_facet_query,
            hard_facet_query=bool(expanded_hard_query_facet_tokens),
            title_artist_query=title_artist_query,
            fallback_lexical_used=fallback_lexical_used,
        )
        query_metrics = dict(search_metrics.get("result_quality", {}) or {})
        payload = {
            "query": query,
            "results": final_results,
            "query_metrics": query_metrics,
            "search_metrics": search_metrics,
            "fallback_lexical_used": bool(fallback_lexical_used),
        }
        self._cache_set(
            self._result_cache,
            result_cache_key,
            {
                "results": [dict(row) for row in final_results],
                "query_metrics": dict(query_metrics),
                "search_metrics": dict(search_metrics),
                "fallback_lexical_used": bool(fallback_lexical_used),
            },
            self._max_result_cache,
        )
        return payload