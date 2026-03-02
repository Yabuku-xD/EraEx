import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from config import settings
from src.core.text_embeddings import embedding_handler
from src.core.media_metadata import (
    as_optional_bool,
    build_track_embedding_text_context_first,
    is_youtube_id,
    thumb_from_video_id,
    thumbnail_candidates,
)


class FastDPP:
    # Initialize class state.
    def __init__(self, kernel_matrix=None):
        """
        Initialize the instance state.
        
        This method implements the init step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        self.kernel = kernel_matrix

    # Handle decompose kernel.
    def decompose_kernel(self, similarity_matrix, quality_scores):
        """
        Execute decompose kernel.
        
        This method implements the decompose kernel step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        quality = np.array(quality_scores).reshape(-1, 1)
        return similarity_matrix * np.dot(quality, quality.T)

    # Handle greedy selection.
    def greedy_selection(self, kernel_l, k=10):
        """
        Execute greedy selection.
        
        This method implements the greedy selection step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        n = kernel_l.shape[0]
        if k > n:
            return list(range(n))
        cis = np.zeros((k, n))
        di2 = np.diag(kernel_l).copy()
        selected_items = []
        for j in range(k):
            best_i = np.argmax(di2)
            if di2[best_i] < 1e-10:
                break
            selected_items.append(best_i)
            sqrt_val = math.sqrt(di2[best_i])
            cis[j, best_i] = sqrt_val
            row = kernel_l[best_i, :]
            dot_prod = np.dot(cis[:j, best_i], cis[:j, :]) if j > 0 else np.zeros(n)
            cis[j, :] = (row - dot_prod) / sqrt_val
            di2 -= cis[j, :] ** 2
            di2[best_i] = -np.inf
        return selected_items


class SimilarityManager:
    # Initialize class state.
    def __init__(self, metadata_path=None, metadata=None):
        """
        Initialize the instance state.
        
        This method implements the init step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        self.metadata = metadata if metadata is not None else {}
        if not self.metadata:
            self.load_metadata(metadata_path)
        self.cached_embeddings = {}
        self.cached_audio_features = {}
        self.audio_model = None
        self.steering = None
        self._load_cached_features()

    # Load cached features.
    def _load_cached_features(self):
        """
        Load cached features.
        
        This method implements the load cached features step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        path = settings.CACHE_DIR / "audio_features.json"
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for key, value in data.items():
                    if isinstance(value, dict):
                        arr = np.array(
                            [
                                float(value.get("brightness", 0.0)),
                                float(value.get("energy", 0.0)),
                                float(value.get("mood", 0.0)),
                                float(value.get("tempo", 0.0)),
                            ],
                            dtype=np.float32,
                        )
                    else:
                        arr = np.array(value, dtype=np.float32)
                        if len(arr) < 4:
                            arr = np.pad(arr, (0, 4 - len(arr)))
                        arr = arr[:4]
                    self.cached_audio_features[key] = arr
            except Exception:
                self.cached_audio_features = {}

    # Internal helper to ensure audio steering.
    def _ensure_audio_steering(self):
        """
        Execute ensure audio steering.
        
        This method implements the ensure audio steering step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if self.steering is not None:
            return True
        try:
            from src.core.audio_processing import ActivationSteering, AudioTransformer
        except Exception:
            return False
        self.audio_model = AudioTransformer()
        self.steering = ActivationSteering(self.audio_model)
        self.steering.load_dummy_vectors()
        return True

    # Load metadata.
    def load_metadata(self, path=None):
        """
        Load metadata.
        
        This method implements the load metadata step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        path = path or settings.INDEX_DIR / "metadata.json"
        if Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            return
        if settings.MOCK_MODE:
            self.metadata = {
                str(i): {
                    "title": f"Mock Song {i}",
                    "artist_name": f"Artist {i % 5}",
                    "year": 2015 + i % 5,
                    "deezer_tags": ["pop", "rock"] if i % 2 == 0 else ["jazz", "soul"],
                    "deezer_playcount": 1000 * (50 - i),
                    "views": 2000 * (50 - i),
                }
                for i in range(50)
            }
            return
        self.metadata = {}

    # Get track info.
    def get_track_info(self, track_id):
        """
        Get track info.
        
        This method implements the get track info step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        return self.metadata.get(str(track_id), {})

    # Internal helper to parse list.
    @staticmethod
    def _parse_list(value):
        """
        Parse list.
        
        This method implements the parse list step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return []
            try:
                parsed = json.loads(value.replace("'", '"'))
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        return []

    # Get text for embedding.
    def get_text_for_embedding(self, track_id):
        """
        Get text for embedding.
        
        This method implements the get text for embedding step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        info = self.get_track_info(track_id)
        return build_track_embedding_text_context_first(info)

    # Get vector.
    def _get_vector(self, track_id, text):
        """
        Get vector.
        
        This method implements the get vector step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if track_id not in self.cached_embeddings:
            self.cached_embeddings[track_id] = embedding_handler.encode([text])[0]
        return self.cached_embeddings[track_id]

    # Get embedding similarity.
    def get_embedding_similarity(self, id1, id2):
        """
        Get embedding similarity.
        
        This method implements the get embedding similarity step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        txt1 = self.get_text_for_embedding(id1)
        txt2 = self.get_text_for_embedding(id2)
        vec1 = self._get_vector(id1, txt1)
        vec2 = self._get_vector(id2, txt2)
        return float(np.dot(vec1, vec2))

    # Handle precompute embeddings.
    def precompute_embeddings(self, track_ids):
        """
        Execute precompute embeddings.
        
        This method implements the precompute embeddings step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        texts = [self.get_text_for_embedding(track_id) for track_id in track_ids]
        missing_indices = [i for i, tid in enumerate(track_ids) if tid not in self.cached_embeddings]
        if missing_indices:
            vectors = embedding_handler.encode([texts[i] for i in missing_indices], batch_size=32)
            for i, idx in enumerate(missing_indices):
                self.cached_embeddings[track_ids[idx]] = vectors[i]
        missing_audio = [track_id for track_id in track_ids if track_id not in self.cached_audio_features]
        if missing_audio:
            # Fast path: use audio feature scores already present in metadata (tempo/energy/brightness/mood).
            # Avoid expensive local audio file extraction during runtime recommendation requests.
            unresolved = []
            for track_id in missing_audio:
                info = self.get_track_info(track_id)
                try:
                    vector = np.array(
                        [
                            float(info.get("brightness", 0.0) or 0.0),
                            float(info.get("energy", 0.0) or 0.0),
                            float(info.get("mood", 0.0) or 0.0),
                            float(info.get("tempo", 0.0) or 0.0),
                        ],
                        dtype=np.float32,
                    )
                except Exception:
                    vector = np.zeros(4, dtype=np.float32)
                if float(np.abs(vector).sum()) > 0.0:
                    norm = np.linalg.norm(vector)
                    if norm > 0:
                        vector /= norm
                    self.cached_audio_features[track_id] = vector
                else:
                    unresolved.append(track_id)

            if unresolved:
                # Runtime fallback should stay cheap/reliable. Use zeros instead of trying mp3 extraction.
                # (Local extraction can be done offline and cached to audio_features.json if needed.)
                for track_id in unresolved:
                    self.cached_audio_features[track_id] = np.zeros(4, dtype=np.float32)

    # Get mood similarity.
    def get_mood_similarity(self, id1, id2):
        """
        Get mood similarity.
        
        This method implements the get mood similarity step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        t1 = self._parse_list(self.get_track_info(id1).get("deezer_tags", []))
        t2 = self._parse_list(self.get_track_info(id2).get("deezer_tags", []))
        s1 = {value.lower() for value in t1}
        s2 = {value.lower() for value in t2}
        overlap = 0.0 if not s1 or not s2 else len(s1.intersection(s2)) / min(len(s1), len(s2))
        text_sim = self.get_embedding_similarity(id1, id2)
        f1 = self.cached_audio_features.get(id1, np.zeros(4))
        f2 = self.cached_audio_features.get(id2, np.zeros(4))
        audio_sim = float(np.dot(f1, f2))
        return 0.4 * text_sim + 0.4 * audio_sim + 0.2 * overlap

    # Get era similarity.
    def get_era_similarity(self, id1, id2, sigma=3.0):
        """
        Get era similarity.
        
        This method implements the get era similarity step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        y1 = self.get_track_info(id1).get("year", 0)
        y2 = self.get_track_info(id2).get("year", 0)
        try:
            y1, y2 = int(y1), int(y2)
        except Exception:
            return 0.0
        if y1 == 0 or y2 == 0:
            return 0.0
        return math.exp(-abs(y1 - y2) ** 2 / (2 * sigma**2))

    # Get style similarity.
    def get_style_similarity(self, id1, id2):
        """
        Get style similarity.
        
        This method implements the get style similarity step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        info1 = self.get_track_info(id1)
        info2 = self.get_track_info(id2)
        if info1.get("artist_name", "").lower() == info2.get("artist_name", "").lower():
            return 1.0
        g1 = self._parse_list(info1.get("genres", []))
        g2 = self._parse_list(info2.get("genres", []))
        if not g1 and not g2:
            g1 = self._parse_list(info1.get("deezer_tags", []))
            g2 = self._parse_list(info2.get("deezer_tags", []))
        s1 = {genre.lower() for genre in g1}
        s2 = {genre.lower() for genre in g2}
        if not s1 or not s2:
            return 0.0
        return len(s1.intersection(s2)) / min(len(s1), len(s2)) * 0.5

    # Compute kernel.
    def compute_kernel(self, candidate_ids, weights=None):
        """
        Compute kernel.
        
        This method implements the compute kernel step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if weights is None:
            weights = {"mood": 1.0, "era": 1.0, "style": 1.0}
        n = len(candidate_ids)
        self.precompute_embeddings(candidate_ids)
        kernel = np.zeros((n, n))
        denom = sum(weights.values()) or 1.0
        for i in range(n):
            for j in range(i, n):
                id1, id2 = candidate_ids[i], candidate_ids[j]
                if i == j:
                    sim = 1.0
                else:
                    sim = (
                        weights["mood"] * self.get_mood_similarity(id1, id2)
                        + weights["era"] * self.get_era_similarity(id1, id2)
                        + weights["style"] * self.get_style_similarity(id1, id2)
                    ) / denom
                kernel[i, j] = sim
                kernel[j, i] = sim
        return kernel


class ColdStartHandler:
    LIVE_BOT_PROFILES = {
        "rnb_bot": {
            "label": "R&B Bot",
            "tokens": {
                "rnb",
                "soul",
                "neosoul",
                "smooth",
                "romantic",
                "late",
                "night",
                "slow",
                "jam",
            },
        },
        "hiphop_bot": {
            "label": "Hip-Hop Bot",
            "tokens": {
                "hip",
                "hop",
                "rap",
                "trap",
                "drill",
                "808",
                "bars",
                "freestyle",
            },
        },
        "rock_bot": {
            "label": "Rock Bot",
            "tokens": {
                "rock",
                "metal",
                "punk",
                "guitar",
                "alt",
                "indie",
                "grunge",
                "band",
            },
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
        },
    }

    # Initialize class state.
    def __init__(self, metadata_path=None, metadata=None):
        """
        Initialize the instance state.
        
        This method implements the init step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        self.sim_manager = SimilarityManager(metadata_path, metadata=metadata)
        self.dpp = FastDPP()
        self.metadata = self.sim_manager.metadata
        self.popularity_ranking = []
        self.track_quality = {}
        self.track_popularity_percentile = {}
        self.track_quality_min = 0.0
        self.track_quality_max = 1.0
        self.video_id_to_track_id = {}
        self.artist_index = defaultdict(list)
        self.tag_index = defaultdict(list)
        self.year_index = defaultdict(list)
        self.last_quota_stats = {}
        self._build_popularity()
        self._build_id_lookup()
        self._build_retrieval_indexes()

    # Internal helper to fallback cover.
    @staticmethod
    def _fallback_cover(track_id, meta):
        """
        Execute fallback cover.
        
        This method implements the fallback cover step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        cover = (
            meta.get("cover_url")
            or meta.get("thumbnail")
            or meta.get("album_cover")
        )
        if cover:
            return cover
        meta_video_id = meta.get("video_id") or ""
        thumb = thumb_from_video_id(meta_video_id)
        if thumb:
            return thumb
        return ""

    # Prefer canonical recommendation track_id for playback when it is already a YouTube ID.
    @staticmethod
    def _playback_video_id(track_id, meta):
        """
        Execute playback video id.
        
        This method implements the playback video id step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        sid = str(track_id or "").strip()
        meta_video_id = str((meta or {}).get("video_id", "") or "").strip()
        if is_youtube_id(sid):
            return sid
        if is_youtube_id(meta_video_id):
            return meta_video_id
        return meta_video_id

    # Build cover candidates for frontend fallback loading.
    @staticmethod
    def _cover_candidates(track_id, meta):
        """
        Execute cover candidates.
        
        This method implements the cover candidates step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        candidates = []
        raw_candidates = meta.get("thumbnail_candidates") or meta.get("cover_candidates") or []
        if isinstance(raw_candidates, list):
            for url in raw_candidates:
                if url:
                    candidates.append(url)
        for key in ("cover_url", "thumbnail", "album_cover"):
            url = meta.get(key)
            if url:
                candidates.append(url)
        video_id = str(meta.get("video_id") or "")
        candidates.extend(thumbnail_candidates(video_id, None))

        deduped = []
        seen = set()
        for url in candidates:
            if not url:
                continue
            if url in seen:
                continue
            seen.add(url)
            deduped.append(url)
        return deduped

    # DPP may stop early when residual gain collapses; backfill with best remaining scores.
    @staticmethod
    def _fill_selection_count(selected_indices, scores, k):
        """
        Execute fill selection count.
        
        This method implements the fill selection count step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        target = max(0, int(k or 0))
        picked = []
        seen = set()
        for raw_idx in list(selected_indices or []):
            try:
                idx = int(raw_idx)
            except Exception:
                continue
            if idx < 0 or idx in seen:
                continue
            seen.add(idx)
            picked.append(idx)
            if len(picked) >= target:
                return picked[:target]
        ranked = sorted(
            range(len(scores or [])),
            key=lambda i: float(scores[i]),
            reverse=True,
        )
        for idx in ranked:
            if idx in seen:
                continue
            seen.add(idx)
            picked.append(idx)
            if len(picked) >= target:
                break
        return picked[:target]

    # Fast top-k selection with lightweight artist diversity. Used when DPP is disabled.
    def _fast_topk_diverse_indices(self, candidate_ids, scores, k):
        """
        Execute fast topk diverse indices.
        
        This method implements the fast topk diverse indices step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        target = max(1, int(k or 1))
        if not candidate_ids:
            return []
        ranked = sorted(
            range(len(scores or [])),
            key=lambda i: float(scores[i]),
            reverse=True,
        )
        max_per_artist = int(getattr(settings, "RECO_FAST_MAX_PER_ARTIST", 2) or 2)
        selected = []
        artist_counts = defaultdict(int)
        used = set()

        for idx in ranked:
            if idx in used:
                continue
            used.add(idx)
            sid = str(candidate_ids[idx] or "").strip()
            meta = self.sim_manager.metadata.get(sid, {}) if sid else {}
            artist = str(meta.get("artist_name", "") or "").strip().lower()
            if artist and artist_counts[artist] >= max_per_artist:
                continue
            selected.append(idx)
            if artist:
                artist_counts[artist] += 1
            if len(selected) >= target:
                break

        if len(selected) < target:
            for idx in ranked:
                if idx in selected:
                    continue
                selected.append(idx)
                if len(selected) >= target:
                    break
        return selected[:target]

    # Build popularity.
    def _build_popularity(self):
        """
        Build popularity.
        
        This method implements the build popularity step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        scored = []
        for song_key, meta in self.metadata.items():
            views = float(meta.get("views", 0) or 0)
            playcount = float(meta.get("deezer_playcount", 0) or 0)
            rank = float(meta.get("deezer_rank", meta.get("rank", 0)) or 0)
            if views > 0 or playcount > 0:
                score = float(np.log1p(views * 0.7 + playcount * 0.3))
            elif rank > 0:
                score = float(1.0 / np.log1p(rank))
            else:
                score = 0.0
            scored.append((song_key, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        self.popularity_ranking = scored
        self.track_quality = {str(sid): float(score) for sid, score in scored}
        if scored:
            quality_values = [float(score) for _sid, score in scored]
            self.track_quality_min = float(min(quality_values))
            self.track_quality_max = float(max(quality_values))
        else:
            self.track_quality_min = 0.0
            self.track_quality_max = 1.0
        self.track_popularity_percentile = {}
        if scored:
            denom = float(max(1, len(scored) - 1))
            for idx, (sid, _score) in enumerate(scored):
                self.track_popularity_percentile[str(sid)] = float(idx) / denom

    # Build canonical lookup so profile history can use either track ids or video ids.
    def _build_id_lookup(self):
        """
        Build id lookup.
        
        This method implements the build id lookup step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        mapping = {}
        for sid, _score in self.popularity_ranking:
            key = str(sid or "").strip()
            if key:
                mapping.setdefault(key, key)
            meta = self.sim_manager.get_track_info(key) or {}
            vid = str(meta.get("video_id", "") or "").strip()
            if vid:
                mapping.setdefault(vid, key)
        self.video_id_to_track_id = mapping

    def _canonical_song_id(self, song_id):
        """
        Execute canonical song id.
        
        This method implements the canonical song id step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        sid = str(song_id or "").strip()
        if not sid:
            return ""
        if sid in self.metadata:
            return sid
        mapped = self.video_id_to_track_id.get(sid)
        if mapped:
            return str(mapped)
        return sid

    @staticmethod
    def _track_year(meta):
        """
        Execute track year.
        
        This method implements the track year step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        try:
            year = int(meta.get("year", 0) or 0)
        except Exception:
            year = 0
        return year if year > 0 else 0

    def _track_tokens(self, meta):
        """
        Execute track tokens.
        
        This method implements the track tokens step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        values = []
        for key in ("genres", "deezer_tags"):
            parsed = self.sim_manager._parse_list((meta or {}).get(key, []))
            if parsed:
                values.extend(parsed)
        out = []
        seen = set()
        for raw in values:
            token = str(raw or "").strip().lower()
            if not token or token in seen:
                continue
            seen.add(token)
            out.append(token)
        return out

    # Build lightweight retrieval indexes ordered by popularity.
    def _build_retrieval_indexes(self):
        """
        Build retrieval indexes.
        
        This method implements the build retrieval indexes step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        self.artist_index = defaultdict(list)
        self.tag_index = defaultdict(list)
        self.year_index = defaultdict(list)
        for sid, _score in self.popularity_ranking:
            meta = self.sim_manager.get_track_info(sid) or {}
            artist = str(meta.get("artist_name", "") or "").strip().lower()
            if artist:
                self.artist_index[artist].append(sid)
            year = self._track_year(meta)
            if year > 0:
                self.year_index[year].append(sid)
            for token in self._track_tokens(meta):
                self.tag_index[token].append(sid)

    # Get candidate pool.
    def get_candidate_pool(self, n=50, exclude_ids=None):
        """
        Get candidate pool.
        
        This method implements the get candidate pool step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        candidates = []
        exclude = set(exclude_ids or [])
        for sid, score in self.popularity_ranking:
            if sid not in exclude:
                candidates.append((sid, score))
                if len(candidates) >= n:
                    break
        return candidates

    @staticmethod
    def _clamp(value, low, high):
        """
        Execute clamp.
        
        This method implements the clamp step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        return max(low, min(high, float(value)))

    @staticmethod
    def _normalize_text_token(value):
        """
        Normalize text token.
        
        This method implements the normalize text token step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        return " ".join(re.findall("[a-z0-9]+", str(value or "").lower()))

    def _popularity_percentile(self, track_id):
        """
        Execute popularity percentile.
        
        This method implements the popularity percentile step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        sid = str(track_id or "").strip()
        if not sid:
            return 0.5
        return float(self.track_popularity_percentile.get(sid, 0.5))

    def _quality_score_normalized(self, track_id):
        """
        Execute quality score normalized.
        
        This method implements the quality score normalized step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        sid = str(track_id or "").strip()
        if not sid:
            return 0.0
        score = float(self.track_quality.get(sid, 0.0))
        low = float(self.track_quality_min)
        high = float(self.track_quality_max)
        denom = max(1e-9, high - low)
        return self._clamp((score - low) / denom, 0.0, 1.0)

    def _long_tail_strength(self, track_id):
        """
        Execute long tail strength.
        
        This method implements the long tail strength step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        pct = self._popularity_percentile(track_id)
        # Use a smoother ramp so long-tail does not collapse to all-zeros unless
        # recommendations are truly chart-head only.
        return self._clamp((pct - 0.18) / 0.82, 0.0, 1.0)

    def _mainstream_strength(self, track_id):
        """
        Execute mainstream strength.
        
        This method implements the mainstream strength step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        pct = self._popularity_percentile(track_id)
        return self._clamp((0.24 - pct) / 0.24, 0.0, 1.0)

    @staticmethod
    def _simple_tokens(value):
        """
        Execute simple tokens.
        
        This method implements the simple tokens step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        return set(re.findall("[a-z0-9]+", str(value or "").lower()))

    def _meta_bot_tokens(self, meta):
        """
        Execute meta bot tokens.
        
        This method implements the meta bot tokens step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        info = meta if isinstance(meta, dict) else {}
        tokens = set()
        for key in ("title", "artist_name", "description"):
            tokens.update(self._simple_tokens(info.get(key, "")))
        for token in self._track_tokens(info):
            tokens.update(self._simple_tokens(token))
        return tokens

    @staticmethod
    def _bot_overlap_score(candidate_tokens, profile_tokens):
        """
        Execute bot overlap score.
        
        This method implements the bot overlap score step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if not candidate_tokens or not profile_tokens:
            return 0.0
        inter = len(candidate_tokens.intersection(profile_tokens))
        if inter <= 0:
            return 0.0
        denom = max(1.0, min(float(len(candidate_tokens)), float(len(profile_tokens))))
        return float(inter) / denom

    # Infer the dominant taste-bots from user behavior history (fast token routing).
    def _resolve_active_profile_bots(self, liked_history, playlist_history, played_history):
        """
        Resolve active profile bots.
        
        This method implements the resolve active profile bots step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        seed_ids = []
        for sid in list(liked_history or [])[:12]:
            seed_ids.append(self._canonical_song_id(sid))
        for sid in list(playlist_history or [])[:12]:
            seed_ids.append(self._canonical_song_id(sid))
        for sid in list(played_history or [])[:14]:
            seed_ids.append(self._canonical_song_id(sid))
        seed_ids = [sid for sid in seed_ids if sid]
        if not seed_ids:
            return []

        seed_counter = Counter()
        for sid in seed_ids:
            meta = self.sim_manager.get_track_info(sid) or {}
            for token in self._meta_bot_tokens(meta):
                seed_counter[token] += 1
        if not seed_counter:
            return []

        max_count = max(seed_counter.values()) if seed_counter else 1
        min_conf = float(getattr(settings, "RECO_BOT_MIN_CONFIDENCE", 0.24) or 0.24)
        min_hits = int(getattr(settings, "RECO_BOT_MIN_TOKEN_HITS", 2) or 2)
        active = []
        for bot_key, profile in self.LIVE_BOT_PROFILES.items():
            profile_tokens = set(profile.get("tokens", set()))
            if not profile_tokens:
                continue
            weighted_hits = 0.0
            matched = 0
            for token in profile_tokens:
                count = seed_counter.get(token, 0)
                if count <= 0:
                    continue
                matched += 1
                weighted_hits += float(count) / float(max_count)
            if matched <= 0 or matched < min_hits:
                continue
            coverage = float(matched) / float(len(profile_tokens))
            density = weighted_hits / float(len(profile_tokens))
            confidence = self._clamp(0.65 * coverage + 0.35 * density, 0.0, 1.0)
            if confidence < min_conf:
                continue
            active.append(
                {
                    "bot": bot_key,
                    "label": str(profile.get("label", bot_key)),
                    "confidence": float(confidence),
                    "tokens": profile_tokens,
                }
            )
        active.sort(key=lambda row: float(row.get("confidence", 0.0)), reverse=True)
        return active[:2]

    # Compute candidate compatibility with active user bots.
    def _candidate_bot_profile_match(self, track_id, active_bots):
        """
        Execute candidate bot profile match.
        
        This method implements the candidate bot profile match step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        sid = str(track_id or "").strip()
        if not sid or not active_bots:
            return "", 0.0
        meta = self.sim_manager.get_track_info(sid) or {}
        candidate_tokens = self._meta_bot_tokens(meta)
        if not candidate_tokens:
            return "", 0.0

        best_name = ""
        best_score = 0.0
        min_conf = float(getattr(settings, "RECO_BOT_MIN_CONFIDENCE", 0.24) or 0.24)
        min_overlap = float(getattr(settings, "RECO_BOT_MIN_OVERLAP", 0.12) or 0.12)
        for bot in active_bots:
            confidence = float(bot.get("confidence", 0.0) or 0.0)
            if confidence < min_conf:
                continue
            profile_tokens = set(bot.get("tokens", set()))
            overlap = self._bot_overlap_score(candidate_tokens, profile_tokens)
            if overlap < min_overlap:
                continue
            score = confidence * overlap
            if score > best_score:
                best_score = score
                best_name = str(bot.get("label", bot.get("bot", "")))
        return best_name, self._clamp(best_score, 0.0, 1.0)

    def _seed_artist_set(self, liked_history, playlist_history, played_history):
        """
        Execute seed artist set.
        
        This method implements the seed artist set step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        seed_ids = []
        for sid in list(liked_history or [])[:20]:
            seed_ids.append(self._canonical_song_id(sid))
        for sid in list(playlist_history or [])[:20]:
            seed_ids.append(self._canonical_song_id(sid))
        for sid in list(played_history or [])[:18]:
            seed_ids.append(self._canonical_song_id(sid))
        artists = set()
        for sid in seed_ids:
            if not sid:
                continue
            meta = self.sim_manager.get_track_info(sid) or {}
            artist = str(meta.get("artist_name", "") or "").strip().lower()
            if artist:
                artists.add(artist)
        return artists

    # Seed-affinity artists = exact seed artists + artists from seed-neighbor tracks.
    # This is used for quota logic because it is more realistic than exact-artist-only,
    # while still grounded in the user's behavior neighborhood.
    def _seed_affinity_artist_set(self, liked_history, playlist_history, played_history):
        """
        Execute seed affinity artist set.
        
        This method implements the seed affinity artist set step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        seed_ids = []
        seen = set()
        for sid in list(liked_history or [])[:20]:
            cid = self._canonical_song_id(sid)
            if cid and cid not in seen:
                seen.add(cid)
                seed_ids.append(cid)
        for sid in list(playlist_history or [])[:20]:
            cid = self._canonical_song_id(sid)
            if cid and cid not in seen:
                seen.add(cid)
                seed_ids.append(cid)
        for sid in list(played_history or [])[:18]:
            cid = self._canonical_song_id(sid)
            if cid and cid not in seen:
                seen.add(cid)
                seed_ids.append(cid)
        if not seed_ids:
            return set()

        seed_artists = set()
        for sid in seed_ids:
            meta = self.sim_manager.get_track_info(sid) or {}
            artist = str(meta.get("artist_name", "") or "").strip().lower()
            if artist:
                seed_artists.add(artist)

        # Add neighbor artists from a bounded seed neighborhood.
        neighbor_artist_cap = int(
            getattr(settings, "RECO_SEED_NEIGHBOR_ARTIST_CAP", 24) or 24
        )
        per_seed_neighbor_cap = int(
            getattr(settings, "RECO_SEED_NEIGHBOR_PER_SEED", 20) or 20
        )
        added = 0
        affinity_artists = set(seed_artists)
        for seed_id in seed_ids[:12]:
            neighbors = self._collect_seed_neighbors(seed_id)[:per_seed_neighbor_cap]
            for neighbor_id in neighbors:
                meta = self.sim_manager.get_track_info(neighbor_id) or {}
                artist = str(meta.get("artist_name", "") or "").strip().lower()
                if not artist or artist in affinity_artists:
                    continue
                affinity_artists.add(artist)
                added += 1
                if added >= neighbor_artist_cap:
                    return affinity_artists
        return affinity_artists

    def _dedupe_result_rows_by_title_artist(self, rows, limit=None):
        """
        Deduplicate result rows by title artist.
        
        This method implements the dedupe result rows by title artist step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        deduped = []
        seen_keys = set()
        seen_ids = set()
        target = int(limit) if limit is not None else None
        for row in list(rows or []):
            if not isinstance(row, dict):
                continue
            sid = str(row.get("id") or row.get("track_id") or "").strip()
            if sid and sid in seen_ids:
                continue
            title_key = self._normalize_text_token(row.get("title", ""))
            artist_key = self._normalize_text_token(row.get("artist", ""))
            ta_key = (title_key, artist_key)
            if title_key and artist_key and ta_key in seen_keys:
                continue
            if sid:
                seen_ids.add(sid)
            if title_key and artist_key:
                seen_keys.add(ta_key)
            deduped.append(row)
            if target is not None and len(deduped) >= max(1, target):
                break
        return deduped

    def _track_title_artist_key(self, track_id):
        """
        Execute track title artist key.
        
        This method implements the track title artist key step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        sid = self._canonical_song_id(track_id)
        meta = self.sim_manager.get_track_info(sid) or {}
        title = self._normalize_text_token(meta.get("title", ""))
        artist = self._normalize_text_token(meta.get("artist_name", ""))
        if not title and not artist:
            return ("", "")
        return (title, artist)

    # Build final recommendation ordering with explicit quota guarantees.
    def _apply_output_quotas(
        self,
        ordered_indices,
        candidate_ids,
        adjusted_scores,
        liked_history,
        playlist_history,
        played_history,
        k,
    ):
        """
        Apply output quotas.
        
        This method implements the apply output quotas step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        target = max(1, int(k or 1))
        if not ordered_indices:
            return []

        explicit_count = len(liked_history or []) + len(playlist_history or [])
        implicit_count = len(played_history or [])
        explicit_ratio = float(explicit_count) / float(max(1, explicit_count + implicit_count))
        seed_artist_set = self._seed_artist_set(liked_history, playlist_history, played_history)
        seed_affinity_artist_set = self._seed_affinity_artist_set(
            liked_history, playlist_history, played_history
        )
        long_tail_threshold = float(
            getattr(settings, "RECO_LONG_TAIL_BUCKET_THRESHOLD", 0.25) or 0.25
        )
        hard_top20 = bool(target >= 20 and explicit_count > 0)

        if explicit_count >= 3:
            seed_quota = int(round(target * self._clamp(0.34 + 0.18 * explicit_ratio, 0.32, 0.56)))
            long_tail_quota = int(round(target * self._clamp(0.20 + 0.10 * explicit_ratio, 0.18, 0.36)))
            explore_quota = int(round(target * 0.16))
        elif explicit_count > 0:
            seed_quota = int(round(target * 0.24))
            long_tail_quota = int(round(target * 0.16))
            explore_quota = int(round(target * 0.22))
        else:
            seed_quota = 0
            long_tail_quota = int(round(target * 0.20))
            explore_quota = int(round(target * 0.28))

        if hard_top20:
            # User-requested hard constraints for top-20 pages.
            seed_quota = max(seed_quota, 6)
            long_tail_quota = max(long_tail_quota, 4)
            if seed_quota + long_tail_quota > target:
                # Preserve both hard minimums and clip if target is unexpectedly small.
                long_tail_quota = max(0, target - seed_quota)
            explore_quota = max(0, target - seed_quota - long_tail_quota)
        desired_seed_quota = int(max(0, seed_quota))
        desired_long_tail_quota = int(max(0, long_tail_quota))

        quota_total = seed_quota + long_tail_quota + explore_quota
        if quota_total > target:
            overflow = quota_total - target
            for key in ("explore", "long_tail", "seed"):
                if overflow <= 0:
                    break
                if key == "explore" and explore_quota > 0:
                    cut = min(overflow, explore_quota)
                    explore_quota -= cut
                    overflow -= cut
                elif key == "long_tail" and long_tail_quota > 0:
                    cut = min(overflow, long_tail_quota)
                    long_tail_quota -= cut
                    overflow -= cut
                elif key == "seed" and seed_quota > 0:
                    cut = min(overflow, seed_quota)
                    seed_quota -= cut
                    overflow -= cut

        max_per_artist = 1 if hard_top20 else int(getattr(settings, "RECO_FINAL_MAX_PER_ARTIST", 2) or 2)
        if hard_top20:
            # Cap hard targets by what is actually feasible under artist-cap constraints.
            available_seed_artists = set()
            available_long_tail_artists = set()
            for idx in list(ordered_indices or []):
                sid = str(candidate_ids[idx] or "").strip()
                if not sid:
                    continue
                meta = self.sim_manager.get_track_info(sid) or {}
                artist = str(meta.get("artist_name", "") or "").strip().lower()
                if not artist:
                    continue
                if artist in seed_affinity_artist_set:
                    available_seed_artists.add(artist)
                if float(self._long_tail_strength(sid)) > 0.0:
                    available_long_tail_artists.add(artist)
            seed_quota = min(seed_quota, len(available_seed_artists), target)
            long_tail_quota = min(long_tail_quota, len(available_long_tail_artists), max(0, target - seed_quota))
            explore_quota = max(0, target - seed_quota - long_tail_quota)
        effective_seed_quota = int(max(0, seed_quota))
        effective_long_tail_quota = int(max(0, long_tail_quota))
        selected = []
        used_idx = set()
        used_title_artist = set()
        artist_counts = defaultdict(int)

        def _artist_for_idx(idx):
            """
            Execute artist for idx.
            
            This function implements the artist for idx step for this module.
            It is used to keep the broader workflow readable and easier to maintain.
            """
            sid = str(candidate_ids[idx] or "").strip()
            meta = self.sim_manager.get_track_info(sid) or {}
            return str(meta.get("artist_name", "") or "").strip().lower()

        def _can_take(idx, enforce_artist_cap=True):
            """
            Return whether take is allowed.
            
            This function implements the can take step for this module.
            It is used to keep the broader workflow readable and easier to maintain.
            """
            if idx in used_idx:
                return False
            sid = str(candidate_ids[idx] or "").strip()
            if not sid:
                return False
            ta_key = self._track_title_artist_key(sid)
            if ta_key != ("", "") and ta_key in used_title_artist:
                return False
            artist = _artist_for_idx(idx)
            if enforce_artist_cap and artist and artist_counts[artist] >= max_per_artist:
                return False
            return True

        def _take_idx(idx):
            """
            Execute take idx.
            
            This function implements the take idx step for this module.
            It is used to keep the broader workflow readable and easier to maintain.
            """
            sid = str(candidate_ids[idx] or "").strip()
            ta_key = self._track_title_artist_key(sid)
            artist = _artist_for_idx(idx)
            used_idx.add(idx)
            if ta_key != ("", ""):
                used_title_artist.add(ta_key)
            if artist:
                artist_counts[artist] += 1
            selected.append(idx)

        def _drop_idx(idx):
            """
            Execute drop idx.
            
            This function implements the drop idx step for this module.
            It is used to keep the broader workflow readable and easier to maintain.
            """
            if idx not in used_idx:
                return
            sid = str(candidate_ids[idx] or "").strip()
            ta_key = self._track_title_artist_key(sid)
            artist = _artist_for_idx(idx)
            used_idx.discard(idx)
            if ta_key != ("", ""):
                used_title_artist.discard(ta_key)
            if artist:
                next_count = int(artist_counts.get(artist, 0)) - 1
                if next_count > 0:
                    artist_counts[artist] = next_count
                elif artist in artist_counts:
                    del artist_counts[artist]
            try:
                selected.remove(idx)
            except ValueError:
                pass

        def _is_seed_idx(idx):
            """
            Return whether seed idx.
            
            This function implements the is seed idx step for this module.
            It is used to keep the broader workflow readable and easier to maintain.
            """
            return _artist_for_idx(idx) in seed_affinity_artist_set

        def _is_long_tail_idx(idx):
            """
            Return whether long tail idx.
            
            This function implements the is long tail idx step for this module.
            It is used to keep the broader workflow readable and easier to maintain.
            """
            sid = str(candidate_ids[idx] or "").strip()
            if not sid:
                return False
            return float(self._long_tail_strength(sid)) > 0.0

        def _take_bucket(limit, predicate):
            """
            Execute take bucket.
            
            This function implements the take bucket step for this module.
            It is used to keep the broader workflow readable and easier to maintain.
            """
            remaining = max(0, int(limit or 0))
            taken = 0
            if remaining <= 0:
                return 0
            for idx in ordered_indices:
                if remaining <= 0 or len(selected) >= target:
                    break
                if not predicate(idx):
                    continue
                if not _can_take(idx, enforce_artist_cap=True):
                    continue
                _take_idx(idx)
                remaining -= 1
                taken += 1
            return taken

        seed_taken = _take_bucket(
            seed_quota,
            lambda idx: _artist_for_idx(idx) in seed_affinity_artist_set,
        )
        long_tail_taken = _take_bucket(
            long_tail_quota,
            lambda idx: self._long_tail_strength(candidate_ids[idx]) >= long_tail_threshold,
        )
        if hard_top20:
            # If hard quotas are not met on first pass, top-up with progressively
            # relaxed long-tail thresholds while keeping artist-cap constraints.
            remaining_seed = max(0, seed_quota - int(seed_taken or 0))
            if remaining_seed > 0:
                _take_bucket(
                    remaining_seed,
                    lambda idx: _artist_for_idx(idx) in seed_affinity_artist_set,
                )
            remaining_long = max(0, long_tail_quota - int(long_tail_taken or 0))
            if remaining_long > 0:
                for threshold in (long_tail_threshold, 0.20, 0.16, 0.12, 0.08):
                    taken = _take_bucket(
                        remaining_long,
                        lambda idx, t=threshold: self._long_tail_strength(candidate_ids[idx]) >= float(t),
                    )
                    remaining_long = max(0, remaining_long - int(taken or 0))
                    if remaining_long <= 0:
                        break
            if remaining_long > 0:
                # Final hard-mode fallback: take the strongest available long-tail rows
                # relative to this candidate set, even if they do not cross absolute
                # global thresholds for long-tail strength.
                long_tail_ranked = sorted(
                    list(ordered_indices or []),
                    key=lambda idx: float(self._long_tail_strength(candidate_ids[idx])),
                    reverse=True,
                )
                for idx in long_tail_ranked:
                    if remaining_long <= 0 or len(selected) >= target:
                        break
                    if not _can_take(idx, enforce_artist_cap=True):
                        continue
                    _take_idx(idx)
                    remaining_long -= 1

        _take_bucket(
            explore_quota,
            lambda idx: _artist_for_idx(idx) not in seed_affinity_artist_set,
        )

        # Fill remainder using score order, first with artist cap then relaxed.
        enforce_sequence = (True,) if hard_top20 else (True, False)
        for enforce_cap in enforce_sequence:
            if len(selected) >= target:
                break
            for idx in ordered_indices:
                if len(selected) >= target:
                    break
                if not _can_take(idx, enforce_artist_cap=enforce_cap):
                    continue
                _take_idx(idx)

        # Last fallback: pure score sort from all indices if still under target.
        if len(selected) < target:
            score_sorted = sorted(
                range(len(adjusted_scores or [])),
                key=lambda i: float(adjusted_scores[i]),
                reverse=True,
            )
            for idx in score_sorted:
                if len(selected) >= target:
                    break
                if idx in used_idx:
                    continue
                if hard_top20:
                    if not _can_take(idx, enforce_artist_cap=True):
                        continue
                sid = str(candidate_ids[idx] or "").strip()
                ta_key = self._track_title_artist_key(sid)
                if ta_key in used_title_artist:
                    continue
                _take_idx(idx)

        # Post-selection repair:
        # If seed-affinity quota is still short, swap out the lowest-score non-seed
        # tracks for best available unused seed-affinity tracks while preserving:
        # 1) artist cap, 2) title+artist dedupe, 3) long-tail minimum.
        if hard_top20:
            seed_actual_now = sum(1 for idx in selected if _is_seed_idx(idx))
            long_tail_actual_now = sum(1 for idx in selected if _is_long_tail_idx(idx))
            missing_seed = max(0, int(effective_seed_quota) - int(seed_actual_now))
            repair_swaps = 0
            repair_exhausted = False
            if missing_seed > 0:
                seed_candidate_order = sorted(
                    [idx for idx in range(len(candidate_ids or [])) if _is_seed_idx(idx)],
                    key=lambda i: float(adjusted_scores[i]),
                    reverse=True,
                )
                for _step in range(missing_seed):
                    # Select weakest removable non-seed first.
                    victim_candidates = sorted(
                        [idx for idx in list(selected or []) if not _is_seed_idx(idx)],
                        key=lambda i: float(adjusted_scores[i]),
                    )
                    swapped = False
                    for victim_idx in victim_candidates:
                        victim_long_tail = _is_long_tail_idx(victim_idx)
                        if victim_long_tail and int(long_tail_actual_now) <= int(effective_long_tail_quota):
                            continue
                        insert_pos = -1
                        try:
                            insert_pos = selected.index(victim_idx)
                        except Exception:
                            insert_pos = -1
                        _drop_idx(victim_idx)

                        replacement_idx = None
                        for cand_idx in seed_candidate_order:
                            if _can_take(cand_idx, enforce_artist_cap=True):
                                replacement_idx = cand_idx
                                break

                        if replacement_idx is None:
                            # Roll back this victim and try next one.
                            _take_idx(victim_idx)
                            if insert_pos >= 0 and selected and selected[-1] == victim_idx:
                                selected.pop()
                                selected.insert(min(insert_pos, len(selected)), victim_idx)
                            continue

                        _take_idx(replacement_idx)
                        if insert_pos >= 0 and selected and selected[-1] == replacement_idx:
                            selected.pop()
                            selected.insert(min(insert_pos, len(selected)), replacement_idx)
                        repair_swaps += 1
                        seed_actual_now += 1
                        if victim_long_tail:
                            long_tail_actual_now -= 1
                        if _is_long_tail_idx(replacement_idx):
                            long_tail_actual_now += 1
                        swapped = True
                        break
                    if not swapped:
                        repair_exhausted = True
                        break
            # Keep latest counters for stats block below.
            seed_actual_now = sum(1 for idx in selected if _is_seed_idx(idx))
            long_tail_actual_now = sum(1 for idx in selected if _is_long_tail_idx(idx))
            # If swap-repair is exhausted and target is still unmet, downgrade effective
            # target to what is actually feasible under hard constraints.
            if repair_exhausted and int(seed_actual_now) < int(effective_seed_quota):
                effective_seed_quota = int(seed_actual_now)
        else:
            repair_swaps = 0
            repair_exhausted = False
            seed_actual_now = sum(1 for idx in selected if _is_seed_idx(idx))
            long_tail_actual_now = sum(1 for idx in selected if _is_long_tail_idx(idx))

        seed_actual = 0
        long_tail_actual = 0
        for idx in selected:
            sid = str(candidate_ids[idx] or "").strip()
            if not sid:
                continue
            artist = _artist_for_idx(idx)
            if artist and artist in seed_affinity_artist_set:
                seed_actual += 1
            if float(self._long_tail_strength(sid)) > 0.0:
                long_tail_actual += 1
        max_artist_actual = max(artist_counts.values()) if artist_counts else 0
        self.last_quota_stats = {
            "enabled": bool(hard_top20),
            "target_k": int(target),
            "seed_target_desired": int(desired_seed_quota),
            "seed_target_effective": int(effective_seed_quota),
            "seed_actual": int(seed_actual_now if hard_top20 else seed_actual),
            "seed_pool_affinity": int(len(seed_affinity_artist_set)),
            "seed_pool_exact": int(len(seed_artist_set)),
            "long_tail_target_desired": int(desired_long_tail_quota),
            "long_tail_target_effective": int(effective_long_tail_quota),
            "long_tail_actual": int(long_tail_actual_now if hard_top20 else long_tail_actual),
            "max_per_artist_target": int(max_per_artist),
            "max_per_artist_actual": int(max_artist_actual),
            "repair_swaps": int(repair_swaps),
            "repair_exhausted": bool(repair_exhausted),
        }
        return selected[:target]

    def _combined_similarity(self, id1, id2, weights):
        """
        Execute combined similarity.
        
        This method implements the combined similarity step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        denom = sum((weights or {}).values()) or 1.0
        mood = self.sim_manager.get_mood_similarity(id1, id2)
        era = self.sim_manager.get_era_similarity(id1, id2)
        style = self.sim_manager.get_style_similarity(id1, id2)
        sim = (
            float((weights or {}).get("mood", 1.0)) * mood
            + float((weights or {}).get("era", 1.0)) * era
            + float((weights or {}).get("style", 1.0)) * style
        ) / float(denom)
        if math.isnan(sim) or math.isinf(sim):
            return 0.0
        return float(sim)

    def _mean_pairwise_similarity(self, track_ids, feature_weights):
        """
        Execute mean pairwise similarity.
        
        This method implements the mean pairwise similarity step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        ids = [str(sid or "").strip() for sid in list(track_ids or []) if str(sid or "").strip()]
        ids = ids[:5]
        if len(ids) < 2:
            return 0.0
        self.sim_manager.precompute_embeddings(ids)
        total = 0.0
        pairs = 0
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                total += self._combined_similarity(ids[i], ids[j], feature_weights)
                pairs += 1
        if pairs <= 0:
            return 0.0
        return float(total / pairs)

    # Dynamic source weighting for explicit/implicit feedback channels.
    def calculate_dynamic_source_weights(
        self,
        liked_history,
        playlist_history,
        played_history,
        feature_weights,
        disliked_history=None,
    ):
        """
        Execute calculate dynamic source weights.
        
        This method implements the calculate dynamic source weights step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        disliked_count = len(list(disliked_history or []))
        total_behavior = len(liked_history or []) + len(playlist_history or []) + len(played_history or [])
        dislike_ratio = float(disliked_count) / float(max(1, total_behavior + disliked_count))

        channels = {
            "likes": {"ids": list(liked_history or []), "base": 1.45},
            "playlists": {"ids": list(playlist_history or []), "base": 1.25},
            "plays": {"ids": list(played_history or []), "base": 0.95},
        }
        dynamic = {}
        for key, cfg in channels.items():
            ids = cfg["ids"]
            if not ids:
                dynamic[key] = 0.0
                continue
            count_boost = min(1.4, math.log1p(len(ids)) * 0.42)
            coherence = self._mean_pairwise_similarity(ids, feature_weights)
            coherence_boost = max(0.0, coherence) * 0.55
            value = float(cfg["base"]) + count_boost + coherence_boost
            if key == "plays":
                # Heavy dislike ratios indicate noisy passive history; down-weight plays in that case.
                value *= max(0.45, 1.0 - 0.65 * dislike_ratio)
            dynamic[key] = max(0.05, float(value))

        total = sum(dynamic.values())
        if total <= 0:
            return {"likes": 0.4, "playlists": 0.35, "plays": 0.25}
        return {key: float(value) / float(total) for key, value in dynamic.items()}

    def calculate_dynamic_similarity_coefficients(
        self,
        source_weights,
        liked_history,
        playlist_history,
        played_history,
        disliked_history,
    ):
        """
        Execute calculate dynamic similarity coefficients.
        
        This method implements the calculate dynamic similarity coefficients step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        explicit_count = len(liked_history or []) + len(playlist_history or [])
        implicit_count = len(played_history or [])
        disliked_count = len(disliked_history or [])
        explicit_ratio = float(explicit_count) / float(max(1, explicit_count + implicit_count))
        disliked_ratio = float(disliked_count) / float(max(1, explicit_count + implicit_count + disliked_count))

        pos_gain = (
            3.4
            + 2.4 * explicit_ratio
            + 1.0 * float((source_weights or {}).get("likes", 0.0))
            + 0.8 * float((source_weights or {}).get("playlists", 0.0))
        )
        neg_penalty = (
            1.3
            + 2.2 * disliked_ratio
            + 0.6 * float((source_weights or {}).get("plays", 0.0))
        )
        return {
            "positive": self._clamp(pos_gain, 3.0, 6.8),
            "negative": self._clamp(neg_penalty, 1.0, 4.2),
        }

    def _dynamic_channel_mix(
        self,
        liked_history,
        playlist_history,
        played_history,
        disliked_history,
    ):
        """
        Execute dynamic channel mix.
        
        This method implements the dynamic channel mix step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        explicit_count = len(liked_history or []) + len(playlist_history or [])
        implicit_count = len(played_history or [])
        disliked_count = len(disliked_history or [])
        total = max(1, explicit_count + implicit_count)
        explicit_ratio = float(explicit_count) / float(total)

        # When explicit intent exists (likes/playlists), bias hard toward personalized + long-tail.
        if explicit_count >= 3:
            personalized = 0.76 + 0.20 * explicit_ratio
            long_tail = 0.20 + 0.10 * explicit_ratio
            if disliked_count >= 4:
                long_tail += 0.03
            personalized = self._clamp(personalized, 0.70, 0.94)
            long_tail = self._clamp(long_tail, 0.18, 0.34)
            popularity_floor = 0.01
        elif explicit_count > 0:
            personalized = 0.66 + 0.22 * explicit_ratio
            long_tail = 0.16 + 0.10 * explicit_ratio
            if disliked_count >= 5:
                long_tail += 0.03
            personalized = self._clamp(personalized, 0.60, 0.88)
            long_tail = self._clamp(long_tail, 0.14, 0.30)
            popularity_floor = 0.03
        else:
            personalized = 0.50 + 0.20 * explicit_ratio
            long_tail = 0.12 + 0.08 * explicit_ratio
            personalized = self._clamp(personalized, 0.45, 0.78)
            long_tail = self._clamp(long_tail, 0.10, 0.22)
            popularity_floor = 0.10

        popularity = max(popularity_floor, 1.0 - personalized - long_tail)
        mix_total = personalized + long_tail + popularity
        if mix_total <= 0:
            return {"personalized": 0.6, "popularity": 0.25, "long_tail": 0.15}
        return {
            "personalized": personalized / mix_total,
            "popularity": popularity / mix_total,
            "long_tail": long_tail / mix_total,
        }

    def _collect_seed_neighbors(self, seed_id):
        """
        Execute collect seed neighbors.
        
        This method implements the collect seed neighbors step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        sid = self._canonical_song_id(seed_id)
        if not sid:
            return []
        meta = self.sim_manager.get_track_info(sid) or {}
        if not meta:
            return []

        artist_cap = int(getattr(settings, "RECO_PERSONAL_NEIGHBOR_ARTIST_CAP", 36) or 36)
        tag_cap = int(getattr(settings, "RECO_PERSONAL_NEIGHBOR_TAG_CAP", 28) or 28)
        year_cap = int(getattr(settings, "RECO_PERSONAL_NEIGHBOR_YEAR_CAP", 18) or 18)
        max_neighbors = int(getattr(settings, "RECO_PERSONAL_NEIGHBORS_PER_SEED", 180) or 180)

        out = []
        seen = set()

        def _push(raw_id):
            """
            Execute push.
            
            This function implements the push step for this module.
            It is used to keep the broader workflow readable and easier to maintain.
            """
            nid = str(raw_id or "").strip()
            if not nid or nid == sid or nid in seen:
                return False
            seen.add(nid)
            out.append(nid)
            return len(out) >= max_neighbors

        artist = str(meta.get("artist_name", "") or "").strip().lower()
        if artist:
            for neighbor_id in self.artist_index.get(artist, [])[:artist_cap]:
                if _push(neighbor_id):
                    return out

        tokens = self._track_tokens(meta)[:8]
        for token in tokens:
            for neighbor_id in self.tag_index.get(token, [])[:tag_cap]:
                if _push(neighbor_id):
                    return out

        year = self._track_year(meta)
        if year > 0:
            for yr in range(year - 2, year + 3):
                for neighbor_id in self.year_index.get(yr, [])[:year_cap]:
                    if _push(neighbor_id):
                        return out

        return out

    def _build_source_candidate_ids(self, seed_ids, exclude_ids, target_n):
        """
        Build source candidate ids.
        
        This method implements the build source candidate ids step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        seeds = [str(sid or "").strip() for sid in list(seed_ids or []) if str(sid or "").strip()]
        if not seeds:
            return []
        source_seed_cap = int(getattr(settings, "RECO_PERSONAL_SEEDS_PER_SOURCE", 8) or 8)
        seeds = seeds[:source_seed_cap]
        target = max(1, int(target_n or 1))
        exclude = set(exclude_ids or [])
        collected = []
        seen = set(exclude)

        for seed_id in seeds:
            for neighbor_id in self._collect_seed_neighbors(seed_id):
                if neighbor_id in seen:
                    continue
                seen.add(neighbor_id)
                collected.append(neighbor_id)
                if len(collected) >= target:
                    return collected

        if len(collected) < target:
            # Prefer long-tail before chart-head popularity when neighbor coverage is sparse.
            total = len(self.popularity_ranking)
            head_cutoff = max(150, int(total * 0.2))
            for sid, _score in self.popularity_ranking[head_cutoff:]:
                sid = str(sid or "").strip()
                if not sid or sid in seen:
                    continue
                seen.add(sid)
                collected.append(sid)
                if len(collected) >= target:
                    break

        if len(collected) < target:
            for sid, _score in self.popularity_ranking:
                sid = str(sid or "").strip()
                if not sid or sid in seen:
                    continue
                seen.add(sid)
                collected.append(sid)
                if len(collected) >= target:
                    break
        return collected

    def _score_candidates_from_seeds(self, candidate_ids, seed_ids, feature_weights):
        """
        Score candidates from seeds.
        
        This method implements the score candidates from seeds step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        candidates = [str(sid or "").strip() for sid in list(candidate_ids or []) if str(sid or "").strip()]
        seeds = [
            self._canonical_song_id(sid)
            for sid in list(seed_ids or [])
            if str(sid or "").strip()
        ]
        seeds = [sid for sid in seeds if sid]
        if not candidates or not seeds:
            return {}
        source_seed_cap = int(getattr(settings, "RECO_PERSONAL_SEEDS_PER_SOURCE", 8) or 8)
        seeds = seeds[:source_seed_cap]
        self.sim_manager.precompute_embeddings(candidates + seeds)

        scores = {}
        for cid in candidates:
            best = 0.0
            for idx, seed in enumerate(seeds):
                recency_decay = max(0.55, 1.0 - 0.08 * idx)
                sim = self._combined_similarity(cid, seed, feature_weights) * recency_decay
                if sim > best:
                    best = sim
            if best > 0.0:
                scores[cid] = float(best)
        return scores

    def _get_long_tail_pool(self, n=20, exclude_ids=None):
        """
        Get long tail pool.
        
        This method implements the get long tail pool step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        target = max(0, int(n or 0))
        if target <= 0:
            return []
        exclude = set(exclude_ids or [])
        total = len(self.popularity_ranking)
        if total <= 0:
            return []
        head_cutoff = max(150, int(total * 0.2))
        tail_rows = self.popularity_ranking[head_cutoff:]
        out = []
        for sid, score in tail_rows:
            sid = str(sid or "").strip()
            if not sid or sid in exclude:
                continue
            long_tail = self._long_tail_strength(sid)
            quality = self._quality_score_normalized(sid)
            score = 0.72 * long_tail + 0.20 * (1.0 - quality) + 0.08 * quality
            out.append((sid, float(score)))
            if len(out) >= target:
                break
        return out

    # Candidate pool aligned to seed-affinity artists (exact seed + seed-neighbor artists).
    def _get_seed_affinity_pool(self, affinity_artists, n=40, exclude_ids=None):
        """
        Get seed affinity pool.
        
        This method implements the get seed affinity pool step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        target = max(0, int(n or 0))
        if target <= 0:
            return []
        artist_set = {str(v or "").strip().lower() for v in list(affinity_artists or []) if str(v or "").strip()}
        if not artist_set:
            return []
        exclude = set(exclude_ids or [])
        per_artist_cap = int(getattr(settings, "RECO_SEED_AFFINITY_PER_ARTIST_CAP", 8) or 8)
        out = []
        seen = set(exclude)
        artist_rows = {}
        for artist in artist_set:
            rows = []
            for sid in list(self.artist_index.get(artist) or [])[:per_artist_cap]:
                sid = str(sid or "").strip()
                if not sid or sid in seen:
                    continue
                seen.add(sid)
                quality = self._quality_score_normalized(sid)
                long_tail = self._long_tail_strength(sid)
                mainstream = self._mainstream_strength(sid)
                score = 0.48 * quality + 0.36 * long_tail - 0.18 * mainstream + 0.24
                rows.append((sid, float(score)))
            if rows:
                rows.sort(key=lambda row: float(row[1]), reverse=True)
                artist_rows[artist] = rows
        if not artist_rows:
            return []

        # Diversified build: first take best-per-artist, then round-robin deeper
        # layers so seed-affinity pool keeps broad artist coverage.
        ranked_artists = sorted(
            artist_rows.keys(),
            key=lambda a: float(artist_rows[a][0][1]),
            reverse=True,
        )
        used_song = set()
        out = []
        # Layer 0: one per artist.
        for artist in ranked_artists:
            sid, score = artist_rows[artist][0]
            if sid in used_song:
                continue
            used_song.add(sid)
            out.append((sid, float(score)))
            if len(out) >= target:
                return out[:target]
        # Additional layers.
        max_depth = max(len(rows) for rows in artist_rows.values())
        for depth in range(1, max_depth):
            for artist in ranked_artists:
                rows = artist_rows.get(artist) or []
                if depth >= len(rows):
                    continue
                sid, score = rows[depth]
                if sid in used_song:
                    continue
                used_song.add(sid)
                out.append((sid, float(score)))
                if len(out) >= target:
                    return out[:target]
        return out[:target]

    # Build adaptive candidate set from personalized channels + popularity + long-tail.
    def get_personalized_candidate_pool(
        self,
        liked_history,
        playlist_history,
        played_history,
        disliked_history,
        feature_weights,
        source_weights,
        target_n=200,
    ):
        """
        Get personalized candidate pool.
        
        This method implements the get personalized candidate pool step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        target = max(1, int(target_n or 1))
        exclude = set(str(sid or "").strip() for sid in list(disliked_history or []))
        exclude.update(str(sid or "").strip() for sid in list(liked_history or []))
        exclude.update(str(sid or "").strip() for sid in list(playlist_history or []))
        exclude.update(str(sid or "").strip() for sid in list(played_history or []))
        exclude.discard("")

        channel_mix = self._dynamic_channel_mix(
            liked_history,
            playlist_history,
            played_history,
            disliked_history,
        )
        personal_target = max(0, int(round(target * float(channel_mix.get("personalized", 0.6)))))
        long_tail_target = max(0, int(round(target * float(channel_mix.get("long_tail", 0.15)))))
        popularity_target = max(0, target - personal_target - long_tail_target)
        explicit_count = len(liked_history or []) + len(playlist_history or [])
        implicit_count = len(played_history or [])
        explicit_ratio = float(explicit_count) / float(max(1, explicit_count + implicit_count))
        seed_affinity_artist_set = self._seed_affinity_artist_set(
            liked_history, playlist_history, played_history
        )
        hard_top20 = bool(target >= 20 and (len(liked_history or []) + len(playlist_history or [])) > 0)
        if hard_top20:
            # Hard guarantees for top-20 behavior.
            min_seed = 6
            min_long_tail = 4
            personal_target = max(personal_target, min_seed + 2)
            long_tail_target = max(long_tail_target, min_long_tail)
            popularity_target = min(popularity_target, 2)
            total_targets = personal_target + long_tail_target + popularity_target
            if total_targets > target:
                overflow = total_targets - target
                # Trim popularity first, then personal, while keeping long-tail floor.
                cut = min(overflow, popularity_target)
                popularity_target -= cut
                overflow -= cut
                if overflow > 0:
                    removable_personal = max(0, personal_target - min_seed)
                    cut = min(overflow, removable_personal)
                    personal_target -= cut
                    overflow -= cut
                if overflow > 0:
                    removable_long = max(0, long_tail_target - min_long_tail)
                    cut = min(overflow, removable_long)
                    long_tail_target -= cut
                    overflow -= cut
            elif total_targets < target:
                # Allocate remaining capacity back to personalized and long-tail.
                spare = target - total_targets
                personal_target += int(round(spare * 0.65))
                long_tail_target += spare - int(round(spare * 0.65))

        seed_affinity_target = 0
        if explicit_count > 0 and seed_affinity_artist_set:
            seed_affinity_target = int(
                round(target * self._clamp(0.20 + 0.18 * explicit_ratio, 0.18, 0.42))
            )
            if hard_top20:
                seed_affinity_target = max(seed_affinity_target, 6)
            seed_affinity_target = min(seed_affinity_target, personal_target, target)
            personal_target = max(0, personal_target - seed_affinity_target)

        source_lists = {
            "likes": list(liked_history or []),
            "playlists": list(playlist_history or []),
            "plays": list(played_history or []),
        }
        personal_candidate_cap = int(
            getattr(settings, "RECO_PERSONAL_CANDIDATE_CAP", max(220, target * 3)) or max(220, target * 3)
        )
        personal_signal = {}
        for source_name, source_ids in source_lists.items():
            channel_weight = float((source_weights or {}).get(source_name, 0.0))
            if channel_weight <= 0.0 or not source_ids:
                continue
            source_target = max(40, int(personal_candidate_cap * max(0.22, channel_weight)))
            candidate_ids = self._build_source_candidate_ids(source_ids, exclude_ids=exclude, target_n=source_target)
            if not candidate_ids:
                continue
            score_map = self._score_candidates_from_seeds(candidate_ids, source_ids, feature_weights)
            for cid, sim_score in score_map.items():
                personal_signal[cid] = float(personal_signal.get(cid, 0.0)) + channel_weight * float(sim_score)

        explicit_count = len(liked_history or []) + len(playlist_history or [])
        implicit_count = len(played_history or [])
        disliked_count = len(disliked_history or [])
        explicit_ratio = float(explicit_count) / float(max(1, explicit_count + implicit_count))
        disliked_ratio = float(disliked_count) / float(max(1, explicit_count + implicit_count + disliked_count))
        signal_scale = self._clamp(
            4.2 + 1.4 * (explicit_count / float(max(1, explicit_count + len(played_history or [])))),
            3.6,
            5.8,
        )
        mainstream_penalty_weight = self._clamp(
            0.45 + 1.25 * explicit_ratio + 0.40 * disliked_ratio,
            0.40,
            2.10,
        )
        quality_weight = self._clamp(0.06 - 0.055 * explicit_ratio, 0.0, 0.06)
        long_tail_bonus_weight = self._clamp(
            0.65 + 1.05 * explicit_ratio + 0.24 * disliked_ratio,
            0.62,
            1.60,
        )

        personalized_rows = []
        for sid, signal_score in personal_signal.items():
            if sid in exclude:
                continue
            base_quality = self._quality_score_normalized(sid)
            long_tail_bonus = self._long_tail_strength(sid)
            mainstream_penalty = self._mainstream_strength(sid)
            personalized_rows.append(
                (
                    sid,
                    quality_weight * base_quality
                    + signal_scale * float(signal_score)
                    + long_tail_bonus_weight * long_tail_bonus
                    - mainstream_penalty_weight * mainstream_penalty,
                )
            )
        personalized_rows.sort(key=lambda row: row[1], reverse=True)
        seed_affinity_rows = self._get_seed_affinity_pool(
            seed_affinity_artist_set,
            n=max(seed_affinity_target * 10, max(120, target * 4)),
            exclude_ids=exclude,
        )

        popular_seed_rows = self.get_candidate_pool(
            n=max(popularity_target * 3, 0),
            exclude_ids=exclude,
        )
        popular_score_scale = self._clamp(1.0 - 0.88 * explicit_ratio, 0.06, 1.0)
        popular_penalty_scale = self._clamp(0.55 * explicit_ratio + 0.20 * disliked_ratio, 0.0, 0.95)
        popular_rows = []
        for sid, _score in list(popular_seed_rows or []):
            sid = str(sid or "").strip()
            if not sid:
                continue
            quality = self._quality_score_normalized(sid)
            mainstream = self._mainstream_strength(sid)
            long_tail = self._long_tail_strength(sid)
            pop_score = (
                popular_score_scale * quality
                + 0.22 * long_tail
                - popular_penalty_scale * mainstream
            )
            popular_rows.append((sid, float(pop_score)))
        long_tail_rows = self._get_long_tail_pool(
            n=max(long_tail_target * 4, 0),
            exclude_ids=exclude,
        )

        combined = []
        used = set(exclude)

        def _take(rows, limit):
            """
            Execute take.
            
            This function implements the take step for this module.
            It is used to keep the broader workflow readable and easier to maintain.
            """
            added = 0
            for sid, score in list(rows or []):
                sid = str(sid or "").strip()
                if not sid or sid in used:
                    continue
                used.add(sid)
                combined.append((sid, float(score)))
                added += 1
                if added >= limit:
                    break
            return added

        _take(seed_affinity_rows, max(0, seed_affinity_target))
        _take(personalized_rows, max(0, personal_target))
        _take(long_tail_rows, max(0, long_tail_target))
        _take(popular_rows, max(0, popularity_target))

        if len(combined) < target:
            _take(
                seed_affinity_rows,
                target - len(combined),
            )

        if len(combined) < target:
            _take(
                personalized_rows,
                target - len(combined),
            )

        if len(combined) < target:
            _take(
                long_tail_rows,
                target - len(combined),
            )

        if len(combined) < target:
            _take(
                popular_rows,
                target - len(combined),
            )

        if len(combined) < target:
            _take(
                self.get_candidate_pool(n=target * 4, exclude_ids=used),
                target - len(combined),
            )

        return combined[:target]

    # Handle calculate dynamic weights.
    def calculate_dynamic_weights(self, liked_ids):
        """
        Execute calculate dynamic weights.
        
        This method implements the calculate dynamic weights step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if not liked_ids or len(liked_ids) < 2:
            return {"mood": 1.0, "era": 1.0, "style": 1.0}
        recent = liked_ids[-3:]
        infos = [self.sim_manager.get_track_info(i) for i in recent]
        years = [float(info.get("year", 0)) for info in infos if info.get("year")]
        era_weight = 1.0
        if len(years) >= 2:
            std_dev = np.std(years)
            if std_dev < 3.0:
                era_weight = 2.0
            elif std_dev > 10.0:
                era_weight = 0.5
        mood_weight = 1.0
        avg_jaccard = 0.0
        valid_pairs = 0
        for i in range(len(recent)):
            for j in range(i + 1, len(recent)):
                sim = self.sim_manager.get_mood_similarity(recent[i], recent[j])
                avg_jaccard += sim
                valid_pairs += 1
        if valid_pairs > 0:
            avg_jaccard /= valid_pairs
            if avg_jaccard > 0.3:
                mood_weight = 2.0
        style_weight = 1.0
        artists = [info.get("artist_name") for info in infos]
        if len(set(artists)) < len(artists):
            style_weight = 2.0
        total = era_weight + mood_weight + style_weight
        return {
            "mood": mood_weight / total * 3,
            "era": era_weight / total * 3,
            "style": style_weight / total * 3,
        }

    # Get cold start items.
    def get_cold_start_items(self, k=10, exclude_ids=None):
        """
        Get cold start items.
        
        This method implements the get cold start items step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        dpp_select_cap = int(getattr(settings, "RECO_DPP_SELECTION_MAX", 48) or 48)
        dpp_k = max(1, min(int(k or 10), dpp_select_cap))
        # Keep cold-start response fast enough for UI timeouts.
        pool_multiplier = int(getattr(settings, "RECO_COLD_START_POOL_MULTIPLIER", 8) or 8)
        pool_min = int(getattr(settings, "RECO_COLD_START_POOL_MIN", 96) or 96)
        pool_max = int(getattr(settings, "RECO_COLD_START_POOL_MAX", 180) or 180)
        pool_n = max(pool_min, min(pool_max, dpp_k * pool_multiplier))
        candidates = self.get_candidate_pool(n=pool_n, exclude_ids=exclude_ids)
        candidate_ids = [c[0] for c in candidates]
        quality_scores = [c[1] for c in candidates]
        if not candidate_ids:
            return []
        use_dpp = str(getattr(settings, "RECO_USE_DPP", "0") or "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        if use_dpp:
            self.sim_manager.precompute_embeddings(candidate_ids)
            weights = {"mood": 1.0, "era": 1.0, "style": 1.0}
            kernel = self.sim_manager.compute_kernel(candidate_ids, weights)
            l_kernel = self.dpp.decompose_kernel(kernel, quality_scores)
            selected_indices = self.dpp.greedy_selection(l_kernel, dpp_k)
            selected_indices = self._fill_selection_count(selected_indices, quality_scores, dpp_k)
            if int(k or 10) > len(selected_indices):
                selected_indices = self._fill_selection_count(selected_indices, quality_scores, k)
        else:
            selected_indices = self._fast_topk_diverse_indices(candidate_ids, quality_scores, int(k or dpp_k))
        results = []
        for idx in selected_indices:
            sid = candidate_ids[idx]
            meta = self.sim_manager.metadata.get(sid, {})
            results.append(
                {
                    "id": sid,
                    "track_id": sid,
                    "video_id": self._playback_video_id(sid, meta),
                    "title": meta.get("title", ""),
                    "artist": meta.get("artist_name", ""),
                    "description": str(meta.get("description", "") or ""),
                    "instrumental": as_optional_bool(meta.get("instrumental")),
                    "instrumental_confidence": float(meta.get("instrumental_confidence", 0.0) or 0.0),
                    "thumbnail": meta.get("thumbnail", ""),
                    "album_cover": meta.get("album_cover", ""),
                    "cover_url": self._fallback_cover(sid, meta),
                    "cover_candidates": self._cover_candidates(sid, meta),
                    "source": "dpp_cold_start",
                    "score": float(quality_scores[idx]),
                }
            )
        return self._dedupe_result_rows_by_title_artist(results, limit=int(k or 10))

    # Recommend this operation.
    def recommend(
        self,
        liked_ids,
        played_ids,
        k=10,
        n=None,
        disliked_ids=None,
        playlist_track_ids=None,
        skip_feedback=None,
    ):
        """
        Execute recommend.
        
        This method implements the recommend step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        self.last_quota_stats = {}
        if n is not None:
            k = n
        # Preserve observed ordering from the profile store instead of using `set(...)`,
        # which destroys recency and weakens "recent taste" modeling.
        disliked = {
            self._canonical_song_id(song_id)
            for song_id in (disliked_ids or [])
            if self._canonical_song_id(song_id)
        }
        def _ordered_unique(song_ids, exclude=None):
            """
            Execute ordered unique.
            
            This function implements the ordered unique step for this module.
            It is used to keep the broader workflow readable and easier to maintain.
            """
            exclude_set = set(exclude or [])
            out = []
            seen = set()
            for song_id in list(song_ids or []):
                sid = self._canonical_song_id(song_id)
                if not sid or sid in seen or sid in exclude_set:
                    continue
                seen.add(sid)
                out.append(sid)
            return out

        liked_history = _ordered_unique(liked_ids, exclude=disliked)
        playlist_history = _ordered_unique(
            playlist_track_ids,
            exclude=set(disliked).union(liked_history),
        )
        played_history = _ordered_unique(
            played_ids,
            exclude=set(disliked).union(liked_history).union(playlist_history),
        )

        history = []
        seen_history = set()
        for song_id in liked_history + playlist_history + played_history:
            sid = str(song_id or "").strip()
            if not sid or sid in seen_history or sid in disliked:
                continue
            seen_history.add(sid)
            history.append(sid)
        if not history:
            self.last_quota_stats = {
                "enabled": False,
                "target_k": int(k or 10),
                "reason": "cold_start",
            }
            return self.get_cold_start_items(k, exclude_ids=disliked)

        # Seed similarity scoring from explicit positives first, then recent plays.
        recent_positive = (liked_history[:5] + playlist_history[:5] + played_history[:5])[:8]
        recent_disliked = _ordered_unique(disliked_ids)[:5]
        recent_history = recent_positive if recent_positive else history[-5:]

        feedback = skip_feedback if isinstance(skip_feedback, dict) else {}
        raw_next_counts = feedback.get("next_counts") if isinstance(feedback.get("next_counts"), dict) else {}
        raw_prev_counts = feedback.get("prev_counts") if isinstance(feedback.get("prev_counts"), dict) else {}
        raw_early_counts = (
            feedback.get("early_next_counts")
            if isinstance(feedback.get("early_next_counts"), dict)
            else {}
        )
        next_counts = {
            str(sid): int(count)
            for sid, count in raw_next_counts.items()
            if str(sid or "").strip() and int(count or 0) > 0
        }
        prev_counts = {
            str(sid): int(count)
            for sid, count in raw_prev_counts.items()
            if str(sid or "").strip() and int(count or 0) > 0
        }
        early_next_counts = {
            str(sid): int(count)
            for sid, count in raw_early_counts.items()
            if str(sid or "").strip() and int(count or 0) > 0
        }
        skip_artist_penalty = defaultdict(float)
        skip_artist_boost = defaultdict(float)
        for sid, count in next_counts.items():
            meta = self.sim_manager.get_track_info(sid) or {}
            artist = str(meta.get("artist_name", "") or "").strip().lower()
            if not artist:
                continue
            skip_artist_penalty[artist] += min(1.8, 0.22 * float(count))
        for sid, count in early_next_counts.items():
            meta = self.sim_manager.get_track_info(sid) or {}
            artist = str(meta.get("artist_name", "") or "").strip().lower()
            if not artist:
                continue
            skip_artist_penalty[artist] += min(2.2, 0.35 * float(count))
        for sid, count in prev_counts.items():
            meta = self.sim_manager.get_track_info(sid) or {}
            artist = str(meta.get("artist_name", "") or "").strip().lower()
            if not artist:
                continue
            skip_artist_boost[artist] += min(1.6, 0.28 * float(count))

        # Dynamic feature weighting (mood/era/style) stays per-user and per-request.
        feature_seed = liked_history[:3] + playlist_history[:3] + played_history[:2]
        if len(feature_seed) >= 2:
            # calculate_dynamic_weights reads the "last 3" items; reverse so most-recent-ish seeds land there.
            weights = self.calculate_dynamic_weights(list(reversed(feature_seed)))
        else:
            weights = self.calculate_dynamic_weights(history)

        # Fully dynamic source/channel weights for likes/playlists/plays + adaptive similarity gains.
        source_weights = self.calculate_dynamic_source_weights(
            liked_history=liked_history,
            playlist_history=playlist_history,
            played_history=played_history,
            feature_weights=weights,
            disliked_history=list(disliked),
        )
        active_bots = self._resolve_active_profile_bots(
            liked_history=liked_history,
            playlist_history=playlist_history,
            played_history=played_history,
        )
        live_bot_weight = float(
            getattr(settings, "RECO_LIVE_BOT_PROFILE_WEIGHT", 0.28) or 0.28
        )
        live_bot_weight = self._clamp(live_bot_weight, 0.0, 0.85)
        sim_coeffs = self.calculate_dynamic_similarity_coefficients(
            source_weights=source_weights,
            liked_history=liked_history,
            playlist_history=playlist_history,
            played_history=played_history,
            disliked_history=list(disliked),
        )
        explicit_count = len(liked_history or []) + len(playlist_history or [])
        implicit_count = len(played_history or [])
        disliked_count = len(disliked)
        explicit_ratio = float(explicit_count) / float(max(1, explicit_count + implicit_count))
        disliked_ratio = float(disliked_count) / float(max(1, explicit_count + implicit_count + disliked_count))
        seed_affinity_artist_set = self._seed_affinity_artist_set(
            liked_history, playlist_history, played_history
        )
        # Stronger profile => allow bot profile routing to matter a bit more; otherwise keep minimal.
        live_bot_weight *= self._clamp(0.60 + 0.85 * explicit_ratio, 0.55, 1.15)
        quality_score_weight = self._clamp(0.24 - 0.22 * explicit_ratio, 0.0, 0.24)
        long_tail_score_weight = self._clamp(
            0.45 + 1.04 * explicit_ratio + 0.30 * disliked_ratio,
            0.40,
            1.65,
        )
        mainstream_score_penalty_weight = self._clamp(
            0.42 + 1.25 * explicit_ratio + 0.42 * disliked_ratio,
            0.36,
            1.95,
        )
        seed_affinity_boost_weight = self._clamp(
            0.40 + 1.10 * explicit_ratio,
            0.34,
            1.70,
        )
        dpp_select_cap = int(getattr(settings, "RECO_DPP_SELECTION_MAX", 48) or 48)
        dpp_k = max(1, min(int(k or 10), dpp_select_cap))
        adaptive_pool_base = int(getattr(settings, "RECO_ADAPTIVE_POOL_SIZE", 64) or 64)
        adaptive_pool_multiplier = int(
            getattr(settings, "RECO_ADAPTIVE_POOL_MULTIPLIER", 3) or 3
        )
        adaptive_pool_min = int(
            getattr(settings, "RECO_ADAPTIVE_POOL_MIN", 96) or 96
        )
        adaptive_pool_max = int(
            getattr(settings, "RECO_ADAPTIVE_POOL_MAX", 220) or 220
        )
        adaptive_pool_n = max(
            adaptive_pool_base,
            min(adaptive_pool_max, max(adaptive_pool_min, dpp_k * adaptive_pool_multiplier)),
        )
        candidates = self.get_personalized_candidate_pool(
            liked_history=liked_history,
            playlist_history=playlist_history,
            played_history=played_history,
            disliked_history=list(disliked),
            feature_weights=weights,
            source_weights=source_weights,
            target_n=adaptive_pool_n,
        )
        candidate_ids = [c[0] for c in candidates]
        quality_scores = [c[1] for c in candidates]
        if not candidate_ids:
            return []
        if quality_scores:
            q_low = float(min(quality_scores))
            q_high = float(max(quality_scores))
            q_denom = max(1e-9, q_high - q_low)
            quality_scores_norm = [
                self._clamp((float(score) - q_low) / q_denom, 0.0, 1.0)
                for score in quality_scores
            ]
        else:
            quality_scores_norm = []
        # Only recent history is used in similarity scoring; avoid precomputing
        # embeddings for the user's full lifetime play history on every request.
        self.sim_manager.precompute_embeddings(candidate_ids + recent_history + recent_disliked)
        adjusted_scores = []
        candidate_bot_info = {}
        for i, sid in enumerate(candidate_ids):
            max_sim = 0.0
            for hist_id in recent_history:
                mood = self.sim_manager.get_mood_similarity(sid, hist_id)
                era = self.sim_manager.get_era_similarity(sid, hist_id)
                style = self.sim_manager.get_style_similarity(sid, hist_id)
                sim = (
                    weights["mood"] * mood + weights["era"] * era + weights["style"] * style
                ) / 3.0
                if sim > max_sim:
                    max_sim = sim
            max_neg_sim = 0.0
            for neg_id in recent_disliked:
                mood = self.sim_manager.get_mood_similarity(sid, neg_id)
                era = self.sim_manager.get_era_similarity(sid, neg_id)
                style = self.sim_manager.get_style_similarity(sid, neg_id)
                neg_sim = (
                    weights["mood"] * mood + weights["era"] * era + weights["style"] * style
                ) / 3.0
                if neg_sim > max_neg_sim:
                    max_neg_sim = neg_sim
            artist = str((self.sim_manager.get_track_info(sid) or {}).get("artist_name", "") or "").strip().lower()
            track_next = float(next_counts.get(sid, 0))
            track_early = float(early_next_counts.get(sid, 0))
            track_prev = float(prev_counts.get(sid, 0))
            track_skip_penalty = min(3.2, 0.34 * track_next + 0.62 * track_early)
            track_replay_boost = min(2.0, 0.42 * track_prev)
            artist_skip_penalty = min(2.4, float(skip_artist_penalty.get(artist, 0.0)))
            artist_replay_boost = min(1.6, float(skip_artist_boost.get(artist, 0.0)))
            seed_affinity_boost = (
                seed_affinity_boost_weight if artist and artist in seed_affinity_artist_set else 0.0
            )
            long_tail_bonus = self._long_tail_strength(sid)
            mainstream_penalty = self._mainstream_strength(sid)
            bot_name, bot_match = self._candidate_bot_profile_match(sid, active_bots)
            bot_boost = live_bot_weight * float(bot_match)
            candidate_bot_info[sid] = {
                "bot_profile": bot_name,
                "bot_match_score": float(bot_match),
                "bot_boost": float(bot_boost),
            }
            adjusted_scores.append(
                quality_score_weight * quality_scores_norm[i]
                + float(sim_coeffs.get("positive", 5.0)) * max_sim
                - float(sim_coeffs.get("negative", 2.0)) * max_neg_sim
                + long_tail_score_weight * long_tail_bonus
                + track_replay_boost
                + artist_replay_boost
                + seed_affinity_boost
                + bot_boost
                - track_skip_penalty
                - artist_skip_penalty
                - mainstream_score_penalty_weight * mainstream_penalty
            )
        use_dpp = str(getattr(settings, "RECO_USE_DPP", "0") or "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        if use_dpp:
            kernel = self.sim_manager.compute_kernel(candidate_ids, weights)
            l_kernel = self.dpp.decompose_kernel(kernel, adjusted_scores)
            selected_indices = self.dpp.greedy_selection(l_kernel, dpp_k)
            selected_indices = self._fill_selection_count(selected_indices, adjusted_scores, dpp_k)
            if int(k or 10) > len(selected_indices):
                selected_indices = self._fill_selection_count(selected_indices, adjusted_scores, k)
        else:
            selected_indices = self._fast_topk_diverse_indices(candidate_ids, adjusted_scores, int(k or dpp_k))
        ranked_indices = sorted(
            range(len(adjusted_scores or [])),
            key=lambda i: float(adjusted_scores[i]),
            reverse=True,
        )
        # Merge diversification order with score order, then enforce output quotas and hard dedupe.
        merged_order = []
        merged_seen = set()
        for idx in list(selected_indices or []) + list(ranked_indices or []):
            try:
                norm_idx = int(idx)
            except Exception:
                continue
            if norm_idx < 0 or norm_idx in merged_seen:
                continue
            merged_seen.add(norm_idx)
            merged_order.append(norm_idx)
        selected_indices = self._apply_output_quotas(
            ordered_indices=merged_order,
            candidate_ids=candidate_ids,
            adjusted_scores=adjusted_scores,
            liked_history=liked_history,
            playlist_history=playlist_history,
            played_history=played_history,
            k=int(k or dpp_k),
        )
        results = []
        for idx in selected_indices:
            sid = candidate_ids[idx]
            meta = self.sim_manager.metadata.get(sid, {})
            bot_info = candidate_bot_info.get(sid, {})
            results.append(
                {
                    "id": sid,
                    "track_id": sid,
                    "video_id": self._playback_video_id(sid, meta),
                    "title": meta.get("title", ""),
                    "artist": meta.get("artist_name", ""),
                    "description": str(meta.get("description", "") or ""),
                    "instrumental": as_optional_bool(meta.get("instrumental")),
                    "instrumental_confidence": float(meta.get("instrumental_confidence", 0.0) or 0.0),
                    "thumbnail": meta.get("thumbnail", ""),
                    "album_cover": meta.get("album_cover", ""),
                    "cover_url": self._fallback_cover(sid, meta),
                    "cover_candidates": self._cover_candidates(sid, meta),
                    "source": "dpp_adaptive",
                    "score": float(adjusted_scores[idx]),
                    "bot_profile": str(bot_info.get("bot_profile", "") or ""),
                    "bot_match_score": float(bot_info.get("bot_match_score", 0.0) or 0.0),
                    "bot_boost": float(bot_info.get("bot_boost", 0.0) or 0.0),
                }
            )
        return self._dedupe_result_rows_by_title_artist(results, limit=int(k or dpp_k))

    # Get trending.
    def get_trending(self, n=10):
        """
        Get trending.
        
        This method implements the get trending step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        return self.get_cold_start_items(k=n)
