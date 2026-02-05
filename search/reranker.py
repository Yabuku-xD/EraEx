import numpy as np
from sentence_transformers import CrossEncoder
from typing import Optional
import json
from pathlib import Path
import math

from config.settings import DATA_DIR


CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-TinyBERT-L-2-v2"


class SearchReranker:
    def __init__(self):
        self.cross_encoder = None
        self.tag_similarity = {}
        self._loaded = False
    
    def load(self):
        if self._loaded:
            return
        
        print("Loading cross-encoder model...")
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        
        tag_path = DATA_DIR / "tag_similarity.json"
        if tag_path.exists():
            print("Loading tag similarity matrix...")
            with open(tag_path, 'r') as f:
                self.tag_similarity = json.load(f)
            print(f"  Loaded {len(self.tag_similarity)} tag mappings")
        
        self._loaded = True
        print("Reranker ready!")
    
    def expand_query_tags(self, query: str, max_expansions: int = 5) -> list:
        query_lower = query.lower()
        expansions = []
        
        for tag in self.tag_similarity:
            if tag in query_lower:
                similar = self.tag_similarity[tag]
                for neighbor, score in similar[:3]:
                    if neighbor not in query_lower and score > 0.1:
                        expansions.append((neighbor, score))
        
        expansions.sort(key=lambda x: x[1], reverse=True)
        return [t for t, s in expansions[:max_expansions]]
    
    def popularity_score(self, playback_count: int) -> float:
        if not playback_count or playback_count <= 0:
            return 0.0
        return min(math.log10(playback_count + 1) / 7.0, 1.0)
    
    def rerank_with_cross_encoder(
        self,
        query: str,
        candidates: list,
        top_k: int = 20
    ) -> list:
        if not self.cross_encoder or not candidates:
            return candidates[:top_k]
        
        pairs = []
        for c in candidates:
            title = c.get('title') or ''
            artist = c.get('artist') or ''
            genre = c.get('genre') or ''
            doc_text = f"{title} by {artist}. {genre}"
            pairs.append([query, doc_text])
        
        scores = self.cross_encoder.predict(pairs)
        
        for i, c in enumerate(candidates):
            c['cross_score'] = float(scores[i])
        
        candidates.sort(key=lambda x: x['cross_score'], reverse=True)
        return candidates[:top_k]
    
    def rerank_full_pipeline(
        self,
        query: str,
        candidates: list,
        semantic_weight: float = 0.5,
        popularity_weight: float = 0.2,
        cross_encoder_weight: float = 0.3,
        top_k: int = 20
    ) -> list:
        if not candidates:
            return []
        
        if not self._loaded:
            self.load()
        
        expanded_tags = self.expand_query_tags(query)
        
        for c in candidates:
            tag_boost = 0.0
            genre = c.get('genre') or ''
            tags = c.get('tags') or ''
            track_tags = (genre + ' ' + tags).lower()
            for exp_tag in expanded_tags:
                if exp_tag in track_tags:
                    tag_boost += 0.1
            c['tag_boost'] = min(tag_boost, 0.3)
        
        for c in candidates:
            c['pop_score'] = self.popularity_score(c.get('playback_count', 0))
        
        rerank_limit = 30
        if len(candidates) <= rerank_limit:
            candidates = self.rerank_with_cross_encoder(query, candidates, len(candidates))
        else:
            top_candidates = self.rerank_with_cross_encoder(query, candidates[:rerank_limit], rerank_limit)
            remaining = candidates[rerank_limit:]
            for c in remaining:
                c['cross_score'] = -10.0
            candidates = top_candidates + remaining
        
        for c in candidates:
            semantic = c.get('score', 0)
            if semantic > 2:
                semantic = 1.0
            else:
                semantic = max(0, min(semantic, 1.0))
            
            cross = (c.get('cross_score', 0) + 10) / 20
            cross = max(0, min(cross, 1.0))
            
            pop = c.get('pop_score', 0)
            tag = c.get('tag_boost', 0)
            
            final = (
                semantic_weight * semantic +
                popularity_weight * pop +
                cross_encoder_weight * cross +
                0.1 * tag
            )
            c['final_score'] = final
        
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        return candidates[:top_k]


reranker = SearchReranker()