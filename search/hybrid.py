import numpy as np
import polars as pl
from pathlib import Path

from config.settings import EMBEDDINGS_DIR, INDEXES_DIR, PROCESSED_DIR, EMBEDDING_MODEL, YEAR_RANGE
from index.faiss_index import load_index, search_index
from search.mood_lexicon import expand_mood_query
from search.cache import embedding_cache
from search.mood_classifier import mood_classifier
from search.fusion import reciprocal_rank_fusion
from search.mmr import mmr_by_metadata
from embed.hyde_templates import get_hyde_expansion


class MusicSearcher:
    def __init__(self):
        self.model = None
        self.indexes = {}
        self.id_maps = {}
        self.metadata = {}
        self.bm25 = None
        self._loaded = False
    
    def load(self):
        if self._loaded:
            return
        
        print("Loading ColBERT model...")
        from embed.colbert_embedder import get_embedder
        self.model = get_embedder(use_colbert=True)
        self.model.load()
        
        for year in YEAR_RANGE:
            index_path = INDEXES_DIR / f"faiss_{year}.index"
            ids_path = EMBEDDINGS_DIR / f"ids_{year}.parquet"
            meta_path = PROCESSED_DIR / "music_ready" / f"year={year}" / "data.parquet"
            
            if not index_path.exists():
                continue
            
            print(f"Loading {year}...")
            self.indexes[year] = load_index(index_path)
            
            ids_df = pl.read_parquet(ids_path)
            self.id_maps[year] = ids_df["track_id"].to_list()
            
            if meta_path.exists():
                self.metadata[year] = pl.read_parquet(meta_path)
        
        bm25_path = INDEXES_DIR / "bm25_index.pkl"
        if bm25_path.exists():
            print("Loading BM25 index...")
            from search.bm25_index import bm25_index
            bm25_index.load(bm25_path)
            self.bm25 = bm25_index
        
        self._loaded = True
        print(f"Loaded {len(self.indexes)} year indexes")
        print(f"Cache status: {embedding_cache.stats()}")
    
    def embed_query(self, query: str) -> np.ndarray:
        cached = embedding_cache.get(query)
        if cached is not None:
            return cached
        
        if hasattr(self.model, 'embed_query_pooled'):
            embedding = self.model.embed_query_pooled(query)
        else:
            embedding = self.model.embed_query(query)
        embedding = embedding.astype(np.float32)
        
        embedding_cache.set(query, embedding)
        
        return embedding
    
    def search_by_artist(self, artist_query: str, years: list, k: int = 100) -> list:
        artist_lower = artist_query.lower().strip()
        results = []
        
        for year in years:
            if year not in self.metadata:
                continue
            
            df = self.metadata[year].with_row_index("_idx")
            
            matches = df.filter(
                pl.col("artist").str.to_lowercase().str.contains(artist_lower, literal=True)
            )
            
            for row in matches.iter_rows(named=True):
                results.append({
                    "track_id": row["track_id"],
                    "score": 2.0,
                    "year": year,
                    "index": row["_idx"],
                    "title": row.get("title", ""),
                    "artist": row.get("artist", ""),
                    "genre": row.get("genre", ""),
                    "permalink_url": row.get("permalink_url", ""),
                    "playback_count": row.get("playback_count", 0),
                    "match_type": "artist"
                })
        
        results.sort(key=lambda x: x.get("playback_count", 0) or 0, reverse=True)
        return results[:k]
    
    def search_bm25(self, query: str, k: int = 200) -> list:
        if self.bm25 is None:
            return []
        
        return self.bm25.search(query, top_k=k)
    
    def search_year(self, query_embedding: np.ndarray, year: int, k: int = 100, n_probe: int = 10) -> list:
        if year not in self.indexes:
            return []
        
        scores, indices = search_index(self.indexes[year], query_embedding, k=k, n_probe=n_probe)
        
        results = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            track_id = self.id_maps[year][idx]
            results.append({
                "track_id": track_id,
                "score": float(score),
                "year": year,
                "index": int(idx)
            })
        
        return results
    
    def search(
        self,
        query: str,
        years: list = None,
        k: int = 50,
        k_per_year: int = 200,
        n_probe: int = 10,
        use_reranker: bool = True,
        use_hybrid: bool = True,
        use_mmr: bool = True,
        mmr_lambda: float = 0.7
    ) -> list:
        if not self._loaded:
            self.load()
        
        if years is None:
            years = list(self.indexes.keys())
        
        query_intent = mood_classifier.classify(query)
        
        if query_intent == "artist":
            artist_results = self.search_by_artist(query, years, k=k)
            if len(artist_results) >= k // 2:
                return artist_results[:k]
        
        if query_intent == "mood":
            expanded_query = get_hyde_expansion(query)
        else:
            expanded_query = expand_mood_query(query)
        
        query_embedding = self.embed_query(expanded_query)
        
        faiss_results = []
        for year in years:
            if year in self.indexes:
                year_results = self.search_year(query_embedding, year, k=k_per_year, n_probe=n_probe)
                faiss_results.extend(year_results)
        
        faiss_results = self.enrich_results(faiss_results)
        faiss_results.sort(key=lambda x: x["score"], reverse=True)
        
        if use_hybrid and self.bm25 is not None:
            bm25_results = self.search_bm25(expanded_query, k=200)
            bm25_results = self.enrich_bm25_results(bm25_results)
            
            combined = reciprocal_rank_fusion(
                [faiss_results[:200], bm25_results],
                k=60
            )
        else:
            combined = faiss_results
        
        if query_intent == "artist":
            artist_results = self.search_by_artist(query, years, k=k)
            seen_ids = set(r["track_id"] for r in artist_results)
            for r in combined:
                if r["track_id"] not in seen_ids:
                    seen_ids.add(r["track_id"])
                    artist_results.append(r)
            combined = artist_results
        
        combined.sort(key=lambda x: x.get("rrf_score", x.get("score", 0)), reverse=True)
        
        if use_reranker and len(combined) > 10:
            from search.reranker import reranker
            combined = reranker.rerank_full_pipeline(query, combined[:150], top_k=k * 2 if use_mmr else k)
        
        if use_mmr and len(combined) > k:
            combined = mmr_by_metadata(combined, lambda_param=mmr_lambda, top_k=k)
        
        return combined[:k]
    
    def enrich_bm25_results(self, results: list) -> list:
        enriched = []
        
        track_to_meta = {}
        for year in self.metadata:
            df = self.metadata[year].with_row_index("_idx")
            for row in df.iter_rows(named=True):
                track_to_meta[str(row.get("track_id", ""))] = {
                    "year": year,
                    "index": row["_idx"],
                    "title": row.get("title", ""),
                    "artist": row.get("artist", ""),
                    "genre": row.get("genre", ""),
                    "permalink_url": row.get("permalink_url", ""),
                    "playback_count": row.get("playback_count", 0)
                }
        
        for r in results:
            track_id = str(r.get("track_id", ""))
            if track_id in track_to_meta:
                meta = track_to_meta[track_id]
                r.update(meta)
                r["score"] = r.get("bm25_score", 0)
            enriched.append(r)
        
        return enriched
    
    def enrich_results(self, results: list) -> list:
        enriched = []
        for r in results:
            if "title" in r and r["title"]:
                enriched.append(r)
                continue
            
            year = r["year"]
            idx = r["index"]
            
            if year in self.metadata:
                meta_df = self.metadata[year]
                if idx < meta_df.height:
                    row = meta_df.row(idx, named=True)
                    r["title"] = row.get("title", "")
                    r["artist"] = row.get("artist", "")
                    r["genre"] = row.get("genre", "")
                    r["permalink_url"] = row.get("permalink_url", "")
                    r["playback_count"] = row.get("playback_count", 0)
            
            enriched.append(r)
        
        return enriched


searcher = MusicSearcher()