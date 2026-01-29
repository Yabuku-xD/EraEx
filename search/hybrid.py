import numpy as np
import polars as pl
from pathlib import Path
from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDINGS_DIR, INDEXES_DIR, PROCESSED_DIR, EMBEDDING_MODEL, YEAR_RANGE
from index.faiss_index import load_index, search_index


class MusicSearcher:
    def __init__(self):
        self.model = None
        self.indexes = {}
        self.id_maps = {}
        self.metadata = {}
        self._loaded = False
    
    def load(self):
        if self._loaded:
            return
        
        print("Loading search model...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        
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
        
        self._loaded = True
        print(f"Loaded {len(self.indexes)} year indexes")
    
    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode(query, normalize_embeddings=True)
        return embedding.astype(np.float32)
    
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
        use_reranker: bool = True
    ) -> list:
        if not self._loaded:
            self.load()
        
        if years is None:
            years = list(self.indexes.keys())
        
        artist_results = self.search_by_artist(query, years, k=k)
        
        if len(artist_results) >= k:
            return artist_results[:k]
        
        query_embedding = self.embed_query(query)
        
        all_results = []
        for year in years:
            if year in self.indexes:
                year_results = self.search_year(query_embedding, year, k=k_per_year, n_probe=n_probe)
                all_results.extend(year_results)
        
        all_results = self.enrich_results(all_results)
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        seen_ids = set(r["track_id"] for r in artist_results)
        combined = list(artist_results)
        
        for r in all_results:
            if r["track_id"] not in seen_ids:
                seen_ids.add(r["track_id"])
                combined.append(r)
        
        combined.sort(key=lambda x: x["score"], reverse=True)
        
        if use_reranker and len(combined) > 10:
            from search.reranker import reranker
            combined = reranker.rerank_full_pipeline(query, combined[:150], top_k=k)
        
        return combined[:k]
    
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