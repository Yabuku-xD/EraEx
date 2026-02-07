import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from src.config.settings import EMBEDDINGS_DIR, INDEXES_DIR, MUSIC_READY_DIR, YEARS


class DataLoader:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if DataLoader._initialized:
            return
        self.indexes: Dict[int, faiss.Index] = {}
        self.embeddings: Dict[int, np.ndarray] = {}
        self.ids: Dict[int, pd.DataFrame] = {}
        self.metadata: Dict[int, pd.DataFrame] = {}
        self._loading = False
        self._loaded_years = set()
        DataLoader._initialized = True

    def is_ready(self) -> bool:
        return len(self._loaded_years) > 0

    def get_loaded_years(self) -> List[int]:
        return sorted(list(self._loaded_years))

    def load_year(self, year: int) -> bool:
        if year in self._loaded_years:
            return True
        if year not in YEARS:
            return False
        index_path = INDEXES_DIR / f"faiss_{year}.index"
        ids_path = EMBEDDINGS_DIR / f"ids_{year}.parquet"
        metadata_path = MUSIC_READY_DIR / f"year={year}" / "data.parquet"
        if not index_path.exists():
            print(f"Index not found: {index_path}")
            return False
        if not ids_path.exists():
            print(f"IDs not found: {ids_path}")
            return False
        try:
            print(f"Loading FAISS index for {year}...")
            self.indexes[year] = faiss.read_index(str(index_path))
            print(f"Loading ID mappings for {year}...")
            self.ids[year] = pd.read_parquet(ids_path)
            if metadata_path.exists():
                print(f"Loading metadata for {year}...")
                self.metadata[year] = pd.read_parquet(metadata_path)
            else:
                print(f"Metadata not found for {year}, using IDs only")
                self.metadata[year] = self.ids[year]
            self._loaded_years.add(year)
            print(f"Loaded year {year}: {self.indexes[year].ntotal} vectors")
            return True
        except Exception as e:
            print(f"Error loading year {year}: {e}")
            return False

    def load_all_years(self) -> int:
        loaded = 0
        self._loading = True
        for year in YEARS:
            if self.load_year(year):
                loaded += 1
        self._loading = False
        return loaded

    def search_year(
        self,
        year: int,
        query_embedding: np.ndarray,
        top_k: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        if year not in self.indexes:
            return np.array([]), np.array([])
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype("float32")
        faiss.normalize_L2(query_embedding)
        distances, indices = self.indexes[year].search(query_embedding, top_k)
        return distances[0], indices[0]

    def get_track_info(self, year: int, idx: int) -> Optional[dict]:
        if year not in self.metadata:
            return None
        if year not in self.ids:
            return None
        try:
            if idx < 0 or idx >= len(self.ids[year]):
                return None
            id_row = self.ids[year].iloc[idx]
            track_id = id_row.get("id") or id_row.get("track_id") or idx
            if year in self.metadata and len(self.metadata[year]) > 0:
                meta_matches = self.metadata[year][
                    self.metadata[year].get("id", self.metadata[year].get("track_id", pd.Series())) == track_id
                ]
                if len(meta_matches) > 0:
                    return meta_matches.iloc[0].to_dict()
            return id_row.to_dict()
        except Exception as e:
            print(f"Error getting track info: {e}")
            return None

    def get_tracks_batch(
        self,
        year: int,
        indices: np.ndarray,
        distances: np.ndarray
    ) -> List[dict]:
        results = []
        if year not in self.ids:
            return results
        ids_df = self.ids[year]
        meta_df = self.metadata.get(year, ids_df)
        for i, (idx, dist) in enumerate(zip(indices, distances)):
            if idx < 0 or idx >= len(ids_df):
                continue
            try:
                id_row = ids_df.iloc[idx]
                track_data = id_row.to_dict()
                track_data["_distance"] = float(dist)
                track_data["_year"] = year
                track_data["_idx"] = int(idx)
                results.append(track_data)
            except Exception:
                continue
        return results


data_loader = DataLoader()
