import numpy as np
from typing import List, Tuple
from src.utils.data_loader import data_loader
from src.search.embedder import embedder

from src.config import settings
import os

_BAD_VECTOR = None

def keyword_search_years(
    query: str,
    year_start: int,
    year_end: int,
    top_k: int = 50
) -> List[dict]:
    query_lower = query.lower().strip()
    all_results = []
    
    # 1. Collect all candidates
    for year in range(year_start, year_end + 1):
        if year not in data_loader.metadata:
            continue
        meta_df = data_loader.metadata[year]
        artist_col = meta_df.get("artist", None)
        title_col = meta_df.get("title", None)

        mask = None
        if artist_col is not None:
            mask = meta_df["artist"].fillna("").str.lower().str.contains(query_lower, regex=False)
        
        if title_col is not None:
            title_mask = meta_df["title"].fillna("").str.lower().str.contains(query_lower, regex=False)
            if mask is not None:
                mask = mask | title_mask
            else:
                mask = title_mask
                
        if mask is not None:
            try:
                matched_df = meta_df[mask].copy()
                if "playback_count" in matched_df.columns:
                    matched_df["playback_count"] = matched_df["playback_count"].fillna(0).astype(int)
                    matched_df = matched_df.sort_values("playback_count", ascending=False)
                
                top_indices = matched_df.head(top_k).index.tolist()
                
                for idx in top_indices:
                    try:
                        track_data = meta_df.iloc[idx].to_dict()
                        track_data["_distance"] = 0.0 # Keyword match
                        track_data["_year"] = year
                        track_data["_idx"] = int(idx)
                        track_data["_keyword_match"] = True
                        all_results.append(track_data)
                    except Exception:
                        continue
            except Exception:
                continue

    # 2. Global Sort by Popularity
    all_results.sort(key=lambda x: x.get("playback_count", 0) or 0, reverse=True)
    return all_results[:top_k]


def filter_semantic(candidates: List[dict]) -> List[dict]:
    """
    Filter a list of candidate tracks using the global _BAD_VECTOR.
    Removes tracks with high similarity to bad concepts.
    """
    global _BAD_VECTOR
    # Ensure _BAD_VECTOR is loaded
    global _BAD_VECTOR
    # Ensure _BAD_VECTOR is loaded
    if _BAD_VECTOR is None:
        try:
            # Load pre-computed vector to avoid identifying SBERT in main process (OOM risk)
            # data/bad_vector.npy
            vec_path = "data/bad_vector.npy"
            if not os.path.exists(vec_path):
                 print(f"DEBUG: Vector file {vec_path} not found!", flush=True)
            else:
                 print(f"DEBUG: Loading Bad Vector from {vec_path}...", flush=True)
                 _BAD_VECTOR = np.load(vec_path)
                 print("DEBUG: Bad Vector Loaded", flush=True)
        except Exception as e:
            print(f"DEBUG ERROR: Failed to load bad vector: {e}", flush=True)
            
    if _BAD_VECTOR is None:
        return candidates
        
    filtered = []
    semantic_threshold = 0.05
    print(f"DEBUG: Filtering {len(candidates)} candidates...", flush=True)
    
    for track in candidates:
        year = track.get("_year")
        idx = track.get("_idx")
        
        # If no index, we can't check semantic vector. Assume safe or fallback?
        if year is None or idx is None:
            filtered.append(track)
            continue
            
        try:
            if year not in data_loader.indexes:
                filtered.append(track)
                continue
                
            index = data_loader.indexes[year]
            if "podcast" in str(track.get("title", "")).lower():
                # Safety net: Explicit podcast check to avoid reliance on vector if it fails
                print(f"DEBUG: Explicitly filtering 'podcast' in title: {track.get('title')}", flush=True)
                continue

            sim = np.dot(vec_norm, _BAD_VECTOR)
            
            if "podcast" in str(track.get("title", "")).lower():
                print(f"DEBUG FILTER: '{track.get('title')[:30]}...' Idx={idx} Sim={sim:.4f}", flush=True)

            if sim > semantic_threshold:
                # Skip this track (It's a podcast/news/etc)
                continue
            
            filtered.append(track)
        except Exception:
            # If error (e.g. index issue), keep it
            filtered.append(track)
            
    return filtered


def search_years(
    query: str,
    year_start: int,
    year_end: int,
    top_k: int = 50
) -> List[dict]:
    # 1. Keyword Search
    keyword_results = keyword_search_years(query, year_start, year_end, top_k)
    
    # 2. FILTER KEYWORD RESULTS SEMANTICALLY (CRITICAL FIX)
    keyword_results = filter_semantic(keyword_results)
    
    if len(keyword_results) >= top_k:
        return keyword_results
        
    # 3. Semantic Search
    query_embedding = embedder.encode(query)
    
    # Ensure _BAD_VECTOR is initialized for the loop below
    global _BAD_VECTOR
    if _BAD_VECTOR is None:
         # Trigger init via helper if needed, or just let the loop handle it
         # But loop needs _BAD_VECTOR variable.
         # Calling filter_semantic([]) forces init.
         filter_semantic([])
            
    semantic_results = []
    semantic_threshold = 0.08
    
    for year in range(year_start, year_end + 1):
        if year not in data_loader.indexes:
            continue
            
        distances, indices = data_loader.search_year(year, query_embedding, top_k * 2) 
        
        # Filter by Semantic Similarity (Optimized Loop)
        valid_indices = []
        valid_distances = []
        
        if _BAD_VECTOR is not None:
            index = data_loader.indexes[year]
            for i, idx in enumerate(indices):
                if idx < 0: continue
                try:
                    cand_vec = index.reconstruct(int(idx))
                    cand_norm = cand_vec / np.linalg.norm(cand_vec)
                    sim = np.dot(cand_norm, _BAD_VECTOR)
                    
                    if sim > semantic_threshold:
                        continue
                        
                    valid_indices.append(idx)
                    valid_distances.append(distances[i])
                except Exception:
                    valid_indices.append(idx)
                    valid_distances.append(distances[i])
        else:
            valid_indices = indices.tolist()
            valid_distances = distances.tolist()
            
        tracks = data_loader.get_tracks_batch(
            year, 
            np.array(valid_indices)[:top_k], 
            np.array(valid_distances)[:top_k]
        )
        semantic_results.extend(tracks)
        
    semantic_results.sort(key=lambda x: x.get("_distance", float("inf")))
    combined = keyword_results + semantic_results
    return combined


def merge_candidates(
    results: List[dict],
    top_k: int = 100
) -> List[dict]:
    seen_ids = set()
    unique_results = []
    for track in results:
        track_id = track.get("id") or track.get("track_id") or track.get("permalink_url")
        if track_id and track_id not in seen_ids:
            seen_ids.add(track_id)
            unique_results.append(track)
        if len(unique_results) >= top_k:
            break
    return unique_results