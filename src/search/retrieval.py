import numpy as np
from typing import List, Tuple
from src.utils.data_loader import data_loader
from src.search.embedder import embedder


def search_years(
    query: str,
    year_start: int,
    year_end: int,
    top_k: int = 50
) -> List[dict]:
    query_embedding = embedder.encode(query)
    all_results = []
    for year in range(year_start, year_end + 1):
        if year not in data_loader.indexes:
            continue
        distances, indices = data_loader.search_year(year, query_embedding, top_k)
        tracks = data_loader.get_tracks_batch(year, indices, distances)
        all_results.extend(tracks)
    all_results.sort(key=lambda x: x.get("_distance", float("inf")))
    return all_results


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