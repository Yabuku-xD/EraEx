import math
from typing import List
from src.config.settings import MMR_LAMBDA, MIN_ARTIST_GAP


def compute_similarity(track1: dict, track2: dict) -> float:
    artist1 = str(track1.get("artist", "")).lower().strip()
    artist2 = str(track2.get("artist", "")).lower().strip()
    artist_sim = 1.0 if artist1 == artist2 and artist1 else 0.0
    score_sim = 0.0
    s1 = track1.get("_final_score", 0.5)
    s2 = track2.get("_final_score", 0.5)
    if s1 > 0 and s2 > 0:
        score_sim = 1.0 - abs(s1 - s2)
    return 0.6 * artist_sim + 0.4 * score_sim


def mmr_rerank(
    candidates: List[dict],
    top_k: int = 50,
    lambda_param: float = MMR_LAMBDA
) -> List[dict]:
    if not candidates:
        return []
    if len(candidates) <= top_k:
        return candidates
    selected = []
    remaining = list(candidates)
    best = max(remaining, key=lambda x: x.get("_final_score", 0))
    selected.append(best)
    remaining.remove(best)
    while len(selected) < top_k and remaining:
        mmr_scores = []
        for candidate in remaining:
            relevance = candidate.get("_final_score", 0)
            max_sim = max(
                compute_similarity(candidate, sel)
                for sel in selected
            )
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            mmr_scores.append((candidate, mmr_score))
        best_candidate = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(best_candidate)
        remaining.remove(best_candidate)
    return selected


def apply_artist_spacing(
    ranked: List[dict],
    min_gap: int = MIN_ARTIST_GAP
) -> List[dict]:
    result = []
    artist_last_pos = {}
    deferred = []
    for track in ranked:
        artist = str(track.get("artist", "")).lower().strip()
        if not artist:
            result.append(track)
            continue
        if artist in artist_last_pos:
            gap = len(result) - artist_last_pos[artist]
            if gap < min_gap:
                deferred.append(track)
                continue
        result.append(track)
        artist_last_pos[artist] = len(result) - 1
    for track in deferred:
        artist = str(track.get("artist", "")).lower().strip()
        if artist in artist_last_pos:
            gap = len(result) - artist_last_pos[artist]
            if gap >= min_gap:
                result.append(track)
                artist_last_pos[artist] = len(result) - 1
        else:
            result.append(track)
            artist_last_pos[artist] = len(result) - 1
    return result


def rerank(
    candidates: List[dict],
    top_k: int = 50
) -> List[dict]:
    mmr_result = mmr_rerank(candidates, top_k=top_k * 2)
    spaced_result = apply_artist_spacing(mmr_result)
    return spaced_result[:top_k]