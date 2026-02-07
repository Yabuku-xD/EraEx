import math
from typing import List
from src.config.settings import WEIGHTS
from src.config.mood_lexicon import extract_mood_from_query, detect_query_type


def compute_text_score(distance: float) -> float:
    return 1.0 / (1.0 + distance)


def compute_mood_score(
    query_mood: dict,
    track_mood: dict
) -> float:
    q_val = query_mood.get("valence", 0.0)
    q_aro = query_mood.get("arousal", 0.0)
    t_val = track_mood.get("valence", 0.0)
    t_aro = track_mood.get("arousal", 0.0)
    distance = math.sqrt((q_val - t_val) ** 2 + (q_aro - t_aro) ** 2)
    max_distance = math.sqrt(8)
    return 1.0 - (distance / max_distance)


def compute_popularity_score(
    playback_count: int,
    max_playback: int = 1000000
) -> float:
    if playback_count <= 0:
        return 0.0
    return math.log(1 + playback_count) / math.log(1 + max_playback)


def infer_track_mood(track: dict) -> dict:
    title = str(track.get("title", "")).lower()
    tags = str(track.get("tags", "")).lower()
    description = str(track.get("description", "")).lower()
    combined = f"{title} {tags} {description}"
    return extract_mood_from_query(combined)


def score_candidates(
    candidates: List[dict],
    query: str
) -> List[dict]:
    query_type = detect_query_type(query)
    query_mood = extract_mood_from_query(query)
    weights = WEIGHTS.get(query_type, WEIGHTS["default"])
    max_playback = 1
    for track in candidates:
        playback = track.get("playback_count", 0) or 0
        if playback > max_playback:
            max_playback = playback
    scored = []
    for track in candidates:
        text_score = compute_text_score(track.get("_distance", 1.0))
        track_mood = infer_track_mood(track)
        mood_score = compute_mood_score(query_mood, track_mood)
        playback = track.get("playback_count", 0) or 0
        popularity_score = compute_popularity_score(playback, max_playback)
        final_score = (
            weights["text"] * text_score +
            weights["mood"] * mood_score +
            weights["popularity"] * popularity_score
        )
        track["_text_score"] = text_score
        track["_mood_score"] = mood_score
        track["_popularity_score"] = popularity_score
        track["_final_score"] = final_score
        track["_query_type"] = query_type
        scored.append(track)
    scored.sort(key=lambda x: x.get("_final_score", 0), reverse=True)
    return scored