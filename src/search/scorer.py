import math
from typing import List
from src.config.settings import WEIGHTS
from src.config.mood_lexicon import extract_mood_from_query, detect_query_type
from src.config import settings


def compute_text_score(distance: float) -> float:
    return 1.0 / (1.0 + distance)


def compute_keyword_match_score(query: str, track: dict) -> float:
    query_lower = query.lower().strip()
    query_tokens = set(query_lower.split())
    artist = str(track.get("artist", "")).lower()
    title = str(track.get("title", "")).lower()
    if query_lower in artist or artist in query_lower:
        return 1.0
    if query_lower in title:
        return 0.9
    artist_tokens = set(artist.split())
    title_tokens = set(title.split())
    artist_overlap = len(query_tokens & artist_tokens) / max(len(query_tokens), 1)
    title_overlap = len(query_tokens & title_tokens) / max(len(query_tokens), 1)
    return max(artist_overlap * 0.8, title_overlap * 0.7)


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


def compute_genre_penalty(track: dict) -> float:
    # Returns multiplier: 1.0 = no penalty, 0.0 = terrible
    title = str(track.get("title", "")).lower()
    tags = str(track.get("tags", "")).lower()
    genre = str(track.get("genre", "")).lower()
    artist = str(track.get("artist", "")).lower()

    bad_keywords = settings.BLOCKLIST.get("keywords", [])
    blocked_artists = settings.BLOCKLIST.get("artists", [])
    blocked_titles = settings.BLOCKLIST.get("titles", [])

    for kw in bad_keywords:
        if kw in genre or kw in tags or (kw in title and "mix" not in title) or kw in artist:
             return 0.0 # Kill it

    for blk_art in blocked_artists:
        if blk_art in artist:
            return 0.0

    for blk_title in blocked_titles:
        if blk_title in title:
            return 0.0

    duration = track.get("duration", 0) # ms
    
    if duration == 0:
        return 0.2 # Penalize missing duration (but don't kill, might be valid)
        
    if duration > 420000: # 7 minutes
        if "mix" in title or "set" in title:
             return 0.5
        else:
             return 0.0

    return 1.0

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
    
    query_length = len(query.split())
    scored = []
    for track in candidates:
        text_score = compute_text_score(track.get("_distance", 1.0))
        keyword_score = compute_keyword_match_score(query, track)
        track_mood = infer_track_mood(track)
        mood_score = compute_mood_score(query_mood, track_mood)
        playback = track.get("playback_count", 0) or 0
        popularity_score = compute_popularity_score(playback, max_playback)
        genre_penalty = compute_genre_penalty(track)

        # Keyword match logic
        if keyword_score > 0.8 and query_length <= 3:
            # Strong exact match on short query -> Keyword dominates
            final_score = keyword_score * 0.5 + text_score * 0.2 + popularity_score * 0.3
        elif keyword_score > 0.5:
             # Partial match or long query -> Balanced
             final_score = keyword_score * 0.4 + text_score * 0.2 + popularity_score * 0.2 + mood_score * 0.2
        else:
            # Semantic/Mood based
            final_score = (
                weights["text"] * text_score +
                weights["mood"] * mood_score +
                weights["popularity"] * popularity_score
            )
        
        final_score *= genre_penalty
        
        track["_text_score"] = text_score
        track["_keyword_score"] = keyword_score
        track["_mood_score"] = mood_score
        track["_popularity_score"] = popularity_score
        track["_final_score"] = final_score
        track["_query_type"] = query_type

        if final_score > 0.001:
            scored.append(track)
            
    scored.sort(key=lambda x: x.get("_final_score", 0), reverse=True)
    return scored