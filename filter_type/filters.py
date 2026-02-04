import re
import polars as pl

# Block genres that are not music tracks
# Source: SoundCloud Content Categories
# Reference: https://developers.soundcloud.com/docs/api/reference#tracks
BLOCK_GENRES = {"sports", "spoken words", "spoken word"}

# Patterns to exclude non-music content (DJ sets, podcasts, interviews, etc.)
# Sources:
# - Wikipedia: Podcast naming conventions
#   https://en.wikipedia.org/wiki/Podcast
# - Common DJ mix and radio show naming patterns
EXCLUDE_PATTERNS = {
    "mix": r"\b(dj\s*mix|mixtape|mix\s*tape|megamix|minimix|set\s*mix)\b",
    "ep_episode": r"\b(ep|episode)\s*[\.:#-]?\s*\d+",
    "season_episode": r"\bs\s*\d+\s*e\s*\d+",
    "podcast": r"\bpodcast\b",
    "interview": r"\binterview\b",
    "with_guest": r"\bwith\s+guest\b",
    "full_album": r"\bfull\s+album\b",
    "radio_show": r"\bradio\s*(show|program|episode)\b",
    "live_set": r"\blive\s*set\b",
    "continuous": r"\bcontinuous\s*mix\b",
}

# Music indicator keywords for track classification
# Source: Music terminology glossary
# Reference: https://en.wikipedia.org/wiki/Glossary_of_music_terminology
MUSIC_INDICATORS = {
    "remix": r"\bremix\b",
    "cover": r"\bcover\b",
    "original": r"\boriginal\b",
    "feat": r"\b(feat|ft|featuring)\b",
    "prod": r"\b(prod|produced)\s*(by|\.)\b",
    "instrumental": r"\binstrumental\b",
    "acoustic": r"\bacoustic\b",
    "vocals": r"\bvocals?\b",
    "lyrics": r"\blyrics?\b",
    "beat": r"\bbeat\b",
    "track": r"\btrack\b",
    "song": r"\bsong\b",
}

# Music genre taxonomy for classification
# Sources:
# - Wikipedia: List of music genres and styles
#   https://en.wikipedia.org/wiki/List_of_music_genres_and_styles
# - Every Noise at Once: https://everynoise.com/
# - SoundCloud genre categories
MUSIC_GENRES = {
    "electronic", "hip hop rap", "house", "techno", "dubstep", "drum bass",
    "ambient", "trance", "r&b soul", "pop", "rock", "metal", "jazz", "blues",
    "classical", "reggae", "country", "folk singer songwriter", "latin",
    "disco", "funk", "indie", "alternative", "punk", "dance edm", "trap",
    "deep house", "progressive house", "minimal", "lo-fi", "chillout",
}


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    return text.lower().strip()


def is_blocked_genre(genre: str) -> bool:
    if genre is None:
        return False
    return normalize_text(genre) in BLOCK_GENRES


def matches_exclude_pattern(text: str) -> bool:
    if text is None:
        return False
    text_lower = normalize_text(text)
    for pattern in EXCLUDE_PATTERNS.values():
        if re.search(pattern, text_lower):
            return True
    return False


def count_music_indicators(text: str) -> int:
    if text is None:
        return 0
    text_lower = normalize_text(text)
    count = 0
    for pattern in MUSIC_INDICATORS.values():
        if re.search(pattern, text_lower):
            count += 1
    return count


def is_likely_music(genre: str, title: str, description: str) -> bool:
    genre_lower = normalize_text(genre) if genre else ""
    if genre_lower in MUSIC_GENRES:
        return True
    
    combined_text = f"{title or ''} {description or ''}"
    music_score = count_music_indicators(combined_text)
    
    if music_score >= 2:
        return True
    
    return False


def infer_tags_from_content(title: str, genre: str, description: str) -> str:
    inferred = []
    combined = f"{title or ''} {description or ''}"
    combined_lower = normalize_text(combined)
    
    for tag_name, pattern in MUSIC_INDICATORS.items():
        if re.search(pattern, combined_lower):
            inferred.append(tag_name)
    
    if genre:
        genre_lower = normalize_text(genre)
        if genre_lower in MUSIC_GENRES:
            inferred.append(genre_lower.replace(" ", "-"))
    
    return " ".join(inferred) if inferred else None


def classify_content_type(row: dict) -> str:
    genre = row.get("genre")
    title = row.get("title", "")
    description = row.get("description", "")
    tags = row.get("tags", "")
    
    all_text = f"{title} {description} {tags}"
    
    if is_blocked_genre(genre):
        return "other_audio"
    
    if matches_exclude_pattern(title):
        return "other_audio"
    
    if matches_exclude_pattern(all_text):
        title_lower = normalize_text(title)
        if re.search(r"\bremix\b", title_lower):
            return "music_track"
        if re.search(r"\b(dj\s*mix|mixtape|live\s*set|megamix)\b", title_lower):
            return "other_audio"
    
    if is_likely_music(genre, title, description):
        return "music_track"
    
    return "music_track"


def filter_dataset(df: pl.DataFrame) -> tuple:
    df = df.with_columns([
        pl.col("title").fill_null("").alias("title_clean"),
        pl.col("genre").fill_null("").alias("genre_clean"),
        pl.col("description").fill_null("").alias("desc_clean"),
        pl.col("tags").fill_null("").alias("tags_clean"),
    ])
    
    df = df.with_columns([
        pl.col("title_clean").str.to_lowercase().alias("title_l"),
        pl.col("genre_clean").str.to_lowercase().alias("genre_l"),
    ])
    
    blocked_genre_mask = pl.col("genre_l").is_in(list(BLOCK_GENRES))
    
    mix_pattern = r"\b(dj\s*mix|mixtape|mix\s*tape|megamix|minimix|set\s*mix|live\s*set|continuous\s*mix)\b"
    dj_mix_mask = pl.col("title_l").str.contains(mix_pattern)
    
    podcast_pattern = r"\b(podcast|episode\s*\d+|ep\s*\d+|s\d+\s*e\d+)\b"
    podcast_mask = pl.col("title_l").str.contains(podcast_pattern)
    
    interview_pattern = r"\b(interview|with\s+guest)\b"
    interview_mask = pl.col("title_l").str.contains(interview_pattern)
    
    album_pattern = r"\bfull\s+album\b"
    album_mask = pl.col("title_l").str.contains(album_pattern)
    
    remix_mask = pl.col("title_l").str.contains(r"\bremix\b")
    
    exclude_mask = (
        blocked_genre_mask | 
        (dj_mix_mask & ~remix_mask) | 
        podcast_mask | 
        interview_mask | 
        album_mask
    )
    
    music_df = df.filter(~exclude_mask)
    other_df = df.filter(exclude_mask)
    
    drop_cols = ["title_clean", "genre_clean", "desc_clean", "tags_clean", "title_l", "genre_l"]
    music_df = music_df.drop([c for c in drop_cols if c in music_df.columns])
    other_df = other_df.drop([c for c in drop_cols if c in other_df.columns])
    
    return music_df, other_df


def infer_missing_tags(df: pl.DataFrame) -> pl.DataFrame:
    def infer_tag(row):
        if row["tags"] is not None and row["tags"] != "":
            return row["tags"]
        
        inferred = []
        title = normalize_text(row.get("title", ""))
        genre = normalize_text(row.get("genre", ""))
        desc = normalize_text(row.get("description", ""))
        
        combined = f"{title} {desc}"
        
        for tag_name, pattern in MUSIC_INDICATORS.items():
            if re.search(pattern, combined):
                inferred.append(tag_name)
        
        if genre and genre in MUSIC_GENRES:
            inferred.append(genre.replace(" ", "-"))
        
        return " ".join(inferred) if inferred else None
    
    rows = df.to_dicts()
    new_tags = [infer_tag(row) for row in rows]
    
    return df.with_columns([
        pl.Series("inferred_tags", new_tags)
    ])