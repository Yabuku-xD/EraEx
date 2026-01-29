import polars as pl

CANONICAL_SCHEMA = {
    "track_id": pl.Utf8,
    "title": pl.Utf8,
    "artist": pl.Utf8,
    "year": pl.Int32,
    "created_at": pl.Utf8,
    "genre": pl.Utf8,
    "tags": pl.Utf8,
    "description": pl.Utf8,
    "playback_count": pl.Int64,
    "permalink_url": pl.Utf8,
    "extracted_vibe_text": pl.Utf8,
    "ingest_date": pl.Date,
}

REQUIRED_COLUMNS = [
    "title",
    "artist",
    "genre",
    "tags",
    "description",
    "playback_count",
    "permalink_url",
]