import re
import polars as pl

from config.settings import MAX_DESC_LENGTH


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_urls(text: str) -> str:
    if text is None:
        return ""
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def normalize_title(title: str) -> str:
    if title is None:
        return ""
    title = title.lower()
    title = re.sub(r"[^\w\s]", " ", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def normalize_tags(tags: str) -> str:
    if tags is None or tags == "":
        return ""
    
    separators = r"[,;|/\n]+"
    tag_list = re.split(separators, tags.lower())
    
    cleaned = []
    junk_tokens = {"soundcloud", "source", "iphone", "android", "3rdparty", "recorder"}
    
    for tag in tag_list:
        tag = tag.strip()
        if len(tag) <= 1:
            continue
        if tag in junk_tokens:
            continue
        if tag not in cleaned:
            cleaned.append(tag)
    
    return " ".join(cleaned)


def trim_text(text: str, max_length: int = MAX_DESC_LENGTH) -> str:
    if text is None:
        return ""
    text = normalize_text(text)
    if len(text) <= max_length:
        return text
    trimmed = text[:max_length]
    last_space = trimmed.rfind(" ")
    if last_space > max_length * 0.7:
        trimmed = trimmed[:last_space]
    return trimmed.strip()


def clean_description(desc: str) -> str:
    if desc is None:
        return ""
    desc = remove_urls(desc)
    desc = trim_text(desc, MAX_DESC_LENGTH)
    return desc


def clean_vibe(vibe: str) -> str:
    if vibe is None:
        return ""
    vibe = remove_urls(vibe)
    vibe = trim_text(vibe, MAX_DESC_LENGTH)
    return vibe


def build_doc_text(row: dict) -> str:
    parts = []
    
    title = row.get("title", "") or ""
    if title:
        parts.append(f"TITLE: {normalize_text(title)}")
    
    artist = row.get("artist", "") or ""
    if artist:
        parts.append(f"ARTIST: {normalize_text(artist)}")
    
    genre = row.get("genre", "") or ""
    if genre:
        parts.append(f"GENRE: {genre} {genre}")
    
    tags = row.get("tags", "") or ""
    inferred = row.get("inferred_tags", "") or ""
    combined_tags = f"{tags} {inferred}".strip()
    if combined_tags:
        normalized_tags = normalize_tags(combined_tags)
        if normalized_tags:
            parts.append(f"TAGS: {normalized_tags} {normalized_tags}")
    
    vibe = row.get("extracted_vibe_text", "") or ""
    if vibe:
        parts.append(f"VIBE: {clean_vibe(vibe)}")
    
    desc = row.get("description", "") or ""
    if desc:
        parts.append(f"DESC: {clean_description(desc)}")
    
    year = row.get("year", "")
    if year:
        parts.append(f"YEAR: {year}")
    
    return " ".join(parts)


def dedup_by_permalink(df: pl.DataFrame) -> pl.DataFrame:
    before = df.height
    df = df.unique(subset=["permalink_url"], keep="first")
    after = df.height
    print(f"  Dedup by permalink: {before:,} -> {after:,} (removed {before - after:,})")
    return df


def dedup_by_title_artist_year(df: pl.DataFrame) -> pl.DataFrame:
    before = df.height
    
    df = df.with_columns([
        pl.col("title").map_elements(normalize_title, return_dtype=pl.Utf8).alias("title_norm"),
        pl.col("artist").fill_null("").str.to_lowercase().str.strip_chars().alias("artist_norm"),
    ])
    
    df = df.with_columns([
        (
            pl.col("description").str.len_chars().fill_null(0) +
            pl.col("extracted_vibe_text").str.len_chars().fill_null(0) +
            pl.col("tags").str.len_chars().fill_null(0)
        ).alias("text_quality")
    ])
    
    df = df.sort("text_quality", descending=True)
    df = df.unique(subset=["title_norm", "artist_norm", "year"], keep="first")
    df = df.drop(["title_norm", "artist_norm", "text_quality"])
    
    after = df.height
    print(f"  Dedup by title+artist+year: {before:,} -> {after:,} (removed {before - after:,})")
    return df


def preprocess_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    df = dedup_by_permalink(df)
    df = dedup_by_title_artist_year(df)
    
    print("  Building doc_text_music...")
    rows = df.to_dicts()
    doc_texts = [build_doc_text(row) for row in rows]
    
    df = df.with_columns([
        pl.Series("doc_text_music", doc_texts)
    ])
    
    return df