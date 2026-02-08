"""
EXTERNAL DATA SOURCES:
----------------------

1. VALENCE-AROUSAL LEXICON
   Source: Warriner, A.B., Kuperman, V., & Brysbaert, M. (2013).
           "Norms of valence, arousal, and dominance for 13,915 English lemmas."
           Behavior Research Methods, 45(4), 1191-1207.
   URL: https://github.com/JULIELab/XANEW
   DOI: https://doi.org/10.3758/s13428-012-0314-x
   Description: Human-rated emotional scores on a 1-9 scale for valence (pleasant/unpleasant)
                and arousal (calm/excited). Normalized to [-1, 1] range.

2. GENRE TAXONOMY (Fallback)
   Source: MusicBrainz - Open Music Encyclopedia
   URL: https://musicbrainz.org/
   API: https://raw.githubusercontent.com/metabrainz/musicbrainz-server/master/root/static/scripts/common/constants/genres.json
   License: Creative Commons CC0
   Description: Community-curated genre taxonomy from the world's largest open music database.

3. MUSICAL INSTRUMENTS
   Source: Wikidata - Free Knowledge Base
   URL: https://www.wikidata.org/
   Query Endpoint: https://query.wikidata.org/sparql
   Query: All instances/subclasses of "musical instrument" (Q34379)
   License: Creative Commons CC0
   Description: Structured knowledge graph containing 500+ musical instrument names.

Usage:
    python -m scripts.generate_mood_lexicon

Output Files:
    - data/mood_lexicon.json     (13,905 words with valence/arousal)
    - data/genre_keywords.json   (extracted from dataset, frequency >= 100)
    - data/attribute_keywords.json (instruments matched against Wikidata)
"""

import json
import pandas as pd
import re
from pathlib import Path
from io import StringIO
import requests


WARRINER_URL = "https://raw.githubusercontent.com/JULIELab/XANEW/master/Ratings_Warriner_et_al.csv"
MUSICBRAINZ_GENRES_URL = "https://raw.githubusercontent.com/metabrainz/musicbrainz-server/master/root/static/scripts/common/constants/genres.json"
WIKIDATA_INSTRUMENTS_SPARQL = """
SELECT DISTINCT ?instrumentLabel WHERE {
  ?instrument wdt:P31/wdt:P279* wd:Q34379.
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT 500
"""


def download_warriner_lexicon() -> pd.DataFrame:
    print("Downloading Warriner et al. (2013) VAD lexicon...")
    response = requests.get(WARRINER_URL, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
    print(f"Downloaded {len(df)} words from Warriner lexicon")
    return df


def normalize_score(value: float, min_val: float = 1.0, max_val: float = 9.0) -> float:
    return round(((value - min_val) / (max_val - min_val)) * 2 - 1, 3)


def build_mood_lexicon(df: pd.DataFrame) -> dict:
    lexicon = {}
    for _, row in df.iterrows():
        word = str(row.get("Word", "")).lower().strip()
        if not word:
            continue
        valence_raw = row.get("V.Mean.Sum", 5.0)
        arousal_raw = row.get("A.Mean.Sum", 5.0)
        valence = normalize_score(valence_raw)
        arousal = normalize_score(arousal_raw)
        lexicon[word] = {"valence": valence, "arousal": arousal}
    return lexicon


def is_valid_genre(genre: str) -> bool:
    if not genre or len(genre) < 2 or len(genre) > 50:
        return False
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9\s\-&/]+$', genre):
        return False
    return True


def download_genre_taxonomy() -> list:
    print("Downloading MusicBrainz genre taxonomy...")
    try:
        response = requests.get(MUSICBRAINZ_GENRES_URL, timeout=30)
        response.raise_for_status()
        genres_data = response.json()
        genres = []
        for item in genres_data:
            if isinstance(item, dict) and "name" in item:
                genres.append(item["name"].lower())
            elif isinstance(item, str):
                genres.append(item.lower())
        print(f"Downloaded {len(genres)} genres from MusicBrainz")
        return genres
    except Exception as e:
        print(f"Failed to download MusicBrainz genres: {e}")
        return []


def download_instrument_list() -> set:
    print("Downloading instrument list from Wikidata...")
    try:
        url = "https://query.wikidata.org/sparql"
        headers = {"Accept": "application/sparql-results+json", "User-Agent": "EraEx/1.0"}
        response = requests.get(url, params={"query": WIKIDATA_INSTRUMENTS_SPARQL}, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
        instruments = set()
        for result in data.get("results", {}).get("bindings", []):
            label = result.get("instrumentLabel", {}).get("value", "")
            if label and len(label) > 2:
                instruments.add(label.lower())
        print(f"Downloaded {len(instruments)} instruments from Wikidata")
        return instruments
    except Exception as e:
        print(f"Failed to download Wikidata instruments: {e}")
        return set()


def extract_genre_keywords(data_dir: Path, min_frequency: int = 100) -> list:
    genre_counts = {}
    music_ready_dir = data_dir / "processed" / "music_ready"
    if not music_ready_dir.exists():
        return download_genre_taxonomy()
    for year_dir in music_ready_dir.iterdir():
        if not year_dir.is_dir():
            continue
        parquet_file = year_dir / "data.parquet"
        if parquet_file.exists():
            try:
                df = pd.read_parquet(parquet_file, columns=["genre"])
                if "genre" in df.columns:
                    for g in df["genre"].dropna():
                        if isinstance(g, str):
                            g_clean = g.strip().lower()
                            if is_valid_genre(g_clean):
                                genre_counts[g_clean] = genre_counts.get(g_clean, 0) + 1
            except Exception as e:
                print(f"Error reading {parquet_file}: {e}")
    frequent_genres = [g for g, count in genre_counts.items() if count >= min_frequency]
    if len(frequent_genres) < 10:
        return download_genre_taxonomy()
    return sorted(frequent_genres)


def extract_attribute_keywords(data_dir: Path, min_frequency: int = 50) -> list:
    instrument_reference = download_instrument_list()
    if not instrument_reference:
        return []
    attribute_counts = {}
    music_ready_dir = data_dir / "processed" / "music_ready"
    if not music_ready_dir.exists():
        return sorted(list(instrument_reference))[:50]
    for year_dir in music_ready_dir.iterdir():
        if not year_dir.is_dir():
            continue
        parquet_file = year_dir / "data.parquet"
        if parquet_file.exists():
            try:
                df = pd.read_parquet(parquet_file, columns=["tags"])
                if "tags" in df.columns:
                    for tags_str in df["tags"].dropna():
                        if isinstance(tags_str, str):
                            for tag in tags_str.split(","):
                                tag = tag.strip().lower()
                                if tag in instrument_reference:
                                    attribute_counts[tag] = attribute_counts.get(tag, 0) + 1
            except Exception:
                pass
    frequent_attrs = [a for a, count in attribute_counts.items() if count >= min_frequency]
    if len(frequent_attrs) < 5:
        return sorted(list(instrument_reference))[:50]
    return sorted(frequent_attrs)


def generate_all_lexicons(data_dir: Path):
    output_dir = data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    warriner_df = download_warriner_lexicon()
    mood_lexicon = build_mood_lexicon(warriner_df)
    mood_path = output_dir / "mood_lexicon.json"
    with open(mood_path, "w", encoding="utf-8") as f:
        json.dump(mood_lexicon, f, indent=2, ensure_ascii=False)
    print(f"Saved mood lexicon with {len(mood_lexicon)} words to {mood_path}")
    genre_keywords = extract_genre_keywords(data_dir)
    genre_path = output_dir / "genre_keywords.json"
    with open(genre_path, "w", encoding="utf-8") as f:
        json.dump(genre_keywords, f, indent=2)
    print(f"Saved {len(genre_keywords)} genre keywords to {genre_path}")
    attribute_keywords = extract_attribute_keywords(data_dir)
    attribute_path = output_dir / "attribute_keywords.json"
    with open(attribute_path, "w", encoding="utf-8") as f:
        json.dump(attribute_keywords, f, indent=2)
    print(f"Saved {len(attribute_keywords)} attribute keywords to {attribute_path}")
    return {
        "mood_lexicon": mood_lexicon,
        "genre_keywords": genre_keywords,
        "attribute_keywords": attribute_keywords
    }


if __name__ == "__main__":
    from src.config.settings import DATA_DIR
    generate_all_lexicons(DATA_DIR)