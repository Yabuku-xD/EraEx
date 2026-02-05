from pathlib import Path
import re
import polars as pl
import os
import nltk
from nltk.corpus import wordnet as wn
from guessit import guessit
from joblib import Parallel, delayed
from tqdm import tqdm

# Setup
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

PROJECT_DIR = Path.cwd() # Run from root
PROCESSED_DIR = PROJECT_DIR / 'data' / 'processed'
MUSIC_DIR = PROCESSED_DIR / 'music_tracks'
READY_DIR = PROCESSED_DIR / 'music_ready'

MUSIC_DIR.mkdir(parents=True, exist_ok=True)
READY_DIR.mkdir(parents=True, exist_ok=True)

def get_automated_genres():
    genres = set()
    try:
        base = wn.synset('musical_style.n.01')
        for syn in base.closure(lambda s: s.hyponyms()):
            for lemma in syn.lemmas():
                name = lemma.name().replace('_', ' ').lower()
                genres.add(name)
    except Exception as e:
        print(f"Error fetching genres: {e}")
        return set()
    return genres

MUSIC_GENRES = get_automated_genres()
print(f"Loaded {len(MUSIC_GENRES)} genres.")

def is_music_track(row) -> bool:
    title = str(row.get('title', '')).lower()
    genre = str(row.get('genre', '')).lower()
    
    g = guessit(title)
    if g.get('type') == 'episode' or 'season' in g or 'episode' in g:
        return False

    if genre and (genre in MUSIC_GENRES):
        return True
    
    return True

def filter_music_tracks_parallel(df: pl.DataFrame) -> tuple:
    print("  > Starting parallel filtering...")
    rows = df.to_dicts()
    
    # Try 2 jobs for debug, keep low overhead
    mask = Parallel(n_jobs=2, batch_size=10)(
        delayed(is_music_track)(row) for row in tqdm(rows, desc="Filtering")
    )
    
    music_rows = [r for r, m in zip(rows, mask) if m]
    other_rows = [r for r, m in zip(rows, mask) if not m]
    
    music_df = pl.DataFrame(music_rows, schema=df.schema)
    other_df = pl.DataFrame(other_rows, schema=df.schema)
    
    return music_df, other_df

def process_year(year):
    year_dir = PROCESSED_DIR / f'year={year}'
    if not year_dir.exists():
        print(f"Year dir {year_dir} not found")
        return None
    
    print(f'Processing {year}...')
    parquet_files = [str(f) for f in year_dir.glob('*.parquet')]
    if not parquet_files:
        print("No parquet files")
        return None
    
    try:
        df = pl.scan_parquet(parquet_files).collect()
    except Exception as e:
        print(f"Scan failed: {e}. Fallback...")
        dfs = [pl.read_parquet(f) for f in parquet_files]
        df = pl.concat(dfs, how='diagonal')

    # DEBUG: LIMIT ROWS
    print(f"  Total rows: {df.height}")
    df = df.head(100) 
    print("  (Debug mode: subset to 100 rows)")
    
    before = df.height
    music_df, other_df = filter_music_tracks_parallel(df)
    print(f'  Filtered: {before:,} -> {music_df.height:,} music tracks')
    
    if music_df.height > 0:
        out_dir = READY_DIR / f'year={year}'
        out_dir.mkdir(parents=True, exist_ok=True)
        music_df.write_parquet(out_dir / 'data.parquet')
        print(f'  Saved: {out_dir / "data.parquet"}')
    else:
        print("  0 rows remaining, nothing to save.")

if __name__ == "__main__":
    process_year(2012)
