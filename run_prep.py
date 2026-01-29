from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import polars as pl
from config.settings import PROCESSED_DIR, YEAR_RANGE
from prep.preprocess import preprocess_dataframe


def check_existing_output():
    output_dir = PROCESSED_DIR / "music_ready"
    existing_years = []
    for year in YEAR_RANGE:
        output_file = output_dir / f"year={year}" / "data.parquet"
        if output_file.exists():
            existing_years.append(year)
    return existing_years


def main():
    music_dir = PROCESSED_DIR / "music_tracks"
    output_dir = PROCESSED_DIR / "music_ready"
    
    existing = check_existing_output()
    if len(existing) == len(YEAR_RANGE):
        print("All preprocessed years already exist. Skipping preprocessing.")
        print(f"Existing: {existing}")
        print("Delete data/processed/music_ready/ to re-run.")
        return 0
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_before = 0
    total_after = 0
    skipped_years = []
    
    for year in YEAR_RANGE:
        out_file = output_dir / f"year={year}" / "data.parquet"
        if out_file.exists():
            skipped_years.append(year)
            continue
        
        year_dir = music_dir / f"year={year}"
        if not year_dir.exists():
            continue
        
        parquet_file = year_dir / "data.parquet"
        if not parquet_file.exists():
            continue
        
        print(f"\nProcessing {year}...")
        df = pl.read_parquet(parquet_file)
        before = df.height
        total_before += before
        
        df = preprocess_dataframe(df)
        after = df.height
        total_after += after
        
        out_dir = output_dir / f"year={year}"
        out_dir.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_dir / "data.parquet")
        
        print(f"  Final: {after:,} rows")
        
        sample = df.head(1)
        if sample.height > 0:
            doc_text = sample["doc_text_music"][0]
            print(f"  Sample doc_text: {doc_text[:200]}...")
    
    if skipped_years:
        print(f"\nSkipped existing years: {skipped_years}")
    
    if total_before > 0:
        print("\n" + "=" * 50)
        print("PREPROCESSING COMPLETE")
        print("=" * 50)
        print(f"Total before: {total_before:,}")
        print(f"Total after: {total_after:,}")
        print(f"Dedup rate: {(total_before - total_after) / total_before * 100:.2f}%")
        print(f"Output: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())