from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import polars as pl
from config.settings import PROCESSED_DIR, YEAR_RANGE
from filter_type.filters import filter_dataset, infer_missing_tags


def check_existing_output():
    music_dir = PROCESSED_DIR / "music_tracks"
    existing_years = []
    for year in YEAR_RANGE:
        output_file = music_dir / f"year={year}" / "data.parquet"
        if output_file.exists():
            existing_years.append(year)
    return existing_years


def main():
    music_dir = PROCESSED_DIR / "music_tracks"
    other_dir = PROCESSED_DIR / "other_audio"
    
    existing = check_existing_output()
    if len(existing) == len(YEAR_RANGE):
        print("All filtered years already exist. Skipping filtering.")
        print(f"Existing: {existing}")
        print("Delete data/processed/music_tracks/ to re-run.")
        return 0
    
    music_dir.mkdir(parents=True, exist_ok=True)
    other_dir.mkdir(parents=True, exist_ok=True)
    
    total_music = 0
    total_other = 0
    total_tags_inferred = 0
    skipped_years = []
    
    for year in YEAR_RANGE:
        music_out = music_dir / f"year={year}" / "data.parquet"
        if music_out.exists():
            skipped_years.append(year)
            continue
        
        year_dir = PROCESSED_DIR / f"year={year}"
        if not year_dir.exists():
            continue
        
        parquet_files = list(year_dir.glob("*.parquet"))
        if not parquet_files:
            continue
        
        print(f"\nProcessing {year}...")
        df = pl.concat([pl.read_parquet(f) for f in parquet_files])
        original_count = df.height
        
        music_df, other_df = filter_dataset(df)
        
        null_tags_before = music_df.filter(pl.col("tags").is_null()).height
        music_df = infer_missing_tags(music_df)
        null_tags_after = music_df.filter(
            (pl.col("tags").is_null()) & (pl.col("inferred_tags").is_null())
        ).height
        tags_inferred = null_tags_before - null_tags_after
        
        music_out_dir = music_dir / f"year={year}"
        other_out_dir = other_dir / f"year={year}"
        music_out_dir.mkdir(parents=True, exist_ok=True)
        other_out_dir.mkdir(parents=True, exist_ok=True)
        
        music_df.write_parquet(music_out_dir / "data.parquet")
        other_df.write_parquet(other_out_dir / "data.parquet")
        
        total_music += music_df.height
        total_other += other_df.height
        total_tags_inferred += tags_inferred
        
        print(f"  Original: {original_count:,}")
        print(f"  Music tracks: {music_df.height:,}")
        print(f"  Other audio: {other_df.height:,}")
        print(f"  Tags inferred: {tags_inferred:,}")
    
    if skipped_years:
        print(f"\nSkipped existing years: {skipped_years}")
    
    if total_music > 0 or total_other > 0:
        print("\n" + "=" * 50)
        print("FILTERING COMPLETE")
        print("=" * 50)
        print(f"Total music tracks: {total_music:,}")
        print(f"Total other audio: {total_other:,}")
        print(f"Total tags inferred: {total_tags_inferred:,}")
        if total_music + total_other > 0:
            print(f"Removal rate: {total_other / (total_music + total_other) * 100:.2f}%")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())