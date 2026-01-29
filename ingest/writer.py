import hashlib
from datetime import date
from pathlib import Path

import polars as pl

from config.settings import PROCESSED_DIR, YEAR_RANGE


def generate_track_id(row: dict) -> str:
    if row.get("soundcloud_id"):
        return str(row["soundcloud_id"])
    permalink = row.get("permalink_url", "")
    return hashlib.sha1(permalink.encode()).hexdigest()[:16]


def ingest_csv_to_parquet(
    input_path: Path,
    output_dir: Path = PROCESSED_DIR,
    chunk_size: int = 500_000,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {year: 0 for year in YEAR_RANGE}
    stats["skipped"] = 0
    
    reader = pl.read_csv_batched(input_path, batch_size=chunk_size)
    
    batch_num = 0
    while True:
        batches = reader.next_batches(1)
        if not batches:
            break
        
        df = batches[0]
        batch_num += 1
        
        df = df.with_columns([
            pl.lit(date.today()).alias("ingest_date"),
        ])
        
        if "year" not in df.columns and "created_at" in df.columns:
            df = df.with_columns([
                pl.col("created_at").str.slice(0, 4).cast(pl.Int32).alias("year")
            ])
        
        if "track_id" not in df.columns:
            if "soundcloud_id" in df.columns:
                df = df.with_columns([
                    pl.col("soundcloud_id").cast(pl.Utf8).alias("track_id")
                ])
            elif "permalink_url" in df.columns:
                df = df.with_columns([
                    pl.col("permalink_url").hash().cast(pl.Utf8).alias("track_id")
                ])
        
        for year in YEAR_RANGE:
            year_df = df.filter(pl.col("year") == year)
            if year_df.height == 0:
                continue
            
            year_dir = output_dir / f"year={year}"
            year_dir.mkdir(parents=True, exist_ok=True)
            
            out_path = year_dir / f"batch_{batch_num:05d}.parquet"
            year_df.write_parquet(out_path)
            stats[year] += year_df.height
        
        outside_range = df.filter(
            (pl.col("year") < min(YEAR_RANGE)) | (pl.col("year") > max(YEAR_RANGE))
        )
        stats["skipped"] += outside_range.height
    
    return stats


def get_parquet_paths(processed_dir: Path = PROCESSED_DIR, year: int = None) -> list:
    if year:
        year_dir = processed_dir / f"year={year}"
        if year_dir.exists():
            return list(year_dir.glob("*.parquet"))
        return []
    return list(processed_dir.glob("year=*/*.parquet"))


def load_year_partition(year: int, processed_dir: Path = PROCESSED_DIR) -> pl.DataFrame:
    paths = get_parquet_paths(processed_dir, year)
    if not paths:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(p) for p in paths])