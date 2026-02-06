from pathlib import Path
import polars as pl

base_dir = Path(__file__).parent
ready_dir = base_dir / "data" / "processed" / "music_ready"
csv_dir = base_dir / "data" / "after_filtering"
csv_dir.mkdir(parents=True, exist_ok=True)

year_range = range(2012, 2019)

for year in year_range:
    parquet_path = ready_dir / f"year={year}" / "data.parquet"
    if not parquet_path.exists():
        print(f"{year}: not found, skipping")
        continue

    df = pl.read_parquet(parquet_path)
    csv_path = csv_dir / f"{year}.csv"
    df.write_csv(csv_path)
    print(f"{year}: {df.height:,} rows -> {csv_path}")

print(f"\nDone. CSVs saved to {csv_dir}")