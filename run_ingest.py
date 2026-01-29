from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from ingest.writer import ingest_csv_to_parquet
from config.settings import RAW_DIR, PROCESSED_DIR, YEAR_RANGE


def check_existing_output():
    existing_years = []
    for year in YEAR_RANGE:
        year_dir = PROCESSED_DIR / f"year={year}"
        if year_dir.exists() and list(year_dir.glob("*.parquet")):
            existing_years.append(year)
    return existing_years


def main():
    input_csv = RAW_DIR / "dataset.csv"
    
    if not input_csv.exists():
        print(f"Error: {input_csv} not found")
        return 1
    
    existing = check_existing_output()
    if len(existing) == len(YEAR_RANGE):
        print("All year partitions already exist. Skipping ingest.")
        print(f"Existing: {existing}")
        print("Delete data/processed/year=* folders to re-run.")
        return 0
    
    if existing:
        print(f"Some years already exist: {existing}")
        print("Will only process missing years.")
    
    print(f"Starting ingest from {input_csv}")
    print(f"Output directory: {PROCESSED_DIR}")
    
    stats = ingest_csv_to_parquet(input_csv, PROCESSED_DIR)
    
    print("\nIngest complete. Row counts by year:")
    for year in sorted(y for y in stats.keys() if y != "skipped"):
        print(f"  {year}: {stats[year]:,}")
    print(f"  Skipped (outside 2012-2018): {stats['skipped']:,}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())