from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import polars as pl
from config.settings import PROCESSED_DIR, EMBEDDINGS_DIR, YEAR_RANGE
from embed.embedder import load_model, embed_texts, save_embeddings


def check_existing_output():
    existing_years = []
    for year in YEAR_RANGE:
        emb_file = EMBEDDINGS_DIR / f"embeddings_{year}.npy"
        ids_file = EMBEDDINGS_DIR / f"ids_{year}.parquet"
        if emb_file.exists() and ids_file.exists():
            existing_years.append(year)
    return existing_years


def main():
    music_ready = PROCESSED_DIR / "music_ready"
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    existing = check_existing_output()
    if len(existing) == len(YEAR_RANGE):
        print("All embeddings already exist. Skipping.")
        print(f"Existing: {existing}")
        print("Delete data/embeddings/ to re-run.")
        return 0
    
    model = load_model()
    
    total_embedded = 0
    skipped_years = []
    
    for year in YEAR_RANGE:
        emb_file = EMBEDDINGS_DIR / f"embeddings_{year}.npy"
        ids_file = EMBEDDINGS_DIR / f"ids_{year}.parquet"
        
        if emb_file.exists() and ids_file.exists():
            skipped_years.append(year)
            continue
        
        parquet_file = music_ready / f"year={year}" / "data.parquet"
        if not parquet_file.exists():
            print(f"Skipping {year}: no data file")
            continue
        
        print(f"\nProcessing {year}...")
        df = pl.read_parquet(parquet_file)
        
        texts = df["doc_text_music"].to_list()
        track_ids = df["track_id"].to_list()
        
        print(f"  Embedding {len(texts):,} texts...")
        embeddings = embed_texts(model, texts, batch_size=256)
        
        save_embeddings(embeddings, emb_file)
        
        ids_df = pl.DataFrame({"track_id": track_ids})
        ids_df.write_parquet(ids_file)
        print(f"  Saved IDs: {ids_file}")
        
        total_embedded += len(texts)
    
    if skipped_years:
        print(f"\nSkipped existing years: {skipped_years}")
    
    print("\n" + "=" * 50)
    print("EMBEDDING COMPLETE")
    print("=" * 50)
    print(f"Total embedded: {total_embedded:,}")
    print(f"Output: {EMBEDDINGS_DIR}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())