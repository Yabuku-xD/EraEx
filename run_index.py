from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from config.settings import EMBEDDINGS_DIR, INDEXES_DIR, YEAR_RANGE
from index.faiss_index import build_ivf_pq_index, save_index


def check_existing_output():
    existing_years = []
    for year in YEAR_RANGE:
        index_file = INDEXES_DIR / f"faiss_{year}.index"
        if index_file.exists():
            existing_years.append(year)
    return existing_years


def main():
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    
    existing = check_existing_output()
    if len(existing) == len(YEAR_RANGE):
        print("All FAISS indexes already exist. Skipping.")
        print(f"Existing: {existing}")
        print("Delete data/indexes/ to re-run.")
        return 0
    
    skipped_years = []
    built_years = []
    
    for year in YEAR_RANGE:
        index_file = INDEXES_DIR / f"faiss_{year}.index"
        
        if index_file.exists():
            skipped_years.append(year)
            continue
        
        emb_file = EMBEDDINGS_DIR / f"embeddings_{year}.npy"
        if not emb_file.exists():
            print(f"Skipping {year}: no embeddings file")
            continue
        
        print(f"\nBuilding index for {year}...")
        embeddings = np.load(emb_file)
        print(f"  Loaded {embeddings.shape[0]:,} embeddings ({embeddings.shape[1]}d)")
        
        index = build_ivf_pq_index(embeddings)
        save_index(index, index_file)
        built_years.append(year)
    
    if skipped_years:
        print(f"\nSkipped existing years: {skipped_years}")
    
    print("\n" + "=" * 50)
    print("INDEXING COMPLETE")
    print("=" * 50)
    print(f"Built indexes: {built_years}")
    print(f"Output: {INDEXES_DIR}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())