import polars as pl
from pathlib import Path
from collections import defaultdict
import json
import re

from config.settings import PROCESSED_DIR, DATA_DIR, YEAR_RANGE


def normalize_tag(tag: str) -> str:
    tag = tag.lower().strip()
    tag = re.sub(r'[^a-z0-9\s-]', '', tag)
    return tag.strip()


def extract_tags(tags_str: str) -> list:
    if not tags_str:
        return []
    separators = r'[,;|/\n]+'
    raw_tags = re.split(separators, tags_str.lower())
    cleaned = []
    junk = {'soundcloud', 'source', 'iphone', 'android', '3rdparty', 'recorder', ''}
    for t in raw_tags:
        t = normalize_tag(t)
        if len(t) > 1 and t not in junk:
            cleaned.append(t)
    return cleaned[:10]


def build_tag_cooccurrence():
    print("Building tag co-occurrence matrix...")
    
    cooccurrence = defaultdict(lambda: defaultdict(int))
    tag_counts = defaultdict(int)
    
    music_ready_dir = PROCESSED_DIR / "music_ready"
    
    total_tracks = 0
    for year in YEAR_RANGE:
        year_path = music_ready_dir / f"year={year}" / "data.parquet"
        if not year_path.exists():
            continue
        
        print(f"Processing {year}...")
        df = pl.read_parquet(year_path, columns=["tags", "genre"])
        
        for row in df.iter_rows(named=True):
            tags = extract_tags(row.get("tags", "") or "")
            genre = row.get("genre", "")
            if genre:
                genre_tag = normalize_tag(genre)
                if genre_tag and len(genre_tag) > 1:
                    tags.append(genre_tag)
            
            tags = list(set(tags))[:10]
            
            for tag in tags:
                tag_counts[tag] += 1
            
            for i, tag1 in enumerate(tags):
                for tag2 in tags[i+1:]:
                    cooccurrence[tag1][tag2] += 1
                    cooccurrence[tag2][tag1] += 1
            
            total_tracks += 1
        
        print(f"  Processed {df.height:,} tracks")
    
    print(f"\nTotal tracks processed: {total_tracks:,}")
    print(f"Unique tags: {len(tag_counts):,}")
    
    min_count = 100
    filtered_tags = {t for t, c in tag_counts.items() if c >= min_count}
    print(f"Tags with {min_count}+ occurrences: {len(filtered_tags):,}")
    
    similar_tags = {}
    for tag in filtered_tags:
        neighbors = cooccurrence.get(tag, {})
        scored = []
        for neighbor, count in neighbors.items():
            if neighbor in filtered_tags and neighbor != tag:
                score = count / (tag_counts[tag] + tag_counts[neighbor] - count + 1)
                if score > 0.05:
                    scored.append((neighbor, round(score, 4)))
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored:
            similar_tags[tag] = scored[:10]
    
    output_path = DATA_DIR / "tag_similarity.json"
    with open(output_path, 'w') as f:
        json.dump(similar_tags, f, indent=2)
    
    print(f"\nSaved to: {output_path}")
    print(f"Tags with similarities: {len(similar_tags):,}")
    
    print("\nSample similarities:")
    samples = ["chill", "electronic", "hip hop", "ambient", "trap"]
    for sample in samples:
        if sample in similar_tags:
            top3 = similar_tags[sample][:3]
            print(f"  {sample}: {top3}")


if __name__ == "__main__":
    build_tag_cooccurrence()