import argparse
import json
import os
import shutil
import sqlite3
import time
from pathlib import Path

from src.core.media_metadata import (
    YouTubeMetadataEnricher,
    as_optional_bool,
)


# Enrich metadata with ytdlp.
def enrich_metadata_with_ytdlp(
    metadata_path: Path,
    limit=0,
    start=0,
    only_missing=True,
    save_every=200,
    sleep_ms=0,
):
    """
    Execute enrich metadata with ytdlp.
    
    This function implements the enrich metadata with ytdlp step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    if not metadata_path.exists():
        print(f"[FAIL] Missing metadata file: {metadata_path}")
        return
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    track_ids = list(metadata.keys())
    if start > 0:
        track_ids = track_ids[start:]
    if limit > 0:
        track_ids = track_ids[:limit]

    enricher = YouTubeMetadataEnricher()
    print(
        f"[INFO] Loaded {len(metadata)} tracks. "
        f"Processing {len(track_ids)} from offset {start}. yt-dlp enabled={enricher.enabled}"
    )

    updated = 0
    visited = 0
    for track_id in track_ids:
        visited += 1
        meta = metadata.get(track_id, {})
        has_description = bool(str(meta.get("description", "") or "").strip())
        has_instrumental = as_optional_bool(meta.get("instrumental")) is not None
        if only_missing and has_description and has_instrumental:
            continue

        enriched = enricher.enrich(
            track_id=track_id,
            title=str(meta.get("title", "") or ""),
            artist=str(meta.get("artist_name", "") or ""),
        )
        if enriched:
            if enriched.get("description"):
                meta["description"] = str(enriched.get("description", ""))
            if "instrumental" in enriched:
                meta["instrumental"] = bool(enriched.get("instrumental"))
            if "instrumental_confidence" in enriched:
                meta["instrumental_confidence"] = float(
                    enriched.get("instrumental_confidence", 0.0)
                )
            meta["description_source"] = str(enriched.get("source", "yt-dlp"))
            metadata[track_id] = meta
            updated += 1

        if updated > 0 and updated % save_every == 0:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False)
            print(f"[INFO] Saved checkpoint. visited={visited} updated={updated}")

        if sleep_ms > 0:
            time.sleep(float(sleep_ms) / 1000.0)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
    print(f"[DONE] visited={visited} updated={updated} file={metadata_path}")


# Extract audio features.
def extract_audio_features(
    csv_path: Path,
    cache_path: Path,
    temp_audio_dir: Path,
    save_interval=50,
):
    """
    Extract audio features.
    
    This function implements the extract audio features step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    import pandas as pd
    from tqdm import tqdm
    from src.core.audio_processing import ActivationSteering, AudioDownloader, AudioTransformer

    temp_audio_dir.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  AUDIO FEATURE EXTRACTION (Storage Efficient)")
    print("  Downloads -> Extracts -> Deletes Audio")
    print("=" * 60)
    print(f"Loading dataset from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} tracks.")
    except Exception as exc:
        print(f"Failed to load dataset: {exc}")
        return

    audio_features = {}
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                audio_features = json.load(f)
            print(f"Resuming from {len(audio_features)} processed items.")
        except Exception:
            print("Cache file corrupted or empty. Starting fresh.")

    downloader = AudioDownloader(str(temp_audio_dir))
    model = AudioTransformer()
    steering = ActivationSteering(model)
    steering.load_dummy_vectors()

    process_df = df.sort_values("Deezer Rank") if "Deezer Rank" in df.columns else df
    changes_made = 0
    print("\nStarting processing loop...")
    for _, row in tqdm(process_df.iterrows(), total=len(df)):
        track_id = str(row["Track ID"])
        if track_id in audio_features:
            continue
        query = f"{row['Artist Name']} - {row['Track Title']}"
        try:
            path = downloader.download_preview(query, track_id)
            if path and os.path.exists(path):
                try:
                    feats = steering.extract_features(path)
                    audio_features[track_id] = {key: float(value) for key, value in feats.items()}
                except Exception:
                    audio_features[track_id] = {}
                try:
                    os.remove(path)
                except Exception:
                    pass
            else:
                audio_features[track_id] = {}

            changes_made += 1
            if changes_made >= save_interval:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(audio_features, f)
                changes_made = 0
        except KeyboardInterrupt:
            print("\nStopping gracefully...")
            break
        except Exception as exc:
            print(f"Error on {track_id}: {exc}")
            audio_features[track_id] = {}

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(audio_features, f)
    print(f"\nSaved {len(audio_features)} features to {cache_path}")
    print("Done.")


# Run recommendation simulation.
def run_recommendation_simulation():
    """
    Run recommendation simulation.
    
    This function implements the run recommendation simulation step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    from src.recommendation.recommendation_engine import ColdStartHandler

    print("Initializing Cold Start Handler...")
    handler = ColdStartHandler()
    if not handler.metadata:
        print("Error: Metadata not loaded. Cannot run simulation.")
        return

    bots = [
        {"name": "2015 Fan", "type": "era", "target": 2015, "tol": 1},
        {"name": "Pop Fan", "type": "mood", "target": "pop"},
        {"name": "Rock Fan", "type": "mood", "target": "rock"},
    ]
    results = {}

    for bot in bots:
        print(f"\nRunning simulation for {bot['name']}...")
        history = []
        mrr_scores = []
        for step in range(5):
            print(f"  Step {step}: generating recommendations...")
            recs = handler.recommend(liked_ids=history, played_ids=[], k=10)
            relevant_indices = []
            for i, item in enumerate(recs):
                sid = item["id"]
                meta = handler.sim_manager.get_track_info(sid)
                is_relevant = False
                if bot["type"] == "era":
                    year = meta.get("year", 0)
                    try:
                        if abs(int(year) - bot["target"]) <= bot["tol"]:
                            is_relevant = True
                    except Exception:
                        pass
                elif bot["type"] == "mood":
                    tags = meta.get("deezer_tags", [])
                    if isinstance(tags, str):
                        tags = json.loads(tags.replace("'", '"'))
                    if any((bot["target"] in str(tag).lower() for tag in tags)):
                        is_relevant = True
                if is_relevant:
                    relevant_indices.append(i)

            if relevant_indices:
                first_rank = relevant_indices[0] + 1
                mrr = 1.0 / first_rank
                clicked_id = recs[relevant_indices[0]]["id"]
                history.append(clicked_id)
                print(
                    f"    -> Bot clicked '{recs[relevant_indices[0]]['title']}' (Rank {first_rank})"
                )
            else:
                mrr = 0.0
                if recs:
                    history.append(recs[0]["id"])
                    print(f"    -> Bot clicked '{recs[0]['title']}' (fallback)")

            mrr_scores.append(mrr)

        avg_mrr = float(sum(mrr_scores) / len(mrr_scores)) if mrr_scores else 0.0
        results[bot["name"]] = avg_mrr
        print(f"  Final MRR: {avg_mrr:.4f}")

    print("\nSimulation Results (MRR over 5 steps):")
    for name, score in results.items():
        print(f"{name}: {score:.4f}")


# Run subset pipeline test.
def run_subset_pipeline_test(csv_path: Path, subset_size=5, audio_dir=Path("data/audio_test_subset")):
    """
    Run subset pipeline test.
    
    This function implements the run subset pipeline test step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity
    from tqdm import tqdm

    from src.core.audio_processing import ActivationSteering, AudioDownloader, AudioTransformer
    from src.core.media_metadata import build_track_embedding_text
    from src.core.text_embeddings import embedding_handler

    print("=" * 60)
    print("  REL DATA SUBSET PIPELINE TEST (Real Models)")
    print("=" * 60)
    print("\n[1/5] Loading dataset...")
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} tracks.")
    except Exception as exc:
        print(f"Failed to load dataset: {exc}")
        return

    print(f"\n[2/5] Selecting top {subset_size} items (by rank)...")
    subset = df.sort_values("Deezer Rank").head(subset_size) if "Deezer Rank" in df.columns else df.head(subset_size)
    for _, row in subset.iterrows():
        print(f" - {row['Track Title']} by {row['Artist Name']}")

    print("\n[3/5] Generating BGE-M3 embeddings...")
    texts = []
    for _, row in subset.iterrows():
        try:
            tags = json.loads(row.get("Deezer Tags", "[]").replace("'", '"'))
        except Exception:
            tags = []
        instrumental = as_optional_bool(
            row.get("Instrumental", row.get("instrumental", ""))
        )
        texts.append(
            build_track_embedding_text(
                title=str(row.get("Track Title", row.get("title", "Unknown"))),
                artist=str(row.get("Artist Name", row.get("artist_name", "Unknown"))),
                tags=tags,
                description=str(
                    row.get(
                        "Description",
                        row.get("description", ""),
                    )
                    or ""
                ),
                instrumental=instrumental,
            )
        )
    try:
        embeddings = embedding_handler.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings shape: {embeddings.shape}")
    except Exception as exc:
        print(f"[ERROR] Embedding generation failed: {exc}")
        return

    print("\n[4/5] Processing audio (download + steering)...")
    downloader = AudioDownloader(str(audio_dir))
    model = AudioTransformer()
    steering = ActivationSteering(model)
    steering.load_dummy_vectors()
    for _, row in tqdm(subset.iterrows(), total=len(subset)):
        track_id = str(row["Track ID"])
        query = f"{row['Artist Name']} - {row['Track Title']}"
        audio_path = downloader.download_preview(query, track_id)
        if audio_path and os.path.exists(audio_path):
            try:
                feats = steering.extract_features(audio_path)
                print(f"    {track_id}: {feats}")
            except Exception as exc:
                print(f"    {track_id}: extraction failed: {exc}")
        else:
            print(f"    {track_id}: download failed")

    print("\n[5/5] Pairwise similarity matrix:")
    sim_matrix = cosine_similarity(embeddings)
    print(sim_matrix)
    print("\n[SUCCESS] Subset pipeline test complete.")


# Clean unused artifacts.
def clean_unused_artifacts(
    root: Path,
    delete_dirs=None,
    remove_pycache=True,
    prune_stale_api_cache=True,
    dry_run=False,
):
    """
    Execute clean unused artifacts.
    
    This function implements the clean unused artifacts step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    delete_dirs = delete_dirs or [
        root / "data" / "audio_temp",
        root / "data" / "audio_test_subset",
        root / "data" / "audio_test_tmp",
        root / "data" / "audio_cache",
        root / "__pycache__",
        root / "scripts" / "__pycache__",
        root / "tests" / "__pycache__",
        root / "src" / "encoding" / "__pycache__",
        root / "src" / "core" / "__pycache__",
        root / "src" / "recommendation" / "__pycache__",
        root / "src" / "pipeline" / "__pycache__",
        root / "src" / "search" / "__pycache__",
        root / "src" / "api" / "__pycache__",
        root / "src" / "user" / "__pycache__",
    ]

    removed = []
    for directory in delete_dirs:
        if not directory.exists():
            continue
        if dry_run:
            print(f"[DRY-RUN] would remove: {directory}")
            removed.append(str(directory))
            continue
        shutil.rmtree(directory, ignore_errors=True)
        print(f"[REMOVED] {directory}")
        removed.append(str(directory))

    if remove_pycache:
        for pyc in root.rglob("*.pyc"):
            if dry_run:
                print(f"[DRY-RUN] would remove: {pyc}")
                removed.append(str(pyc))
                continue
            try:
                pyc.unlink()
                removed.append(str(pyc))
            except Exception:
                pass

    if prune_stale_api_cache:
        api_cache_path = root / "data" / "cache" / "api_cache.db"
        if api_cache_path.exists():
            should_remove = False
            try:
                conn = sqlite3.connect(str(api_cache_path))
                table_row = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='ytdlp_track_meta'"
                ).fetchone()
                conn.close()
                should_remove = table_row is None
            except Exception:
                should_remove = True

            if should_remove:
                if dry_run:
                    print(f"[DRY-RUN] would remove stale api cache: {api_cache_path}")
                    removed.append(str(api_cache_path))
                else:
                    try:
                        api_cache_path.unlink()
                        print(f"[REMOVED] stale api cache: {api_cache_path}")
                        removed.append(str(api_cache_path))
                    except Exception:
                        pass

    print(f"[DONE] removed_entries={len(removed)}")
    return removed


# Run this operation.
def main():
    """
    Run the command entry point.
    
    This function implements the main step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    parser = argparse.ArgumentParser(description="EraEx maintenance CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    enrich = sub.add_parser("enrich", help="Enrich metadata with yt-dlp-derived fields.")
    enrich.add_argument("--metadata", type=Path, default=Path("data/indexes/metadata.json"))
    enrich.add_argument("--limit", type=int, default=0)
    enrich.add_argument("--start", type=int, default=0)
    enrich.add_argument("--only-missing", action="store_true")
    enrich.add_argument("--save-every", type=int, default=200)
    enrich.add_argument("--sleep-ms", type=int, default=0)

    audio_features = sub.add_parser(
        "audio-features",
        help="Download previews and extract audio feature cache.",
    )
    audio_features.add_argument("--csv", type=Path, default=Path("data/EraEx_Dataset_Final.csv"))
    audio_features.add_argument("--cache", type=Path, default=Path("data/cache/audio_features.json"))
    audio_features.add_argument("--temp-dir", type=Path, default=Path("data/audio_temp"))
    audio_features.add_argument("--save-every", type=int, default=50)

    simulate = sub.add_parser("simulate", help="Run cold-start recommendation simulation.")

    subset = sub.add_parser("subset-test", help="Run small end-to-end subset pipeline test.")
    subset.add_argument("--csv", type=Path, default=Path("data/EraEx_Dataset_Final.csv"))
    subset.add_argument("--subset-size", type=int, default=5)
    subset.add_argument("--audio-dir", type=Path, default=Path("data/audio_test_subset"))

    clean = sub.add_parser(
        "clean",
        help="Remove temp/unused local artifacts (audio temp folders, __pycache__, .pyc).",
    )
    clean.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.command == "enrich":
        enrich_metadata_with_ytdlp(
            metadata_path=args.metadata,
            limit=args.limit,
            start=args.start,
            only_missing=bool(args.only_missing),
            save_every=max(args.save_every, 1),
            sleep_ms=max(args.sleep_ms, 0),
        )
        return

    if args.command == "audio-features":
        extract_audio_features(
            csv_path=args.csv,
            cache_path=args.cache,
            temp_audio_dir=args.temp_dir,
            save_interval=max(args.save_every, 1),
        )
        return

    if args.command == "simulate":
        run_recommendation_simulation()
        return

    if args.command == "subset-test":
        run_subset_pipeline_test(
            csv_path=args.csv,
            subset_size=max(args.subset_size, 1),
            audio_dir=args.audio_dir,
        )
        return

    if args.command == "clean":
        clean_unused_artifacts(root=Path.cwd(), dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()