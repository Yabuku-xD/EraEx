import sys
import argparse
from pathlib import Path
import json
import logging

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import settings
from src.recommendation.recommendation_engine import ColdStartHandler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Run this operation.
def main():
    """
    Run the command entry point.
    
    This function implements the main step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    parser = argparse.ArgumentParser(description="Run SOTA Cold Start Recommendation Pipeline")
    parser.add_argument("--user_id", type=str, help="User ID to simulate (optional)")
    parser.add_argument("--liked", type=str, nargs="*", help="List of liked song IDs")
    parser.add_argument("--played", type=str, nargs="*", help="List of played song IDs")
    parser.add_argument("--mock", action="store_true", help="Run in Mock Mode (No Embeddings)")
    args = parser.parse_args()
    if args.mock:
        logger.info("[MODE] Mocking Embeddings & Data for Testing...")
        settings.MOCK_MODE = True
    logger.info("Initializing Cold Start Handler (BGE-M3 + DPP + Overlap)...")
    try:
        if args.mock:
            pass
        handler = ColdStartHandler()
    except Exception as e:
        logger.error(f"Failed to initialize handler: {e}")
        return
    liked = args.liked or []
    played = args.played or []
    if len(liked) == 1 and liked[0].startswith("["):
        liked = json.loads(liked[0])
    if len(played) == 1 and played[0].startswith("["):
        played = json.loads(played[0])
    if not liked and (not played):
        logger.info("Generating Pure Cold Start Recommendations...")
        recs = handler.get_cold_start_items(k=10)
    else:
        logger.info(
            f"Generating Hybrid Recommendations (History: {len(liked)} likes, {len(played)} plays)..."
        )
        recs = handler.recommend(liked, played, k=10)
    print(f"\nTop 10 Recommendations:")
    print("-" * 50)
    for i, r in enumerate(recs):
        print(f"{i + 1}. {r['title']} - {r['artist']}")
        print(f"    Score: {r['score']:.4f} | Source: {r['source']}")
        print(f"    Cover: {r['cover_url']}")
    print("-" * 50)


if __name__ == "__main__":
    main()