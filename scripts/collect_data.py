import os
import sys
import argparse
import pandas as pd
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.lastfm import LastFMCollector
from src.data.deezer import DeezerCollector

def main():
    parser = argparse.ArgumentParser(description="Collect and enrich music data")
    parser.add_argument("--users", type=str, help="Comma-separated list of Last.fm usernames")
    parser.add_argument("--limit", type=int, default=200, help="Max pages per user")
    parser.add_argument("--output", type=str, default="data/raw/scrobbles.parquet", help="Output path")
    
    args = parser.parse_args()
    
    load_dotenv()
    api_key = os.getenv("LASTFM_API_KEY")
    if not api_key:
        print("Error: LASTFM_API_KEY not found in environment variables.")
        return

    users = args.users.split(",") if args.users else ["rj"] # Default to 'rj' (Last.fm founder) for testing
    
    # 1. Collect User History
    print(f"Starting collection for {len(users)} users...")
    collector = LastFMCollector(api_key)
    
    dfs = []
    for user in users:
        df = collector.collect_user_history(user.strip(), max_pages=args.limit)
        dfs.append(df)
        
    if not dfs:
        print("No data collected.")
        return

    all_scrobbles = pd.concat(dfs, ignore_index=True)
    print(f"Collected {len(all_scrobbles)} scrobbles.")
    
    # 2. Enrich with Deezer Metadata
    print("Enriching with Deezer metadata...")
    deezer = DeezerCollector()
    enriched_df = deezer.enrich_tracks(all_scrobbles) # Calls enrich_tracks method in DeezerCollector class
    
    # 3. Save
    output_dir = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    enriched_df.to_parquet(args.output)
    print(f"Saved enriched data to {args.output}")

if __name__ == "__main__":
    main()
