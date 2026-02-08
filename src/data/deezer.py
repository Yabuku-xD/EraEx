import requests
import pandas as pd
import time
from typing import Dict, Optional, List

class DeezerCollector:
    def __init__(self):
        self.base_url = "https://api.deezer.com"

    def search_track(self, artist: str, track: str) -> Optional[Dict]:
        """Search for a track on Deezer."""
        try:
            # Clean up query strings to improve match rate
            artist = artist.split(' feat.')[0].split(' ft.')[0]
            track = track.split(' (')[0].split(' - ')[0]
            
            url = f"{self.base_url}/search"
            params = {'q': f'artist:"{artist}" track:"{track}"'}
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get('total', 0) > 0:
                return data['data'][0]
        except Exception as e:
            print(f"Error searching for {artist} - {track}: {e}")
            
        return None

    def get_track_details(self, track_id: int) -> Dict:
        """Get detailed metadata for a track."""
        try:
            url = f"{self.base_url}/track/{track_id}"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching details for track {track_id}: {e}")
            return {}

    def enrich_tracks(self, tracks_df: pd.DataFrame, batch_size: int = 100) -> pd.DataFrame:
        """
        Enrich a DataFrame of tracks (must have 'artist' and 'track' columns) 
        with Deezer metadata.
        """
        # Work on unique tracks to save API calls
        unique_tracks = tracks_df[['artist', 'track']].drop_duplicates()
        print(f"enriching {len(unique_tracks)} unique tracks...")
        
        deezer_data = []
        
        for idx, row in unique_tracks.iterrows():
            result = self.search_track(row['artist'], row['track'])
            
            if result:
                meta = {
                    'artist': row['artist'],
                    'track': row['track'],
                    'deezer_id': result['id'],
                    'deezer_title': result['title'],  # Store fetched title for verification
                    'deezer_artist': result['artist']['name'],
                    'album': result['album']['title'],
                    'duration': result['duration'],
                    'rank': result.get('rank', 0),
                    'preview_url': result.get('preview', ''),
                    'bpm': 0.0, # Will need detailed fetch for this if search doesn't return it
                    'release_date': '',
                    'cover_medium': result['album'].get('cover_medium', '')
                }
                
                # Optional: Fetch detailed info for BPM and release date logic if needed
                # For now, keep it fast with just search results
                
                deezer_data.append(meta)
            
            # Rate limiting
            time.sleep(0.05)
            
            if len(deezer_data) % batch_size == 0 and len(deezer_data) > 0:
                print(f"Found matches for {len(deezer_data)} tracks...")

        deezer_df = pd.DataFrame(deezer_data)
        
        print(f"Enrichment complete. Found matches for {len(deezer_df)}/{len(unique_tracks)} tracks.")
        
        if deezer_df.empty:
            return tracks_df
            
        # Merge back to original dataframe
        enriched_df = tracks_df.merge(
            deezer_df, 
            on=['artist', 'track'], 
            how='left'
        )
        

        return enriched_df

    def get_genres(self) -> List[str]:
        """Fetch list of genres from Deezer API."""
        try:
            url = f"{self.base_url}/genre"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            genres = []
            if 'data' in data:
                for item in data['data']:
                    genres.append(item['name'].lower())
                    

            # Add some common sub-genres not always in top-level list
            # but we want to seed these as valid if API misses them
            seed_genres = {'trap', 'lo-fi', 'house', 'techno', 'dubstep', 'ambient', 'rnb', 'hip hop'}
            
            return list(set(genres).union(seed_genres))
        except Exception as e:
            print(f"Error fetching genres: {e}")
            return []
