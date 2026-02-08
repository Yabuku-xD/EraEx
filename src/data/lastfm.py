import requests
import pandas as pd
import time
import os
from typing import List, Dict, Optional

class LastFMCollector:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://ws.audioscrobbler.com/2.0/"

    def get_user_tracks(self, user: str, limit: int = 200, page: int = 1) -> Dict:
        """Fetch recent tracks for a Last.fm user."""
        params = {
            'method': 'user.getRecentTracks',
            'user': user,
            'api_key': self.api_key,
            'format': 'json',
            'limit': limit,
            'page': page
        }
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        return response.json()

    def collect_user_history(self, user: str, max_pages: int = 200) -> pd.DataFrame:
        """Collect full listening history for a user."""
        all_tracks = []
        
        print(f"Collecting history for user: {user}")
        
        for page in range(1, max_pages + 1):
            try:
                data = self.get_user_tracks(user, page=page)
                
                if 'recenttracks' not in data or 'track' not in data['recenttracks']:
                    break
                    
                tracks = data['recenttracks']['track']
                
                if not tracks:
                    break
                
                for track in tracks:
                    if 'date' in track:  # Skip currently playing
                        all_tracks.append({
                            'user': user,
                            'artist': track['artist']['#text'],
                            'track': track['name'],
                            'album': track['album']['#text'],
                            'timestamp': int(track['date']['uts'])
                        })
                
                # Check if we've reached the last page
                attr = data['recenttracks'].get('@attr', {})
                total_pages = int(attr.get('totalPages', 0))
                
                if page >= total_pages:
                    break
                    
                time.sleep(0.2) # Rate limiting
                
            except Exception as e:
                print(f"Error on page {page} for user {user}: {e}")
                break
        
        return pd.DataFrame(all_tracks)

    def get_top_tags(self, artist: str, track: str) -> List[str]:
        """Get top tags for a track."""
        try:
            params = {
                'method': 'track.getTopTags',
                'artist': artist,
                'track': track,
                'api_key': self.api_key,
                'format': 'json'
            }
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if 'toptags' in data and 'tag' in data['toptags']:
                return [tag['name'] for tag in data['toptags']['tag'][:5]]
        except Exception:
            pass
        return []
