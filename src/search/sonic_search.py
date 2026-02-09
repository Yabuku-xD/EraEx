import pickle
import numpy as np
import os
import glob
import random
from src.ranking.nostalgia import NostalgiaFilter
from src.audio.semantic import SemanticEncoder

class SonicSearch:
    def __init__(self, index_dir='data/indices'):
        self.index_dir = index_dir
        self.data = []
        self.nostalgia = NostalgiaFilter()
        print("Loading Semantic Encoder for Search...")
        self.encoder = SemanticEncoder() 
        self.load_index()

    def load_index(self):
        """Loads and merges all sonic_*.pkl files from the index directory."""
        pattern = os.path.join(self.index_dir, 'sonic_*.pkl')
        files = glob.glob(pattern)
        
        if not files:
            old_path = os.path.join(self.index_dir, 'sonic_index.pkl')
            if os.path.exists(old_path):
                files = [old_path]
        
        if not files:
            print("No Sonic Index files found. Please build them first.")
            self.data = []
            return
            
        for f in sorted(files):
            try:
                with open(f, 'rb') as fp:
                    year_data = pickle.load(fp)
                    self.data.extend(year_data)
                    print(f"  Loaded {os.path.basename(f)}: {len(year_data)} tracks")
            except Exception as e:
                print(f"  Error loading {f}: {e}")
        
        print(f"Sonic Index Total: {len(self.data)} tracks from {len(files)} file(s).")


    def search_by_text(self, query_text, limit=10):
        """Encodes text query and searches for semantic matches."""
        target_vec = self.encoder.encode(query_text)
        # We use alpha=0.0 (100% Semantic) for text queries
        # But we pass the raw text for Keyword Boosting (Hybrid Search)
        return self.search_by_vector(target_audio_vec=None, target_semantic_vec=target_vec, limit=limit, alpha=0.0, boost_text=query_text)

    def _cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def search_by_vector(self, target_audio_vec, target_semantic_vec, limit=10, alpha=0.5, boost_text=None):
        """
        Finds nearest neighbors.
        alpha: Weight for Audio (0.0 to 1.0). 1.0 = Audio Only, 0.0 = Semantic Only.
        """
        scored_candidates = []
        
        for track in self.data:
            # 1. NOSTALGIA CHECK (Strict Guardrail)
            # Even if the index was built carefully, we check again.
            if not self.nostalgia.is_in_era(track.get('release_date')):
                continue

            # ---------------------------------------------------------
            # ERA-WEIGHTED TANIMOTO FUSION (Proprietary Algorithm)
            # ---------------------------------------------------------
            # Instead of basic Cosine Similarity, we use Tanimoto Coefficient
            # which is a generalized Jaccard Index for continuous vectors.
            # Formula: T(A,B) = (A . B) / (||A||^2 + ||B||^2 - A . B)
            # This penalizes magnitude differences more strictly than Cosine.
            # ---------------------------------------------------------

            t_score = 0.0
            
            # 1. Semantic Tanimoto (The "Meaning")
            if target_semantic_vec is not None and 'semantic_vector' in track:
                v1, v2 = target_semantic_vec, track['semantic_vector']
                dot = np.dot(v1, v2)
                mag1 = np.dot(v1, v1)
                mag2 = np.dot(v2, v2)
                # Tanimoto Similarity
                s_sim = dot / (mag1 + mag2 - dot + 1e-9) 
                t_score += s_sim * (1 - alpha)
                
            # 2. Audio Tanimoto (The "Vibe")
            if target_audio_vec is not None and 'audio_vector' in track:
                v1, v2 = target_audio_vec, track['audio_vector']
                dot = np.dot(v1, v2)
                mag1 = np.dot(v1, v1)
                mag2 = np.dot(v2, v2)
                a_sim = dot / (mag1 + mag2 - dot + 1e-9)
                t_score += a_sim * alpha

            # 3. Era-Weighted Temporal Decay
            # We slightly boost tracks from "Peak Era" (2014-2016) or penalize edge years
            try:
                track_year = int(track.get('release_date', '2015')[:4])
                # Gaussian decay from 2016 (Peak Nostalgia)
                temporal_weight = 1.0 + 0.1 * np.exp(-0.5 * ((track_year - 2016)**2))
                t_score *= temporal_weight
            except:
                pass

            # 4. Keyword Boosting (Hybrid Search)
            # If the user typed "PARTYNEXTDOOR", we want to heavily boost tracks
            # where the Artist Name contains "partynextdoor".
            # This fixes SBERT drifting to "Party" keyworks generally.
            if boost_text:
                q_norm = boost_text.lower()
                # Check Artist
                t_artist = track.get('artist', '')
                if isinstance(t_artist, dict): t_artist = t_artist.get('name', '')
                t_artist_norm = str(t_artist).lower()
                
                if q_norm in t_artist_norm:
                    t_score += 2.0 # Massive conditional boost for Artist Match
                
                # Check Title (smaller boost)
                t_title_norm = track.get('title', '').lower()
                if q_norm in t_title_norm:
                    t_score += 0.5
                
            scored_candidates.append((t_score, track))
            
        # Sort desc
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Deduplication & Fetching
        import requests
        results = []
        seen_keys = set()
        
        # We iterate through more candidates to fill the limit after dedup
        for s, t in scored_candidates:
            if len(results) >= limit:
                break
                
            # Create a dedup key (Title + Artist)
            # Normalize: lowercase, remove punctuation
            title_norm = ''.join(c for c in t.get('title', '').lower() if c.isalnum())
            artist_norm = ''.join(c for c in t.get('artist', '').lower() if c.isalnum())
            dedup_key = f"{title_norm}_{artist_norm}"
            
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            track_id = t['id']
            cover_url = ''
            preview_url = t.get('preview', '')
            try:
                r = requests.get(f'https://api.deezer.com/track/{track_id}', timeout=2)
                if r.status_code == 200:
                    data = r.json()
                    cover_url = data.get('album', {}).get('cover_medium', '')
                    preview_url = data.get('preview', preview_url)
            except:
                pass
            results.append({
                'id': track_id,
                'title': t['title'],
                'title_short': t.get('title_short', t['title']),
                'artist': {'name': t['artist']} if isinstance(t['artist'], str) else t['artist'],
                'album': {'cover_medium': cover_url},
                'score': float(s),
                'preview': preview_url
            })
        return results

    def serendipity_search(self, limit=5):
        """
        The 'Feeling Lucky' Engine.
        Generates a random audio vector and finds closest matches.
        """
        if not self.data:
            return []
            
        # 1. Generate Dream Vector
        # We look at the first track to get dimensions
        sample_vec = self.data[0]['audio_vector']
        dims = sample_vec.shape[0]
        
        # Create random vector (Gaussian distribution)
        dream_vec = np.random.normal(0, 1, size=dims)
        
        # 2. Search (Audio Only)
        # We want to find tracks that sound like this random 'dream'
        return self.search_by_vector(dream_vec, None, limit=limit, alpha=1.0)
