import pickle
import numpy as np
import os
import random
from src.ranking.nostalgia import NostalgiaFilter
from src.audio.semantic import SemanticEncoder

class SonicSearch:
    def __init__(self, index_path='data/indices/sonic_index.pkl'):
        self.index_path = index_path
        self.data = []
        self.nostalgia = NostalgiaFilter()
        print("Loading Semantic Encoder for Search...")
        self.encoder = SemanticEncoder() 
        self.load_index()

    def load_index(self):
        """Loads the list of dicts from pickle."""
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, 'rb') as f:
                    self.data = pickle.load(f)
                print(f"Sonic Index loaded: {len(self.data)} tracks.")
            except Exception as e:
                print(f"Error loading index: {e}")
                self.data = []
        else:
            print("Sonic Index not found. Please build it first.")
            self.data = []

    def search_by_text(self, query_text, limit=10):
        """Encodes text query and searches for semantic matches."""
        target_vec = self.encoder.encode(query_text)
        # We use alpha=0.0 (100% Semantic) for text queries
        # Unless we later support 'audio query'
        return self.search_by_vector(target_audio_vec=None, target_semantic_vec=target_vec, limit=limit, alpha=0.0)

    def _cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def search_by_vector(self, target_audio_vec, target_semantic_vec, limit=10, alpha=0.5):
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

            score = 0.0
            
            # Audio Score
            if target_audio_vec is not None and 'audio_vector' in track:
                a_score = self._cosine_similarity(target_audio_vec, track['audio_vector'])
                score += a_score * alpha
                
            # Semantic Score
            if target_semantic_vec is not None and 'semantic_vector' in track:
                s_score = self._cosine_similarity(target_semantic_vec, track['semantic_vector'])
                score += s_score * (1 - alpha)
                
            scored_candidates.append((score, track))
            
        # Sort desc
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Return top N
        return [
            {
                'id': t['id'],
                'title': t['title'],
                'artist': t['artist'],
                'score': float(s), # Cast numpy float to native float
                'preview': t.get('preview')
            }
            for s, t in scored_candidates[:limit]
        ]

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
