from typing import List, Tuple
import numpy as np
from src.models.als import ALSModel
from src.search.index import VectorIndex

class CandidateGenerator:
    def __init__(self, model_path: str, index_path: str):
        self.model = ALSModel()
        self.model.load(model_path) # Load 'model_path' (pickle file) into self.model
        
        # We need the embedding dimension from the model to init the index
        # But we can just load the index directly if we assume it was built correctly
        # Actually VectorIndex.load() doesn't need dimension
        self.index = VectorIndex(dimension=64) # Dimension placeholder, will be overwritten by read_index
        self.index.load(index_path)

    def get_candidates(self, user_id: int, k: int = 500) -> List[Tuple[int, float]]:
        """
        Get candidates for a user using their vector from ALS model.
        Returns list of (track_id, score).
        """
        try:
            user_vector = self.model.get_user_vector(user_id)
            candidates = self.index.search(user_vector, k=k)
            return candidates
        except Exception as e:
            print(f"Error getting candidates for user {user_id}: {e}")
            return []

    def get_similar_items(self, item_id: int, k: int = 50) -> List[Tuple[int, float]]:
        """
        Get similar items (item-to-item).
        """
        try:
            item_vector = self.model.get_item_vector(item_id)
            candidates = self.index.search(item_vector, k=k)
            return candidates
        except Exception as e:
            print(f"Error getting similar items for item {item_id}: {e}")
            return []
