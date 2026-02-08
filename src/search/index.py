import faiss
import numpy as np
import pickle
import os

class VectorIndex:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension) # Inner Product (Cosine Similarity if normalized)
        self.id_map = {} # Internal ID -> Deezer ID

    def build(self, item_factors: np.ndarray, item_ids: list):
        """
        Build index from item factors.
        item_factors: numpy array of shape (n_items, dimension)
        item_ids: list of external IDs corresponding to rows in item_factors
        """
        if item_factors.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.dimension}, got {item_factors.shape[1]}")
            
        # Normalize for cosine similarity
        faiss.normalize_L2(item_factors)
        
        self.index.add(item_factors.astype('float32'))
        self.id_map = {i: item_id for i, item_id in enumerate(item_ids)}

    def search(self, query_vector: np.ndarray, k: int = 50):
        """
        Search for nearest neighbors.
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
            
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector.astype('float32'), k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append((self.id_map[idx], float(dist)))
                
        return results

    def save(self, path: str):
        """Save index and ID map."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path + ".index")
        with open(path + ".map", 'wb') as f:
            pickle.dump(self.id_map, f)

    def load(self, path: str):
        """Load index and ID map."""
        self.index = faiss.read_index(path + ".index")
        with open(path + ".map", 'rb') as f:
            self.id_map = pickle.load(f)
