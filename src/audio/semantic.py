from sentence_transformers import SentenceTransformer
import torch

class SemanticEncoder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the SBERT model.
        'all-MiniLM-L6-v2' is chosen for speed/quality balance (perfect for CPU/Mediocre PC).
        """
        print(f"Loading SBERT model: {model_name}...")
        try:
            self.model = SentenceTransformer(model_name)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)
            print(f"Model loaded on {self.device}.")
        except Exception as e:
            print(f"Error loading SBERT model: {e}")
            self.model = None

    def encode(self, text):
        """
        Encodes a string or list of strings into a vector.
        Returns numpy array.
        """
        if not self.model:
            return None
        
        try:
            # Normalize text
            if isinstance(text, str):
                text = text.strip().lower()
                
            embeddings = self.model.encode(text)
            return embeddings
        except Exception as e:
            print(f"Error encoding text: {e}")
            return None

    def compute_similarity(self, text1, text2):
        """Computes cosine similarity between two texts."""
        if not self.model:
            return 0.0
            
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        
        from sentence_transformers import util
        # Convert back to tensor for util.cos_sim
        return util.cos_sim(emb1, emb2).item()
