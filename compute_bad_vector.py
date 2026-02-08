import numpy as np
from sentence_transformers import SentenceTransformer
from src.config.settings import BLOCKLIST, SBERT_MODEL_NAME
import json

def compute():
    print("Loading config...")
    bad_concepts = BLOCKLIST.get("bad_concepts", "")
    if not bad_concepts:
        print("No bad_concepts found!")
        return

    print(f"Loading SBERT model: {SBERT_MODEL_NAME}...")
    model = SentenceTransformer(SBERT_MODEL_NAME)
    
    print(f"Encoding concepts: {bad_concepts[:50]}...")
    vec = model.encode(bad_concepts, convert_to_numpy=True)
    vec = vec / np.linalg.norm(vec) # Normalize
    
    output_path = "data/bad_vector.npy"
    np.save(output_path, vec)
    print(f"Saved normalized vector to {output_path}")

if __name__ == "__main__":
    compute()