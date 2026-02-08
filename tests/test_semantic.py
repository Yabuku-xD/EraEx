import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio.semantic import SemanticEncoder 

def test():
    print("Initializing Semantic Encoder...")
    encoder = SemanticEncoder()
    
    if not encoder.model:
        print("Failed to load model.")
        return

    # Test Case 1: Lyric Encoding
    lyrics = "I'm up all night to get lucky, she's up all night to the sun"
    print(f"\nEncoding Lyrics: '{lyrics}'")
    vec = encoder.encode(lyrics)
    print(f"Vector Shape: {vec.shape} (Should be 384 for MiniLM)")
    
    # Test Case 2: Similarity
    t1 = "Heavy metal song about saving the whales"
    t2 = "Aggressive rock track regarding semantic environmentalism"
    t3 = "Happy pop song about dancing"
    
    print("\nComputing Semantic Similarity:")
    sim1 = encoder.compute_similarity(t1, t2)
    print(f"'{t1}' vs '{t2}': {sim1:.4f} (High?)")
    
    sim2 = encoder.compute_similarity(t1, t3)
    print(f"'{t1}' vs '{t3}': {sim2:.4f} (Low?)")
    
    if sim1 > sim2:
        print("\nSUCCESS: SBERT correctly identified the semantic context!")
    else:
        print("\nFAILURE: SBERT failed context check.")

if __name__ == "__main__":
    test()
