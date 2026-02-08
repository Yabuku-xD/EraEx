import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio.processor import AudioProcessor
import numpy as np

# Test URL (Deezer Preview for "Get Lucky" - Daft Punk)
# Or we can search for one. Let's use a hardcoded one or fetch one dynamically.
# Hardcoded valid preview URL (as of 2024, might expire, but usually stable)
# Actually, let's fetch a live one to be safe.
import requests

def get_test_url(query="Daft Punk Get Lucky"):
    resp = requests.get("https://api.deezer.com/search", params={'q': query, 'limit': 1})
    data = resp.json().get('data', [])
    if data:
        return data[0]['preview'], data[0]['title']
    return None, None

def test():
    print("Initialize AudioProcessor...")
    processor = AudioProcessor()
    
    print("Fetching test track...")
    url, title = get_test_url()
    if not url:
        print("Could not find test track.")
        return

    print(f"Analyzing: {title}")
    print(f"URL: {url}")
    
    result = processor.analyze_url(url)
    
    if result:
        vec = result['vector']
        print("\n--- Audio Analysis Result ---")
        print(f"Vector Shape: {vec.shape} (Should be ~33)")
        print(f"Detected Tempo: {result['details']['tempo']:.2f} BPM")
        print(f"Vector Preview: {vec[:5]}...")
        print("Success! Audio pipeline is working.")
    else:
        print("Analysis failed.")

if __name__ == "__main__":
    test()
