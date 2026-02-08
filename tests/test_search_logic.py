
import sys
import os
sys.path.append(os.getcwd())
from src.ranking.nostalgia import NostalgiaFilter

nf = NostalgiaFilter()

queries = [
    "chill trap dark rnb", # Should remain GENERAL, not just "chill"
    "I'm feeling sad",     # Should detect MOOD: sad
    "Sad songs",           # Should detect MOOD: sad
    "Partynextdoor",       # Should remain GENERAL/ARTIST
    "love songs",          # Should detect MOOD: love
    "driving late at night with vibes" # Complex sentence -> Sentiment or General
]

print("Testing Search Logic:")
for q in queries:
    res = nf.get_search_query(q)
    print(f"Query: '{q}'\n -> Type: {res['type']}\n -> Term: {res['query']}\n")
    if res['type'] == 'general':
        extracted = nf.extract_genres(q)
        print(f" -> Extracted Genres for Fallback: {extracted}\n")
