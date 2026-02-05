import nltk
from nltk.corpus import wordnet as wn

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def get_all_genres():
    genres = set()
    try:
        # Start from 'musical_style' (music genre)
        base = wn.synset('musical_style.n.01')
        # Get full closure of hyponyms (sub-genres)
        for syn in base.closure(lambda s: s.hyponyms()):
            for lemma in syn.lemmas():
                name = lemma.name().replace('_', ' ').lower()
                genres.add(name)
    except Exception as e:
        print(f"Error: {e}")
    return sorted(list(genres))

genres = get_all_genres()
print(f"Found {len(genres)} genres from WordNet")
print(f"Examples: {genres[:10]}")

# Test if key genres are there
check = ['rock', 'pop', 'jazz', 'hip hop', 'techno']
found = [g for g in check if g in genres or any(g in x for x in genres)]
print(f"Direct Matches: {found}")