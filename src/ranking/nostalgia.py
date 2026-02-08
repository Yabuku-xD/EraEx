
from typing import List, Dict, Optional
import datetime
from textblob import TextBlob
import nltk
from nltk.corpus import wordnet
from src.data.deezer import DeezerCollector

class NostalgiaFilter:
    def __init__(self, start_year: int = 2012, end_year: int = 2018):
        self.start_year = start_year
        self.end_year = end_year
        
        # Initialize Deezer Collector to fetch official genres
        self.deezer = DeezerCollector()
        
        # DYNAMIC: Fetch genres from API
        print("Fetching genres from Deezer API...")
        self.valid_genres = set(self.deezer.get_genres())
        print(f"Loaded {len(self.valid_genres)} genres.")
        
        # Ensure NLTK data is available (wordnet)
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')

    def get_search_query(self, user_query: str) -> dict:
        """
        Parse user query to determine intent (Mood, Artist, or General).
        Returns a dict with 'type' and 'query'.
        """
        user_query = user_query.lower().strip()
        words = user_query.split()
        
        # 0. Check for complex descriptive queries with known genres
        extracted_genres = self.extract_genres(user_query)
        if len(extracted_genres) > 0 and len(words) > 2:
             # If we found known genres in a long query, use the original query (let Deezer handle it)
             # or separate it logic if we wanted strictly genre based.
             # For now, let's treat "chill trap" as General but we know extracted genres exists for fallback.
             return {'type': 'general', 'query': user_query, 'original': user_query}

        # 1. Check for Mood/Sentiment using WordNet & TextBlob
        # Instead of hardcoded keywords, we check if the words are "feelings"
        
        blob = TextBlob(user_query)
        
        # Heuristic: If short query (1-2 words) and it's an ADJECTIVE or related to 'feeling'
        # we treat it as a mood.
        if len(words) <= 2:
            for word in words:
                # Check synsets for "feeling", "emotion", or if it is an adjective
                synsets = wordnet.synsets(word)
                is_adj = any(s.pos() == 'a' or s.pos() == 's' for s in synsets)
                
                # Check if it connects to "emotion" or "feeling" hypernyms?
                # Simplify: if it's an adjective and NOT a genre, assume Mood.
                if is_adj and word not in self.valid_genres:
                    # It's likely a mood (e.g., "sad", "happy", "nostalgic")
                    # We search for the mood itself as a keyword, as Deezer playlists often use these terms.
                    # e.g., "sad songs"
                    return {'type': 'mood', 'query': f'{word} songs', 'mood': word}

        # 2. Sentiment Analysis (Implicit Mood for sentences)
        polarity = blob.sentiment.polarity
        if polarity < -0.2:
            return {'type': 'mood', 'query': 'genre:"indie"', 'mood': 'sad (detected)', 'sentiment': polarity}
        elif polarity > 0.2:
            return {'type': 'mood', 'query': 'genre:"pop"', 'mood': 'happy (detected)', 'sentiment': polarity}
        
        # 3. Default to General Search
        return {'type': 'general', 'query': user_query}

    def is_in_era(self, release_date_str: str) -> bool:
        """Check if a track fits the 2012-2018 era."""
        if not release_date_str:
            return False
            
        try:
            # Handles 'YYYY-MM-DD' or just 'YYYY'
            year = int(release_date_str[:4])
            return self.start_year <= year <= self.end_year
        except ValueError:
            return False

    def filter_tracks(self, tracks: List[Dict]) -> List[Dict]:
        """Filter a list of tracks to only include those from the target era."""
        filtered = []
        for track in tracks:
            # Check metadata location (it might be nested or flat)
            meta = track.get('metadata', track)
            date = meta.get('release_date', '')
            
            if self.is_in_era(date):
                filtered.append(track)
                
        return filtered

    def extract_genres(self, query: str) -> List[str]:
        """Extract known genres from a query string."""
        query = query.lower()
        found_genres = []
        
        words = set(query.split())
        for genre in self.valid_genres:
            if genre in query: 
                 found_genres.append(genre)
        
        return list(set(found_genres))

