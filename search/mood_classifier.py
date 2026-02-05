"""
Query Intent Classifier - Automated using NRCLex

Uses NRC Emotion Lexicon to detect if query is mood-based.

Sources:
- NRC Emotion Lexicon (EmoLex) by Dr. Saif Mohammad
  https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
"""

from typing import Literal
import re

try:
    from nrclex import NRCLex
    NRCLEX_AVAILABLE = True
except ImportError:
    NRCLEX_AVAILABLE = False

QueryIntent = Literal["mood", "artist", "genre", "mixed"]

GENRE_INDICATORS = {
    "hip hop", "hip-hop", "rap", "r&b", "rnb", "electronic", "edm",
    "house", "techno", "dubstep", "trap", "lo-fi", "lofi", "ambient",
    "rock", "metal", "indie", "pop", "jazz", "soul", "funk",
    "reggae", "classical", "country", "folk", "blues", "drill",
    "phonk", "grunge", "punk", "disco", "trance", "dnb", "drum and bass"
}

ARTIST_PATTERNS = [
    r"^[A-Z][a-z]+\s+[A-Z][a-z]+$",
    r"^[A-Z][a-z]+$",
    r"\b(by|from|feat|ft\.?|featuring)\s+\w+",
]


class MoodClassifier:
    def __init__(self):
        self._compiled_artist_patterns = [re.compile(p, re.IGNORECASE) for p in ARTIST_PATTERNS]
    
    def _has_emotions(self, text: str) -> bool:
        if not NRCLEX_AVAILABLE:
            return False
        
        try:
            emotion = NRCLex(text)
            scores = emotion.raw_emotion_scores
            return sum(scores.values()) > 0
        except Exception:
            return False
    
    def _emotion_strength(self, text: str) -> float:
        if not NRCLEX_AVAILABLE:
            return 0.0
        
        try:
            emotion = NRCLex(text)
            freqs = emotion.affect_frequencies
            return sum(freqs.values())
        except Exception:
            return 0.0
    
    def _has_genre(self, text: str) -> bool:
        text_lower = text.lower()
        return any(g in text_lower for g in GENRE_INDICATORS)
    
    def _looks_like_artist(self, text: str) -> bool:
        for p in self._compiled_artist_patterns:
            if p.match(text):
                return True
        return False
    
    def classify(self, query: str) -> QueryIntent:
        has_emotion = self._has_emotions(query)
        emotion_strength = self._emotion_strength(query)
        has_genre = self._has_genre(query)
        looks_like_artist = self._looks_like_artist(query)
        
        mood_phrases = ["i feel", "feeling", "vibes", "vibe", "mood", "when i", "for when"]
        query_lower = query.lower()
        has_mood_phrase = any(phrase in query_lower for phrase in mood_phrases)
        
        if has_mood_phrase or emotion_strength > 0.3:
            return "mood"
        
        if has_emotion and not has_genre and not looks_like_artist:
            return "mood"
        
        if has_genre and not has_emotion:
            return "genre"
        
        if looks_like_artist and not has_emotion and not has_genre:
            return "artist"
        
        if has_emotion:
            return "mood"
        
        return "mixed"
    
    def get_mood_expansion(self, query: str) -> str:
        if not NRCLEX_AVAILABLE:
            return query
        
        try:
            emotion = NRCLex(query)
            
            expansions = []
            
            emotions = emotion.raw_emotion_scores
            top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
            for e, score in top_emotions:
                if score > 0:
                    expansions.append(e)
            
            affect_words = list(emotion.affect_dict.keys())[:5]
            for word in affect_words:
                if word not in expansions:
                    expansions.append(word)
            
            if expansions:
                return f"{query} {' '.join(expansions[:6])}"
            
        except Exception:
            pass
        
        return query


mood_classifier = MoodClassifier()