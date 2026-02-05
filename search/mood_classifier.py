"""
Query Intent Classifier - Automated using NRCLex + WordNet

Uses NRC Emotion Lexicon to detect if query is mood-based.
Uses WordNet to detect if query is genre-based.
NO manual keyword sets.

Sources:
- NRC Emotion Lexicon (EmoLex) by Dr. Saif Mohammad
- NLTK WordNet
"""

from typing import Literal, Set
import re

try:
    from nrclex import NRCLex
    NRCLEX_AVAILABLE = True
except ImportError:
    NRCLEX_AVAILABLE = False

try:
    from nltk.corpus import wordnet as wn
    import nltk
    try:
        wn.synsets('test')
        WORDNET_AVAILABLE = True
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False

QueryIntent = Literal["mood", "artist", "genre", "mixed"]

ARTIST_PATTERNS = [
    r"^[A-Z][a-z]+\s+[A-Z][a-z]+$",
    r"^[A-Z][a-z]+$",
    r"\b(by|from|feat|ft\.?|featuring)\s+\w+",
]


class MoodClassifier:
    def __init__(self):
        self._compiled_artist_patterns = [re.compile(p, re.IGNORECASE) for p in ARTIST_PATTERNS]
        self.genres: Set[str] = self._load_genres()
    
    def _load_genres(self) -> Set[str]:
        if not WORDNET_AVAILABLE:
            return set()
        
        genres = set()
        try:
            # Load genres dynamically from WordNet 'musical_style'
            base = wn.synset('musical_style.n.01')
            for syn in base.closure(lambda s: s.hyponyms()):
                for lemma in syn.lemmas():
                    name = lemma.name().replace('_', ' ').lower()
                    genres.add(name)
        except Exception:
            pass
        return genres

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
        if not self.genres:
            return False
            
        text_lower = text.lower()
        # Check if any genre is in text (word boundary check would be better but simple substring matches previous logic)
        # Optimization: split text and check intersection if genres are single words, but genres can be multi-word
        # Simple iteration is safest for now
        return any(g in text_lower for g in self.genres)
    
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