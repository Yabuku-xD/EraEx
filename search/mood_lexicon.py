"""
NRC Emotion Lexicon Integration - Fully Automated

Uses NRCLex library (14,000+ word lexicon) for emotion detection.
Uses NLTK WordNet for synonym expansion.

Sources:
- NRC Emotion Lexicon (EmoLex) by Dr. Saif Mohammad
  https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
  
- NLTK WordNet
  https://wordnet.princeton.edu/
"""

from typing import Dict, List
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


def analyze_emotions(text: str) -> Dict[str, float]:
    if not NRCLEX_AVAILABLE:
        return {}
    try:
        emotion = NRCLex(text)
        return emotion.affect_frequencies
    except Exception:
        return {}


def get_raw_emotions(text: str) -> Dict[str, int]:
    if not NRCLEX_AVAILABLE:
        return {}
    try:
        emotion = NRCLex(text)
        return emotion.raw_emotion_scores
    except Exception:
        return {}


def get_affect_words(text: str) -> List[str]:
    if not NRCLEX_AVAILABLE:
        return []
    try:
        emotion = NRCLex(text)
        return list(emotion.affect_dict.keys())
    except Exception:
        return []


def get_top_emotions(text: str, threshold: float = 0.1, max_emotions: int = 3) -> List[str]:
    emotions = analyze_emotions(text)
    filtered = [(e, s) for e, s in emotions.items() if s >= threshold]
    filtered.sort(key=lambda x: x[1], reverse=True)
    return [e for e, s in filtered[:max_emotions]]


def get_synonyms(word: str, max_synonyms: int = 5) -> List[str]:
    if not WORDNET_AVAILABLE:
        return []
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace('_', ' ')
            if name.lower() != word.lower():
                synonyms.add(name.lower())
            if len(synonyms) >= max_synonyms:
                return list(synonyms)
    return list(synonyms)


def expand_mood_query(query: str, max_expansions: int = 10) -> str:
    expansions = []
    
    emotions = get_raw_emotions(query)
    if emotions:
        top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
        for emotion, score in top_emotions:
            if score > 0:
                expansions.append(emotion)
    
    affect_words = get_affect_words(query)
    for word in affect_words[:5]:
        if word not in expansions:
            expansions.append(word)
    
    if WORDNET_AVAILABLE:
        words = re.findall(r'\b[a-zA-Z]{4,}\b', query.lower())
        for word in words[:3]:
            syns = get_synonyms(word, max_synonyms=2)
            for syn in syns:
                if syn not in expansions and syn not in query.lower():
                    expansions.append(syn)
    
    expansions = expansions[:max_expansions]
    
    if expansions:
        return f"{query} {' '.join(expansions)}"
    return query


def detect_mood_intent(query: str) -> bool:
    if NRCLEX_AVAILABLE:
        emotions = analyze_emotions(query)
        if any(score > 0.1 for score in emotions.values()):
            return True
    return False


def get_emotion_summary(query: str) -> dict:
    if not NRCLEX_AVAILABLE:
        return {"error": "NRCLex not installed"}
    try:
        emotion = NRCLex(query)
        return {
            "top_emotions": get_top_emotions(query),
            "raw_scores": emotion.raw_emotion_scores,
            "frequencies": emotion.affect_frequencies,
            "affect_words": list(emotion.affect_dict.keys()),
        }
    except Exception as e:
        return {"error": str(e)}