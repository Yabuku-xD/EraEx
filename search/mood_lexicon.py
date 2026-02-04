"""
NRC Emotion Lexicon Integration for EraEx Search

Uses the NRCLex library to detect emotions in queries and expand
them with music-relevant descriptors.

NRC Emotion Lexicon (EmoLex):
- 8 emotions: anger, fear, anticipation, trust, surprise, sadness, joy, disgust
- 2 sentiments: positive, negative
- Source: Dr. Saif Mohammad, National Research Council Canada
- https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm

Requirements:
    pip install nrclex
"""

from typing import Dict, List, Tuple

# Try to import NRCLex, provide fallback if not installed
try:
    from nrclex import NRCLex
    NRCLEX_AVAILABLE = True
except ImportError:
    NRCLEX_AVAILABLE = False
    print("⚠ NRCLex not installed. Run: pip install nrclex")


# NRC Emotion -> Music descriptor mappings
# Maps the 8 NRC emotions + 2 sentiments to music-relevant search terms
EMOTION_TO_MUSIC = {
    # Primary NRC Emotions
    "anger": ["aggressive", "heavy", "intense", "hard", "metal", "rage"],
    "fear": ["dark", "eerie", "atmospheric", "haunting", "tense"],
    "anticipation": ["building", "epic", "progressive", "dynamic"],
    "trust": ["warm", "comforting", "acoustic", "folk", "gentle"],
    "surprise": ["experimental", "unexpected", "avant-garde", "eclectic"],
    "sadness": ["melancholy", "emotional", "heartbreak", "slow", "crying", "sad r&b"],
    "joy": ["upbeat", "happy", "dance", "feel good", "party", "energetic"],
    "disgust": ["dark", "industrial", "gritty", "harsh"],
    
    # NRC Sentiments
    "positive": ["uplifting", "happy", "bright", "feel good"],
    "negative": ["dark", "sad", "melancholy", "moody"],
}

# Additional context-based expansions for common mood phrases
CONTEXT_EXPANSIONS = {
    # Relationship contexts
    "ex": ["heartbreak", "breakup", "sad r&b", "emotional"],
    "girlfriend": ["love", "romantic", "r&b", "slow jam"],
    "boyfriend": ["love", "romantic", "r&b", "slow jam"],
    "breakup": ["sad", "heartbreak", "crying", "emotional"],
    "love": ["romantic", "r&b", "slow", "passionate"],
    
    # Activity contexts
    "party": ["dance", "edm", "hype", "club", "bass"],
    "study": ["lo-fi", "instrumental", "focus", "ambient"],
    "workout": ["energetic", "hype", "motivation", "bass"],
    "sleep": ["ambient", "calm", "peaceful", "soft"],
    "driving": ["road trip", "cruising", "night drive"],
    
    # Time contexts
    "night": ["late night", "nocturnal", "ambient", "moody"],
    "morning": ["calm", "peaceful", "uplifting", "fresh"],
    "summer": ["beach", "tropical", "feel good", "dance"],
    
    # Mood descriptors that need expansion
    "nostalgic": ["throwback", "memories", "old school", "vintage", "retro"],
    "chill": ["lo-fi", "ambient", "relaxed", "mellow"],
    "vibes": [],  # Just a suffix, no expansion
}


def analyze_emotions(text: str) -> Dict[str, float]:
    """
    Analyze text using NRC Emotion Lexicon.
    
    Returns dict of emotion -> score (frequency).
    """
    if not NRCLEX_AVAILABLE:
        return {}
    
    try:
        emotion = NRCLex(text)
        return emotion.affect_frequencies
    except Exception:
        return {}


def get_top_emotions(text: str, threshold: float = 0.1, max_emotions: int = 3) -> List[str]:
    """
    Get top detected emotions from text.
    
    Args:
        text: Input text to analyze
        threshold: Minimum score to include emotion
        max_emotions: Maximum number of emotions to return
    
    Returns:
        List of emotion names (e.g., ["sadness", "fear"])
    """
    emotions = analyze_emotions(text)
    
    # Filter by threshold and sort by score
    filtered = [(e, s) for e, s in emotions.items() if s >= threshold]
    filtered.sort(key=lambda x: x[1], reverse=True)
    
    return [e for e, s in filtered[:max_emotions]]


def expand_mood_query(query: str, max_expansions: int = 6) -> str:
    """
    Expand query using NRC Emotion Lexicon + context mappings.
    
    This function:
    1. Detects emotions in the query using NRCLex
    2. Maps detected emotions to music descriptors
    3. Adds context-based expansions for keywords like "ex", "party", etc.
    
    Args:
        query: Original user query
        max_expansions: Maximum descriptor words to add
        
    Returns:
        Expanded query with music descriptors
        
    Example:
        Input:  "I'm feeling nostalgic cause my ex girlfriend is back"
        NRC:    Detects "sadness", "anticipation"
        Output: "I'm feeling nostalgic cause my ex girlfriend is back 
                 melancholy emotional heartbreak throwback memories"
    """
    query_lower = query.lower()
    expansions = []
    
    # 1. NRC Emotion-based expansion
    top_emotions = get_top_emotions(query)
    for emotion in top_emotions:
        if emotion in EMOTION_TO_MUSIC:
            # Add top 2 descriptors per emotion
            for desc in EMOTION_TO_MUSIC[emotion][:2]:
                if desc not in query_lower and desc not in expansions:
                    expansions.append(desc)
    
    # 2. Context-based expansion (for keywords not caught by NRC)
    for keyword, descriptors in CONTEXT_EXPANSIONS.items():
        if keyword in query_lower and descriptors:
            for desc in descriptors[:2]:
                if desc not in query_lower and desc not in expansions:
                    expansions.append(desc)
    
    # Limit total expansions
    expansions = expansions[:max_expansions]
    
    if expansions:
        return f"{query} {' '.join(expansions)}"
    return query


def detect_mood_intent(query: str) -> bool:
    """
    Detect if a query is mood/emotion-based vs. specific artist search.
    """
    # Check for NRC emotions
    if NRCLEX_AVAILABLE:
        emotions = analyze_emotions(query)
        if any(score > 0.1 for score in emotions.values()):
            return True
    
    # Check for mood keywords
    mood_keywords = [
        "feeling", "vibes", "mood", "when", "for",
        "sad", "happy", "chill", "emotional", "nostalgic"
    ]
    query_lower = query.lower()
    return any(kw in query_lower for kw in mood_keywords)


def get_emotion_summary(query: str) -> dict:
    """
    Get a summary of detected emotions for debugging/display.
    """
    if not NRCLEX_AVAILABLE:
        return {"error": "NRCLex not installed"}
    
    try:
        emotion = NRCLex(query)
        return {
            "top_emotions": get_top_emotions(query),
            "frequencies": emotion.affect_frequencies,
            "word_count": len(query.split()),
        }
    except Exception as e:
        return {"error": str(e)}
