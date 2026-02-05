"""
HYDE (Hypothetical Document Expansion) - Fully Automated

Uses NRCLex + WordNet to automatically expand queries with relevant music terms.
NO manual dictionaries - everything is generated from lexical resources.

Sources:
- NRC Emotion Lexicon: 14,000+ words with emotion annotations
  https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
  
- NLTK WordNet: Lexical database with synonyms, hypernyms, related terms
  https://wordnet.princeton.edu/
  
- Gao et al. (2023). "Precise Zero-Shot Dense Retrieval without Relevance Labels."
  ACL 2023. arXiv:2212.10496
"""

from typing import Dict, List, Set, Optional
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

def get_synonyms(word: str, max_synonyms: int = 5) -> Set[str]:
    if not WORDNET_AVAILABLE:
        return set()
    
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace('_', ' ')
            if name.lower() != word.lower():
                synonyms.add(name.lower())
            if len(synonyms) >= max_synonyms:
                return synonyms
    
    return synonyms

def get_related_terms(word: str, max_terms: int = 5) -> Set[str]:
    if not WORDNET_AVAILABLE:
        return set()
    
    related = set()
    for syn in wn.synsets(word):
        for hyper in syn.hypernyms():
            for lemma in hyper.lemmas():
                related.add(lemma.name().replace('_', ' ').lower())
        for hypo in syn.hyponyms():
            for lemma in hypo.lemmas():
                related.add(lemma.name().replace('_', ' ').lower())
        
        if len(related) >= max_terms:
            break
    
    return set(list(related)[:max_terms])

def detect_emotions(text: str) -> Dict[str, float]:
    if not NRCLEX_AVAILABLE:
        return {}
    
    try:
        emotion = NRCLex(text)
        scores = emotion.raw_emotion_scores
        return {k: v for k, v in scores.items() if v > 0}
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

def expand_with_wordnet(query: str) -> str:
    words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
    
    expansions = set()
    for word in words:
        synonyms = get_synonyms(word, max_synonyms=3)
        expansions.update(synonyms)
        
        related = get_related_terms(word, max_terms=2)
        expansions.update(related)
    
    if expansions:
        expansion_text = ' '.join(list(expansions)[:10])
        return f"{query} {expansion_text}"
    
    return query

def get_hyde_expansion(query: str) -> str:
    expanded_parts = [query]
    
    emotions = detect_emotions(query)
    if emotions:
        emotion_terms = ' '.join(emotions.keys())
        expanded_parts.append(emotion_terms)
    
    affect_words = get_affect_words(query)
    if affect_words:
        expanded_parts.append(' '.join(affect_words[:5]))
    
    words = re.findall(r'\b[a-zA-Z]{4,}\b', query.lower())
    all_synonyms = set()
    all_related = set()
    
    for word in words[:5]:
        all_synonyms.update(get_synonyms(word, max_synonyms=3))
        all_related.update(get_related_terms(word, max_terms=2))
    
    if all_synonyms:
        expanded_parts.append(' '.join(list(all_synonyms)[:8]))
    
    if all_related:
        expanded_parts.append(' '.join(list(all_related)[:5]))
    
    return ' '.join(expanded_parts)

def analyze_query(query: str) -> Dict:
    emotions = detect_emotions(query)
    affect_words = get_affect_words(query)
    words = re.findall(r'\b[a-zA-Z]{4,}\b', query.lower())
    
    synonyms = set()
    related = set()
    for word in words[:3]:
        synonyms.update(get_synonyms(word, max_synonyms=3))
        related.update(get_related_terms(word, max_terms=2))
    
    return {
        "query": query,
        "emotions": emotions,
        "affect_words": affect_words,
        "synonyms": list(synonyms),
        "related_terms": list(related),
        "expanded": get_hyde_expansion(query),
    }

def get_all_emotions() -> List[str]:
    return ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation", "positive", "negative"]
