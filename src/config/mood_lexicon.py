import json
from pathlib import Path
from typing import Optional, List

_mood_lexicon: Optional[dict] = None
_genre_keywords: Optional[List[str]] = None
_attribute_keywords: Optional[List[str]] = None


def _get_data_dir() -> Path:
    from src.config.settings import DATA_DIR
    return DATA_DIR


def load_mood_lexicon() -> dict:
    global _mood_lexicon
    if _mood_lexicon is not None:
        return _mood_lexicon
    path = _get_data_dir() / "mood_lexicon.json"
    if not path.exists():
        print(f"Mood lexicon not found at {path}, generating from Warriner dataset...")
        from scripts.generate_mood_lexicon import generate_all_lexicons
        result = generate_all_lexicons(_get_data_dir())
        _mood_lexicon = result["mood_lexicon"]
    else:
        with open(path, "r", encoding="utf-8") as f:
            _mood_lexicon = json.load(f)
        print(f"Loaded mood lexicon with {len(_mood_lexicon)} words from {path}")
    return _mood_lexicon


def load_genre_keywords() -> List[str]:
    global _genre_keywords
    if _genre_keywords is not None:
        return _genre_keywords
    path = _get_data_dir() / "genre_keywords.json"
    if not path.exists():
        print(f"Genre keywords not found at {path}, generating...")
        from scripts.generate_mood_lexicon import generate_all_lexicons
        result = generate_all_lexicons(_get_data_dir())
        _genre_keywords = result["genre_keywords"]
    else:
        with open(path, "r", encoding="utf-8") as f:
            _genre_keywords = json.load(f)
        print(f"Loaded {len(_genre_keywords)} genre keywords from {path}")
    return _genre_keywords


def load_attribute_keywords() -> List[str]:
    global _attribute_keywords
    if _attribute_keywords is not None:
        return _attribute_keywords
    path = _get_data_dir() / "attribute_keywords.json"
    if not path.exists():
        print(f"Attribute keywords not found at {path}, generating...")
        from scripts.generate_mood_lexicon import generate_all_lexicons
        result = generate_all_lexicons(_get_data_dir())
        _attribute_keywords = result["attribute_keywords"]
    else:
        with open(path, "r", encoding="utf-8") as f:
            _attribute_keywords = json.load(f)
        print(f"Loaded {len(_attribute_keywords)} attribute keywords from {path}")
    return _attribute_keywords


def extract_mood_from_query(query_text: str) -> dict:
    lexicon = load_mood_lexicon()
    tokens = query_text.lower().split()
    matched_moods = []
    for token in tokens:
        if token in lexicon:
            matched_moods.append(lexicon[token])
    if not matched_moods:
        return {"valence": 0.0, "arousal": 0.0}
    avg_valence = sum(m["valence"] for m in matched_moods) / len(matched_moods)
    avg_arousal = sum(m["arousal"] for m in matched_moods) / len(matched_moods)
    return {"valence": avg_valence, "arousal": avg_arousal}


def detect_query_type(query_text: str) -> str:
    lexicon = load_mood_lexicon()
    genre_keywords = load_genre_keywords()
    attribute_keywords = load_attribute_keywords()
    query_lower = query_text.lower()
    tokens = query_lower.split()
    has_mood = any(token in lexicon for token in tokens)
    has_genre = any(kw in query_lower for kw in genre_keywords)
    has_attribute = any(kw in query_lower for kw in attribute_keywords)
    if has_mood and not has_genre and not has_attribute:
        return "mood-heavy"
    elif has_genre:
        return "genre"
    elif has_attribute:
        return "text-heavy"
    else:
        return "default"