from typing import List, Dict

class HybridRanker:
    def __init__(self):
        # In a real implementation, we might load weights or models here
        self.weights = {
            'cf': 0.7,      # Collaborative Filtering
            'content': 0.3  # Content-based (genres, etc.)
        }

    def score(self, user_id: int, candidates: List[Dict], user_prefs: Dict) -> List[Dict]:
        """
        Rank candidates based on a hybrid score of CF + Content.
        candidates: List of dicts with 'cf_score' and 'metadata'.
        user_prefs: Dict of genre preferences e.g. {'Pop': 0.8}
        """
        ranked = []
        
        for item in candidates:
            # 1. Normalize CF Score (assuming 0-1 or similar)
            cf_score = item.get('cf_score', 0.0)
            
            # 2. Compute Content Score
            # Simple match: Does item genre match user top genres?
            meta = item.get('metadata', {})
            genre = meta.get('genre', 'Unknown').lower()
            
            content_score = 0.0
            for pref_genre, weight in user_prefs.items():
                if pref_genre.lower() in genre:
                    content_score += weight
            
            # 3. Weighted Sum
            final_score = (cf_score * self.weights['cf']) + (content_score * self.weights['content'])
            
            # Add to result
            item['final_score'] = final_score
            ranked.append(item)
            
        # Sort descending by final score
        ranked.sort(key=lambda x: x['final_score'], reverse=True)
        return ranked
