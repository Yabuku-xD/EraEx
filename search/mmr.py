import numpy as np
from typing import List, Dict


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidates: List[Dict],
    lambda_param: float = 0.7,
    top_k: int = 20,
    diversity_field: str = "genre"
) -> List[Dict]:
    if len(candidates) == 0:
        return []
    
    if len(candidates) <= top_k:
        return candidates
    
    query_embedding = query_embedding.flatten()
    if candidate_embeddings.ndim == 1:
        candidate_embeddings = candidate_embeddings.reshape(1, -1)
    
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
    cand_norms = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-9)
    
    relevance_scores = cand_norms @ query_norm
    
    selected_indices = []
    selected_embeddings = []
    remaining_indices = list(range(len(candidates)))
    
    while len(selected_indices) < top_k and remaining_indices:
        if not selected_indices:
            best_idx = remaining_indices[np.argmax([relevance_scores[i] for i in remaining_indices])]
        else:
            best_score = float('-inf')
            best_idx = remaining_indices[0]
            
            selected_emb_matrix = np.array(selected_embeddings)
            
            for idx in remaining_indices:
                rel = relevance_scores[idx]
                
                cand_emb = cand_norms[idx]
                similarities = selected_emb_matrix @ cand_emb
                max_sim = np.max(similarities)
                
                mmr_score = lambda_param * rel - (1 - lambda_param) * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
        
        selected_indices.append(best_idx)
        selected_embeddings.append(cand_norms[best_idx])
        remaining_indices.remove(best_idx)
    
    return [candidates[i] for i in selected_indices]


def mmr_by_metadata(
    candidates: List[Dict],
    lambda_param: float = 0.7,
    top_k: int = 20
) -> List[Dict]:
    if len(candidates) <= top_k:
        return candidates
    
    selected = []
    remaining = list(candidates)
    
    while len(selected) < top_k and remaining:
        if not selected:
            selected.append(remaining.pop(0))
            continue
        
        best_score = float('-inf')
        best_idx = 0
        
        selected_genres = {r.get("genre", "").lower() for r in selected}
        selected_artists = {r.get("artist", "").lower() for r in selected}
        
        for i, cand in enumerate(remaining):
            rel = cand.get("final_score", cand.get("score", 0))
            
            cand_genre = cand.get("genre", "").lower()
            cand_artist = cand.get("artist", "").lower()
            
            genre_penalty = 0.3 if cand_genre and cand_genre in selected_genres else 0
            artist_penalty = 0.5 if cand_artist and cand_artist in selected_artists else 0
            
            diversity_penalty = max(genre_penalty, artist_penalty)
            
            mmr_score = lambda_param * rel - (1 - lambda_param) * diversity_penalty
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
        
        selected.append(remaining.pop(best_idx))
    
    return selected
