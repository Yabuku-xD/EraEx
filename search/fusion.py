from typing import List, Dict, Tuple


def reciprocal_rank_fusion(
    result_lists: List[List[Dict]],
    k: int = 60,
    id_field: str = "track_id"
) -> List[Dict]:
    fused_scores: Dict[str, float] = {}
    result_map: Dict[str, Dict] = {}
    
    for results in result_lists:
        for rank, result in enumerate(results):
            doc_id = result.get(id_field)
            if doc_id is None:
                continue
            
            rrf_score = 1.0 / (k + rank + 1)
            
            if doc_id in fused_scores:
                fused_scores[doc_id] += rrf_score
            else:
                fused_scores[doc_id] = rrf_score
                result_map[doc_id] = result.copy()
    
    for doc_id, score in fused_scores.items():
        result_map[doc_id]["rrf_score"] = score
    
    sorted_results = sorted(
        result_map.values(),
        key=lambda x: x.get("rrf_score", 0),
        reverse=True
    )
    
    return sorted_results


def weighted_rrf(
    result_lists: List[Tuple[List[Dict], float]],
    k: int = 60,
    id_field: str = "track_id"
) -> List[Dict]:
    fused_scores: Dict[str, float] = {}
    result_map: Dict[str, Dict] = {}
    
    for results, weight in result_lists:
        for rank, result in enumerate(results):
            doc_id = result.get(id_field)
            if doc_id is None:
                continue
            
            rrf_score = weight * (1.0 / (k + rank + 1))
            
            if doc_id in fused_scores:
                fused_scores[doc_id] += rrf_score
            else:
                fused_scores[doc_id] = rrf_score
                result_map[doc_id] = result.copy()
    
    for doc_id, score in fused_scores.items():
        result_map[doc_id]["rrf_score"] = score
    
    sorted_results = sorted(
        result_map.values(),
        key=lambda x: x.get("rrf_score", 0),
        reverse=True
    )
    
    return sorted_results
