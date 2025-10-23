
from typing import List, Tuple
import numpy as np

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)

def mmr(query_vec: List[float], docs: List[str], doc_vecs: List[List[float]], k: int = 8, lambda_mult: float = 0.7) -> List[int]:
    q = np.array(query_vec, dtype=float)
    vecs = [np.array(v, dtype=float) for v in doc_vecs]
    selected: List[int] = []
    candidates = list(range(len(docs)))

    while candidates and len(selected) < k:
        best_i, best_score = -1, -1e9
        for i in candidates:
            relevance = _cosine(q, vecs[i])
            diversity = 0.0 if not selected else max(_cosine(vecs[i], vecs[j]) for j in selected)
            score = lambda_mult * relevance - (1 - lambda_mult) * diversity
            if score > best_score:
                best_score, best_i = score, i
        selected.append(best_i)
        candidates.remove(best_i)
    return selected
