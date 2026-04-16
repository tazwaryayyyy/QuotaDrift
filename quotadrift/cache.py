from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticCache:
    def __init__(self, threshold: float = 0.92):
        # Using the same small, fast model as memory.py for embedding consistency
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self._store: list[dict] = []  # {vec, response, model, timestamp}
        self._threshold = threshold
        self.hits = 0
        self.total_queries = 0

    def get(self, query: str) -> dict | None:
        self.total_queries += 1
        if not self._store:
            return None

        q_vec = self._embedder.encode(query)
        best_score = -1
        best_item = None

        for item in self._store:
            # Cosine similarity: (A . B) / (||A|| * ||B||)
            norm_q = np.linalg.norm(q_vec)
            norm_i = np.linalg.norm(item["vec"])
            score = float(np.dot(q_vec, item["vec"]) / (norm_q * norm_i + 1e-8))

            if score > best_score:
                best_score = score
                best_item = item

        if best_score >= self._threshold:
            self.hits += 1
            return {
                "response": best_item["response"],
                "model": best_item["model"],
                "cached": True,
                "similarity": round(best_score, 3),
            }
        return None

    def set(self, query: str, response: str, model: str):
        vec = self._embedder.encode(query)
        self._store.append(
            {
                "vec": vec,
                "response": response,
                "model": model,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        # Keep last 200 entries to prevent memory bloat
        if len(self._store) > 200:
            self._store = self._store[-200:]

    def clear(self):
        self._store = []
        self.hits = 0
        self.total_queries = 0

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def stats(self) -> dict:
        perc = (self.hits / self.total_queries * 100) if self.total_queries > 0 else 0
        return {
            "hits": self.hits,
            "total": self.total_queries,
            "percentage": round(perc, 1),
            "size": self.size,
        }


# Global instance
_cache = SemanticCache(threshold=0.92)


def get_cache() -> SemanticCache:
    return _cache
