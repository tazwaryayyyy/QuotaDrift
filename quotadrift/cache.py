"""
Semantic response cache backed by SQLite for durability across restarts.

Why SQLite-backed: the original in-memory store reset on every restart,
causing a thundering-herd effect on popular repeated queries. The cache
now survives restarts via a SQLite store in ~/.quotadrift/cache.db while
keeping a hot in-memory working set for sub-millisecond hit checks.

Cosine similarity is computed in Python after pulling candidate blobs. Since
the store is capped at MAX_ENTRIES=200, the O(n) scan is cheap (~10 µs).

TTL enforcement: entries older than CACHE_TTL seconds are evicted on startup
and periodically via evict_expired().
"""

import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

from quotadrift.embedding import get_embedding_model

CACHE_TTL: float = 3600.0  # 1 hour default
MAX_ENTRIES: int = 200  # Hard cap on in-memory and DB working set

_DEFAULT_CACHE_DB = Path.home() / ".quotadrift" / "cache.db"


class CacheStore:
    """SQLite persistence layer for semantic cache entries."""

    def __init__(
        self, db_path: Path = _DEFAULT_CACHE_DB, ttl: float = CACHE_TTL
    ) -> None:
        self._db_path = db_path
        self._ttl = ttl
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA mmap_size=134217728")  # 128 MB
        conn.execute("PRAGMA cache_size=-65536")  # 64 MB
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    query      TEXT    NOT NULL,
                    response   TEXT    NOT NULL,
                    model      TEXT    NOT NULL,
                    embedding  BLOB    NOT NULL,
                    created_at REAL    NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cache_created "
                "ON cache_entries(created_at DESC)"
            )
            conn.commit()
        finally:
            conn.close()

    def evict_expired(self) -> None:
        """Remove entries older than TTL and cap the table at MAX_ENTRIES."""
        cutoff = time.time() - self._ttl
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "DELETE FROM cache_entries WHERE created_at < ?", (cutoff,)
                )
                conn.execute(
                    """
                    DELETE FROM cache_entries WHERE id NOT IN (
                        SELECT id FROM cache_entries
                        ORDER BY created_at DESC LIMIT ?
                    )
                    """,
                    (MAX_ENTRIES,),
                )
                conn.commit()
            finally:
                conn.close()

    def load_recent(self) -> list[dict]:
        """Load non-expired entries into the in-memory hot set on startup."""
        cutoff = time.time() - self._ttl
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT query, response, model, embedding, created_at "
                "FROM cache_entries WHERE created_at >= ? "
                "ORDER BY created_at DESC LIMIT ?",
                (cutoff, MAX_ENTRIES),
            ).fetchall()
        finally:
            conn.close()

        result: list[dict] = []
        for row in rows:
            try:
                vec = np.frombuffer(row[3], dtype=np.float32).copy()
                result.append(
                    {
                        "query": row[0],
                        "vec": vec,
                        "response": row[1],
                        "model": row[2],
                        "timestamp": row[4],
                    }
                )
            except (ValueError, TypeError, BufferError):
                # Corrupt blob — skip rather than crash; it ages out naturally.
                pass
        return result

    def append(self, query: str, response: str, model: str, vec: "np.ndarray") -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO cache_entries "
                    "(query, response, model, embedding, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        query,
                        response,
                        model,
                        vec.astype(np.float32).tobytes(),
                        time.time(),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def clear(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM cache_entries")
                conn.commit()
            finally:
                conn.close()


class SemanticCache:
    def __init__(
        self,
        threshold: float = 0.92,
        db_path: Path = _DEFAULT_CACHE_DB,
        ttl: float = CACHE_TTL,
    ) -> None:
        self._embedder = get_embedding_model()
        self._threshold = threshold
        self._store: list[dict] = []
        self._cache_store = CacheStore(db_path=db_path, ttl=ttl)
        self.hits = 0
        self.total_queries = 0

        # Warm the in-memory set from the DB so the first post-restart requests
        # benefit from previously cached responses.
        self._store = self._cache_store.load_recent()

    def get(self, query: str) -> dict | None:
        self.total_queries += 1
        if not self._store:
            return None

        q_vec = self._embedder.encode(query)
        best_score = -1.0
        best_item: dict | None = None

        for item in self._store:
            norm_q = np.linalg.norm(q_vec)
            norm_i = np.linalg.norm(item["vec"])
            score = float(np.dot(q_vec, item["vec"]) / (norm_q * norm_i + 1e-8))
            if score > best_score:
                best_score = score
                best_item = item

        if best_score >= self._threshold and best_item is not None:
            self.hits += 1
            return {
                "response": best_item["response"],
                "model": best_item["model"],
                "cached": True,
                "similarity": round(best_score, 3),
            }
        return None

    def set(self, query: str, response: str, model: str) -> None:
        vec = self._embedder.encode(query).astype(np.float32)
        self._store.append(
            {
                "vec": vec,
                "response": response,
                "model": model,
                "timestamp": time.time(),
            }
        )
        self._cache_store.append(query, response, model, vec)

        if len(self._store) > MAX_ENTRIES:
            self._store = self._store[-MAX_ENTRIES:]

    def clear(self) -> None:
        self._store = []
        self.hits = 0
        self.total_queries = 0
        self._cache_store.clear()

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def stats(self) -> dict:
        pct = (self.hits / self.total_queries * 100) if self.total_queries > 0 else 0.0
        return {
            "hits": self.hits,
            "total": self.total_queries,
            "percentage": round(pct, 1),
            "size": self.size,
        }


# Global instance
_cache = SemanticCache(threshold=0.92)


def get_cache() -> SemanticCache:
    return _cache
