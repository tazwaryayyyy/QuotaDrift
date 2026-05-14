"""
Process-wide SentenceTransformer singleton.

Both cache.py and memory.py previously instantiated 'all-MiniLM-L6-v2'
independently, doubling the ~80 MB resident set for zero benefit. This
module owns the single instance and hands out the same reference to all
callers.

Double-checked locking (threading.Lock inside an outer None-guard) ensures
exactly one model is loaded even when multiple modules call
get_embedding_model() concurrently at import time in a threaded ASGI worker.
"""

import threading

from sentence_transformers import SentenceTransformer

_instance: SentenceTransformer | None = None
_lock = threading.Lock()


def get_embedding_model() -> SentenceTransformer:
    """Return the shared SentenceTransformer instance, loading it on first call."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = SentenceTransformer("all-MiniLM-L6-v2")
    return _instance
