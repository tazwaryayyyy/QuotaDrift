"""
Tests for cache.py — CacheStore persistence, TTL eviction, and SemanticCache behaviour.
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from quotadrift.cache import MAX_ENTRIES, CacheStore, SemanticCache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(tmp_path: Path, ttl: float = 3600.0) -> CacheStore:
    return CacheStore(db_path=tmp_path / "cache.db", ttl=ttl)


def _dummy_vec(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(384).astype(np.float32)


# ---------------------------------------------------------------------------
# Basic persistence
# ---------------------------------------------------------------------------


def test_append_and_load_round_trip(tmp_path):
    """Entries written via append() must be retrievable via load_recent()."""
    store = _make_store(tmp_path)
    vec = _dummy_vec(1)
    store.append("what is Python", "Python is a language.", "primary", vec)

    rows = store.load_recent()
    assert len(rows) == 1
    assert rows[0]["query"] == "what is Python"
    assert rows[0]["response"] == "Python is a language."
    assert rows[0]["model"] == "primary"
    np.testing.assert_array_almost_equal(rows[0]["vec"], vec, decimal=5)


def test_load_recent_survives_new_store_instance(tmp_path):
    """Data written in one CacheStore instance must be visible from a second
    instance pointing at the same file — confirming SQLite durability."""
    vec = _dummy_vec(2)
    store_a = _make_store(tmp_path)
    store_a.append("hello", "hi there", "secondary", vec)

    store_b = _make_store(tmp_path)
    rows = store_b.load_recent()
    assert any(r["query"] == "hello" for r in rows)


# ---------------------------------------------------------------------------
# TTL eviction (the test the first pass was missing)
# ---------------------------------------------------------------------------


def test_evict_expired_removes_stale_entries(tmp_path):
    """evict_expired() must delete entries older than the TTL."""
    # Use a tiny TTL so we can expire entries immediately.
    store = _make_store(tmp_path, ttl=0.05)  # 50 ms
    store.append("stale query", "stale response", "primary", _dummy_vec(3))

    # Wait until the TTL lapses.
    time.sleep(0.1)

    store.evict_expired()

    rows = store.load_recent()
    assert len(rows) == 0, (
        f"Expected 0 rows after TTL eviction but got {len(rows)}. "
        "evict_expired() is not honouring the TTL."
    )


def test_fresh_entry_survives_eviction(tmp_path):
    """A just-written entry must NOT be deleted by evict_expired()."""
    store = _make_store(tmp_path, ttl=3600.0)
    store.append("fresh query", "fresh response", "primary", _dummy_vec(4))

    store.evict_expired()

    rows = store.load_recent()
    assert len(rows) == 1, "A fresh entry should survive evict_expired()."


def test_load_recent_respects_ttl_without_explicit_evict(tmp_path):
    """load_recent() filters by TTL at query time so stale rows are invisible
    even if evict_expired() has not been called yet."""
    store = _make_store(tmp_path, ttl=0.05)
    store.append("aged", "old response", "primary", _dummy_vec(5))

    time.sleep(0.1)

    rows = store.load_recent()
    assert len(rows) == 0, (
        "load_recent() must filter out entries beyond the TTL cutoff."
    )


# ---------------------------------------------------------------------------
# MAX_ENTRIES cap
# ---------------------------------------------------------------------------


def test_evict_expired_caps_at_max_entries(tmp_path):
    """After inserting more than MAX_ENTRIES rows, evict_expired() must trim
    the table down to MAX_ENTRIES most-recent entries."""
    store = _make_store(tmp_path, ttl=3600.0)
    for i in range(MAX_ENTRIES + 10):
        store.append(f"query_{i}", f"response_{i}", "primary", _dummy_vec(i))

    store.evict_expired()

    rows = store.load_recent()
    assert len(rows) <= MAX_ENTRIES, (
        f"Expected at most {MAX_ENTRIES} rows after eviction cap, got {len(rows)}."
    )


# ---------------------------------------------------------------------------
# SemanticCache integration (smoke test — no live embedder)
# ---------------------------------------------------------------------------


def test_semantic_cache_set_and_hit(tmp_path):
    """set() followed by get() for an identical query must return a cache hit."""
    # Patch the embedder so this test does not load SentenceTransformer.
    fixed_vec = _dummy_vec(99)
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = fixed_vec

    with patch("quotadrift.cache.get_embedding_model", return_value=mock_embedder):
        sc = SemanticCache(threshold=0.90, db_path=tmp_path / "sc.db", ttl=3600.0)
        sc.set("what is gravity", "It is a force.", "primary")

        hit = sc.get("what is gravity")

    assert hit is not None, "Identical-vector query must be a cache hit."
    assert hit["cached"] is True
    assert hit["response"] == "It is a force."


def test_semantic_cache_miss_below_threshold(tmp_path):
    """A query whose embedding is orthogonal to all cached embeddings must miss."""
    mock_embedder = MagicMock()
    # set() encodes with vec_a; get() encodes with vec_b (orthogonal → cosine ≈ 0)
    vec_a = np.zeros(384, dtype=np.float32)
    vec_a[0] = 1.0
    vec_b = np.zeros(384, dtype=np.float32)
    vec_b[1] = 1.0
    mock_embedder.encode.side_effect = [vec_a, vec_b]

    with patch("quotadrift.cache.get_embedding_model", return_value=mock_embedder):
        sc = SemanticCache(threshold=0.90, db_path=tmp_path / "sc2.db", ttl=3600.0)
        sc.set("topic A", "response A", "primary")

        miss = sc.get("completely different topic")

    assert miss is None, "Orthogonal embedding must not exceed the cosine threshold."
