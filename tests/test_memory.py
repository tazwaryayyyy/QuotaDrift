"""
Tests for memory.py — BM25 rebuild efficiency and SQLite WAL mode.
"""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helper: isolated HybridSearcher with a real temp DB
# ---------------------------------------------------------------------------


def _make_searcher(project_id: int, db_path: Path):  # noqa: ARG001 — kept for future use
    """Instantiate a HybridSearcher with an empty temp DB."""

    # Temporarily override the DB_PATH constant in the memory module so the
    # HybridSearcher reads from our temp file, not the real switchboard.db.
    import quotadrift.memory as mem_module

    original_db = mem_module.DB_PATH
    mem_module.DB_PATH = db_path

    # Re-initialize the schema on the temp DB.
    mem_module.init_db()

    try:
        from quotadrift.memory import HybridSearcher

        searcher = HybridSearcher(project_id)
    finally:
        mem_module.DB_PATH = original_db

    return searcher


# ---------------------------------------------------------------------------
# BM25 rebuild efficiency test (FIX #4)
# ---------------------------------------------------------------------------


def test_bulk_load_triggers_single_bm25_rebuild(tmp_path):
    """Loading N documents from the DB must build the BM25 index exactly once,
    not N times (which was the pre-fix O(n²) behaviour).

    We patch rank_bm25.BM25Okapi and assert the constructor is called once
    regardless of how many documents are in the project_files table.
    """
    db_path = tmp_path / "switchboard.db"

    import quotadrift.memory as mem_module

    original_db = mem_module.DB_PATH
    mem_module.DB_PATH = db_path
    mem_module.init_db()

    # Insert 5 documents into the project_files table directly.
    conn = sqlite3.connect(str(db_path))
    try:
        # Ensure there is a project row first.
        conn.execute(
            "INSERT OR IGNORE INTO projects (id, name) VALUES (?, ?)",
            (1, "test_project"),
        )
        for i in range(5):
            conn.execute(
                "INSERT INTO project_files "
                "(project_id, filename, content, indexed_at) VALUES (?, ?, ?, ?)",
                (1, f"file_{i}.py", f"def func_{i}(): pass", 0.0),
            )
        conn.commit()
    finally:
        conn.close()

    try:
        with patch("quotadrift.memory.BM25Okapi") as mock_bm25:
            mock_bm25.return_value = MagicMock()
            from quotadrift.memory import HybridSearcher

            HybridSearcher(1)  # construction exercises bulk load

            # The BM25Okapi constructor must be called exactly ONCE after the
            # bulk load, not once per document.
            assert mock_bm25.call_count == 1, (
                f"Expected BM25Okapi to be constructed once after bulk load, "
                f"but it was called {mock_bm25.call_count} times. "
                f"This indicates the O(n²) rebuild regression is present."
            )
    finally:
        mem_module.DB_PATH = original_db


# ---------------------------------------------------------------------------
# WAL mode test (FIX #5)
# ---------------------------------------------------------------------------


def test_wal_mode_active_on_connection(tmp_path):
    """Every connection opened via _db() must have journal_mode=WAL active."""
    db_path = tmp_path / "switchboard.db"

    import quotadrift.memory as mem_module

    original_db = mem_module.DB_PATH
    mem_module.DB_PATH = db_path

    try:
        # init_db() applies the pragma and commits.
        mem_module.init_db()

        # _db() is a module-level context manager; direct access is intentional in tests.
        with mem_module._db() as conn:  # pylint: disable=protected-access
            row = conn.execute("PRAGMA journal_mode").fetchone()
            journal_mode = row[0] if row else "unknown"

        assert journal_mode == "wal", (
            f"Expected journal_mode=wal but got '{journal_mode}'. "
            "The _db() context manager must set PRAGMA journal_mode=WAL."
        )
    finally:
        mem_module.DB_PATH = original_db
