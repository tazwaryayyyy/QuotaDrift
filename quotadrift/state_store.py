"""
SQLite-backed persistence layer for circuit breaker state and model health.

Why this exists: ModelManager operates with in-memory deques and dataclasses.
On restart, all learned reliability scores and circuit breaker positions reset
to defaults, causing provider scoring to start blind. This module persists a
compact summary (aggregates, not raw sliding windows) that gives the engine
a warm start reflecting real production history.

Schema stores one row per model slot. Writes are O(n_models) with tiny rows —
each upsert is a single round-trip to a WAL-mode SQLite file in ~/.quotadrift/.
"""

import logging
import sqlite3
import threading
import time
from pathlib import Path

logger = logging.getLogger("state_store")

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA busy_timeout=5000;
PRAGMA mmap_size=134217728;
PRAGMA cache_size=-32768;

CREATE TABLE IF NOT EXISTS model_state (
    model         TEXT    PRIMARY KEY,
    circuit_state TEXT    NOT NULL DEFAULT 'closed',
    last_failure  REAL,
    failure_count INTEGER NOT NULL DEFAULT 0,
    latency_avg   REAL    NOT NULL DEFAULT 0.0,
    reliability   REAL    NOT NULL DEFAULT 1.0,
    request_count INTEGER NOT NULL DEFAULT 0,
    updated_at    REAL    NOT NULL
);
"""


class StateStore:
    """Persist and restore model slot state across process restarts."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        # Ensure ~/.quotadrift/ exists before sqlite3.connect() tries to open the file.
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Per-connection pragmas; journal_mode and mmap_size persist to the DB
        # file after first set, but busy_timeout and cache_size are session-only.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA mmap_size=134217728")
        conn.execute("PRAGMA cache_size=-32768")
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_all(self) -> dict[str, dict]:
        """Return all persisted model states keyed by slot name."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute("SELECT * FROM model_state").fetchall()
                return {r["model"]: dict(r) for r in rows}
            finally:
                conn.close()

    def load(self, model: str) -> dict | None:
        """Return persisted state for a single slot, or None if not yet stored."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM model_state WHERE model = ?", (model,)
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def save(self, model: str, state: dict) -> None:
        """Upsert the state dict for a single model slot."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO model_state
                        (model, circuit_state, last_failure, failure_count,
                         latency_avg, reliability, request_count, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(model) DO UPDATE SET
                        circuit_state = excluded.circuit_state,
                        last_failure  = excluded.last_failure,
                        failure_count = excluded.failure_count,
                        latency_avg   = excluded.latency_avg,
                        reliability   = excluded.reliability,
                        request_count = excluded.request_count,
                        updated_at    = excluded.updated_at
                    """,
                    (
                        model,
                        state["circuit_state"],
                        state.get("last_failure"),
                        state["failure_count"],
                        state["latency_avg"],
                        state["reliability"],
                        state["request_count"],
                        time.time(),
                    ),
                )
                conn.commit()
            finally:
                conn.close()
