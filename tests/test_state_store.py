"""
Tests for state_store.StateStore — round-trip persistence and sentinel values.
"""
# pylint: disable=redefined-outer-name  # pytest fixture injection intentionally
# shadows the outer fixture function name — this is correct pytest usage.

import pytest

from quotadrift.state_store import StateStore


@pytest.fixture
def store(tmp_path):
    """Fresh StateStore backed by a temp SQLite file."""
    return StateStore(tmp_path / "state.db")


# ---------------------------------------------------------------------------
# Round-trip persistence
# ---------------------------------------------------------------------------


def test_persistence_round_trip(tmp_path):
    """State saved to a StateStore must be readable from a new instance
    pointing at the same file, confirming that data is durably written and
    not just held in memory.
    """
    db_path = tmp_path / "state.db"
    state = {
        "circuit_state": "open",
        "failure_count": 3,
        "last_failure": 1700000000.0,
        "latency_avg": 420.5,
        "reliability": 0.88,
        "request_count": 47,
    }

    # Write via first instance.
    store_a = StateStore(db_path)
    store_a.save("primary", state)

    # Read via a completely new instance (no shared in-memory state).
    store_b = StateStore(db_path)
    loaded = store_b.load("primary")

    assert loaded is not None, "load() returned None — state was not persisted"
    assert loaded["circuit_state"] == state["circuit_state"]
    assert loaded["failure_count"] == state["failure_count"]
    assert abs(loaded["last_failure"] - state["last_failure"]) < 0.001
    assert abs(loaded["latency_avg"] - state["latency_avg"]) < 0.001
    assert abs(loaded["reliability"] - state["reliability"]) < 0.001
    assert loaded["request_count"] == state["request_count"]


def test_save_then_update_reflects_new_values(store):
    """Calling save() twice for the same model must update, not duplicate."""
    store.save(
        "primary",
        {
            "circuit_state": "closed",
            "failure_count": 0,
            "last_failure": None,
            "latency_avg": 300.0,
            "reliability": 0.95,
            "request_count": 20,
        },
    )
    store.save(
        "primary",
        {
            "circuit_state": "open",
            "failure_count": 3,
            "last_failure": 1700000000.0,
            "latency_avg": 900.0,
            "reliability": 0.70,
            "request_count": 25,
        },
    )

    loaded = store.load("primary")
    assert loaded["circuit_state"] == "open"
    assert loaded["failure_count"] == 3


# ---------------------------------------------------------------------------
# Missing model sentinel
# ---------------------------------------------------------------------------


def test_unknown_model_returns_none(store):
    """load() for a model that has never been saved must return None."""
    result = store.load("nonexistent_slot_xyz")
    assert result is None, (
        f"Expected None for an unknown model slot, but got: {result!r}"
    )


# ---------------------------------------------------------------------------
# load_all
# ---------------------------------------------------------------------------


def test_load_all_returns_all_saved_models(store):
    """load_all() must return one entry per saved model slot."""
    store.save(
        "primary",
        {
            "circuit_state": "closed",
            "failure_count": 0,
            "last_failure": None,
            "latency_avg": 300.0,
            "reliability": 0.95,
            "request_count": 10,
        },
    )
    store.save(
        "secondary",
        {
            "circuit_state": "closed",
            "failure_count": 0,
            "last_failure": None,
            "latency_avg": 500.0,
            "reliability": 0.90,
            "request_count": 5,
        },
    )

    all_rows = store.load_all()
    assert "primary" in all_rows
    assert "secondary" in all_rows
    assert all_rows["primary"]["circuit_state"] == "closed"
