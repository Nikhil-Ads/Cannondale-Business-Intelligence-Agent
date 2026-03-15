"""Tests for src/backend/feedback_db.py using a temporary DB path."""

import pathlib
import pytest
import src.backend.feedback_db as fdb


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    """Redirect DB_PATH to a temp file for each test."""
    db_file = tmp_path / "feedback_test.db"
    monkeypatch.setattr(fdb, "DB_PATH", db_file)
    yield db_file


def test_init_creates_table(tmp_db):
    fdb.init_db()
    import sqlite3
    with sqlite3.connect(tmp_db) as conn:
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
    assert "feedback" in tables


def test_save_feedback_up(tmp_db):
    fdb.init_db()
    fdb.save_feedback("sess1", 1, "What is a Synapse?", "The Synapse is...", "up")
    import sqlite3
    with sqlite3.connect(tmp_db) as conn:
        row = conn.execute("SELECT * FROM feedback").fetchone()
    assert row is not None
    # rating is at index 5 (id, session_id, ai_index, question, answer_snippet, rating, created_at)
    assert row[5] == "up"
    assert row[1] == "sess1"
    assert row[2] == 1


def test_save_feedback_down(tmp_db):
    fdb.init_db()
    fdb.save_feedback("sess2", 2, "Compare models?", "Carbon 1 vs...", "down")
    import sqlite3
    with sqlite3.connect(tmp_db) as conn:
        row = conn.execute("SELECT * FROM feedback").fetchone()
    assert row[5] == "down"


def test_duplicate_ignored(tmp_db):
    fdb.init_db()
    fdb.save_feedback("sess3", 1, "Q?", "A.", "up")
    fdb.save_feedback("sess3", 1, "Q?", "A.", "down")  # same session_id + ai_index
    import sqlite3
    with sqlite3.connect(tmp_db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    assert count == 1


def test_get_summary_empty(tmp_db):
    fdb.init_db()
    summary = fdb.get_summary()
    assert summary == {"total": 0, "ups": 0, "downs": 0, "satisfaction_pct": 0.0}


def test_get_summary_with_data(tmp_db):
    fdb.init_db()
    fdb.save_feedback("s1", 1, "Q1", "A1", "up")
    fdb.save_feedback("s1", 2, "Q2", "A2", "up")
    fdb.save_feedback("s1", 3, "Q3", "A3", "down")
    summary = fdb.get_summary()
    assert summary["total"] == 3
    assert summary["ups"] == 2
    assert summary["downs"] == 1
    assert summary["satisfaction_pct"] == round(2 / 3 * 100, 1)


def test_save_feedback_never_raises(tmp_db):
    # DB not initialised — should not raise
    fdb.save_feedback(None, None, None, None, "up")  # type: ignore[arg-type]
    fdb.save_feedback("s", 1, "q", "a", "invalid_rating")


def test_get_recent_returns_list(tmp_db):
    fdb.init_db()
    fdb.save_feedback("s1", 1, "Q1", "A1", "up")
    rows = fdb.get_recent(limit=10)
    assert isinstance(rows, list)
    assert len(rows) == 1
    assert rows[0]["rating"] == "up"
    assert rows[0]["session_id"] == "s1"
