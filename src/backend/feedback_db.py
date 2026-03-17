"""SQLite persistence layer for user feedback events."""

import logging
import sqlite3
import pathlib
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DB_PATH = pathlib.Path(__file__).resolve().parents[2] / "res" / "data" / "feedback.db"


def init_db() -> None:
    """Create the feedback table if it does not exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL,
                ai_index    INTEGER NOT NULL,
                question    TEXT,
                answer_snippet TEXT,
                rating      TEXT    NOT NULL CHECK(rating IN ('up', 'down')),
                created_at  TEXT    NOT NULL,
                UNIQUE(session_id, ai_index)
            )
        """)
        conn.commit()


def save_feedback(
    session_id: str,
    ai_index: int,
    question: str,
    answer_snippet: str,
    rating: str,
) -> None:
    """Insert a feedback event. Silently ignores duplicate (session_id, ai_index)."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """INSERT OR IGNORE INTO feedback
                   (session_id, ai_index, question, answer_snippet, rating, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, ai_index, (question or "")[:500], (answer_snippet or "")[:200],
                 rating, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
    except Exception:
        logger.debug("Failed to save feedback", exc_info=True)


def get_summary() -> dict:
    """Return aggregate feedback stats."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                "SELECT COUNT(*) total, SUM(rating='up') ups, SUM(rating='down') downs FROM feedback"
            ).fetchone()
        total, ups, downs = row
        total = total or 0
        ups = ups or 0
        downs = downs or 0
        return {
            "total": total,
            "ups": ups,
            "downs": downs,
            "satisfaction_pct": round(ups / total * 100, 1) if total else 0.0,
        }
    except Exception:
        logger.debug("Failed to retrieve feedback summary", exc_info=True)
        return {"total": 0, "ups": 0, "downs": 0, "satisfaction_pct": 0.0}


def get_recent(limit: int = 50) -> list[dict]:
    """Return the most recent feedback rows as dicts."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM feedback ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []
