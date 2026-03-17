import json
import logging
import pathlib
import re
from datetime import datetime

HISTORY_DIR = pathlib.Path(__file__).resolve().parents[2] / "res" / "data" / "chat_history"

logger = logging.getLogger(__name__)

_SAFE_SESSION_ID = re.compile(r'^[a-zA-Z0-9_\-]+$')


def save_history(session_id: str, messages: list) -> None:
    """Serialize chat messages to JSON file keyed by session_id."""
    if not _SAFE_SESSION_ID.match(session_id):
        raise ValueError(f"Invalid session_id: {session_id!r}")
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = HISTORY_DIR / f"{session_id}.json"
    data = [{"type": m.type, "content": m.content} for m in messages]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def load_history(session_id: str) -> list[dict]:
    """Load chat messages from JSON file. Returns empty list if not found."""
    path = HISTORY_DIR / f"{session_id}.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("Corrupted chat history file %s; returning empty history.", path)
        return []


def list_sessions() -> list[dict]:
    """Return list of saved sessions with id and last-modified time."""
    if not HISTORY_DIR.exists():
        return []
    sessions = []
    files_with_mtime = [(p, p.stat().st_mtime) for p in HISTORY_DIR.glob("*.json")]
    for p, mtime in sorted(files_with_mtime, key=lambda x: x[1], reverse=True):
        sessions.append({
            "id": p.stem,
            "modified": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M"),
        })
    return sessions
