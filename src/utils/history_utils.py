import json
import pathlib
from datetime import datetime

HISTORY_DIR = pathlib.Path(__file__).resolve().parents[2] / "res" / "data" / "chat_history"


def save_history(session_id: str, messages: list) -> None:
    """Serialize chat messages to JSON file keyed by session_id."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = HISTORY_DIR / f"{session_id}.json"
    data = [{"type": m.type, "content": m.content} for m in messages]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def load_history(session_id: str) -> list[dict]:
    """Load chat messages from JSON file. Returns empty list if not found."""
    path = HISTORY_DIR / f"{session_id}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def list_sessions() -> list[dict]:
    """Return list of saved sessions with id and last-modified time."""
    if not HISTORY_DIR.exists():
        return []
    sessions = []
    for p in sorted(HISTORY_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        sessions.append({
            "id": p.stem,
            "modified": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        })
    return sessions
