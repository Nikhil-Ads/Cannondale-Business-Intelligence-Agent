"""Utility functions for generating follow-up question suggestions."""

import json
import re

try:
    import yaml
    import pathlib

    _BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
    _CREDENTIALS_PATH = _BASE_DIR / "credentials.yml"

    import os
    os.environ.setdefault(
        "OPENAI_API_KEY",
        yaml.safe_load(open(_CREDENTIALS_PATH))["openai"],
    )
except Exception:
    pass  # credentials may already be set via env var


def parse_followup_response(text: str) -> list[str]:
    """Parse a model response into a list of follow-up question strings.

    Handles numbered lists ("1. ..."), bullet lists ("- ..."),
    comma-separated values, or JSON arrays.  Returns up to 3 non-empty strings.
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    # Try JSON array first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            items = [str(s).strip() for s in parsed if str(s).strip()]
            return items[:3]
    except (json.JSONDecodeError, ValueError):
        pass

    # Try numbered list: "1. ...", "2) ...", etc.
    numbered = re.findall(r'^\s*\d+[\.\)]\s+(.+)', text, re.MULTILINE)
    if numbered:
        return [s.strip() for s in numbered if s.strip()][:3]

    # Try bullet list: "- ..." or "* ..."
    bullets = re.findall(r'^\s*[-*]\s+(.+)', text, re.MULTILINE)
    if bullets:
        return [s.strip() for s in bullets if s.strip()][:3]

    # Try line-by-line (plain list, one item per line)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 2:
        return lines[:3]

    # Comma-separated fallback
    parts = [p.strip() for p in text.split(',') if p.strip()]
    if len(parts) >= 2:
        return parts[:3]

    return []


def generate_followup_suggestions(
    question: str,
    answer: str,
    model: str = "gpt-4o-mini",
) -> list[str]:
    """Generate 2-3 follow-up question suggestions using an OpenAI chat model.

    Returns an empty list on any failure — never raises.
    """
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=model, temperature=0.3)

        prompt = (
            "Based on the following question and answer about Cannondale bicycles, "
            "suggest 2-3 short follow-up questions the user might want to ask next.\n\n"
            f"Question: {question}\n\n"
            f"Answer: {answer}\n\n"
            "Return a numbered list of 2-3 concise follow-up questions only. "
            "No explanations, no preamble — just the numbered list."
        )

        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        return parse_followup_response(text)
    except Exception:
        return []
