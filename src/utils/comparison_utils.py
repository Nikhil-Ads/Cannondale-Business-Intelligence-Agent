"""Utility functions for detecting bike comparison intent and extracting bike names."""

import re

_COMPARISON_PATTERNS = [
    r'\bcompare\b.+\b(?:vs?|versus|and|with|to)\b',
    r'\b(?:vs?|versus)\b',
    r'\bdifferences?\s+between\b',
    r'\bhow\s+does\b.+\bcompare\b',
    r'\bwhich\s+is\s+better\b',
    r'\bside[\s\-]by[\s\-]side\b',
    r'\bpros?\s+and\s+cons?\b.+\bvs?\b',
    r'\bcontrast\b',
]

_COMPARISON_RE = re.compile(
    '|'.join(_COMPARISON_PATTERNS),
    re.IGNORECASE,
)


def detect_comparison_intent(question: str) -> bool:
    """Return True if the question is asking to compare two or more bikes.

    Uses regex heuristics for speed and reliability -- no LLM call needed.
    """
    return bool(_COMPARISON_RE.search(question))


def extract_bike_names(question: str, model: str = "gpt-4o-mini") -> list[str]:
    """Extract bike model names from a comparison question using an LLM.

    Uses a deterministic (temperature=0) LLM call for reliable name extraction
    since model names vary significantly (e.g. "Carbon 1", "LAB71",
    "Synapse Carbon 3 SmartSense").

    Returns a list of 2+ bike name strings. Returns an empty list on failure.
    """
    try:
        from langchain_openai import ChatOpenAI  # noqa: PLC0415

        llm = ChatOpenAI(model=model, temperature=0)

        prompt = (
            "You are a Cannondale product expert. Extract ALL bike model names "
            "from the following question. Return ONLY a Python-style list of "
            "strings with the exact model names as mentioned by the user "
            "(e.g. [\"Synapse Carbon 1\", \"Synapse Carbon 2\"]). "
            "Do not add any explanation, commentary, or extra text.\n\n"
            f"Question: {question}"
        )

        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        text = text.strip()

        # Parse the list representation returned by the LLM
        import ast
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(n).strip() for n in parsed if str(n).strip()]

        return []
    except (ValueError, SyntaxError, ImportError, RuntimeError):
        return []
