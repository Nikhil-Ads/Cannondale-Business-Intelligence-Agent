"""
Utility functions for computing confidence levels from ChromaDB retrieval
similarity scores.
"""

HIGH_THRESHOLD = 0.75
MEDIUM_THRESHOLD = 0.50


def compute_confidence(scores: list[float]) -> dict:
    """
    Compute a confidence level from a list of 0-1 relevance scores.

    Parameters
    ----------
    scores : list[float]
        List of similarity scores in the [0, 1] range returned by ChromaDB.

    Returns
    -------
    dict with keys:
        level     : str   — "High", "Medium", or "Low"
        emoji     : str   — "🟢", "🟡", or "🔴"
        top_score : float — max score rounded to 3 decimal places
    """
    if not scores:
        return {"level": "Low", "emoji": "🔴", "top_score": 0.0}

    top = round(max(scores), 3)

    if top >= HIGH_THRESHOLD:
        return {"level": "High", "emoji": "🟢", "top_score": top}
    elif top >= MEDIUM_THRESHOLD:
        return {"level": "Medium", "emoji": "🟡", "top_score": top}
    else:
        return {"level": "Low", "emoji": "🔴", "top_score": top}
