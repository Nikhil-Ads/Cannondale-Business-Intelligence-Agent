"""Unit tests for src/utils/confidence_utils.py"""

import importlib.util
import pathlib
import sys
import pytest

# Import directly from file to avoid the src/utils/__init__.py side-effects
# (which transitively loads langchain_chroma and other heavy dependencies)
_MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "src" / "utils" / "confidence_utils.py"
_spec = importlib.util.spec_from_file_location("confidence_utils", _MODULE_PATH)
if _spec is None or _spec.loader is None:
    pytest.skip("Could not load confidence_utils module", allow_module_level=True)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

compute_confidence = _mod.compute_confidence
HIGH_THRESHOLD = _mod.HIGH_THRESHOLD
MEDIUM_THRESHOLD = _mod.MEDIUM_THRESHOLD


def test_high_confidence():
    result = compute_confidence([0.9, 0.8, 0.85])
    assert result["level"] == "High"
    assert result["emoji"] == "🟢"
    assert result["top_score"] == 0.9


def test_medium_confidence():
    result = compute_confidence([0.6, 0.55, 0.5])
    assert result["level"] == "Medium"
    assert result["emoji"] == "🟡"
    assert result["top_score"] == 0.6


def test_low_confidence():
    result = compute_confidence([0.3, 0.2, 0.1])
    assert result["level"] == "Low"
    assert result["emoji"] == "🔴"
    assert result["top_score"] == 0.3


def test_empty_list():
    result = compute_confidence([])
    assert result["level"] == "Low"
    assert result["emoji"] == "🔴"
    assert result["top_score"] == 0.0


def test_boundary_high_threshold():
    """Score exactly at HIGH_THRESHOLD should return High."""
    result = compute_confidence([HIGH_THRESHOLD])
    assert result["level"] == "High"
    assert result["emoji"] == "🟢"
    assert result["top_score"] == HIGH_THRESHOLD


def test_just_below_high_threshold():
    """Score just below HIGH_THRESHOLD should return Medium (assuming >= MEDIUM_THRESHOLD)."""
    score = round(HIGH_THRESHOLD - 0.001, 3)
    result = compute_confidence([score])
    assert result["level"] == "Medium"
    assert result["emoji"] == "🟡"


def test_boundary_medium_threshold():
    """Score exactly at MEDIUM_THRESHOLD should return Medium."""
    result = compute_confidence([MEDIUM_THRESHOLD])
    assert result["level"] == "Medium"
    assert result["emoji"] == "🟡"
    assert result["top_score"] == MEDIUM_THRESHOLD


def test_just_below_medium_threshold():
    """Score just below MEDIUM_THRESHOLD should return Low."""
    score = round(MEDIUM_THRESHOLD - 0.001, 3)
    result = compute_confidence([score])
    assert result["level"] == "Low"
    assert result["emoji"] == "🔴"


def test_top_score_rounded_to_3dp():
    result = compute_confidence([0.123456789])
    assert result["top_score"] == 0.123


def test_uses_max_score():
    """Confidence level is determined by the highest score in the list."""
    result = compute_confidence([0.2, 0.8, 0.4])
    assert result["top_score"] == 0.8
    assert result["level"] == "High"
