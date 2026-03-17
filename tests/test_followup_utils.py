"""Unit tests for src/utils/followup_utils.parse_followup_response."""

import sys
import pathlib

# Ensure project root is on sys.path
_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.followup_utils import parse_followup_response


def test_numbered_list_basic():
    text = "1. What bikes are available?\n2. What is the price range?"
    result = parse_followup_response(text)
    assert result == ["What bikes are available?", "What is the price range?"]


def test_numbered_list_three_items():
    text = "1. What is the lightest model?\n2. Which has disc brakes?\n3. What is LAB71?"
    result = parse_followup_response(text)
    assert len(result) == 3
    assert result[0] == "What is the lightest model?"
    assert result[2] == "What is LAB71?"


def test_numbered_list_max_three_items_truncated():
    text = "1. Q one?\n2. Q two?\n3. Q three?\n4. Q four?"
    result = parse_followup_response(text)
    assert len(result) == 3
    assert "Q four?" not in result


def test_bullet_list():
    text = "- Option A\n- Option B"
    result = parse_followup_response(text)
    assert result == ["Option A", "Option B"]


def test_bullet_list_asterisk():
    text = "* First question?\n* Second question?\n* Third question?"
    result = parse_followup_response(text)
    assert len(result) == 3
    assert result[0] == "First question?"


def test_json_array():
    text = '["What is the price?", "Which model is fastest?", "Are there electric options?"]'
    result = parse_followup_response(text)
    assert len(result) == 3
    assert result[0] == "What is the price?"


def test_plain_lines():
    text = "How does SmartSense work?\nWhat is the carbon grade?"
    result = parse_followup_response(text)
    assert "How does SmartSense work?" in result
    assert "What is the carbon grade?" in result


def test_empty_string_returns_empty():
    assert parse_followup_response("") == []


def test_whitespace_only_returns_empty():
    assert parse_followup_response("   \n\n  ") == []


def test_strips_whitespace_from_items():
    text = "1.   Spaced question?   \n2.  Another one?  "
    result = parse_followup_response(text)
    assert result[0] == "Spaced question?"
    assert result[1] == "Another one?"


def test_numbered_list_with_paren():
    text = "1) First?\n2) Second?\n3) Third?"
    result = parse_followup_response(text)
    assert len(result) == 3
    assert result[0] == "First?"
