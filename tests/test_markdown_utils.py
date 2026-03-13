"""Unit tests for src.utils.markdown_utils.sanitize_markdown."""

import sys
import pathlib

# Ensure project root is on sys.path
_BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from src.utils.markdown_utils import sanitize_markdown


def test_unclosed_bold():
    """**text* should become **text**."""
    assert sanitize_markdown("**$3,000*") == "**$3,000**"


def test_unclosed_bold_inline():
    """Unclosed bold inside a sentence should be fixed."""
    assert sanitize_markdown("Price is **$3,000* per unit.") == "Price is **$3,000** per unit."


def test_stray_asterisk_mid_sentence():
    """Stray asterisks around punctuation should be removed."""
    assert sanitize_markdown("text *.* more text") == "text . more text"


def test_clean_bold_passthrough():
    """Valid **bold** should pass through unchanged."""
    assert sanitize_markdown("**bold**") == "**bold**"


def test_clean_italic_passthrough():
    """Valid *italic* should pass through unchanged."""
    assert sanitize_markdown("*italic*") == "*italic*"


def test_bullet_list_dash_unaffected():
    """Dash bullet list lines should be unaffected."""
    assert sanitize_markdown("- item") == "- item"


def test_bullet_list_asterisk_unaffected():
    """Asterisk bullet list lines should be unaffected."""
    assert sanitize_markdown("* item") == "* item"


def test_plain_text_unaffected():
    """Plain text with no asterisks should be unchanged."""
    assert sanitize_markdown("Hello world") == "Hello world"


def test_header_unaffected():
    """Markdown headers should be unaffected."""
    assert sanitize_markdown("## Heading") == "## Heading"


def test_inline_code_unaffected():
    """Inline code spans should not be modified."""
    result = sanitize_markdown("Use `**bold**` in markdown.")
    assert "`**bold**`" in result


def test_fenced_code_block_unaffected():
    """Fenced code blocks should not be modified."""
    code = "```\n**unclosed* bold\n```"
    assert sanitize_markdown(code) == code


def test_bold_and_italic_together():
    """Both bold and italic in the same text should be preserved."""
    text = "Use **bold** and *italic* together."
    assert sanitize_markdown(text) == text


def test_multiline_preserves_structure():
    """Multi-line text preserves all valid formatting."""
    text = "## Section\n\n**bold** text and *italic* text.\n\n- item one\n- item two"
    assert sanitize_markdown(text) == text


def test_stray_asterisk_at_end():
    """A lone trailing asterisk should be removed."""
    assert sanitize_markdown("some text*") == "some text"


def test_multiple_unclosed_bolds():
    """Multiple unclosed bold spans on different lines should all be fixed."""
    text = "**price* here\n**value* there"
    result = sanitize_markdown(text)
    assert result == "**price** here\n**value** there"
