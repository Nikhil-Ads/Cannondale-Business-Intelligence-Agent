"""Utilities for sanitizing potentially malformed markdown from LLM output."""

import re


def sanitize_markdown(text: str) -> str:
    """Sanitize potentially malformed markdown returned by an LLM.

    Fixes:
    - Unclosed bold markers: **text* → **text**
    - Stray single asterisks not part of valid *italic* or **bold** syntax

    Preserves:
    - **bold**, *italic*, - bullet lists, ## headers, code blocks
    """
    # Stash code blocks to avoid modifying them
    _stash: list[str] = []

    def _stash_block(m: re.Match) -> str:
        _stash.append(m.group(0))
        return f'\x00S{len(_stash) - 1}\x00'

    # Fenced code blocks first (multi-line), then inline code
    text = re.sub(r'```[\s\S]*?```', _stash_block, text)
    text = re.sub(r'`[^`\n]+`', _stash_block, text)

    # Fix unclosed bold: **text* → **text**
    # (?<!\w) ensures we only match opening ** (not the closing ** of valid bold).
    # Matches ** then content without * or newline (non-greedy),
    # then a single * not followed by another *.
    text = re.sub(r'(?<!\w)\*\*([^*\n]+?)\*(?!\*)', r'**\1**', text)

    # Remove stray single asterisks line by line
    text = '\n'.join(_fix_line(line) for line in text.split('\n'))

    # Restore stashed code blocks
    for i, block in enumerate(_stash):
        text = text.replace(f'\x00S{i}\x00', block)

    return text


def _fix_line(line: str) -> str:
    """Remove stray single asterisks from one line of text."""
    # Leave bullet list lines untouched (- item or * item)
    if re.match(r'^\s*[-*]\s+\S', line):
        return line

    # Replace ** with a placeholder so singles are easier to identify
    line = line.replace('**', '\x00B\x00')

    # Protect valid italic spans: *content* where content has ≥1 word char
    line = re.sub(r'\*([^*\n]*\w[^*\n]*)\*', '\x00I\x00\\1\x00E\x00', line)

    # Any remaining * is stray — remove it
    line = line.replace('*', '')

    # Restore placeholders
    line = line.replace('\x00B\x00', '**')
    line = line.replace('\x00I\x00', '*')
    line = line.replace('\x00E\x00', '*')

    return line
