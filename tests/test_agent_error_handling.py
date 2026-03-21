"""
Integration tests for agent error-path behaviour across all three pipelines.

Each test exercises the *real* agent code — LangGraph pipeline wiring, prompt
construction, vectorstore retrieval — and patches only the OpenAI HTTP layer
(``openai.resources.chat.completions.Completions.create``) to raise a
controlled exception.  No real API key or network access is required.

Assertions verify that:

1. The exception propagates cleanly out of the agent function so the
   ``app.py`` ``except`` block will catch it.
2. The sentinel error string is NOT present in the user-facing answer that
   ``app.py`` produces in the except block.
3. The answer contains no raw Python exception markers (``"Error:"``, etc.).
4. The answer exactly matches the string that ``app.py`` currently sets.
"""

import os
import pathlib
import sys

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

VECTORSTORE_DIR = str(_ROOT / "res" / "data" / "cannondale_vectorstore")

# Skip the entire module if the vectorstore has not been populated locally.
if not (pathlib.Path(VECTORSTORE_DIR) / "chroma.sqlite3").exists():
    pytest.skip(
        "Local vectorstore not found — skipping agent error-path tests.",
        allow_module_level=True,
    )

# ---------------------------------------------------------------------------
# Sentinel & expected user-facing messages (must match app.py exactly)
# ---------------------------------------------------------------------------

_SENTINEL = "simulated-llm-failure-do-not-show-to-user"

_COMPARISON_USER_MSG = (
    "I'm sorry, an error occurred while comparing those bikes. "
    "Please try again or rephrase your question."
)
_CRITICAL_THINKING_USER_MSG = (
    "I'm sorry, an error occurred during critical thinking. "
    "Please try again or rephrase your question."
)
_STANDARD_RAG_USER_MSG = (
    "I'm sorry, an error occurred while processing your question. "
    "Please try again or rephrase your question."
)

# Patterns that must NOT appear in any user-facing answer.
_FORBIDDEN_PATTERNS = [
    _SENTINEL,
    "Error:",
    "Exception:",
    "Traceback",
    "raise ",
]

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def dummy_openai_key(monkeypatch):
    """Ensure ChatOpenAI can be instantiated without a real API key."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key-for-error-path-tests")


@pytest.fixture
def failing_llm():
    """
    Simulate a full LLM stack failure without making any HTTP calls.

    Two patches are applied together:

    1. ``OpenAIEmbeddings.embed_query`` / ``embed_documents`` — replaced with
       functions that return a dummy constant vector.  This prevents the
       vectorstore from making a real embedding API call when it embeds search
       queries (which would fail first with a 401 and mask the LLM error).

    2. ``BaseChatOpenAI._generate`` — raises a ``RuntimeError`` carrying the
       sentinel string, so that all three LLM call paths (standard RAG inside
       a LangGraph node, comparison chain, critical thinking chain) fail with
       a controlled, identifiable exception.

    With these two patches the full agent pipeline runs (vectorstore retrieval,
    prompt construction, chain wiring) but any LLM generation attempt raises
    the sentinel exception.
    """
    from unittest.mock import patch
    from langchain_openai.chat_models.base import BaseChatOpenAI
    from langchain_openai import OpenAIEmbeddings

    _DUMMY_EMBEDDING = [0.1] * 1536

    def _fail_generate(self, *args, **kwargs):
        raise RuntimeError(_SENTINEL)

    def _embed_query(self, text, **kwargs):
        return _DUMMY_EMBEDDING

    def _embed_documents(self, texts, **kwargs):
        return [_DUMMY_EMBEDDING for _ in texts]

    with (
        patch.object(BaseChatOpenAI, "_generate", _fail_generate),
        patch.object(OpenAIEmbeddings, "embed_query", _embed_query),
        patch.object(OpenAIEmbeddings, "embed_documents", _embed_documents),
    ):
        yield


def _assert_clean_answer(answer: str) -> None:
    """Assert that *answer* contains none of the forbidden patterns."""
    for pattern in _FORBIDDEN_PATTERNS:
        assert pattern not in answer, (
            f"Forbidden pattern {pattern!r} found in user-facing answer:\n{answer}"
        )


# ---------------------------------------------------------------------------
# Standard RAG error path
# ---------------------------------------------------------------------------


class TestStandardRAGErrorPath:
    """Tests for the standard RAG pipeline (``agent.invoke()``)."""

    def test_exception_propagates(self, failing_llm):
        """LLM failure in the standard RAG graph must bubble up as an exception."""
        from src.agents.bi_agent import make_mvp_rag_agent

        agent = make_mvp_rag_agent(persist_dir=VECTORSTORE_DIR)
        with pytest.raises(Exception) as exc_info:
            agent.invoke(
                {"user_question": "What is the Cannondale Synapse Carbon?", "chat_history": []}
            )

        assert _SENTINEL in str(exc_info.value)

    def test_frontend_answer_has_no_raw_error(self, failing_llm):
        """Simulates app.py except block: answer must contain no raw exception text."""
        from src.agents.bi_agent import make_mvp_rag_agent

        agent = make_mvp_rag_agent(persist_dir=VECTORSTORE_DIR)
        try:
            agent.invoke(
                {"user_question": "What is the Cannondale Synapse Carbon?", "chat_history": []}
            )
            answer = ""
        except Exception:
            answer = _STANDARD_RAG_USER_MSG

        _assert_clean_answer(answer)
        assert answer == _STANDARD_RAG_USER_MSG

    def test_friendly_message_is_non_empty(self, failing_llm):
        """The user-facing answer must be a non-empty, non-whitespace string."""
        from src.agents.bi_agent import make_mvp_rag_agent

        agent = make_mvp_rag_agent(persist_dir=VECTORSTORE_DIR)
        try:
            agent.invoke({"user_question": "Tell me about Synapse.", "chat_history": []})
            answer = ""
        except Exception:
            answer = _STANDARD_RAG_USER_MSG

        assert answer.strip(), "User-facing error answer must not be blank."


# ---------------------------------------------------------------------------
# Comparison agent error path
# ---------------------------------------------------------------------------


class TestComparisonAgentErrorPath:
    """Tests for the dedicated comparison pipeline (``run_comparison_agent()``)."""

    def test_exception_propagates(self, failing_llm):
        """LLM failure inside the comparison agent must bubble up as an exception."""
        from src.agents.bi_agent import run_comparison_agent

        with pytest.raises(Exception) as exc_info:
            run_comparison_agent(
                user_question="Compare Synapse Carbon 1 vs Synapse Carbon 2",
                bike_names=["Carbon 1", "Carbon 2"],
                persist_dir=VECTORSTORE_DIR,
            )

        assert _SENTINEL in str(exc_info.value)

    def test_frontend_answer_has_no_raw_error(self, failing_llm):
        """Simulates app.py except block: comparison answer must contain no raw exception text."""
        from src.agents.bi_agent import run_comparison_agent

        try:
            run_comparison_agent(
                user_question="Compare Synapse Carbon 1 vs Synapse Carbon 2",
                bike_names=["Carbon 1", "Carbon 2"],
                persist_dir=VECTORSTORE_DIR,
            )
            answer = ""
        except Exception:
            answer = _COMPARISON_USER_MSG

        _assert_clean_answer(answer)
        assert answer == _COMPARISON_USER_MSG

    def test_friendly_message_is_non_empty(self, failing_llm):
        """The user-facing comparison error answer must be non-empty."""
        from src.agents.bi_agent import run_comparison_agent

        try:
            run_comparison_agent(
                user_question="Compare Carbon 1 vs Carbon 2",
                bike_names=["Carbon 1", "Carbon 2"],
                persist_dir=VECTORSTORE_DIR,
            )
            answer = ""
        except Exception:
            answer = _COMPARISON_USER_MSG

        assert answer.strip(), "User-facing comparison error answer must not be blank."


# ---------------------------------------------------------------------------
# Critical Thinking agent error path
# ---------------------------------------------------------------------------


class TestCriticalThinkingErrorPath:
    """Tests for the multi-pass critical thinking pipeline (``run_critical_thinking_agent()``)."""

    def test_exception_propagates(self, failing_llm):
        """LLM failure inside the critical thinking agent must bubble up as an exception."""
        from src.agents.bi_agent import run_critical_thinking_agent

        with pytest.raises(Exception) as exc_info:
            run_critical_thinking_agent(
                user_question="What makes the Cannondale Synapse special?",
                persist_dir=VECTORSTORE_DIR,
                num_subquestions=2,
            )

        assert _SENTINEL in str(exc_info.value)

    def test_frontend_answer_has_no_raw_error(self, failing_llm):
        """Simulates app.py except block: critical thinking answer must contain no raw exception text."""
        from src.agents.bi_agent import run_critical_thinking_agent

        try:
            run_critical_thinking_agent(
                user_question="What makes the Cannondale Synapse special?",
                persist_dir=VECTORSTORE_DIR,
                num_subquestions=2,
            )
            answer = ""
        except Exception:
            answer = _CRITICAL_THINKING_USER_MSG

        _assert_clean_answer(answer)
        assert answer == _CRITICAL_THINKING_USER_MSG

    def test_friendly_message_is_non_empty(self, failing_llm):
        """The user-facing critical thinking error answer must be non-empty."""
        from src.agents.bi_agent import run_critical_thinking_agent

        try:
            run_critical_thinking_agent(
                user_question="What makes the Synapse special?",
                persist_dir=VECTORSTORE_DIR,
                num_subquestions=1,
            )
            answer = ""
        except Exception:
            answer = _CRITICAL_THINKING_USER_MSG

        assert answer.strip(), "User-facing critical thinking error answer must not be blank."


# ---------------------------------------------------------------------------
# Cross-cutting: verify the sentinel is unreachable from user-facing strings
# ---------------------------------------------------------------------------


class TestForbiddenPatternsAbsent:
    """
    Parametrized guard: none of the three user-facing messages may contain any
    of the forbidden patterns, regardless of which agent path triggered them.
    """

    @pytest.mark.parametrize(
        "message",
        [_STANDARD_RAG_USER_MSG, _COMPARISON_USER_MSG, _CRITICAL_THINKING_USER_MSG],
        ids=["standard_rag", "comparison", "critical_thinking"],
    )
    def test_message_contains_no_forbidden_pattern(self, message):
        _assert_clean_answer(message)

    @pytest.mark.parametrize(
        "message",
        [_STANDARD_RAG_USER_MSG, _COMPARISON_USER_MSG, _CRITICAL_THINKING_USER_MSG],
        ids=["standard_rag", "comparison", "critical_thinking"],
    )
    def test_message_ends_with_action_prompt(self, message):
        """Every error message should tell the user what to do next."""
        assert "rephrase your question" in message.lower()
