"""
MVP Business Intelligence Agent using LangGraph.

Implements a simple RAG pipeline:
    retrieve  -->  generate_answer  -->  END

Also supports a multi-pass Critical Thinking mode:
    decompose --> [retrieve + answer sub-question] x N --> critique --> synthesize --> END

Model is configurable via make_mvp_rag_agent(model=...).
Supports optional <chart_data> JSON blocks for front-end visualizations.
"""

import sys
import pathlib

# Ensure the project root is on sys.path so `src.*` imports work
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import re
from typing import List, Tuple
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph

from src.utils.db_utils import get_chroma_vectorstore, get_retriever

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gpt-4o-mini"

AVAILABLE_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]

DEFAULT_PERSIST_DIR = str(
    pathlib.Path(__file__).resolve().parents[2]
    / "res"
    / "data"
    / "cannondale_vectorstore"
)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

QA_SYSTEM_PROMPT = """\
You are an expert assistant specializing in Cannondale Synapse bicycles.
Use the following pieces of retrieved context from the Cannondale website
to answer questions about their bikes.

When answering:
- Be specific about model names, features, and specifications.
- Compare different models when asked.
- Mention relevant technologies (like SmartSense) and their benefits.
- Provide helpful recommendations based on the user's needs.
- If you don't know something, say so honestly.
- Always be enthusiastic and helpful about cycling!

Context:
{context}

Provide clear, detailed explanations that help customers make informed decisions.

When your response contains comparable data across multiple models — such as
prices, weights, component specs, or feature lists — optionally append a single
<chart_data> JSON block at the very end of your response (after all prose) so
the UI can render an interactive chart. Use this format:

For bar/line charts:
<chart_data>
{{"type": "bar", "title": "...", "x": ["Model A", "Model B"], "y": [1299, 1999], "labels": {{"x": "Model", "y": "Price (USD)"}}}}
</chart_data>

For tabular comparisons:
<chart_data>
{{"type": "table", "title": "...", "columns": ["Model", "Price", "Weight"], "rows": [["Carbon 1", "$3999", "8.1 kg"], ["Carbon 2", "$2999", "8.4 kg"]]}}
</chart_data>

Only include the <chart_data> block when it genuinely adds value (skip it for
simple factual questions). Do not include markdown fences or extra text inside
the block — only valid JSON."""

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{user_question}"),
    ]
)


# ---------------------------------------------------------------------------
# Critical Thinking prompts
# ---------------------------------------------------------------------------

DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert analyst specialising in Cannondale Synapse bicycles. "
        "Your job is to decompose a user's question into {n_subquestions} clear, "
        "focused sub-questions that together fully cover the original question. "
        "Return ONLY a numbered list of sub-questions, nothing else."
    )),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "Question: {user_question}"),
])

SUBQUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert assistant specialising in Cannondale Synapse bicycles. "
        "Use the retrieved context below to answer the specific sub-question. "
        "Be concise and factual.\n\nContext:\n{context}"
    )),
    ("human", "Sub-question: {sub_question}"),
])

CRITIQUE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a critical reviewer. Given the original question and the "
        "sub-question answers below, identify: gaps in the analysis, "
        "any contradictions, and what additional context would improve "
        "the final answer. Be brief and specific."
    )),
    ("human", (
        "Original question: {user_question}\n\n"
        "Sub-question answers:\n{sub_answers}"
    )),
])

SYNTHESIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert assistant specialising in Cannondale Synapse bicycles. "
        "Synthesize the sub-question answers and critique into a single, "
        "comprehensive, well-structured answer to the original question. "
        "Do not mention the reasoning process — just deliver the best possible answer. "
        "When your answer contains comparable data across models (prices, specs, "
        "features), optionally append a <chart_data> JSON block at the very end "
        "using format: "
        "{{'type': 'bar'|'line'|'table', 'title': '...', 'x': [...], 'y': [...], "
        "'labels': {{'x': '...', 'y': '...'}}}} for charts or "
        "{{'type': 'table', 'title': '...', 'columns': [...], 'rows': [[...]]}} "
        "for tables. Only include it when it adds real value."
    )),
    ("human", (
        "Original question: {user_question}\n\n"
        "Sub-question answers:\n{sub_answers}\n\n"
        "Critique / gaps:\n{critique}"
    )),
])


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    """State flowing through the LangGraph RAG pipeline."""

    user_question: str
    chat_history: list  # list of LangChain BaseMessage objects (HumanMessage/AIMessage)
    retrieved_docs: list
    answer: str
    sources: list  # deduplicated source URLs from retrieved docs


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def make_mvp_rag_agent(persist_dir: str = DEFAULT_PERSIST_DIR, model: str = DEFAULT_MODEL):
    """
    Build and compile the MVP RAG agent graph.

    Parameters
    ----------
    persist_dir : str
        Path to the persisted Chroma vector store directory.
    model : str
        OpenAI chat model name to use for answer generation.
        Defaults to DEFAULT_MODEL.

    Returns
    -------
    CompiledStateGraph
        Ready-to-invoke LangGraph application.
    """

    # Retriever
    vectorstore = get_chroma_vectorstore(persist_dir=persist_dir)
    retriever = get_retriever(vectorstore, k=5)

    # LLM (configurable)
    llm = ChatOpenAI(model=model, temperature=0.7)

    # Chain for answer generation
    answer_chain = QA_PROMPT | llm

    # ------------------------------------------------------------------
    # Node functions
    # ------------------------------------------------------------------

    def retrieve(state: GraphState) -> dict:
        """Retrieve relevant documents from the vector store."""
        print("--- RETRIEVE ---")
        question = state["user_question"]
        docs = retriever.invoke(question)

        # Extract and deduplicate source URLs from document metadata
        seen = set()
        sources = []
        for doc in docs:
            src = doc.metadata.get("source") or doc.metadata.get("url") or doc.metadata.get("Source")
            if src and src not in seen and src != "generated_toc":
                seen.add(src)
                sources.append(src)

        return {"retrieved_docs": docs, "sources": sources}

    def generate_answer(state: GraphState) -> dict:
        """Generate a text answer from the retrieved context."""
        print("--- GENERATE ANSWER ---")
        docs = state["retrieved_docs"]
        context = "\n\n".join(doc.page_content for doc in docs)
        question = state["user_question"]

        response = answer_chain.invoke(
            {"context": context, "user_question": question, "chat_history": state.get("chat_history", [])}
        )

        return {"answer": response.content}

    # ------------------------------------------------------------------
    # Build graph
    # ------------------------------------------------------------------

    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate_answer", generate_answer)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Critical Thinking Agent (multi-pass)
# ---------------------------------------------------------------------------

def run_critical_thinking_agent(
    user_question: str,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    model: str = DEFAULT_MODEL,
    num_subquestions: int = 3,
    chat_history: list = None,
) -> dict:
    """
    Multi-pass critical thinking agent.

    Stages:
      1. Decompose question into sub-questions
      2..N. RAG retrieval + answer each sub-question
      N+1. Critique — find gaps / contradictions
      N+2. Synthesize into final answer

    Parameters
    ----------
    user_question : str
        The user's original question.
    persist_dir : str
        Path to the persisted Chroma vector store.
    model : str
        OpenAI model to use.
    num_subquestions : int
        Number of sub-questions to decompose the query into (min 1, max 5).
        Total stages = num_subquestions + 3 (decompose, critique, synthesize).

    Returns
    -------
    dict with keys:
        answer     : str   — final synthesized answer
        sources    : list  — deduplicated source URLs
        reasoning  : list  — list of (label, text) tuples for display
    """
    n_subquestions = max(1, min(5, num_subquestions))

    vectorstore = get_chroma_vectorstore(persist_dir=persist_dir)
    retriever = get_retriever(vectorstore, k=5)
    llm = ChatOpenAI(model=model, temperature=0.7)

    reasoning: List[Tuple[str, str]] = []
    all_sources: List[str] = []

    # ------------------------------------------------------------------
    # Pass 1 — Decompose
    # ------------------------------------------------------------------
    print(f"--- CRITICAL THINKING: DECOMPOSE (n_subquestions={n_subquestions}) ---")
    decompose_chain = DECOMPOSE_PROMPT | llm
    decompose_response = decompose_chain.invoke({
        "user_question": user_question,
        "n_subquestions": n_subquestions,
        "chat_history": chat_history or [],
    })
    decompose_text = decompose_response.content.strip()
    reasoning.append(("🔍 Pass 1 — Decompose", decompose_text))

    # Parse sub-questions from numbered list; handle "1.", "1)", "1:" formats
    sub_questions = [
        m.group(1).strip()
        for line in decompose_text.splitlines()
        if (m := re.match(r'^\d+[.):\-]\s*(.+)', line.strip()))
    ]
    if not sub_questions:
        sub_questions = [user_question]

    # ------------------------------------------------------------------
    # Passes 2..N-1 — Research each sub-question
    # ------------------------------------------------------------------
    sub_answers_parts = []
    subq_chain = SUBQUESTION_PROMPT | llm

    for i, sub_q in enumerate(sub_questions, start=1):
        print(f"--- CRITICAL THINKING: SUB-QUESTION {i} ---")
        docs = retriever.invoke(sub_q)

        # Collect sources
        seen = set()
        for doc in docs:
            src = doc.metadata.get("source") or doc.metadata.get("url") or doc.metadata.get("Source")
            if src and src not in seen and src != "generated_toc":
                seen.add(src)
                if src not in all_sources:
                    all_sources.append(src)

        context = "\n\n".join(doc.page_content for doc in docs)
        sub_response = subq_chain.invoke({"context": context, "sub_question": sub_q})
        sub_answer = sub_response.content.strip()
        sub_answers_parts.append(f"**Q{i}: {sub_q}**\n{sub_answer}")
        reasoning.append((f"🔬 Pass {i + 1} — Sub-question {i}", f"**{sub_q}**\n\n{sub_answer}"))

    sub_answers_text = "\n\n".join(sub_answers_parts)

    # ------------------------------------------------------------------
    # Pass N-1 — Critique
    # ------------------------------------------------------------------
    print("--- CRITICAL THINKING: CRITIQUE ---")
    critique_chain = CRITIQUE_PROMPT | llm
    critique_response = critique_chain.invoke({
        "user_question": user_question,
        "sub_answers": sub_answers_text,
    })
    critique_text = critique_response.content.strip()
    critique_pass_num = len(sub_questions) + 2
    reasoning.append((f"🧐 Pass {critique_pass_num} — Critique", critique_text))

    # ------------------------------------------------------------------
    # Pass N — Synthesize
    # ------------------------------------------------------------------
    print("--- CRITICAL THINKING: SYNTHESIZE ---")
    synthesize_chain = SYNTHESIZE_PROMPT | llm
    synthesize_response = synthesize_chain.invoke({
        "user_question": user_question,
        "sub_answers": sub_answers_text,
        "critique": critique_text,
    })
    final_answer = synthesize_response.content.strip()
    synth_pass_num = critique_pass_num + 1
    reasoning.append((f"✅ Pass {synth_pass_num} — Synthesize", final_answer))

    return {
        "answer": final_answer,
        "sources": all_sources,
        "reasoning": reasoning,
    }
