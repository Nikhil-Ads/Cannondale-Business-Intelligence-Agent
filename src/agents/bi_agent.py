"""
MVP Business Intelligence Agent using LangGraph.

Implements a simple RAG pipeline:
    retrieve  -->  generate_answer  -->  END

Fixed model: gpt-4o-mini  (no user-facing option to change).
Text-only insights (no charts, no tables).
"""

import sys
import pathlib

# Ensure the project root is on sys.path so `src.*` imports work
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from src.utils.db_utils import get_chroma_vectorstore, get_retriever

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIXED_MODEL = "gpt-4o-mini"

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

Provide clear, detailed explanations that help customers make informed
decisions. Return text-only insights — no charts, no tables."""

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM_PROMPT),
        ("human", "{user_question}"),
    ]
)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    """State flowing through the LangGraph RAG pipeline."""

    user_question: str
    retrieved_docs: list
    answer: str


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def make_mvp_rag_agent(persist_dir: str = DEFAULT_PERSIST_DIR):
    """
    Build and compile the MVP RAG agent graph.

    Parameters
    ----------
    persist_dir : str
        Path to the persisted Chroma vector store directory.

    Returns
    -------
    CompiledStateGraph
        Ready-to-invoke LangGraph application.
    """

    # Retriever
    vectorstore = get_chroma_vectorstore(persist_dir=persist_dir)
    retriever = get_retriever(vectorstore, k=5)

    # LLM (fixed)
    llm = ChatOpenAI(model=FIXED_MODEL, temperature=0.7)

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
        return {"retrieved_docs": docs}

    def generate_answer(state: GraphState) -> dict:
        """Generate a text answer from the retrieved context."""
        print("--- GENERATE ANSWER ---")
        docs = state["retrieved_docs"]
        context = "\n\n".join(doc.page_content for doc in docs)
        question = state["user_question"]

        response = answer_chain.invoke(
            {"context": context, "user_question": question}
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
