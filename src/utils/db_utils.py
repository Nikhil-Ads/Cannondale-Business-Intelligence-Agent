"""
Utility functions for connecting to the Chroma vector store used by the MVP
BI Agent.
"""

import pathlib

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Default persist directory (relative to the project root)
DEFAULT_PERSIST_DIR = str(
    pathlib.Path(__file__).resolve().parents[2] / "res" / "data" / "cannondale_vectorstore"
)

DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"


def get_chroma_vectorstore(
    persist_dir: str = DEFAULT_PERSIST_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Chroma:
    """
    Load and return a Chroma vector store from a persisted directory.

    Parameters
    ----------
    persist_dir : str
        Path to the persisted Chroma directory.
    embedding_model : str
        OpenAI embedding model name.

    Returns
    -------
    Chroma
        A Chroma vector store instance ready for queries.
    """
    embedding_function = OpenAIEmbeddings(model=embedding_model)

    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_function,
    )

    return vectorstore


def get_retriever(vectorstore: Chroma, k: int = 5):
    """
    Create a retriever from a Chroma vector store.

    Parameters
    ----------
    vectorstore : Chroma
        The Chroma vector store to build a retriever from.
    k : int
        Number of top results to retrieve.

    Returns
    -------
    VectorStoreRetriever
        A LangChain retriever.
    """
    return vectorstore.as_retriever(search_kwargs={"k": k})
