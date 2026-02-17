"""
One-time script to scrape Cannondale Synapse product pages and build
a Chroma vector store for the MVP BI Agent.

Usage (from project root):
    python scripts/create_cannondale_db.py
"""

import os
import re
import copy
import pathlib

import yaml
import pandas as pd
import nest_asyncio

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]  # project root

CSV_PATH = BASE_DIR / "res" / "cannondale_synapse_products.csv"
PERSIST_DIR = str(BASE_DIR / "res" / "data" / "cannondale_vectorstore")
CREDENTIALS_PATH = BASE_DIR / "credentials.yml"
EMBEDDING_MODEL = "text-embedding-ada-002"

# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open(CREDENTIALS_PATH))["openai"]

# ---------------------------------------------------------------------------
# Text cleaning (adapted from challenge_02_cannondale.py)
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Remove web navigation artifacts, normalise whitespace, filter noise."""

    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"Toggle navigation.*?Cannondale", "", text, flags=re.DOTALL)
    text = re.sub(r"© Cannondale.*", "", text, flags=re.DOTALL)
    text = re.sub(r"Skip to main content", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Cookie Policy.*", "", text, flags=re.DOTALL)

    text = text.replace("\xa0", " ")
    text = text.replace("\u2019", "'")
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "--")

    text = re.sub(
        r"[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\'\"\+\=\*\/\\\<\>\#\@\%\&\$]",
        " ",
        text,
    )

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    # 1. Read product CSV
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} products from {CSV_PATH}")

    # 2. Load web pages asynchronously
    nest_asyncio.apply()
    loader = WebBaseLoader(df["website"].tolist())
    documents = loader.aload()
    print(f"Loaded {len(documents)} web pages")

    # 3. Clean documents
    documents_clean = copy.deepcopy(documents)
    for doc in documents_clean:
        doc.page_content = clean_text(doc.page_content)
    print(f"Cleaned {len(documents_clean)} documents")

    # 4. Add metadata from CSV and prepend to page content
    documents_with_metadata = copy.deepcopy(documents_clean)

    for doc in documents_with_metadata:
        source = doc.metadata.get("source", "")
        matching = df[df["website"] == source]

        if not matching.empty:
            doc.metadata["product_id"] = matching["product_id"].values[0]
            doc.metadata["description"] = matching["description"].values[0]

        title = doc.metadata.get("title", "Unknown Title")
        source_url = doc.metadata.get("source", "Unknown URL")
        description = doc.metadata.get("description", "Unknown Product")

        doc.page_content = (
            f"Title: {title}\n"
            f"Description: {description}\n"
            f"Source: {source_url}\n\n"
            f"{doc.page_content}"
        )

    print(f"Enriched {len(documents_with_metadata)} documents with metadata")

    # 5. Create Table of Contents document
    toc_content = (
        "# Cannondale Synapse Product Catalog\n\n"
        "This database contains information about the following "
        "Cannondale Synapse bicycle models:\n\n"
    )

    for i, doc in enumerate(documents_with_metadata):
        desc = doc.metadata.get("description", "Unknown Product")
        url = doc.metadata.get("source", "")
        toc_content += f"{i + 1}. {desc}\n   URL: {url}\n\n"

    toc_content += (
        "\n## About the Synapse Series\n"
        "The Cannondale Synapse is an endurance road bike series designed for "
        "comfort and performance on long rides. These bikes feature various "
        "carbon configurations, SmartSense technology options, and are suitable "
        "for riders looking for a versatile road bike that excels in comfort "
        "without sacrificing speed.\n"
    )

    toc_document = Document(
        page_content=toc_content,
        metadata={
            "title": "Cannondale Synapse Product Catalog - Table of Contents",
            "source": "generated_toc",
            "description": "Complete catalog of all Cannondale Synapse products in database",
            "type": "table_of_contents",
        },
    )

    documents_with_metadata.insert(0, toc_document)
    print(f"Total documents (including TOC): {len(documents_with_metadata)}")

    # 6. Create embeddings and persist Chroma vector store
    embedding_function = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    Chroma.from_documents(
        documents_with_metadata,
        embedding=embedding_function,
        persist_directory=PERSIST_DIR,
    )

    print(f"Vector store created at: {PERSIST_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
