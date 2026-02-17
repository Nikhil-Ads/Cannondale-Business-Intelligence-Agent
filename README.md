# Business Intelligence Agent

An AI-powered business intelligence agent that answers questions about Cannondale Synapse bicycles. It scrapes product data from the Cannondale website, builds a vector knowledge base, and provides text-based insights through a conversational Streamlit interface using Retrieval-Augmented Generation (RAG).

## Tech Stack

| Category | Technologies |
|----------|--------------|
| **AI / LLM** | LangChain, LangGraph, OpenAI (GPT-4o-mini) |
| **Vector Store** | ChromaDB |
| **Embeddings** | OpenAI text-embedding-ada-002 |
| **Web Scraping** | LangChain WebBaseLoader |
| **Web UI** | Streamlit |
| **Data** | Pandas |
| **Config** | PyYAML |

## Project Structure

```
Business Intelligence Agent/
├── scripts/
│   └── create_cannondale_db.py       # Web scraping script to build vector store
├── src/
│   ├── agents/
│   │   └── bi_agent_mvp.py           # LangGraph RAG agent
│   ├── utils/
│   │   ├── __init__.py
│   │   └── db_utils.py               # Chroma vector store connection utilities
│   └── frontend/
│       └── app.py                    # Streamlit chat application
├── res/
│   ├── cannondale_synapse_products.csv   # Product URLs for scraping
│   └── data/
│       └── cannondale_vectorstore/       # Generated vector store (created by script)
├── credentials.yml                       # OpenAI API key (not committed)
├── env.yaml                              # Conda environment spec
└── requirements.txt                      # pip dependencies
```

## Prerequisites

- **Python** 3.10+
- **OpenAI API key** — stored in `credentials.yml` under the `openai` key

## Setup

### Option 1: Conda

1. Create and activate the conda environment from `env.yaml`:

   ```bash
   conda env create -f env.yaml
   conda activate bi_agent_dev_langchain_latest
   ```

2. Add a `credentials.yml` in the project root with your OpenAI API key:

   ```yaml
   openai: "your-openai-api-key-here"
   ```

### Option 2: Python venv

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   # source .venv/bin/activate      # Linux / macOS
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Add a `credentials.yml` in the project root:

   ```yaml
   openai: "your-openai-api-key-here"
   ```

## Running the Application

### Step 1: Build the Vector Store (one-time setup)

Run the web scraping script from the project root to scrape Cannondale product pages and create the Chroma vector store:

```bash
python scripts/create_cannondale_db.py
```

This reads `res/cannondale_synapse_products.csv`, scrapes each product page, cleans the text, enriches it with metadata, and persists the vector store to `res/data/cannondale_vectorstore/`.

### Step 2: Launch the Streamlit App

```bash
streamlit run src/frontend/app.py
```

## Usage

Once the app is running, ask questions in the chat interface. The agent retrieves relevant product information from the vector store and returns text-based insights.

**Example questions:**
- What are the main differences between Synapse models?
- Which Synapse bike is best for long-distance riding?
- Tell me about the SmartSense technology
- What's the difference between Carbon 1 and Carbon 2?
- What is the LAB71 series and how is it different?
- Compare the Carbon 3 SmartSense with the Carbon 4
- Which model would you recommend for a beginner?
