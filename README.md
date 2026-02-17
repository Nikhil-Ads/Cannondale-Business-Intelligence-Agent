# Business Intelligence Agent

The Business Intelligence Agent performs data analysis and presents business insights through text or visualizations, answering questions for business requirements and analysis. It connects to SQL databases, interprets natural language queries, executes SQL, and returns results as tables or interactive Plotly charts—serving as an AI copilot for BI, customer analytics, and data visualization.

## Tech Stack

| Category | Technologies |
|----------|--------------|
| **AI / LLM** | LangChain, LangGraph, OpenAI (GPT-4o, GPT-4.1, etc.) |
| **BI Agent** | ai-data-science-team |
| **Web UI** | Streamlit |
| **Visualization** | Plotly |
| **Data** | Pandas, SQLAlchemy |
| **Database** | SQLite (configurable) |
| **Config** | PyYAML |

The agent is implemented in `src/agents/bi_agent_final.py`, which uses the `make_business_intelligence_agent` from the ai-data-science-team package to orchestrate SQL generation, execution, and visualization via LangChain/LangGraph and OpenAI.

## Prerequisites

- **Python** 3.10+
- **OpenAI API key** — stored in `credentials.yml` under the `openai` key
- **Databases** — SQLite files (e.g. `database/leads_scored.db`, `challenges/challenge_03_connect_bikes_database/bikeshop_database.sqlite`) or other SQLAlchemy-supported databases

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

3. Run the Streamlit app from the project root:

   ```bash
   streamlit run src/agents/bi_agent_final.py
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

4. Run the Streamlit app from the project root:

   ```bash
   streamlit run src/agents/bi_agent_final.py
   ```

## Usage

After starting the app, you can:

1. Choose an **OpenAI model** (e.g. gpt-4.1-nano, gpt-4o) from the sidebar.
2. Choose a **Database** connection (e.g. Leads, Bikes).
3. Ask questions in natural language.

**Example questions:**
- What tables are in the database?
- What does the transactions table contain?
- What is the average p1 lead score of leads by member rating?
- What are the top 5 product sales revenue by product name? Make a donut chart.
- What are the total sales by month-year? Make a chart of sales over time.

The agent responds with tables, interactive charts, or explanatory text depending on the question.
