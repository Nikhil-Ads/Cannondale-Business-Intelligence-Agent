"""
Streamlit frontend for the MVP BI Agent.

Run from the project root:
    streamlit run src/frontend/app.py
"""

import os
import sys
import pathlib

# Ensure the project root is on sys.path so `src.*` imports work
_BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

import json
import logging
import re
import uuid
import yaml
import streamlit as st
from src.utils.history_utils import save_history, load_history, list_sessions  # noqa: E402
from src.backend.feedback_db import init_db as _init_feedback_db, save_feedback as _save_feedback  # noqa: E402
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project paths & credentials
# ---------------------------------------------------------------------------

BASE_DIR = _BASE_DIR
CREDENTIALS_PATH = BASE_DIR / "credentials.yml"
VECTORSTORE_DIR = str(BASE_DIR / "res" / "data" / "cannondale_vectorstore")

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open(CREDENTIALS_PATH))["openai"]

# Import after env var is set so OpenAI clients pick it up
from src.agents.bi_agent import make_mvp_rag_agent, run_critical_thinking_agent, run_comparison_agent, AVAILABLE_MODELS, DEFAULT_MODEL  # noqa: E402
from src.utils.markdown_utils import sanitize_markdown  # noqa: E402
from src.utils.followup_utils import generate_followup_suggestions  # noqa: E402
from src.utils.comparison_utils import detect_comparison_intent, extract_bike_names  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="BI Agent MVP - Cannondale Expert", layout="wide")

# ---------------------------------------------------------------------------
# Theme: light/dark toggle (persisted in session_state)
# ---------------------------------------------------------------------------

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

# Persist session_id in the URL query params so it survives page refreshes.
# On first load: generate a new ID and write it to the URL.
# On refresh: read it back from the URL (session_state is reset but query_params survive).
if "session_id" not in st.session_state:
    _qp_id = st.query_params.get("session_id")
    if _qp_id:
        st.session_state.session_id = _qp_id
    else:
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.query_params["session_id"] = st.session_state.session_id

# Inject theme CSS early so it applies to the whole app.
# Use high specificity and !important so we override Streamlit's defaults.
# Also style the Dark Mode toggle so it's clearly visible.
_DARK_CSS = """
<style>
/* ---- Root and main area (high specificity) ---- */
.stApp,
.stAppViewContainer,
[data-testid="stAppViewContainer"],
section[data-testid="stAppViewContainer"],
.stApp section.main,
.stApp [data-testid="stVerticalBlock"] {
    background-color: #0e1117 !important;
}
.stApp .main .block-container {
    background-color: #0e1117 !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}
/* ---- Main content text ---- */
.stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label, .stApp span,
.stApp .stMarkdown p, .stApp .stMarkdown li,
.stApp .stMarkdown td, .stApp .stMarkdown th,
.stApp [data-testid="stExpander"] label,
.stApp [data-testid="stExpander"] p {
    color: #fafafa !important;
}
/* ---- Sidebar ---- */
.stApp [data-testid="stSidebar"],
.stApp [data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] {
    background-color: #1a1a2e !important;
}
.stApp [data-testid="stSidebar"] .stMarkdown,
.stApp [data-testid="stSidebar"] label,
.stApp [data-testid="stSidebar"] p,
.stApp [data-testid="stSidebar"] span {
    color: #e0e0e0 !important;
}
/* ---- Dark Mode toggle: make it clearly visible in sidebar ---- */
.stApp [data-testid="stSidebar"] [data-testid="stCheckbox"] label,
.stApp [data-testid="stSidebar"] [data-testid="stToggle"] label,
.stApp [data-testid="stSidebar"] label[data-testid="stCheckboxLabel"] {
    font-weight: 700 !important;
    color: #fff !important;
}
.stApp [data-testid="stSidebar"] [data-testid="stCheckbox"] input,
.stApp [data-testid="stSidebar"] [data-testid="stToggle"] input,
.stApp [data-testid="stSidebar"] input[type="checkbox"] {
    accent-color: #6ee7b7 !important;
}
/* Toggle/checkbox widget container in sidebar - visible track */
.stApp [data-testid="stSidebar"] [data-testid="stCheckbox"],
.stApp [data-testid="stSidebar"] [data-testid="stToggle"] {
    padding: 0.5rem 0.75rem !important;
    margin-bottom: 0.5rem !important;
    border-radius: 8px !important;
    background-color: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
}
/* Theme section container (heading + toggle) - dark */
.stApp [data-testid="stSidebar"] [data-testid="stVerticalBlock"]:first-of-type {
    padding: 0.5rem 0.75rem !important;
    margin-bottom: 0.75rem !important;
    border-radius: 8px !important;
    background-color: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
}
/* ---- Chat messages and expanders ---- */
.stApp [data-testid="stChatMessage"],
.stApp .stChatMessage {
    background-color: #262730 !important;
}
.stApp [data-testid="stChatMessage"] p,
.stApp .stChatMessage .stMarkdown p {
    color: #fafafa !important;
}
.stApp [data-testid="stExpander"],
.stApp .stExpander {
    background-color: #1e1e2e !important;
    border: 1px solid #444 !important;
}
.stApp [data-testid="stExpander"] details,
.stApp [data-testid="stExpander"] summary,
.stApp [data-testid="stExpander"] details > div {
    background-color: #1e1e2e !important;
    color: #fafafa !important;
}
.stApp [data-testid="stExpander"] summary:hover {
    background-color: #2a2d3a !important;
}
/* ---- Top header bar (dark) ---- */
header[data-testid="stHeader"],
header[data-testid="stHeader"] > div,
.stApp header,
.stApp [data-testid="stHeader"],
.stApp [data-testid="stHeader"] > div {
    background-color: #0e1117 !important;
    border-bottom: 1px solid #333 !important;
    color: #e0e0e0 !important;
}
[data-testid="stToolbar"],
[data-testid="stDecoration"],
.stApp [data-testid="stToolbar"],
.stApp [data-testid="stDecoration"] {
    background-color: #0e1117 !important;
}
header[data-testid="stHeader"] button,
header[data-testid="stHeader"] span,
[data-testid="stToolbar"] button,
[data-testid="stToolbar"] span {
    color: #e0e0e0 !important;
}
/* ---- Chat input bottom bar ---- */
.stApp [data-testid="stBottom"],
.stApp [data-testid="stBottom"] > div,
.stApp [data-testid="stBottom"] > div > div {
    background-color: #0e1117 !important;
    border-top: 1px solid #333 !important;
}
.stApp [data-testid="stChatInput"],
.stApp [data-testid="stChatInput"] > div {
    background-color: #262730 !important;
    border: 1px solid #555 !important;
    border-radius: 0.5rem !important;
}
.stApp [data-testid="stChatInput"] textarea {
    background-color: #262730 !important;
    color: #fafafa !important;
    caret-color: #fafafa !important;
}
.stApp [data-testid="stChatInput"] textarea::placeholder {
    color: #9ca3af !important;
    opacity: 1 !important;
}
/* ---- Other inputs ---- */
.stApp .stTextInput input,
.stApp input {
    background-color: #262730 !important;
    color: #fafafa !important;
    border-color: #555 !important;
}
/* ---- Buttons & chips (dark) ---- */
.stApp .stButton > button,
.stApp button[kind="primary"],
.stApp button[kind="secondary"] {
    background-color: #2c2f3a !important;
    color: #e8edf5 !important;
    border: 1px solid #4a5065 !important;
    border-radius: 6px !important;
    box-shadow: none !important;
}
.stApp .stButton > button:hover,
.stApp button[kind="primary"]:hover,
.stApp button[kind="secondary"]:hover {
    background-color: #383c4e !important;
    border-color: #6b7494 !important;
    color: #ffffff !important;
}
.stApp .stButton > button:focus,
.stApp button[kind="primary"]:focus,
.stApp button[kind="secondary"]:focus {
    outline: 2px solid #38bdf8 !important;
    outline-offset: 2px !important;
}
.stApp .stButton > button:disabled,
.stApp button[kind="primary"]:disabled,
.stApp button[kind="secondary"]:disabled {
    background-color: #1e2028 !important;
    color: #5a5f72 !important;
    border-color: #32364a !important;
    cursor: not-allowed !important;
}
/* Download buttons (dark) */
.stApp .stDownloadButton > button {
    background-color: #2c2f3a !important;
    color: #e8edf5 !important;
    border: 1px solid #4a5065 !important;
    border-radius: 6px !important;
}
.stApp .stDownloadButton > button:hover {
    background-color: #383c4e !important;
    border-color: #6b7494 !important;
}
/* Follow-up suggestion chips (dark) */
.stApp [data-testid="stHorizontalBlock"] .stButton > button,
.stApp [data-testid="stVerticalBlock"] .stChatMessage + div .stButton > button {
    background-color: #232635 !important;
    color: #c5cfe8 !important;
    border: 1px solid #3e4560 !important;
    font-size: 0.85rem !important;
}
</style>
"""
_LIGHT_CSS = """
<style>
/* ---- Root and main area ---- */
.stApp,
.stAppViewContainer,
[data-testid="stAppViewContainer"],
section[data-testid="stAppViewContainer"],
.stApp section.main,
.stApp [data-testid="stVerticalBlock"] {
    background-color: #ffffff !important;
}
.stApp .main .block-container {
    background-color: #ffffff !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}
/* ---- Main content text ---- */
.stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label, .stApp span,
.stApp .stMarkdown p, .stApp .stMarkdown li,
.stApp .stMarkdown td, .stApp .stMarkdown th,
.stApp [data-testid="stExpander"] label,
.stApp [data-testid="stExpander"] p {
    color: #31333f !important;
}
/* ---- Sidebar ---- */
.stApp [data-testid="stSidebar"],
.stApp [data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] {
    background-color: #f0f2f6 !important;
}
.stApp [data-testid="stSidebar"] .stMarkdown,
.stApp [data-testid="stSidebar"] label,
.stApp [data-testid="stSidebar"] p,
.stApp [data-testid="stSidebar"] span {
    color: #31333f !important;
}
/* ---- Dark Mode toggle: clearly visible ---- */
.stApp [data-testid="stSidebar"] [data-testid="stCheckbox"] label,
.stApp [data-testid="stSidebar"] [data-testid="stToggle"] label,
.stApp [data-testid="stSidebar"] label[data-testid="stCheckboxLabel"] {
    font-weight: 700 !important;
    color: #1a1a2e !important;
}
.stApp [data-testid="stSidebar"] [data-testid="stCheckbox"] input,
.stApp [data-testid="stSidebar"] [data-testid="stToggle"] input,
.stApp [data-testid="stSidebar"] input[type="checkbox"] {
    accent-color: #0ea5e9 !important;
}
/* Toggle/checkbox widget container - visible box */
.stApp [data-testid="stSidebar"] [data-testid="stCheckbox"],
.stApp [data-testid="stSidebar"] [data-testid="stToggle"] {
    padding: 0.5rem 0.75rem !important;
    margin-bottom: 0.5rem !important;
    border-radius: 8px !important;
    background-color: rgba(0,0,0,0.04) !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
}
/* Theme section container (heading + toggle) - light */
.stApp [data-testid="stSidebar"] [data-testid="stVerticalBlock"]:first-of-type {
    padding: 0.5rem 0.75rem !important;
    margin-bottom: 0.75rem !important;
    border-radius: 8px !important;
    background-color: rgba(0,0,0,0.04) !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
}
/* ---- Chat messages and expanders ---- */
.stApp [data-testid="stChatMessage"],
.stApp .stChatMessage {
    background-color: #f0f2f6 !important;
}
.stApp [data-testid="stChatMessage"] p,
.stApp .stChatMessage .stMarkdown p {
    color: #31333f !important;
}
.stApp [data-testid="stExpander"],
.stApp .stExpander {
    background-color: #fafafa !important;
    border: 1px solid #e0e0e0 !important;
}
.stApp [data-testid="stExpander"] details {
    background-color: #fafafa !important;
    border-radius: 0.5rem !important;
}
.stApp [data-testid="stExpander"] summary {
    background-color: #f8f9fb !important;
    color: #31333f !important;
}
.stApp [data-testid="stExpander"] summary:hover {
    background-color: #edf1f7 !important;
}
.stApp [data-testid="stExpander"] details[open] > summary {
    border-bottom: 1px solid #e0e0e0 !important;
}
.stApp [data-testid="stExpander"] details > div {
    background-color: #ffffff !important;
    color: #31333f !important;
}
/* ---- Top header bar (light) ---- */
header[data-testid="stHeader"],
header[data-testid="stHeader"] > div,
.stApp header,
.stApp [data-testid="stHeader"],
.stApp [data-testid="stHeader"] > div {
    background-color: #f0f2f6 !important;
    border-bottom: 1px solid #e0e0e0 !important;
    color: #31333f !important;
}
[data-testid="stToolbar"],
[data-testid="stDecoration"],
.stApp [data-testid="stToolbar"],
.stApp [data-testid="stDecoration"] {
    background-color: #f0f2f6 !important;
}
header[data-testid="stHeader"] button,
header[data-testid="stHeader"] span,
[data-testid="stToolbar"] button,
[data-testid="stToolbar"] span {
    color: #31333f !important;
}
/* ---- Chat input bottom bar ---- */
.stApp [data-testid="stBottom"],
.stApp [data-testid="stBottom"] > div,
.stApp [data-testid="stBottom"] > div > div {
    background-color: #f8f9fb !important;
    border-top: 1px solid #e0e0e0 !important;
}
.stApp [data-testid="stChatInput"],
.stApp [data-testid="stChatInput"] > div {
    background-color: #ffffff !important;
    border: 1px solid #ccc !important;
    border-radius: 0.5rem !important;
}
.stApp [data-testid="stChatInput"] textarea {
    background-color: #ffffff !important;
    color: #31333f !important;
    caret-color: #31333f !important;
}
.stApp [data-testid="stChatInput"] textarea::placeholder {
    color: #6b7280 !important;
    opacity: 1 !important;
}
/* ---- Other inputs ---- */
.stApp .stTextInput input,
.stApp input {
    background-color: #ffffff !important;
    color: #31333f !important;
    border-color: #ccc !important;
}
/* ---- Buttons & chips (light) ---- */
.stApp .stButton > button,
.stApp button[kind="primary"],
.stApp button[kind="secondary"] {
    background-color: #e8edf5 !important;
    color: #1e3a5f !important;
    border: 1px solid #b0bec5 !important;
    border-radius: 6px !important;
    box-shadow: none !important;
}
.stApp .stButton > button:hover,
.stApp button[kind="primary"]:hover,
.stApp button[kind="secondary"]:hover {
    background-color: #d0dbe8 !important;
    border-color: #8fa4b8 !important;
    color: #1a2d4a !important;
}
.stApp .stButton > button:focus,
.stApp button[kind="primary"]:focus,
.stApp button[kind="secondary"]:focus {
    outline: 2px solid #0ea5e9 !important;
    outline-offset: 2px !important;
}
.stApp .stButton > button:disabled,
.stApp button[kind="primary"]:disabled,
.stApp button[kind="secondary"]:disabled {
    background-color: #f5f7fa !important;
    color: #9aa4b2 !important;
    border-color: #e1e6ed !important;
    cursor: not-allowed !important;
}
/* Download buttons */
.stApp .stDownloadButton > button {
    background-color: #e8edf5 !important;
    color: #1e3a5f !important;
    border: 1px solid #b0bec5 !important;
    border-radius: 6px !important;
}
.stApp .stDownloadButton > button:hover {
    background-color: #d0dbe8 !important;
    border-color: #8fa4b8 !important;
}
/* Follow-up suggestion chips */
.stApp [data-testid="stHorizontalBlock"] .stButton > button,
.stApp [data-testid="stVerticalBlock"] .stChatMessage + div .stButton > button {
    background-color: #f4f7fc !important;
    color: #1c3d5f !important;
    border: 1px solid #c5d3e0 !important;
    font-size: 0.85rem !important;
}
</style>
"""
if st.session_state.dark_mode:
    st.markdown(_DARK_CSS, unsafe_allow_html=True)
else:
    st.markdown(_LIGHT_CSS, unsafe_allow_html=True)

with st.sidebar.container():
    st.markdown("**Theme**")
    st.toggle(
        "Dark Mode",
        value=st.session_state.dark_mode,
        key="dark_mode",
        disabled=st.session_state.is_processing,
        help="Disabled while a response is being generated." if st.session_state.is_processing else None,
    )

# ---------------------------------------------------------------------------
# Sidebar — model selection (persisted in session_state)
# ---------------------------------------------------------------------------

if "selected_model" not in st.session_state:
    st.session_state.selected_model = DEFAULT_MODEL

selected_model = st.sidebar.selectbox(
    "Choose OpenAI model",
    AVAILABLE_MODELS,
    index=AVAILABLE_MODELS.index(st.session_state.selected_model),
)

# Persist selection across reruns; rebuild agent if model changed
if selected_model != st.session_state.selected_model:
    st.session_state.selected_model = selected_model
    # Clear cached agent so it's recreated with the new model
    st.cache_resource.clear()

st.sidebar.markdown(f"**Active model:** `{st.session_state.selected_model}`")

# ---------------------------------------------------------------------------
# Sidebar — Critical Thinking Mode
# ---------------------------------------------------------------------------

st.sidebar.markdown("---")

if "critical_thinking" not in st.session_state:
    st.session_state.critical_thinking = False
if "num_subquestions" not in st.session_state:
    st.session_state.num_subquestions = 3

st.sidebar.toggle("🧠 Critical Thinking Mode", key="critical_thinking")

if st.session_state.critical_thinking:
    st.session_state.num_subquestions = st.sidebar.slider(
        "Sub-questions",
        min_value=1,
        max_value=5,
        value=st.session_state.num_subquestions,
        help="Number of sub-questions to decompose the query into. More = deeper reasoning but slower response.",
    )
    total_stages = st.session_state.num_subquestions + 3
    st.sidebar.caption(f"⚡ {st.session_state.num_subquestions} sub-questions · {total_stages} total stages")

# ---------------------------------------------------------------------------
# Helper functions for Export Chat (defined early, used after msgs is ready)
# ---------------------------------------------------------------------------

import csv as _csv
import io as _io


def _build_chat_history(messages, k=6):
    """Convert recent chat messages to LangChain message objects for context injection.

    k=6 pairs (12 messages) gives enough context for multi-turn follow-ups
    without bloating the prompt for long conversations.
    """
    history = messages[1:]  # drop the greeting message
    history = history[-(k * 2):]  # keep last k human+ai pairs
    result = []
    for msg in history:
        if msg.type == "human":
            result.append(HumanMessage(content=msg.content))
        elif msg.type == "ai":
            result.append(AIMessage(content=msg.content))
    return result


def _build_txt(messages) -> str:
    """Format chat messages as plain text: [Role]: [Content]."""
    lines = []
    for msg in messages:
        role = "Assistant" if msg.type == "ai" else "User"
        lines.append(f"[{role}]: {msg.content}")
    return "\n\n".join(lines)


def _sanitize_csv_cell(value: str) -> str:
    """Sanitize CSV cell to prevent formula injection.

    Prefixes cells starting with =, +, -, or @ with a single quote
    to prevent spreadsheet applications from interpreting them as formulas.
    """
    if value and value[0] in ('=', '+', '-', '@'):
        return "'" + value
    return value


def _build_csv(messages) -> str:
    """Format chat messages as CSV with Role, Content columns."""
    buf = _io.StringIO()
    writer = _csv.writer(buf)
    writer.writerow(["Role", "Content"])
    for msg in messages:
        role = "Assistant" if msg.type == "ai" else "User"
        sanitized_content = _sanitize_csv_cell(msg.content)
        writer.writerow([role, sanitized_content])
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

_CHART_RE = re.compile(r'<chart_data>\s*(.*?)\s*</chart_data>', re.DOTALL)


def _parse_chart_data(text: str):
    """Extract all <chart_data> blocks from text.

    Returns (clean_text, charts_or_None) where charts_or_None is a list of
    parsed chart dicts (one per block) or None when no blocks are found.
    The blocks are stripped from the returned text so prose renders cleanly.
    """
    matches = _CHART_RE.findall(text)
    if not matches:
        return text, None
    clean_text = _CHART_RE.sub('', text).strip()
    charts = []
    for raw in matches:
        try:
            charts.append(json.loads(raw))
        except (json.JSONDecodeError, ValueError):
            pass
    return clean_text, charts if charts else None


def _render_chart(chart_data: dict) -> None:
    """Render a Plotly chart or markdown table from a chart_data dict."""
    chart_type = chart_data.get("type", "bar")
    title = chart_data.get("title", "")
    template = "plotly_dark" if st.session_state.dark_mode else "plotly"

    if chart_type == "table":
        columns = chart_data.get("columns", [])
        rows = chart_data.get("rows", [])
        if columns and rows:
            if title:
                st.markdown(f"**{title}**")
            # Build a markdown table so it inherits the app's theme CSS
            # instead of st.dataframe which has independent styling.
            header = "| " + " | ".join(str(c) for c in columns) + " |"
            separator = "| " + " | ".join("---" for _ in columns) + " |"
            body = "\n".join(
                "| " + " | ".join(str(cell) for cell in row) + " |"
                for row in rows
            )
            st.markdown(f"{header}\n{separator}\n{body}")

    elif chart_type in ("bar", "line"):
        x = chart_data.get("x", [])
        y = chart_data.get("y", [])
        labels = chart_data.get("labels", {})
        if x and y:
            if chart_type == "bar":
                fig = px.bar(x=x, y=y, title=title, labels=labels, template=template)
            else:
                fig = px.line(
                    x=x, y=y, title=title, labels=labels,
                    markers=True, template=template,
                )
            st.plotly_chart(fig, width="stretch")


def _clear_chat():
    st.session_state["langchain_messages"] = []
    st.session_state["msg_sources"] = [[]]
    st.session_state["msg_reasoning"] = [[]]
    st.session_state["msg_charts"] = [None]
    st.session_state["msg_confidence"] = [None]
    st.session_state["msg_followups"] = [None]
    st.session_state["msg_feedback"] = {}
    st.session_state["msg_comparisons"] = [False]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title(f"BI Agent MVP - Cannondale Expert  |  model: `{st.session_state.selected_model}`")

st.markdown(
    "I'm a business intelligence agent that answers questions about "
    "Cannondale Synapse bicycles. Ask me anything — recommendations, "
    "comparisons, features, specifications, and more."
)

# ---------------------------------------------------------------------------
# Example questions
# ---------------------------------------------------------------------------

with st.expander("Example questions"):
    st.markdown(
        """
- What are the main differences between Synapse models?
- Which Synapse bike is best for long-distance riding?
- Tell me about the SmartSense technology
- What's the difference between Carbon 1 and Carbon 2?
- Which bikes have disc brakes?
- What is the LAB71 series and how is it different?
- Compare the Carbon 3 SmartSense with the Carbon 4
- What are the key features of the Synapse series?
- Which model would you recommend for a beginner?
"""
    )

# ---------------------------------------------------------------------------
# Chat history (Streamlit session-based)
# ---------------------------------------------------------------------------

msgs = StreamlitChatMessageHistory(key="langchain_messages")

# Load persisted history if available (before adding greeting)
if "history_loaded" not in st.session_state:
    st.session_state.history_loaded = True
    saved = load_history(st.session_state.session_id)
    if saved:
        lc_msgs = []
        for m in saved:
            if m["type"] == "human":
                lc_msgs.append(HumanMessage(content=m["content"]))
            elif m["type"] == "ai":
                lc_msgs.append(AIMessage(content=m["content"]))
        if lc_msgs:
            msgs.add_messages(lc_msgs)

if len(msgs.messages) == 0:
    msgs.add_ai_message(
        "Hello! I'm your Cannondale Synapse bicycle expert. "
        "How can I help you today?"
    )

# Parallel list that stores source citations for each AI message.
# Index i in msg_sources corresponds to the i-th AI message in msgs.messages.
if "msg_sources" not in st.session_state:
    st.session_state.msg_sources = [[]]  # first entry = greeting (no sources)
if "msg_reasoning" not in st.session_state:
    st.session_state.msg_reasoning = [[]]  # first entry = greeting (no reasoning)
if "msg_charts" not in st.session_state:
    st.session_state.msg_charts = [None]  # first entry = greeting (no chart)
if "msg_confidence" not in st.session_state:
    st.session_state.msg_confidence = [None]  # first entry = greeting (no confidence)
if "msg_followups" not in st.session_state:
    st.session_state.msg_followups = [None]  # first entry = greeting (no follow-ups)
if "msg_feedback" not in st.session_state:
    st.session_state.msg_feedback = {}  # {ai_index: "up" | "down"}
if "msg_comparisons" not in st.session_state:
    st.session_state.msg_comparisons = [False]  # first entry = greeting

# Initialise feedback DB (creates table if needed)
_init_feedback_db()

# ---------------------------------------------------------------------------
# Sidebar — Export Chat History & Clear Chat (placed here so msgs is defined)
# ---------------------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.markdown("**💾 Export Chat**")

_stored_messages = st.session_state.get("langchain_messages", [])
if len(_stored_messages) > 1:
    st.sidebar.download_button(
        label="📥 Export as .txt",
        data=_build_txt(_stored_messages),
        file_name="chat_history.txt",
        mime="text/plain",
        key="export_txt",
        width="stretch",
    )
    st.sidebar.download_button(
        label="📊 Export as .csv",
        data=_build_csv(_stored_messages),
        file_name="chat_history.csv",
        mime="text/csv",
        key="export_csv",
        width="stretch",
    )
else:
    st.sidebar.caption("Start a conversation to enable export.")

st.sidebar.markdown("---")
if st.sidebar.button("💬 New Chat", help="Start a new conversation (saves current)", width="stretch"):
    new_id = str(uuid.uuid4())[:8]
    st.session_state.session_id = new_id
    st.query_params["session_id"] = new_id
    st.session_state.history_loaded = False
    st.session_state.msg_sources = [[]]
    st.session_state.msg_reasoning = [[]]
    st.session_state.msg_charts = [None]
    st.session_state.msg_confidence = [None]
    st.session_state.msg_followups = [None]
    st.session_state.msg_feedback = {}
    st.session_state.msg_comparisons = [False]
    msgs.clear()
    st.rerun()

st.sidebar.caption(f"Session: `{st.session_state.session_id}`")

st.sidebar.button(
    "🗑️ Clear Chat",
    on_click=_clear_chat,
    width="stretch",
    help="Reset the conversation and start fresh",
)

# ---------------------------------------------------------------------------
# Initialise agent (cached so it's created only once per session)
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_agent(model: str):
    return make_mvp_rag_agent(persist_dir=VECTORSTORE_DIR, model=model)

agent = _get_agent(st.session_state.selected_model)

# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------

ai_index = 0  # tracks position in msg_sources, msg_reasoning, msg_charts
for msg in msgs.messages:
    if msg.type == "ai":
        reasoning = (
            st.session_state.msg_reasoning[ai_index]
            if ai_index < len(st.session_state.msg_reasoning)
            else []
        )
        sources = (
            st.session_state.msg_sources[ai_index]
            if ai_index < len(st.session_state.msg_sources)
            else []
        )
        chart_data = (
            st.session_state.msg_charts[ai_index]
            if ai_index < len(st.session_state.msg_charts)
            else None
        )
        confidence = (
            st.session_state.msg_confidence[ai_index]
            if ai_index < len(st.session_state.msg_confidence)
            else None
        )
        is_comparison = (
            st.session_state.msg_comparisons[ai_index]
            if ai_index < len(st.session_state.msg_comparisons)
            else False
        )
        with st.chat_message("ai"):
            if is_comparison:
                st.caption("↔ Side-by-side comparison")
            if reasoning:
                with st.expander("🧠 Reasoning Steps", expanded=False):
                    for label, content in reasoning:
                        st.markdown(f"**{label}**")
                        st.markdown(content)
                        st.markdown("---")
            st.markdown(sanitize_markdown(msg.content))
            if chart_data:
                for _chart in chart_data:
                    _render_chart(_chart)
            if ai_index > 0 and confidence:
                n_sources = len(sources) if sources else 0
                st.caption(
                    f"{confidence['emoji']} {confidence['level']} confidence · {n_sources} sources"
                )
            # Follow-up suggestions
            followups = (
                st.session_state.msg_followups[ai_index]
                if ai_index < len(st.session_state.msg_followups)
                else None
            )
            if ai_index > 0 and followups:
                st.caption("Suggested follow-ups:")
                cols = st.columns(len(followups))
                for i, suggestion in enumerate(followups):
                    with cols[i]:
                        if st.button(suggestion, key=f"followup_{ai_index}_{i}"):
                            st.session_state["queued_question"] = suggestion
                            st.session_state.is_processing = True
                            st.rerun()
            if sources:
                with st.expander("Sources"):
                    for src in sources:
                        st.markdown(f"- [{src}]({src})")
            # Feedback buttons — skip for the greeting message (index 0)
            if ai_index > 0:
                fb = st.session_state.msg_feedback.get(ai_index)
                col1, col2, col3 = st.columns([1, 1, 10])
                # Retrieve context for this ai_index to log with the feedback
                _fb_question = ""
                _fb_answer = msg.content[:200] if hasattr(msg, "content") else ""
                # Walk back through messages to find the preceding human message
                _all = msgs.messages
                _ai_positions = [i for i, m in enumerate(_all) if m.type == "ai"]
                if ai_index < len(_ai_positions):
                    _pos = _ai_positions[ai_index]
                    if _pos > 0 and _all[_pos - 1].type == "human":
                        _fb_question = _all[_pos - 1].content
                with col1:
                    if st.button("👍", key=f"up_{ai_index}", disabled=fb is not None):
                        st.session_state.msg_feedback[ai_index] = "up"
                        _save_feedback(st.session_state.get("session_id", "unknown"), ai_index, _fb_question, _fb_answer, "up")
                        st.rerun()
                with col2:
                    if st.button("👎", key=f"down_{ai_index}", disabled=fb is not None):
                        st.session_state.msg_feedback[ai_index] = "down"
                        _save_feedback(st.session_state.get("session_id", "unknown"), ai_index, _fb_question, _fb_answer, "down")
                        st.rerun()
                if fb == "up":
                    col3.caption("✅ Thanks for the feedback!")
                elif fb == "down":
                    col3.caption("🙏 Thanks — we'll work on improving that!")
        ai_index += 1
    else:
        st.chat_message(msg.type).write(msg.content)

# ---------------------------------------------------------------------------
# Handle user input
# ---------------------------------------------------------------------------

if st.session_state.get("queued_question"):
    question = st.session_state.pop("queued_question")
else:
    _raw_input = st.chat_input(
        "Ask me anything about Cannondale Synapse bikes:", key="query_input"
    )
    if _raw_input:
        # Queue the question and mark as processing so the theme toggle is
        # disabled on the next run (before the agent actually executes).
        st.session_state.queued_question = _raw_input
        st.session_state.is_processing = True
        st.rerun()
    question = None

if question:
    st.chat_message("human").write(question)
    # Build context from prior turns only; current question is passed separately.
    chat_history = _build_chat_history(msgs.messages)
    msgs.add_user_message(question)

    reasoning_steps = []
    confidence = None
    is_comparison_response = False

    # Detect comparison intent before routing to any pipeline
    is_comparison = detect_comparison_intent(question)

    with st.status("Generating answer...", expanded=False) as status:
        # --- Stage 1: Agent invocation ---
        if is_comparison and not st.session_state.critical_thinking:
            status.update(label="Comparing bikes...")
            try:
                bike_names = extract_bike_names(question, model=st.session_state.selected_model)
                if len(bike_names) >= 2:
                    result = run_comparison_agent(
                        user_question=question,
                        bike_names=bike_names,
                        persist_dir=VECTORSTORE_DIR,
                        model=st.session_state.selected_model,
                        chat_history=chat_history,
                    )
                    answer = result["answer"]
                    sources = result.get("sources", [])
                    confidence = result.get("confidence")
                    is_comparison_response = True
                else:
                    # Fallback to standard RAG if bike name extraction fails
                    result = agent.invoke({"user_question": question, "chat_history": chat_history})
                    answer = result["answer"]
                    sources = result.get("sources", [])
                    confidence = result.get("confidence")
            except Exception as e:
                logger.exception("Comparison agent failed")
                answer = (
                    "I'm sorry, an error occurred while comparing those bikes. "
                    f"Please try again.\n\nError: {e}"
                )
                sources = []
        elif st.session_state.critical_thinking:
            n_subq = st.session_state.num_subquestions
            status.update(label=f"🧠 Thinking critically ({n_subq} sub-questions, {n_subq + 3} stages)...")
            try:
                result = run_critical_thinking_agent(
                    user_question=question,
                    persist_dir=VECTORSTORE_DIR,
                    model=st.session_state.selected_model,
                    num_subquestions=n_subq,
                    chat_history=chat_history,
                    is_comparison=is_comparison,
                )
                answer = result["answer"]
                sources = result.get("sources", [])
                reasoning_steps = result.get("reasoning", [])
                confidence = result.get("confidence")
                is_comparison_response = result.get("is_comparison", False)
            except Exception as e:
                logger.exception("Critical thinking agent failed")
                answer = (
                    "I'm sorry, an error occurred during critical thinking. "
                    f"Please try again.\n\nError: {e}"
                )
                sources = []
        else:
            status.update(label="Thinking...")
            try:
                result = agent.invoke({"user_question": question, "chat_history": chat_history})
                answer = result["answer"]
                sources = result.get("sources", [])
                confidence = result.get("confidence")
            except Exception as e:
                logger.exception("Standard inference failed")
                answer = (
                    "I'm sorry, an error occurred while processing your question. "
                    f"Please try again.\n\nError: {e}"
                )
                sources = []

        # --- Stage 2: Post-processing ---
        status.update(label="Processing response...")
        # Parse chart blocks (returns list of dicts or None); store clean prose in history
        clean_answer, chart_data = _parse_chart_data(answer)
        msgs.add_ai_message(clean_answer)
        save_history(st.session_state.session_id, msgs.messages)
        st.session_state.msg_sources.append(sources)
        st.session_state.msg_charts.append(chart_data)
        st.session_state.msg_confidence.append(confidence)
        st.session_state.msg_comparisons.append(is_comparison_response)
        # Store all passes except the final synthesis (which is the answer itself)
        st.session_state.msg_reasoning.append(reasoning_steps[:-1] if reasoning_steps else [])

        # --- Stage 3: Follow-up generation ---
        if clean_answer.startswith("Error") or clean_answer.startswith("I'm sorry, an error"):
            st.session_state.msg_followups.append(None)
        else:
            status.update(label="Generating follow-up suggestions...")
            followups = generate_followup_suggestions(
                question, clean_answer, model=st.session_state.selected_model
            )
            st.session_state.msg_followups.append(followups if followups else None)

        status.update(label="Done!", state="complete", expanded=False)

    st.session_state.is_processing = False
    # Force a rerun so the sidebar Export Chat buttons reflect the updated
    # session state (sidebar renders before messages are added in the same run)
    st.rerun()
