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
import yaml
import streamlit as st
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
from src.agents.bi_agent import make_mvp_rag_agent, run_critical_thinking_agent, AVAILABLE_MODELS, DEFAULT_MODEL  # noqa: E402
import pandas as pd  # noqa: E402
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
</style>
"""
if st.session_state.dark_mode:
    st.markdown(_DARK_CSS, unsafe_allow_html=True)
else:
    st.markdown(_LIGHT_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — model selection (persisted in session_state)
# ---------------------------------------------------------------------------

with st.sidebar.container():
    st.markdown("**Theme**")
    st.toggle("Dark Mode", value=st.session_state.dark_mode, key="dark_mode")

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
    """Extract <chart_data> block from text.

    Returns (clean_text, chart_dict_or_None). The block is stripped from the
    returned text so prose renders without stray JSON.
    """
    match = _CHART_RE.search(text)
    if not match:
        return text, None
    clean_text = _CHART_RE.sub('', text).strip()
    try:
        chart = json.loads(match.group(1))
        return clean_text, chart
    except (json.JSONDecodeError, ValueError):
        return clean_text, None


def _render_chart(chart_data: dict) -> None:
    """Render a Plotly chart or dataframe from a chart_data dict."""
    chart_type = chart_data.get("type", "bar")
    title = chart_data.get("title", "")
    template = "plotly_dark" if st.session_state.dark_mode else "plotly"

    if chart_type == "table":
        columns = chart_data.get("columns", [])
        rows = chart_data.get("rows", [])
        if columns and rows:
            df = pd.DataFrame(rows, columns=columns)
            if title:
                st.caption(f"**{title}**")
            st.dataframe(df, width="stretch")

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
    st.session_state["msg_feedback"] = {}


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
if "msg_feedback" not in st.session_state:
    st.session_state.msg_feedback = {}  # {ai_index: "up" | "down"}

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
        with st.chat_message("ai"):
            if reasoning:
                with st.expander("🧠 Reasoning Steps", expanded=False):
                    for label, content in reasoning:
                        st.markdown(f"**{label}**")
                        st.markdown(content)
                        st.markdown("---")
            st.write(msg.content)
            if chart_data:
                _render_chart(chart_data)
            if sources:
                with st.expander("Sources"):
                    for src in sources:
                        st.markdown(f"- [{src}]({src})")
            # Feedback buttons — skip for the greeting message (index 0)
            if ai_index > 0:
                fb = st.session_state.msg_feedback.get(ai_index)
                col1, col2, col3 = st.columns([1, 1, 10])
                with col1:
                    if st.button("👍", key=f"up_{ai_index}", disabled=fb is not None):
                        st.session_state.msg_feedback[ai_index] = "up"
                        st.rerun()
                with col2:
                    if st.button("👎", key=f"down_{ai_index}", disabled=fb is not None):
                        st.session_state.msg_feedback[ai_index] = "down"
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

if question := st.chat_input(
    "Ask me anything about Cannondale Synapse bikes:", key="query_input"
):
    st.chat_message("human").write(question)
    msgs.add_user_message(question)

    reasoning_steps = []
    chat_history = _build_chat_history(msgs.messages)

    if st.session_state.critical_thinking:
        n_subq = st.session_state.num_subquestions
        spinner_msg = f"🧠 Thinking critically ({n_subq} sub-questions, {n_subq + 3} stages)..."
        with st.spinner(spinner_msg):
            try:
                result = run_critical_thinking_agent(
                    user_question=question,
                    persist_dir=VECTORSTORE_DIR,
                    model=st.session_state.selected_model,
                    num_subquestions=n_subq,
                    chat_history=chat_history,
                )
                answer = result["answer"]
                sources = result.get("sources", [])
                reasoning_steps = result.get("reasoning", [])
            except Exception as e:
                logger.exception("Critical thinking agent failed")
                answer = (
                    "I'm sorry, an error occurred during critical thinking. "
                    f"Please try again.\n\nError: {e}"
                )
                sources = []
    else:
        with st.spinner("Thinking..."):
            try:
                result = agent.invoke({"user_question": question, "chat_history": chat_history})
                answer = result["answer"]
                sources = result.get("sources", [])
            except Exception as e:
                answer = (
                    "I'm sorry, an error occurred while processing your question. "
                    f"Please try again.\n\nError: {e}"
                )
                sources = []

    # Parse optional chart block; store clean prose in history, chart separately
    clean_answer, chart_data = _parse_chart_data(answer)
    msgs.add_ai_message(clean_answer)
    st.session_state.msg_sources.append(sources)
    st.session_state.msg_charts.append(chart_data)
    # Store all passes except the final synthesis (which is the answer itself)
    st.session_state.msg_reasoning.append(reasoning_steps[:-1] if reasoning_steps else [])

    # Force a rerun so the sidebar Export Chat buttons reflect the updated
    # session state (sidebar renders before messages are added in the same run)
    st.rerun()
