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

import yaml
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# ---------------------------------------------------------------------------
# Project paths & credentials
# ---------------------------------------------------------------------------

BASE_DIR = _BASE_DIR
CREDENTIALS_PATH = BASE_DIR / "credentials.yml"
VECTORSTORE_DIR = str(BASE_DIR / "res" / "data" / "cannondale_vectorstore")

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open(CREDENTIALS_PATH))["openai"]

# Import after env var is set so OpenAI clients pick it up
from src.agents.bi_agent import make_mvp_rag_agent, AVAILABLE_MODELS, DEFAULT_MODEL  # noqa: E402

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

ai_index = 0  # tracks position in msg_sources
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)
    if msg.type == "ai":
        sources = (
            st.session_state.msg_sources[ai_index]
            if ai_index < len(st.session_state.msg_sources)
            else []
        )
        if sources:
            with st.expander("Sources"):
                for src in sources:
                    st.markdown(f"- [{src}]({src})")
        ai_index += 1

# ---------------------------------------------------------------------------
# Handle user input
# ---------------------------------------------------------------------------

if question := st.chat_input(
    "Ask me anything about Cannondale Synapse bikes:", key="query_input"
):
    st.chat_message("human").write(question)
    msgs.add_user_message(question)

    with st.spinner("Thinking..."):
        try:
            result = agent.invoke({"user_question": question})
            answer = result["answer"]
            sources = result.get("sources", [])
        except Exception as e:
            answer = (
                "I'm sorry, an error occurred while processing your question. "
                f"Please try again.\n\nError: {e}"
            )
            sources = []

    msgs.add_ai_message(answer)
    st.session_state.msg_sources.append(sources)

    with st.chat_message("ai"):
        st.write(answer)
        if sources:
            with st.expander("Sources"):
                for src in sources:
                    st.markdown(f"- [{src}]({src})")
