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

st.set_page_config(page_title="BI Agent MVP - Cannondale Expert")

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

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

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
        except Exception as e:
            answer = (
                "I'm sorry, an error occurred while processing your question. "
                f"Please try again.\n\nError: {e}"
            )

    msgs.add_ai_message(answer)
    st.chat_message("ai").write(answer)
