import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import io
import logging
import datetime

load_dotenv()

# ─── Logging ─────────────────────────────────────────────────────────────────
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"chatbot_{datetime.date.today()}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("App started")

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="CSV / Excel AI Chatbot", page_icon="📊", layout="wide")

# ─── Session State Init ───────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "df_info" not in st.session_state:
    st.session_state.df_info = ""
if "df_sample" not in st.session_state:
    st.session_state.df_sample = ""

# ─── LangGraph State ─────────────────────────────────────────────────────────
class ChatState(TypedDict):
    messages: list
    dataframe_info: str
    dataframe_sample: str
    user_question: str
    answer: str

# ─── LLM ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.strip() == "your_groq_api_key_here":
        st.error("❌ Add your GROQ_API_KEY to the .env file!")
        st.stop()
    logger.info(f"LLM loaded, key ends in ...{api_key[-4:]}")
    return ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0.2)

def validate_api_key(api_key):
    try:
        test = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0)
        test.invoke([HumanMessage(content="Hi")])
        return True, "✅ API key is working!"
    except Exception as e:
        err = str(e)
        if "401" in err or "authentication" in err.lower():
            return False, "❌ Invalid API key!"
        elif "429" in err:
            return False, "⚠️ Rate limit hit! Wait a moment."
        else:
            return False, f"❌ Error: {err}"

# ─── LangGraph ───────────────────────────────────────────────────────────────
def analyze_question(state: ChatState) -> ChatState:
    llm = get_llm()
    system = f"""You are a data analyst AI. Dataset info:
{state['dataframe_info']}

Sample (first 5 rows):
{state['dataframe_sample']}

Answer concisely based on this data."""
    messages = [SystemMessage(content=system)] + state["messages"]
    logger.info(f"Asking LLM: {state['user_question'][:60]}")
    try:
        response = llm.invoke(messages)
        logger.info(f"LLM replied ({len(response.content)} chars)")
        return {**state, "answer": response.content}
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return {**state, "answer": f"❌ Error: {str(e)}"}

@st.cache_resource
def build_graph():
    g = StateGraph(ChatState)
    g.add_node("analyze", analyze_question)
    g.set_entry_point("analyze")
    g.add_edge("analyze", END)
    return g.compile()

def get_df_info(df):
    stats = f"Shape: {df.shape[0]} rows x {df.shape[1]} cols\n"
    stats += f"Columns: {', '.join(df.columns.tolist())}\n"
    stats += f"Dtypes:\n{df.dtypes.to_string()}\n"
    stats += f"Stats:\n{df.describe().to_string()}"
    return stats, df.head(5).to_string()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 CSV / Excel Chatbot")
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.session_state.df_info, st.session_state.df_sample = get_df_info(df)
            st.session_state.messages = []
            logger.info(f"Loaded: {uploaded_file.name} | {df.shape[0]}x{df.shape[1]}")
            st.success(f"✅ {uploaded_file.name}")
            st.caption(f"{df.shape[0]} rows · {df.shape[1]} columns")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []

    st.markdown("---")
    st.markdown("**🔑 API Key**")
    st.caption("Get free key at console.groq.com")
    if st.button("🔍 Test API Key"):
        key = os.getenv("GROQ_API_KEY", "")
        if not key or key == "your_groq_api_key_here":
            st.error("❌ No key in .env!")
        else:
            with st.spinner("Testing..."):
                ok, msg = validate_api_key(key)
            st.success(msg) if ok else st.error(msg)

# ─── Main Area ────────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 💬 Chat")

    if st.session_state.df is None:
        st.info("👈 Upload a CSV or Excel file from the sidebar to get started.")
    else:
        # ── Render all past messages ──
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # ── Input ──
        user_input = st.chat_input("Ask anything about your data...")

        if user_input:
            # Show user message immediately
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})
            logger.info(f"User: {user_input[:80]}")

            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    lc_messages = []
                    for m in st.session_state.messages:
                        if m["role"] == "user":
                            lc_messages.append(HumanMessage(content=m["content"]))
                        else:
                            lc_messages.append(AIMessage(content=m["content"]))

                    try:
                        result = build_graph().invoke({
                            "messages": lc_messages,
                            "dataframe_info": st.session_state.df_info,
                            "dataframe_sample": st.session_state.df_sample,
                            "user_question": user_input,
                            "answer": ""
                        })
                        answer = result["answer"]
                        logger.info("Graph OK")
                    except Exception as e:
                        answer = f"❌ Error: {e}"
                        logger.error(f"Graph failed: {e}")

                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

with col2:
    if st.session_state.df is not None:
        st.markdown("### 📋 Data Preview")
        preview = st.session_state.df.head(10).copy()
        for c in preview.columns:
            if preview[c].dtype == "object":
                preview[c] = preview[c].astype(str)
        st.dataframe(preview, use_container_width=True, height=400)