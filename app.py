import os
import re
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import io
import logging
import datetime
import traceback

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
st.set_page_config(
    page_title="DataMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface2: #1a1a24;
    --border: #2a2a3a;
    --accent: #7c6af7;
    --accent2: #f7856a;
    --accent3: #6af7c8;
    --text: #e8e8f0;
    --muted: #6b6b80;
    --success: #4ade80;
    --error: #f87171;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Hide streamlit branding */
#MainMenu, footer, header {visibility: hidden;}

/* Main container */
.main .block-container {
    padding: 1.5rem 2rem;
    max-width: 100%;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem;
}

/* Header */
.datamind-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
}

.datamind-logo {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.6rem;
    background: linear-gradient(135deg, var(--accent), var(--accent3));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}

.datamind-tagline {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Chat messages */
.stChatMessage {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-bottom: 0.75rem !important;
    padding: 1rem !important;
}

/* User message */
.stChatMessage[data-testid="chat-message-user"] {
    border-left: 3px solid var(--accent) !important;
}

/* Assistant message */
.stChatMessage[data-testid="chat-message-assistant"] {
    border-left: 3px solid var(--accent3) !important;
}

/* Input */
.stChatInputContainer {
    background: var(--surface) !important;
    border-top: 1px solid var(--border) !important;
    padding: 0.75rem 0 !important;
}

.stChatInput textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
}

.stChatInput textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(124,106,247,0.15) !important;
}

/* Buttons */
.stButton > button {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
    width: 100% !important;
}

.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: rgba(124,106,247,0.08) !important;
}

/* File uploader */
.stFileUploader {
    background: var(--surface2) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 10px !important;
    padding: 0.5rem !important;
}

/* Dataframe */
.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* Metrics */
.stMetric {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 0.75rem !important;
}

.stMetric label {
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

.stMetric [data-testid="metric-container"] > div:nth-child(2) {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.4rem !important;
    color: var(--accent3) !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    gap: 4px !important;
    border-radius: 10px !important;
    padding: 4px !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
}

.stTabs [aria-selected="true"] {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}

/* Select boxes */
.stSelectbox > div > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

/* Suggested questions chip */
.suggestion-chip {
    display: inline-block;
    background: rgba(124,106,247,0.1);
    border: 1px solid rgba(124,106,247,0.3);
    color: #c4baff;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
    margin: 3px;
    cursor: pointer;
}

/* Status badges */
.badge-success {
    background: rgba(74,222,128,0.1);
    border: 1px solid rgba(74,222,128,0.3);
    color: var(--success);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    letter-spacing: 1px;
}

.badge-error {
    background: rgba(248,113,113,0.1);
    border: 1px solid rgba(248,113,113,0.3);
    color: var(--error);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
}

/* Memory pills */
.memory-pill {
    background: rgba(106,247,200,0.08);
    border: 1px solid rgba(106,247,200,0.2);
    border-radius: 8px;
    padding: 6px 10px;
    font-size: 0.75rem;
    color: var(--accent3);
    margin: 4px 0;
    font-family: 'Space Mono', monospace;
}

/* Section label */
.section-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
    margin-bottom: 8px;
    margin-top: 4px;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ───────────────────────────────────────────────────────
defaults = {
    "messages": [],
    "df": None,
    "df_info": "",
    "df_sample": "",
    "df_name": "",
    "dfs": {},           # multi-file: {name: df}
    "memory": [],        # persistent memory facts
    "memory_summary": "", # summarized older conversation
    "turn_count": 0,
    "suggestions": [],
    "col_descriptions": {},
    "chat_export": []
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── LangGraph State ─────────────────────────────────────────────────────────
class ChatState(TypedDict):
    messages: list
    dataframe_info: str
    dataframe_sample: str
    dataframe_context: str      # col descriptions + memory
    user_question: str
    answer: str
    route: str                  # "code" | "viz" | "qa" | "memory"
    code_result: str
    chart_data: Optional[dict]
    memory_facts: List[str]
    memory_summary: str

# ─── LLM ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_llm(temperature=0.2):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.strip() == "your_groq_api_key_here":
        st.error("❌ Add your GROQ_API_KEY to the .env file!")
        st.stop()
    return ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=temperature)

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
            return False, "⚠️ Rate limit — wait a moment."
        else:
            return False, f"❌ Error: {err}"

# ─── Dataset Helpers ──────────────────────────────────────────────────────────
def get_df_info(df: pd.DataFrame):
    buf = io.StringIO()
    df.info(buf=buf)
    stats = f"Shape: {df.shape[0]} rows × {df.shape[1]} cols\n"
    stats += f"Columns: {', '.join(df.columns.tolist())}\n"
    stats += f"Dtypes:\n{df.dtypes.to_string()}\n"
    try:
        stats += f"Stats:\n{df.describe(include='all').to_string()}"
    except Exception:
        pass
    return stats, df.head(5).to_string()

def get_active_df() -> Optional[pd.DataFrame]:
    if st.session_state.df is not None:
        return st.session_state.df
    return None

def safe_exec_pandas(code: str, df: pd.DataFrame) -> tuple[bool, str, any]:
    """Safely execute pandas code and return (success, result_str, raw_result)."""
    local_vars = {"df": df.copy(), "pd": pd, "px": px, "go": go, "plt": plt}
    try:
        exec(compile(code, "<string>", "exec"), {}, local_vars)
        result = local_vars.get("result", None)
        if result is None:
            return True, "Code executed (no `result` variable set).", None
        return True, str(result), result
    except Exception as e:
        return False, f"Error: {traceback.format_exc(limit=3)}", None

# ─── Memory Helpers ───────────────────────────────────────────────────────────
SUMMARIZE_AFTER = 10  # turns before summarizing

def maybe_summarize_memory():
    """Summarize old conversation messages into memory_summary to prevent token bloat."""
    if st.session_state.turn_count > 0 and st.session_state.turn_count % SUMMARIZE_AFTER == 0:
        if len(st.session_state.messages) > SUMMARIZE_AFTER:
            llm = get_llm(temperature=0)
            old_msgs = st.session_state.messages[:-4]  # keep last 4
            history_text = "\n".join([f"{m['role'].upper()}: {m['content'][:200]}" for m in old_msgs])
            summary_prompt = f"""Summarize this data analysis conversation in 3-5 bullet points, focusing on key findings and user preferences:

{history_text}

Be concise. Format: bullet points only."""
            try:
                resp = llm.invoke([HumanMessage(content=summary_prompt)])
                st.session_state.memory_summary = resp.content
                st.session_state.messages = st.session_state.messages[-4:]
                logger.info("Memory summarized")
            except Exception as e:
                logger.error(f"Summary failed: {e}")

def extract_memory_facts(answer: str, question: str) -> List[str]:
    """Extract memorable facts from LLM answer to persist in memory."""
    facts = []
    lower = answer.lower()
    # Simple heuristic: if answer contains numbers/findings, remember them
    if any(kw in lower for kw in ["highest", "lowest", "average", "total", "maximum", "minimum", "trend"]):
        # Keep first 150 chars of insight as memory
        short = answer.strip()[:150].replace("\n", " ")
        facts.append(f"Q: {question[:60]}… → {short}…")
    return facts

# ─── Suggestion Generator ────────────────────────────────────────────────────
def generate_suggestions(df: pd.DataFrame) -> List[str]:
    llm = get_llm(temperature=0.7)
    cols = df.columns.tolist()
    dtypes = df.dtypes.to_dict()
    numeric_cols = [c for c, t in dtypes.items() if "int" in str(t) or "float" in str(t)]
    cat_cols = [c for c, t in dtypes.items() if "object" in str(t)]

    prompt = f"""Given a dataset with columns: {cols}
Numeric columns: {numeric_cols[:5]}
Categorical columns: {cat_cols[:5]}
First row sample: {df.iloc[0].to_dict()}

Generate exactly 4 interesting, specific analytical questions a user might ask about this dataset.
Return ONLY a JSON array of 4 strings, nothing else. Example:
["What is the average X?", "Show distribution of Y", "Which Z has highest W?", "Trend of A over time"]"""

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        raw = resp.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        suggestions = json.loads(raw)
        return suggestions[:4]
    except Exception as e:
        logger.error(f"Suggestions failed: {e}")
        return [
            f"What is the average of {numeric_cols[0] if numeric_cols else 'values'}?",
            f"Show distribution of {cat_cols[0] if cat_cols else 'categories'}",
            "What are the top 5 rows by value?",
            "Are there any missing values?"
        ]

# ─── LangGraph Nodes ─────────────────────────────────────────────────────────

def router_node(state: ChatState) -> ChatState:
    """Route the question to the right processing node."""
    llm = get_llm(temperature=0)
    q = state["user_question"].lower()
    
    # Rule-based fast routing
    viz_keywords = ["chart", "plot", "graph", "visuali", "histogram", "bar chart", "pie", "scatter", "heatmap", "trend", "show me"]
    code_keywords = ["calculate", "compute", "aggregate", "group by", "filter", "count", "sum", "average", "mean", "max", "min", "top ", "bottom ", "how many", "percentage", "correlation"]
    memory_keywords = ["remember", "you said", "earlier", "previous", "last time", "forgot", "recall", "what did"]

    if any(k in q for k in memory_keywords):
        return {**state, "route": "memory"}
    elif any(k in q for k in viz_keywords):
        return {**state, "route": "viz"}
    elif any(k in q for k in code_keywords):
        return {**state, "route": "code"}
    else:
        return {**state, "route": "qa"}

def code_node(state: ChatState) -> ChatState:
    """Generate and execute pandas code to answer the question."""
    llm = get_llm(temperature=0.1)
    df = get_active_df()
    if df is None:
        return {**state, "answer": "No dataset loaded.", "code_result": ""}

    system = f"""You are a Python/pandas expert. Generate executable pandas code to answer the user's question.

Dataset info:
{state['dataframe_info']}

Sample:
{state['dataframe_sample']}

Column descriptions:
{state['dataframe_context']}

RULES:
1. Use variable `df` for the dataframe
2. Store final answer in variable `result`
3. Return ONLY the Python code block, no explanation
4. Handle missing values gracefully
5. For grouped operations, reset_index() if needed

Example format:
```python
result = df.groupby('category')['value'].mean().sort_values(ascending=False).head(10)
```"""

    messages = [SystemMessage(content=system), HumanMessage(content=state["user_question"])]
    
    for attempt in range(3):  # retry loop
        try:
            resp = llm.invoke(messages)
            code = resp.content
            # Extract code block
            match = re.search(r"```(?:python)?\n?(.*?)```", code, re.DOTALL)
            if match:
                code = match.group(1).strip()
            
            success, result_str, raw = safe_exec_pandas(code, df)
            
            if success:
                # Now ask LLM to interpret the result
                interpret_prompt = f"""The user asked: "{state['user_question']}"

I executed this pandas code:
```python
{code}
```

The result was:
{result_str[:1000]}

Provide a clear, insightful interpretation in 2-4 sentences. Highlight key findings. Be specific with numbers."""
                interp = llm.invoke([HumanMessage(content=interpret_prompt)])
                final_answer = f"**📊 Analysis Result:**\n\n```\n{result_str[:500]}\n```\n\n**💡 Insight:** {interp.content}"
                logger.info(f"Code node success (attempt {attempt+1})")
                return {**state, "answer": final_answer, "code_result": result_str}
            else:
                # Feed error back for retry
                messages.append(AIMessage(content=resp.content))
                messages.append(HumanMessage(content=f"That code failed with: {result_str}\n\nPlease fix it."))
                logger.warning(f"Code attempt {attempt+1} failed: {result_str[:100]}")
        except Exception as e:
            logger.error(f"Code node error: {e}")
    
    return {**state, "answer": "❌ Could not execute analysis after 3 attempts. Try rephrasing the question.", "code_result": ""}

def viz_node(state: ChatState) -> ChatState:
    """Generate chart code and data."""
    llm = get_llm(temperature=0.1)
    df = get_active_df()
    if df is None:
        return {**state, "answer": "No dataset loaded.", "chart_data": None}

    system = f"""You are a data visualization expert using plotly express.

Dataset info:
{state['dataframe_info']}

Sample:
{state['dataframe_sample']}

Column descriptions:
{state['dataframe_context']}

Generate Python code using plotly express (imported as `px`) to create a chart.
Store the figure in variable `result`.
Use `df` as the dataframe.
Return ONLY the python code block.

Example:
```python
result = px.bar(df, x='category', y='value', title='Category Values', color='category')
result.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
```"""

    messages = [SystemMessage(content=system), HumanMessage(content=state["user_question"])]
    
    for attempt in range(3):
        try:
            resp = llm.invoke(messages)
            code = resp.content
            match = re.search(r"```(?:python)?\n?(.*?)```", code, re.DOTALL)
            if match:
                code = match.group(1).strip()

            # Always add dark theme update
            if "update_layout" not in code:
                code += "\nresult.update_layout(template='plotly_dark', paper_bgcolor='rgba(17,17,24,0)', plot_bgcolor='rgba(26,26,36,0.5)', font=dict(family='Syne', color='#e8e8f0'))"

            success, result_str, raw = safe_exec_pandas(code, df)
            
            if success and raw is not None and hasattr(raw, 'to_json'):
                chart_json = raw.to_json()
                logger.info(f"Viz node success (attempt {attempt+1})")
                return {**state, "answer": "📈 Chart generated successfully!", "chart_data": {"json": chart_json, "type": "plotly"}}
            else:
                messages.append(AIMessage(content=resp.content))
                messages.append(HumanMessage(content=f"Code failed: {result_str}. Fix it."))
        except Exception as e:
            logger.error(f"Viz node error: {e}")
    
    return {**state, "answer": "❌ Could not generate chart. Try describing the chart differently.", "chart_data": None}

def qa_node(state: ChatState) -> ChatState:
    """Answer general questions about the dataset."""
    llm = get_llm(temperature=0.3)

    memory_context = ""
    if state["memory_summary"]:
        memory_context = f"\n\n## Previous conversation summary:\n{state['memory_summary']}"
    if state["memory_facts"]:
        memory_context += f"\n\n## Remembered facts:\n" + "\n".join([f"- {f}" for f in state["memory_facts"][-10:]])

    system = f"""You are DataMind, an expert data analyst AI assistant.

## Dataset Information:
{state['dataframe_info']}

## Sample Data (first 5 rows):
{state['dataframe_sample']}

## Column Context:
{state['dataframe_context']}
{memory_context}

## Instructions:
- Answer based on the dataset provided
- Be specific, cite column names and values when relevant  
- If you're unsure, say so — don't hallucinate data
- Use markdown formatting for clarity
- Suggest follow-up analyses when appropriate
- Few-shot examples of good answers:
  Q: "What columns are there?" → List them with descriptions and data types
  Q: "Is there missing data?" → Check nulls and report percentages
  Q: "What does this data represent?" → Infer context from column names and values"""

    lc_messages = [SystemMessage(content=system)] + state["messages"]
    
    try:
        resp = llm.invoke(lc_messages)
        logger.info(f"QA node success ({len(resp.content)} chars)")
        return {**state, "answer": resp.content}
    except Exception as e:
        logger.error(f"QA node error: {e}")
        return {**state, "answer": f"❌ Error: {str(e)}"}

def memory_node(state: ChatState) -> ChatState:
    """Handle memory-related queries."""
    facts = state["memory_facts"]
    summary = state["memory_summary"]
    
    if not facts and not summary:
        return {**state, "answer": "🧠 I don't have any specific memories from our conversation yet. Ask me some analysis questions first!"}
    
    memory_text = ""
    if summary:
        memory_text += f"**Conversation Summary:**\n{summary}\n\n"
    if facts:
        memory_text += "**Key Findings I Remember:**\n"
        for f in facts[-10:]:
            memory_text += f"- {f}\n"
    
    llm = get_llm(temperature=0.2)
    prompt = f"""The user asked: "{state['user_question']}"

Here is what I remember from our conversation:
{memory_text}

Answer their question based on this memory. Be helpful and specific."""
    
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        return {**state, "answer": f"🧠 **From Memory:**\n\n{resp.content}"}
    except Exception as e:
        return {**state, "answer": memory_text}

def route_selector(state: ChatState) -> str:
    return state.get("route", "qa")

# ─── Build Graph ─────────────────────────────────────────────────────────────
@st.cache_resource
def build_graph():
    g = StateGraph(ChatState)
    g.add_node("router", router_node)
    g.add_node("code", code_node)
    g.add_node("viz", viz_node)
    g.add_node("qa", qa_node)
    g.add_node("memory", memory_node)
    
    g.set_entry_point("router")
    g.add_conditional_edges("router", route_selector, {
        "code": "code",
        "viz": "viz",
        "qa": "qa",
        "memory": "memory"
    })
    g.add_edge("code", END)
    g.add_edge("viz", END)
    g.add_edge("qa", END)
    g.add_edge("memory", END)
    return g.compile()

# ─── Export Chat ──────────────────────────────────────────────────────────────
def export_chat_as_text():
    lines = [f"DataMind Chat Export — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n", "="*50 + "\n"]
    for m in st.session_state.messages:
        role = "You" if m["role"] == "user" else "DataMind"
        lines.append(f"\n[{role}]\n{m['content']}\n")
    return "\n".join(lines)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="datamind-header"><div><div class="datamind-logo">🧠 DataMind</div><div class="datamind-tagline">AI · Data · Intelligence</div></div></div>', unsafe_allow_html=True)

    # ── File Upload ──
    st.markdown('<div class="section-label">📁 Data Source</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload CSV or Excel files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        for uf in uploaded_files:
            if uf.name not in st.session_state.dfs:
                try:
                    df = pd.read_csv(uf) if uf.name.endswith(".csv") else pd.read_excel(uf)
                    st.session_state.dfs[uf.name] = df
                    logger.info(f"Loaded: {uf.name} | {df.shape}")
                except Exception as e:
                    st.error(f"Error loading {uf.name}: {e}")

    if st.session_state.dfs:
        selected_file = st.selectbox(
            "Active dataset",
            list(st.session_state.dfs.keys()),
            label_visibility="collapsed"
        )
        if selected_file:
            df = st.session_state.dfs[selected_file]
            if st.session_state.df_name != selected_file:
                st.session_state.df = df
                st.session_state.df_name = selected_file
                st.session_state.df_info, st.session_state.df_sample = get_df_info(df)
                st.session_state.messages = []
                st.session_state.memory = []
                st.session_state.memory_summary = ""
                st.session_state.turn_count = 0
                with st.spinner("Generating suggestions..."):
                    st.session_state.suggestions = generate_suggestions(df)
                st.success(f"✅ {selected_file}")
                st.caption(f"{df.shape[0]:,} rows · {df.shape[1]} columns")

    # ── Column Descriptions ──
    if st.session_state.df is not None:
        st.markdown('<div class="section-label" style="margin-top:12px">📝 Column Context</div>', unsafe_allow_html=True)
        with st.expander("Add column descriptions (optional)"):
            df = st.session_state.df
            for col in df.columns[:8]:  # limit to 8 for UX
                desc = st.text_input(
                    f"{col}",
                    value=st.session_state.col_descriptions.get(col, ""),
                    key=f"col_{col}",
                    placeholder="Describe this column..."
                )
                if desc:
                    st.session_state.col_descriptions[col] = desc

    # ── Memory Panel ──
    st.markdown('<div class="section-label" style="margin-top:12px">🧠 Memory</div>', unsafe_allow_html=True)
    if st.session_state.memory:
        with st.expander(f"📌 {len(st.session_state.memory)} facts remembered"):
            for fact in st.session_state.memory[-5:]:
                st.markdown(f'<div class="memory-pill">{fact[:100]}…</div>', unsafe_allow_html=True)
        if st.button("🗑️ Clear Memory"):
            st.session_state.memory = []
            st.session_state.memory_summary = ""
    else:
        st.caption("No memories yet — insights will be saved automatically.")

    # ── Controls ──
    st.markdown('<div class="section-label" style="margin-top:12px">⚙️ Controls</div>', unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.turn_count = 0
    with col_b:
        if st.session_state.messages:
            chat_text = export_chat_as_text()
            st.download_button(
                "💾 Export",
                data=chat_text,
                file_name=f"datamind_chat_{datetime.date.today()}.txt",
                mime="text/plain"
            )

    if st.button("🔍 Test API Key"):
        key = os.getenv("GROQ_API_KEY", "")
        if not key or key == "your_groq_api_key_here":
            st.error("❌ No key in .env!")
        else:
            with st.spinner("Testing..."):
                ok, msg = validate_api_key(key)
            st.success(msg) if ok else st.error(msg)

# ─── Main Area ────────────────────────────────────────────────────────────────
if st.session_state.df is None:
    # Landing state
    st.markdown("""
    <div style='text-align:center; padding: 4rem 2rem;'>
        <div style='font-size:4rem; margin-bottom:1rem;'>🧠</div>
        <div style='font-family:Syne; font-size:2.2rem; font-weight:800; background:linear-gradient(135deg,#7c6af7,#6af7c8); -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>DataMind AI</div>
        <div style='font-family:Space Mono; font-size:0.8rem; color:#6b6b80; letter-spacing:3px; text-transform:uppercase; margin:0.5rem 0 2rem;'>Natural Language · Data Intelligence</div>
        <div style='color:#6b6b80; max-width:480px; margin:0 auto; line-height:1.8;'>Upload a CSV or Excel file from the sidebar to start asking questions about your data in plain English.</div>
    </div>
    <div style='display:flex; gap:1rem; justify-content:center; flex-wrap:wrap; padding: 0 2rem;'>
        <div style='background:#111118; border:1px solid #2a2a3a; border-radius:12px; padding:1.2rem 1.5rem; max-width:200px; text-align:center;'>
            <div style='font-size:1.5rem'>📊</div>
            <div style='font-family:Syne; font-weight:700; margin:6px 0 4px;'>Code Execution</div>
            <div style='font-size:0.75rem; color:#6b6b80;'>Runs real pandas queries</div>
        </div>
        <div style='background:#111118; border:1px solid #2a2a3a; border-radius:12px; padding:1.2rem 1.5rem; max-width:200px; text-align:center;'>
            <div style='font-size:1.5rem'>📈</div>
            <div style='font-family:Syne; font-weight:700; margin:6px 0 4px;'>Visualizations</div>
            <div style='font-size:0.75rem; color:#6b6b80;'>Auto-generates charts</div>
        </div>
        <div style='background:#111118; border:1px solid #2a2a3a; border-radius:12px; padding:1.2rem 1.5rem; max-width:200px; text-align:center;'>
            <div style='font-size:1.5rem'>🧠</div>
            <div style='font-family:Syne; font-weight:700; margin:6px 0 4px;'>Memory</div>
            <div style='font-size:0.75rem; color:#6b6b80;'>Remembers findings</div>
        </div>
        <div style='background:#111118; border:1px solid #2a2a3a; border-radius:12px; padding:1.2rem 1.5rem; max-width:200px; text-align:center;'>
            <div style='font-size:1.5rem'>🔗</div>
            <div style='font-family:Syne; font-weight:700; margin:6px 0 4px;'>Multi-file</div>
            <div style='font-size:0.75rem; color:#6b6b80;'>Switch between datasets</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Layout ──
    chat_col, data_col = st.columns([3, 2])

    with chat_col:
        st.markdown("### 💬 Chat")

        # Suggested questions
        if st.session_state.suggestions and len(st.session_state.messages) == 0:
            st.markdown('<div class="section-label">✨ Suggested Questions</div>', unsafe_allow_html=True)
            sug_cols = st.columns(2)
            for i, sug in enumerate(st.session_state.suggestions):
                with sug_cols[i % 2]:
                    if st.button(sug, key=f"sug_{i}", use_container_width=True):
                        st.session_state["pending_input"] = sug

        # Render chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("chart"):
                    try:
                        import plotly.io as pio
                        fig = pio.from_json(msg["chart"])
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        pass

        # Input
        pending = st.session_state.pop("pending_input", None)
        user_input = st.chat_input("Ask anything about your data…") or pending

        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})
            logger.info(f"User ({st.session_state.df_name}): {user_input[:80]}")

            # Build LC messages (use recent ones + summary)
            lc_messages = []
            for m in st.session_state.messages[-8:]:  # last 8 messages
                if m["role"] == "user":
                    lc_messages.append(HumanMessage(content=m["content"]))
                else:
                    lc_messages.append(AIMessage(content=m["content"]))

            # Build column context
            col_ctx = "\n".join([f"- {c}: {d}" for c, d in st.session_state.col_descriptions.items()])
            if not col_ctx:
                col_ctx = "No column descriptions provided."

            with st.chat_message("assistant"):
                with st.spinner("DataMind is thinking…"):
                    try:
                        result = build_graph().invoke({
                            "messages": lc_messages,
                            "dataframe_info": st.session_state.df_info,
                            "dataframe_sample": st.session_state.df_sample,
                            "dataframe_context": col_ctx,
                            "user_question": user_input,
                            "answer": "",
                            "route": "qa",
                            "code_result": "",
                            "chart_data": None,
                            "memory_facts": st.session_state.memory,
                            "memory_summary": st.session_state.memory_summary,
                        })
                        answer = result["answer"]
                        chart_data = result.get("chart_data")
                        route_used = result.get("route", "qa")
                        logger.info(f"Route: {route_used} | Answer: {len(answer)} chars")
                    except Exception as e:
                        answer = f"❌ Error: {e}"
                        chart_data = None
                        logger.error(f"Graph error: {e}")

                # Route badge
                route_colors = {"code": "#f7856a", "viz": "#6af7c8", "qa": "#7c6af7", "memory": "#f7d46a"}
                route_icons = {"code": "⚙️ Code", "viz": "📈 Viz", "qa": "💬 Q&A", "memory": "🧠 Memory"}
                badge_color = route_colors.get(route_used, "#7c6af7")
                badge_label = route_icons.get(route_used, "💬 Q&A")
                st.markdown(f'<span style="background:rgba(124,106,247,0.1);border:1px solid {badge_color}33;color:{badge_color};border-radius:6px;padding:2px 8px;font-size:0.7rem;font-family:Space Mono">{badge_label}</span>', unsafe_allow_html=True)
                
                st.markdown(answer)
                
                # Render chart if present
                if chart_data and chart_data.get("type") == "plotly":
                    try:
                        import plotly.io as pio
                        fig = pio.from_json(chart_data["json"])
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Chart render error: {e}")

            # Save to session
            msg_entry = {"role": "assistant", "content": answer}
            if chart_data:
                msg_entry["chart"] = chart_data.get("json")
            st.session_state.messages.append(msg_entry)

            # Update memory
            new_facts = extract_memory_facts(answer, user_input)
            st.session_state.memory.extend(new_facts)
            st.session_state.turn_count += 1
            maybe_summarize_memory()

    # ── Data Panel ──
    with data_col:
        df = st.session_state.df
        
        # Metrics row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Rows", f"{df.shape[0]:,}")
        with m2:
            st.metric("Columns", df.shape[1])
        with m3:
            nulls = df.isnull().sum().sum()
            st.metric("Nulls", f"{nulls:,}")

        tab1, tab2, tab3 = st.tabs(["📋 Preview", "📊 Stats", "🔗 Schema"])
        
        with tab1:
            preview = df.head(50).copy()
            for c in preview.columns:
                if preview[c].dtype == "object":
                    preview[c] = preview[c].astype(str)
            st.dataframe(preview, use_container_width=True, height=380)

        with tab2:
            try:
                st.dataframe(df.describe(include="all").T, use_container_width=True, height=380)
            except Exception:
                st.dataframe(df.describe().T, use_container_width=True, height=380)

        with tab3:
            schema_data = []
            for col in df.columns:
                nulls = df[col].isnull().sum()
                unique = df[col].nunique()
                schema_data.append({
                    "Column": col,
                    "Type": str(df[col].dtype),
                    "Nulls": nulls,
                    "Unique": unique,
                    "Description": st.session_state.col_descriptions.get(col, "—")
                })
            st.dataframe(pd.DataFrame(schema_data), use_container_width=True, height=380)