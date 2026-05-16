"""
app.py - Streamlit demo for the RAG agent.
Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from agent import RAGAgent
from memory import load_memory

st.set_page_config(page_title="RAG Agent", page_icon="🔍", layout="wide")

# Hide Streamlit deploy button
st.markdown("""
    <style>
    [data-testid="stToolbar"] {visibility: hidden;}
    .ask-label {font-size: 1.3rem; font-weight: 600; margin-bottom: 0.3rem;}
    .answer-label {font-size: 1.3rem; font-weight: 600; margin-bottom: 0.3rem;}
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("Retrieval Pipeline")
    st.markdown(
        "**Architecture**\n\n"
        "0. Semantic cache — 0 API calls\n"
        "1. Query expansion + semantic search — 1 API call\n"
        "2. BM25 keyword search — 1 API call\n"
        "3. Web search — 2 API calls\n\n"
        "**Features**\n\n"
        "- Re-ranking at every level\n"
        "- Self-correction after every answer\n"
        "- Persistent memory across sessions\n"
        "- Cache invalidation by document fingerprinting\n"
        "- Semantic cache with 0.75 similarity threshold"
    )
    st.markdown("---")
    if st.button("🗑️ Clear cache"):
        from pathlib import Path
        cache_file = Path("logs/cache.jsonl")
        if cache_file.exists():
            cache_file.unlink()
            st.success("Cache cleared.")
        else:
            st.info("Cache is already empty.")
    if st.button("🧹 Clear memory"):
        from pathlib import Path
        memory_file = Path("logs/memory.json")
        if memory_file.exists():
            memory_file.unlink()
            st.success("Memory cleared.")
        else:
            st.info("Memory is already empty.")
    if st.button("📋 Clear history"):
        from pathlib import Path
        log_file = Path("logs/queries.jsonl")
        if log_file.exists():
            log_file.unlink()
            st.success("History cleared.")
        else:
            st.info("History is already empty.")

# Init agent
@st.cache_resource
def get_agent():
    agent = RAGAgent()
    n = agent.load_docs("docs")
    return agent, n

agent, n_chunks = get_agent()

# Header
st.title("🔍 RAG Agent with Escalating Retrieval")
st.caption(
    f"Indexed **{n_chunks} chunks** from `docs/` · "
    "Escalates through 3 retrieval levels until confidence is sufficient · "
    "Re-ranking · Self-correction · Semantic cache · Persistent memory"
)

# Memory indicator
facts = load_memory()
if facts:
    with st.expander(f"🧠 Memory — {len(facts)} known facts"):
        for f in facts:
            st.write(f"- {f}")

# Query input
st.markdown('<div class="ask-label">Ask a question</div>', unsafe_allow_html=True)
query = st.text_input(
    "Ask a question",
    placeholder="e.g. What are the main findings of the paper?",
    label_visibility="collapsed",
)

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Search", type="primary") and query.strip():
    with st.spinner("Searching..."):
        log = agent.ask(query.strip())

    # Auto-add to history
    if query.strip() not in st.session_state.history:
        st.session_state.history.append(query.strip())

    # Answer
    if log.resolved_at_level == 0:
        st.success("⚡ Cache hit — answered instantly from semantic cache")

    st.markdown('<div class="answer-label">Answer</div>', unsafe_allow_html=True)
    display_answer = log.final_answer.replace("[Cached] ", "")
    st.markdown(display_answer)

    # Escalation trace
    st.subheader("Escalation trace")

    rows = log.summary_table()
    df = pd.DataFrame(rows)

    # Add human-readable level name
    level_names = {0: "Cache", 1: "Semantic search", 2: "BM25", 3: "Web search"}
    df.insert(1, "Name", df["Level"].map(level_names).fillna("Unknown"))

    def highlight_resolved(row):
        if row["Level"] == 0:
            return ["background-color: #cce5ff"] * len(row)
        if row["Resolved"] == "✓":
            return ["background-color: #d4edda"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df.style.apply(highlight_resolved, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total tokens", log.total_tokens)
    col2.metric("Total time", f"{log.total_time_ms:.0f} ms")
    col3.metric("Resolved at level", log.resolved_at_level if log.resolved_at_level >= 0 else "4")
    col4.metric("Cache hit", "✓ Yes" if log.resolved_at_level == 0 else "✗ No")

    # Confidence chart — remove per-query chart, timeline shown below

# Query history
if st.session_state.history:
    with st.expander(f"Query history ({len(st.session_state.history)} queries)"):
        for q in reversed(st.session_state.history):
            st.write(f"- {q}")

# ── Confidence timeline ────────────────────────────────────────────────
import json
from pathlib import Path

log_file = Path("logs/queries.jsonl")
if log_file.exists():
    history_rows = []
    query_index = {}  # map query text to x position
    x_counter = [0]

    with open(log_file) as f:
        for line in f:
            try:
                entry = json.loads(line)
                query_text = entry.get("original_query", "")[:30]
                if query_text not in query_index:
                    query_index[query_text] = x_counter[0]
                    x_counter[0] += 1
                x_pos = query_index[query_text]
                iterations = entry.get("iterations", [])
                # Cache hit — no iterations, just resolved_at_level = 0
                if entry.get("resolved_at_level") == 0 and not iterations:
                    history_rows.append({
                        "x": x_pos,
                        "query": query_text,
                        "level": 0,
                        "confidence": 1.0,
                        "strategy": "cache hit",
                    })
                for it in iterations:
                    history_rows.append({
                        "x": x_pos,
                        "query": query_text,
                        "level": it.get("level", 0),
                        "confidence": it.get("confidence_after", 0.0),
                        "strategy": it.get("strategy", "")[:25],
                    })
            except Exception:
                pass

    if history_rows:
        st.subheader("Confidence timeline")
        level_colors = {0: "#28a745", 1: "#1f77b4", 2: "#f0ad4e", 3: "#fd7e14"}
        level_names = {0: "Cache", 1: "Level 1", 2: "Level 2", 3: "Level 3"}
        x_labels = {v: k for k, v in query_index.items()}

        fig = go.Figure()
        for level in [0, 1, 2, 3]:
            level_rows = [r for r in history_rows if r["level"] == level]
            if not level_rows:
                continue
            fig.add_trace(go.Scatter(
                x=[r["x"] for r in level_rows],
                y=[r["confidence"] for r in level_rows],
                mode="markers",
                name=level_names[level],
                marker=dict(size=14, color=level_colors[level]),
                text=[f"{r['query']}<br>{r['strategy']}" for r in level_rows],
                hovertemplate="<b>%{text}</b><br>Confidence: %{y:.2f}<extra></extra>",
            ))

        fig.update_layout(
            yaxis=dict(range=[0, 1.05], title="Confidence"),
            xaxis=dict(
                tickmode="array",
                tickvals=list(x_labels.keys()),
                ticktext=list(x_labels.values()),
                tickangle=-30,
                title="Query",
            ),
            height=400,
            margin=dict(l=40, r=20, t=20, b=120),
            legend=dict(title="Level"),
        )
        st.plotly_chart(fig, use_container_width=True)