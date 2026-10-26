"""
app.py — RetainIQ Assistant v2
Enterprise-grade retention intelligence chatbot.
"""

import html
import os

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads OPENROUTER_API_KEY from a local .env file if there is one
except ImportError:
    pass

import streamlit as st
import time
import datetime
import random
import string
from rag_engine import load_vectorstore, answer_query

st.set_page_config(
    page_title="RetainIQ — Retention Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

def get_api_key():
    """Read the OpenRouter key from the environment or .streamlit/secrets.toml. Never hard-code it."""
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("OPENROUTER_API_KEY", "")
        except Exception:
            key = ""
    return key

API_KEY = get_api_key()

def gen_session_id():
    return "RIQ-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))

if "session_id" not in st.session_state:
    st.session_state.session_id = gen_session_id()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: #080C18 !important;
    color: #E2E8F0 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── SIDEBAR ─────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0C1222 !important;
    border-right: 1px solid rgba(99,179,237,0.12) !important;
}
[data-testid="stSidebarContent"] {
    background: #0C1222 !important;
    padding: 1.4rem 1.1rem !important;
}
[data-testid="stSidebarContent"] * { color: #CBD5E1 !important; }

[data-testid="stSidebarContent"] .stButton > button {
    background: rgba(99,179,237,0.05) !important;
    border: 1px solid rgba(99,179,237,0.12) !important;
    border-radius: 8px !important;
    color: #94A3B8 !important;
    font-size: 0.76rem !important;
    text-align: left !important;
    padding: 0.42rem 0.75rem !important;
    width: 100% !important;
    transition: all 0.18s !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    margin-bottom: 3px !important;
    line-height: 1.4 !important;
    white-space: normal !important;
    height: auto !important;
}
[data-testid="stSidebarContent"] .stButton > button:hover {
    background: rgba(99,179,237,0.1) !important;
    border-color: rgba(99,179,237,0.3) !important;
    color: #63B3ED !important;
    transform: translateX(3px) !important;
}
[data-testid="stSidebarContent"] [data-testid="stMetricValue"] {
    color: #63B3ED !important; font-size: 1.1rem !important;
}
[data-testid="stSidebarContent"] [data-testid="stMetricLabel"] {
    color: #475569 !important; font-size: 0.65rem !important;
}

/* ── TOP NAV ─────────────────────────────────────── */
.topnav {
    background: #0C1222;
    border-bottom: 1px solid rgba(99,179,237,0.12);
    padding: 0.85rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.brand { display: flex; align-items: center; gap: 0.85rem; }
.brand-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #3B82F6, #8B5CF6);
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    box-shadow: 0 4px 14px rgba(59,130,246,0.35);
    flex-shrink: 0;
}
.brand-name {
    font-size: 1rem; font-weight: 800;
    background: linear-gradient(135deg, #63B3ED, #A78BFA);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.brand-sub { font-size: 0.68rem; color: #475569; margin-top: 1px; }
.nav-right { display: flex; align-items: center; gap: 0.75rem; }
.nav-pill {
    background: rgba(99,179,237,0.07);
    border: 1px solid rgba(99,179,237,0.18);
    border-radius: 20px; padding: 3px 10px;
    font-size: 0.67rem; color: #63B3ED; font-weight: 600; letter-spacing: 0.3px;
}
.nav-pill-green {
    background: rgba(52,211,153,0.07);
    border: 1px solid rgba(52,211,153,0.2);
    border-radius: 20px; padding: 3px 10px;
    font-size: 0.67rem; color: #34D399; font-weight: 600;
    display: flex; align-items: center; gap: 4px;
}
.live-dot {
    width: 5px; height: 5px; background: #34D399;
    border-radius: 50%; animation: blink 2s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.25} }
.session-tag {
    font-size: 0.65rem; color: #334155;
    font-family: 'DM Mono', monospace;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 6px; padding: 2px 8px;
}

/* ── PIPELINE BANNER ─────────────────────────────── */
.pipeline {
    background: linear-gradient(135deg, #0D1829, #0F1F35);
    border-bottom: 1px solid rgba(99,179,237,0.1);
    padding: 0.75rem 2rem;
    display: flex;
    align-items: center;
    gap: 0;
    overflow-x: auto;
}
.pipe-step {
    display: flex; align-items: center; gap: 0.5rem;
    background: rgba(99,179,237,0.06);
    border: 1px solid rgba(99,179,237,0.14);
    border-radius: 8px; padding: 0.45rem 0.85rem;
    flex-shrink: 0;
}
.pipe-icon { font-size: 0.85rem; }
.pipe-label { font-size: 0.72rem; font-weight: 700; color: #94A3B8; }
.pipe-sub { font-size: 0.62rem; color: #475569; margin-top: 1px; }
.pipe-arrow {
    color: #1E3A5F; font-size: 1rem; margin: 0 0.5rem;
    flex-shrink: 0; font-weight: 700;
}
.pipe-step-active {
    background: rgba(59,130,246,0.1);
    border-color: rgba(59,130,246,0.3);
}
.pipe-label-active { color: #63B3ED !important; }

/* ── METRICS BAR ─────────────────────────────────── */
.metrics {
    background: #0C1222;
    border-bottom: 1px solid rgba(99,179,237,0.08);
    padding: 0.55rem 2rem;
    display: flex; gap: 1.8rem; align-items: center;
    overflow-x: auto;
}
.m-item { display: flex; align-items: baseline; gap: 0.35rem; flex-shrink: 0; }
.m-val { font-size: 0.88rem; font-weight: 700; color: #63B3ED; font-family: 'DM Mono', monospace; }
.m-lbl { font-size: 0.62rem; color: #334155; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; white-space: nowrap; }
.m-sep { width: 1px; height: 16px; background: rgba(99,179,237,0.1); flex-shrink: 0; }

/* ── CHAT ────────────────────────────────────────── */
.chat-wrap { padding: 1.2rem 2rem 0.5rem; background: #080C18; }

/* Welcome */
.welcome {
    background: linear-gradient(135deg, #0C1527, #0F1E38);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 18px; padding: 2.2rem;
    text-align: center; max-width: 680px;
    margin: 1.5rem auto;
    box-shadow: 0 8px 40px rgba(0,0,0,0.5);
}
.w-icon {
    width: 64px; height: 64px;
    background: linear-gradient(135deg, #3B82F6, #8B5CF6);
    border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.8rem; margin: 0 auto 1.1rem;
    box-shadow: 0 10px 28px rgba(59,130,246,0.3);
}
.w-title {
    font-size: 1.35rem; font-weight: 800;
    background: linear-gradient(135deg, #E2E8F0, #94A3B8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin-bottom: 0.5rem;
}
.w-desc { font-size: 0.84rem; color: #475569; line-height: 1.7; margin-bottom: 1.4rem; }
.topic-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.55rem; text-align: left; }
.topic-card {
    background: rgba(99,179,237,0.05);
    border: 1px solid rgba(99,179,237,0.12);
    border-radius: 10px; padding: 0.65rem 0.85rem;
}
.t-icon { font-size: 0.95rem; margin-bottom: 0.25rem; }
.t-title { font-weight: 700; color: #94A3B8; font-size: 0.78rem; margin-bottom: 0.1rem; }
.t-desc { color: #334155; font-size: 0.7rem; line-height: 1.4; }

/* Messages */
.msg-meta {
    font-size: 0.63rem; color: #334155;
    font-family: 'DM Mono', monospace;
    margin-bottom: 0.2rem;
}
.msg-meta-right { text-align: right; }

/* Source cards */
.src-card {
    background: rgba(59,130,246,0.05);
    border: 1px solid rgba(59,130,246,0.15);
    border-left: 3px solid #3B82F6;
    border-radius: 0 8px 8px 0;
    padding: 0.5rem 0.8rem; margin-bottom: 0.28rem;
}
.src-name { font-weight: 700; color: #63B3ED; font-size: 0.72rem; margin-bottom: 0.15rem; }
.src-excerpt { color: #334155; font-style: italic; line-height: 1.4; font-size: 0.68rem; }

/* Feedback */
.feedback-row {
    display: flex; gap: 0.4rem; margin-top: 0.5rem; align-items: center;
}
.feedback-lbl { font-size: 0.65rem; color: #334155; margin-right: 0.2rem; }

/* OOS */
.oos {
    display: inline-flex; align-items: center; gap: 0.35rem;
    background: rgba(245,158,11,0.07);
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: 6px; padding: 3px 9px;
    font-size: 0.7rem; color: #F59E0B; margin-top: 0.35rem;
}

/* Latency badge */
.latency {
    display: inline-flex; align-items: center; gap: 0.3rem;
    background: rgba(52,211,153,0.06);
    border: 1px solid rgba(52,211,153,0.15);
    border-radius: 6px; padding: 2px 7px;
    font-size: 0.65rem; color: #34D399;
    font-family: 'DM Mono', monospace;
    margin-top: 0.3rem;
}

/* Export button area */
.export-bar {
    background: #0C1222;
    border-top: 1px solid rgba(99,179,237,0.1);
    padding: 0.6rem 2rem;
    display: flex; justify-content: flex-end; gap: 0.5rem;
}

/* Input */
[data-testid="stChatInput"] textarea {
    background: #0C1222 !important;
    border: 1px solid rgba(99,179,237,0.18) !important;
    border-radius: 12px !important;
    color: #E2E8F0 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.88rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
}

/* Sidebar */
.sb-brand {
    display: flex; align-items: center; gap: 0.75rem;
    padding-bottom: 1.1rem; margin-bottom: 1.1rem;
    border-bottom: 1px solid rgba(99,179,237,0.1);
}
.sb-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #3B82F6, #8B5CF6);
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center; font-size: 1rem; flex-shrink: 0;
}
.sb-name {
    font-size: 0.9rem; font-weight: 800;
    background: linear-gradient(135deg, #63B3ED, #A78BFA);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.sb-sub { font-size: 0.65rem; color: #334155 !important; }
.sb-sec { font-size: 0.62rem; font-weight: 700; color: #1E3A5F !important; text-transform: uppercase; letter-spacing: 0.8px; margin: 0.9rem 0 0.4rem; }
.doc-tag {
    display: inline-block;
    background: rgba(99,179,237,0.07); border: 1px solid rgba(99,179,237,0.16);
    border-radius: 5px; padding: 2px 7px; font-size: 0.67rem;
    color: #63B3ED !important; margin: 2px 2px 2px 0; font-weight: 500;
}
.sb-hr { border: none; border-top: 1px solid rgba(99,179,237,0.08); margin: 0.9rem 0; }
.sb-session {
    font-size: 0.62rem; font-family: 'DM Mono', monospace;
    color: #1E3A5F !important; margin-bottom: 0.5rem;
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05);
    border-radius: 6px; padding: 4px 8px;
}
.sb-footer {
    margin-top: 0.9rem;
    background: rgba(99,179,237,0.04);
    border: 1px solid rgba(99,179,237,0.09);
    border-radius: 8px; padding: 0.65rem 0.85rem;
    font-size: 0.67rem; color: #1E3A5F !important; line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)

# ── Load vectorstore ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading RetainIQ knowledge base...")
def get_vs():
    try:
        return load_vectorstore()
    except FileNotFoundError:
        return None

vs = get_vs()
if vs is None:
    st.error("Vector store not found. Run: python ingest.py")
    st.stop()
if not API_KEY:
    st.error("No API key found. Set OPENROUTER_API_KEY in your environment or in .streamlit/secrets.toml (see README).")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="sb-brand">
        <div class="sb-icon">🧠</div>
        <div>
            <div class="sb-name">RetainIQ</div>
            <div class="sb-sub">Retention Intelligence Platform</div>
        </div>
    </div>
    <div class="sb-session">Session · {st.session_state.session_id}</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-sec">Knowledge Base</div>', unsafe_allow_html=True)
    st.markdown("""
    <span class="doc-tag">📊 EDA Findings</span>
    <span class="doc-tag">🔍 SHAP Analysis</span>
    <span class="doc-tag">📋 Retention Playbook</span>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="sb-hr">', unsafe_allow_html=True)

    # Customer ID input
    st.markdown('<div class="sb-sec">Customer Context</div>', unsafe_allow_html=True)
    customer_id = st.text_input("Customer ID (optional)", placeholder="e.g. CUST-4821", label_visibility="collapsed")
    st.session_state["customer_id"] = customer_id.strip()

    st.markdown('<hr class="sb-hr">', unsafe_allow_html=True)
    st.markdown('<div class="sb-sec">Ask RetainIQ</div>', unsafe_allow_html=True)

    questions = [
        "Why are customers in the first 6 months churning?",
        "What is the #1 churn driver in the model?",
        "What actions should I take for a high-risk customer?",
        "How does contract type affect churn?",
        "What does a SHAP value of 2.09 mean?",
        "Which customer profile is highest risk?",
        "What is the ROI of deploying ChurnSense?",
        "Why is spend not predictive of churn?",
        "What is the Tier 1 retention playbook?",
        "How accurate is the ChurnSense model?",
    ]
    for q in questions:
        if st.button(q, key=q, use_container_width=True):
            st.session_state["prefill"] = q

    st.markdown('<hr class="sb-hr">', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.messages = []
            st.session_state.feedback = {}
            st.session_state.session_id = gen_session_id()
            st.rerun()
    with c2:
        st.metric("Messages", len(st.session_state.messages))

    # Export
    if st.session_state.messages:
        transcript = f"RetainIQ Session: {st.session_state.session_id}\n"
        transcript += f"Exported: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        transcript += "="*50 + "\n\n"
        for m in st.session_state.messages:
            role = "Retention agent" if m["role"] == "user" else "RetainIQ"
            transcript += f"[{role}] {m.get('timestamp','')}\n{m['content']}\n\n"
        st.download_button(
            "⬇ Export Chat",
            data=transcript,
            file_name=f"retainiq_{st.session_state.session_id}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.markdown(f"""
    <div class="sb-footer">
        <div style="color:#34D399;margin-bottom:3px;">● System Online</div>
        <div>ChurnSense · FAISS · Claude 3 Haiku</div>
        <div>BARBM222 · EU Business School</div>
    </div>
    """, unsafe_allow_html=True)

# ── Top nav ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="topnav">
    <div class="brand">
        <div class="brand-icon">🧠</div>
        <div>
            <div class="brand-name">RetainIQ Assistant</div>
            <div class="brand-sub">Powered by ChurnSense · RAG Intelligence Layer</div>
        </div>
    </div>
    <div class="nav-right">
        <span class="nav-pill">RAG</span>
        <span class="nav-pill">FAISS</span>
        <span class="nav-pill">SHAP-Aware</span>
        <span class="session-tag">{st.session_state.session_id}</span>
        <span class="nav-pill-green"><span class="live-dot"></span>Live</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Pipeline banner ───────────────────────────────────────────────────────────
st.markdown("""
<div class="pipeline">
    <div class="pipe-step">
        <span class="pipe-icon">📥</span>
        <div>
            <div class="pipe-label">Customer Data</div>
            <div class="pipe-sub">50K records · 22 features</div>
        </div>
    </div>
    <span class="pipe-arrow">→</span>
    <div class="pipe-step">
        <span class="pipe-icon">🤖</span>
        <div>
            <div class="pipe-label">ChurnSense</div>
            <div class="pipe-sub">Logistic Regression · AUC 0.90</div>
        </div>
    </div>
    <span class="pipe-arrow">→</span>
    <div class="pipe-step">
        <span class="pipe-icon">🔍</span>
        <div>
            <div class="pipe-label">SHAP Explainer</div>
            <div class="pipe-sub">Feature-level explanations</div>
        </div>
    </div>
    <span class="pipe-arrow">→</span>
    <div class="pipe-step pipe-step-active">
        <span class="pipe-icon">🧠</span>
        <div>
            <div class="pipe-label pipe-label-active">RetainIQ RAG</div>
            <div class="pipe-sub">You are here</div>
        </div>
    </div>
    <span class="pipe-arrow">→</span>
    <div class="pipe-step">
        <span class="pipe-icon">🎯</span>
        <div>
            <div class="pipe-label">Retention Action</div>
            <div class="pipe-sub">Agent · Email · Offer</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Metrics bar ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="metrics">
    <div class="m-item"><span class="m-val">3</span><span class="m-lbl">Source Docs</span></div>
    <div class="m-sep"></div>
    <div class="m-item"><span class="m-val">0.900</span><span class="m-lbl">AUC-ROC</span></div>
    <div class="m-sep"></div>
    <div class="m-item"><span class="m-val">94%</span><span class="m-lbl">Recall</span></div>
    <div class="m-sep"></div>
    <div class="m-item"><span class="m-val">€1.28M</span><span class="m-lbl">Revenue Protected</span></div>
    <div class="m-sep"></div>
    <div class="m-item"><span class="m-val">&lt;4 days</span><span class="m-lbl">Payback Period</span></div>
    <div class="m-sep"></div>
    <div class="m-item"><span class="m-val">Top-4</span><span class="m-lbl">Chunks Retrieved</span></div>
    <div class="m-sep"></div>
    <div class="m-item"><span class="m-val">all-MiniLM-L6-v2</span><span class="m-lbl">Embeddings</span></div>
</div>
""", unsafe_allow_html=True)

# ── Welcome ───────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="chat-wrap">
    <div class="welcome">
        <div class="w-icon">🧠</div>
        <div class="w-title">RetainIQ Assistant</div>
        <div class="w-desc">
            Your AI intelligence layer built on ChurnSense.<br>
            Ask about churn drivers, SHAP explanations, customer risk profiles,
            and retention strategies — all grounded in your model's own findings.
        </div>
        <div class="topic-grid">
            <div class="topic-card">
                <div class="t-icon">📊</div>
                <div class="t-title">EDA & Cohort Analysis</div>
                <div class="t-desc">Churn rates by segment, tenure cohorts, feature distributions</div>
            </div>
            <div class="topic-card">
                <div class="t-icon">🔍</div>
                <div class="t-title">SHAP Explainability</div>
                <div class="t-desc">Feature importance, individual prediction explanations</div>
            </div>
            <div class="topic-card">
                <div class="t-icon">🎯</div>
                <div class="t-title">Retention Playbook</div>
                <div class="t-desc">Risk-tiered actions, scripts, offers, escalation protocols</div>
            </div>
            <div class="topic-card">
                <div class="t-icon">💰</div>
                <div class="t-title">Business Impact</div>
                <div class="t-desc">ROI, payback period, revenue protection calculations</div>
            </div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

# ── Chat history ──────────────────────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    ts = msg.get("timestamp", "")
    if msg["role"] == "user":
        cid = f" · Customer: {msg.get('customer_id')}" if msg.get("customer_id") else ""
        st.markdown(f'<div class="msg-meta msg-meta-right">You · {ts}{cid}</div>', unsafe_allow_html=True)
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])
    else:
        latency = msg.get("latency", "")
        st.markdown(f'<div class="msg-meta">RetainIQ Assistant · {ts}</div>', unsafe_allow_html=True)
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(msg["content"])

            if msg.get("sources"):
                with st.expander(f"📎 {len(msg['sources'])} source(s) cited", expanded=True):
                    for src in msg["sources"]:
                        st.markdown(f"""
                        <div class="src-card">
                            <div class="src-name">📄 {html.escape(src['label'])}</div>
                            <div class="src-excerpt">{html.escape(src['excerpt'])}</div>
                        </div>
                        """, unsafe_allow_html=True)

            if msg.get("out_of_scope"):
                st.markdown('<div class="oos">⚠ Outside RetainIQ knowledge base</div>', unsafe_allow_html=True)

            if latency:
                st.markdown(f'<div class="latency">⚡ {latency}ms response</div>', unsafe_allow_html=True)

            # Feedback buttons
            fb_key = f"fb_{i}"
            current_fb = st.session_state.feedback.get(fb_key)
            if current_fb is None:
                col1, col2, col3 = st.columns([1, 1, 8])
                with col1:
                    if st.button("👍", key=f"up_{i}"):
                        st.session_state.feedback[fb_key] = "up"
                        st.rerun()
                with col2:
                    if st.button("👎", key=f"dn_{i}"):
                        st.session_state.feedback[fb_key] = "down"
                        st.rerun()
            elif current_fb == "up":
                st.markdown('<span style="font-size:0.72rem;color:#34D399;">✓ Marked as helpful</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span style="font-size:0.72rem;color:#F59E0B;">✓ Feedback recorded</span>', unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", None)
cid = st.session_state.get("customer_id", "")
placeholder = f"Asking about customer {cid}..." if cid else "Ask about churn drivers, SHAP values, risk profiles, or retention strategies..."
query = st.chat_input(placeholder) or prefill

if query:
    ts = datetime.datetime.now().strftime("%H:%M")
    user_msg = {"role": "user", "content": query, "timestamp": ts}
    if cid:
        user_msg["customer_id"] = cid
    st.session_state.messages.append(user_msg)

    st.markdown(f'<div class="msg-meta msg-meta-right">You · {ts}</div>', unsafe_allow_html=True)
    with st.chat_message("user", avatar="👤"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("Searching knowledge base..."):
            t0 = time.time()
            try:
                result = answer_query(vs, query, API_KEY)
                latency = int((time.time() - t0) * 1000)
                ts2 = datetime.datetime.now().strftime("%H:%M")

                st.markdown(result["answer"])

                if result["sources"]:
                    with st.expander(f"📎 {len(result['sources'])} source(s) cited", expanded=True):
                        for src in result["sources"]:
                            st.markdown(f"""
                            <div class="src-card">
                                <div class="src-name">📄 {html.escape(src['label'])}</div>
                                <div class="src-excerpt">{html.escape(src['excerpt'])}</div>
                            </div>
                            """, unsafe_allow_html=True)

                if result["out_of_scope"]:
                    st.markdown('<div class="oos">⚠ Outside RetainIQ knowledge base</div>', unsafe_allow_html=True)

                st.markdown(f'<div class="latency">⚡ {latency}ms response</div>', unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "out_of_scope": result["out_of_scope"],
                    "timestamp": ts2,
                    "latency": latency,
                })

            except Exception as e:
                err = f"Error: {str(e)}"
                st.error(err)
                st.session_state.messages.append({
                    "role": "assistant", "content": err,
                    "sources": [], "out_of_scope": False,
                    "timestamp": datetime.datetime.now().strftime("%H:%M"),
                    "latency": 0,
                })
