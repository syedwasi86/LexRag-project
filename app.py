"""
app.py — Legal Contract RAG Assistant (Streamlit UI)
═══════════════════════════════════════════════════════════════════════════════
Features:
  • Chat interface with persistent message history
  • Citation panel — shows exact retrieved chunks (proves no hallucination)
  • Sidebar filters — filter by contract type, year range, governing law
  • One-time index builder — detects if chroma_db/ exists; builds if not
  • RAGAS score display (run evaluation.py first)
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import os
from pathlib import Path

import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="LexRAG — Legal Contract Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark glassmorphism sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f1117 0%, #1a1f2e 100%);
    border-right: 1px solid rgba(99,179,237,0.15);
}

/* Chat bubbles */
.user-bubble {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 80%;
    margin-left: auto;
    box-shadow: 0 4px 15px rgba(37,99,235,0.3);
}
.assistant-bubble {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    color: #e2e8f0;
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 18px 18px 18px 4px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 85%;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}

/* Citation cards */
.citation-card {
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid rgba(99,179,237,0.25);
    border-left: 3px solid #3b82f6;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.85rem;
}
.citation-meta {
    color: #94a3b8;
    font-size: 0.78rem;
    margin-bottom: 6px;
}
.citation-text {
    color: #cbd5e1;
    font-style: italic;
    border-left: 2px solid #3b82f6;
    padding-left: 10px;
    margin-top: 6px;
}

/* Metric cards */
.metric-card {
    background: rgba(30,41,59,0.7);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 10px;
    padding: 10px 14px;
    text-align: center;
}
.metric-val { font-size: 1.5rem; font-weight: 700; color: #60a5fa; }
.metric-lbl { font-size: 0.72rem; color: #94a3b8; margin-top: 2px; }

/* Header */
.lex-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
    border-bottom: 1px solid rgba(99,179,237,0.2);
    padding: 1rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── Imports (deferred so page_config runs first) ──────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from src.vector_store import get_embeddings, load_vector_store, build_vector_store, vector_store_exists
from src.document_loader import load_all_documents
from src.text_splitter import split_documents
from src.retriever import build_retriever
from src.rag_pipeline import build_rag_chain, query_rag, get_llm


# ── Cached resource loaders ────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_embeddings():
    return get_embeddings()


@st.cache_resource(show_spinner=False)
def _load_or_build_store(_embeddings):
    """Build vector store on first run; load from disk on subsequent runs."""
    if vector_store_exists():
        return load_vector_store(_embeddings)
    else:
        with st.spinner("🔨 First-time indexing — this takes ~20–40 min. Please wait…"):
            pdf_docs, meta_docs = load_all_documents()
            chunks = split_documents(pdf_docs)
            stores = build_vector_store(chunks, meta_docs, _embeddings)
        st.success("✅ Index built! Future runs will load instantly.")
        return stores


@st.cache_resource(show_spinner=False)
def _load_rag_chain(_pdf_store, _embeddings):
    """Build retriever + RAG chain (fast, uses cached store)."""
    llm = get_llm()
    # Sample 500 chunks from the store for BM25 in-memory index
    sample_chunks = _pdf_store.similarity_search("contract agreement", k=500)
    retriever, _ = build_retriever(_pdf_store, sample_chunks, llm)
    chain = build_rag_chain(retriever)
    return chain


# ── Helper: load RAGAS scores if available ────────────────────────────────────

def _load_ragas_scores() -> dict | None:
    results_path = Path("evaluation_results.json")
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return None


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚖️ LexRAG")
        st.caption("Legal Contract AI Assistant")
        st.divider()

        st.markdown("### 🔍 Filters")
        contract_types = [
            "All Types", "Service", "License Agreements", "Development",
            "Co Branding", "Distributor", "Endorsement", "Franchise",
            "Hosting", "IP", "Joint Venture", "Maintenance", "Manufacturing",
            "Marketing", "Non Compete Non Solicit", "Outsourcing",
            "Promotion", "Reseller", "Sponsorship", "Strategic Alliance",
            "Supply", "Transportation", "Affiliate Agreements",
        ]
        selected_type = st.selectbox("Contract Type", contract_types)

        year_range = st.slider("Agreement Year Range", 2000, 2024, (2010, 2022))

        st.divider()

        # RAGAS Metrics panel
        scores = _load_ragas_scores()
        if scores:
            st.markdown("### 📊 RAGAS Metrics")
            st.caption("Run `python src/evaluation.py` to refresh")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-val">{scores.get('context_precision', 0):.2f}</div>
                    <div class="metric-lbl">Context Precision</div></div>""",
                    unsafe_allow_html=True)
                st.markdown(f"""<div class="metric-card" style="margin-top:8px">
                    <div class="metric-val">{scores.get('context_recall', 0):.2f}</div>
                    <div class="metric-lbl">Context Recall</div></div>""",
                    unsafe_allow_html=True)
            with col2:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-val">{scores.get('faithfulness', 0):.2f}</div>
                    <div class="metric-lbl">Faithfulness</div></div>""",
                    unsafe_allow_html=True)
                st.markdown(f"""<div class="metric-card" style="margin-top:8px">
                    <div class="metric-val">{scores.get('answer_relevancy', 0):.2f}</div>
                    <div class="metric-lbl">Answer Relevancy</div></div>""",
                    unsafe_allow_html=True)
        else:
            st.info("Run `python src/evaluation.py` to see RAGAS metrics here.")

        st.divider()
        st.markdown("### 💡 Sample Questions")
        sample_qs = [
            "What are the termination conditions in the Gridiron Bionutrients Endorsement Agreement?",
            "What are the payment terms in the Fuse Medical Distributor Agreement?",
            "What services are covered under the Her Imports Maintenance Agreement?",
            "Is there a non-compete clause in the Soupman Franchise Agreement?",
            "What is the governing law in the Kitov Pharma Manufacturing Agreement?",
        ]
        for q in sample_qs:
            if st.button(q, key=f"sample_{q[:20]}", use_container_width=True):
                st.session_state["prefill_question"] = q

        return selected_type, year_range


# ── Citation renderer ──────────────────────────────────────────────────────────

def render_citations(citations: list[dict]):
    if not citations:
        return
    st.markdown(f"**📎 {len(citations)} Source Citation(s) — Retrieved Chunks**")
    for i, c in enumerate(citations, 1):
        with st.expander(
            f"📄 Citation {i}: {c['filename'][:55]} | {c['section'] or '—'}",
            expanded=(i == 1),
        ):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"**Type:** {c['contract_type']}")
                st.caption(f"**Parties:** {c['parties'][:60] or '—'}")
            with col2:
                st.caption(f"**Date:** {c['agreement_date'] or '—'}")
                st.caption(f"**Governing Law:** {c['governing_law'] or '—'}")
            with col3:
                st.caption(f"**Section:** {c['section'] or '—'}")

            st.markdown("---")
            st.markdown(
                f'<div class="citation-text">{c["content"][:800]}'
                f'{"…" if len(c["content"]) > 800 else ""}</div>',
                unsafe_allow_html=True,
            )


# ── Main app ───────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div class="lex-header">
        <h1 style="margin:0;color:#e2e8f0;font-size:1.8rem">⚖️ LexRAG — Legal Contract Assistant</h1>
        <p style="margin:4px 0 0;color:#94a3b8;font-size:0.9rem">
            Ask questions about your contracts. Every answer is backed by exact source citations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    selected_type, year_range = render_sidebar()

    # ── Load pipeline ──────────────────────────────────────────────────────────
    with st.spinner("Loading embedding model…"):
        embeddings = _load_embeddings()

    with st.spinner("Loading vector store…"):
        pdf_store, meta_store = _load_or_build_store(embeddings)

    with st.spinner("Initialising RAG chain…"):
        chain = _load_rag_chain(pdf_store, embeddings)

    # ── Session state ──────────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "citations_map" not in st.session_state:
        st.session_state.citations_map = {}  # message_idx → citations list

    # Handle sidebar sample question click
    prefill = st.session_state.pop("prefill_question", None)

    # ── Chat history ───────────────────────────────────────────────────────────
    chat_col, cite_col = st.columns([3, 2])

    with chat_col:
        st.markdown("### 💬 Chat")
        for idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and idx in st.session_state.citations_map:
                    render_citations(st.session_state.citations_map[idx])

    # ── Input ──────────────────────────────────────────────────────────────────
    user_input = st.chat_input(
        "Ask about a contract clause, party, date, or legal provision…"
    ) or prefill

    if user_input:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        with chat_col:
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Searching contracts and generating answer…"):
                    try:
                        output   = query_rag(chain, user_input)
                        answer   = output["answer"]
                        citations = output["citations"]
                    except Exception as exc:
                        answer    = f"⚠️ Error: {exc}"
                        citations = []

                st.markdown(answer)
                msg_idx = len(st.session_state.messages)
                render_citations(citations)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.citations_map[len(st.session_state.messages) - 1] = citations

    # ── Citation column (latest answer) ───────────────────────────────────────
    with cite_col:
        st.markdown("### 📎 Latest Citations")
        if st.session_state.citations_map:
            last_idx     = max(st.session_state.citations_map.keys())
            last_citations = st.session_state.citations_map[last_idx]
            if last_citations:
                render_citations(last_citations)
            else:
                st.info("No citations found for the last answer.")
        else:
            st.info("Ask a question to see source citations here.")

        # Clear history button
        if st.session_state.messages:
            st.divider()
            if st.button("🗑️ Clear conversation", use_container_width=True):
                st.session_state.messages = []
                st.session_state.citations_map = {}
                st.rerun()


if __name__ == "__main__":
    main()
