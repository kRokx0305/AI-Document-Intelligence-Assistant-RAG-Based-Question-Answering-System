"""
AI Document Intelligence Assistant — RAG-Based Question Answering System
Interactive Streamlit Application with ChromaDB / FAISS, Grounded QA, Source Citations,
and Live RAG Triad Evaluation Dashboard.
"""

import os
import io
import time
import warnings
from pathlib import Path
import streamlit as st

# Suppress known harmless environment warnings (e.g. macOS LibreSSL urllib3 notice)
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")

# Internal Modules
from src.config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    DEFAULT_RETRIEVAL_TYPE,
    DEFAULT_TEMPERATURE,
    DEFAULT_VECTOR_STORE,
    SAMPLE_DOCS_DIR
)
from src.document_loader import load_document, Document
from src.chunker import chunk_documents
from src.embeddings import get_embedding_provider, OfflineDemoEmbeddings
from src.vectorstore import VectorStoreManager
from src.rag_chain import RAGPipeline
from src.evaluator import RAGEvaluator

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="AI Document Intelligence Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished enterprise look
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .source-box {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #E2E8F0;
        color: #334155;
    }
    .badge-green { background-color: #DCFCE7; color: #166534; }
    .badge-amber { background-color: #FEF3C7; color: #92400E; }
    .badge-red { background-color: #FEE2E2; color: #991B1B; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Session State Initialization
# ==============================================================================
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []  # List of raw Document objects
if "chunks" not in st.session_state:
    st.session_state.chunks = []  # List of chunked Document objects
if "vsm" not in st.session_state:
    st.session_state.vsm = None  # VectorStoreManager instance
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List of chat messages
if "last_eval_result" not in st.session_state:
    st.session_state.last_eval_result = None  # Full evaluation payload

# ==============================================================================
# Sidebar: Setup & Pipeline Configuration
# ==============================================================================
with st.sidebar:
    st.title("⚙️ RAG Configuration")

    st.subheader("1. LLM & Embeddings")
    provider_choice = st.selectbox(
        "Provider",
        options=["OpenAI", "Google Gemini", "Offline Demo Mode"],
        index=0,
        help="Select LLM provider. 'Offline Demo Mode' enables full local indexing & search without API keys."
    )

    api_key_input = ""
    if provider_choice == "OpenAI":
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Enter your sk-... API key or set OPENAI_API_KEY in .env"
        )
        selected_model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
    elif provider_choice == "Google Gemini":
        api_key_input = st.text_input(
            "Google API Key",
            type="password",
            value=os.getenv("GOOGLE_API_KEY", ""),
            help="Enter your Google AI Studio API key or set GOOGLE_API_KEY in .env"
        )
        selected_model = st.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
    else:
        st.info("💡 Running in Offline Demo Mode. Uses deterministic embeddings & heuristic grounded answer generation.")
        selected_model = "local-heuristic"

    st.divider()

    st.subheader("2. Vector Store")
    vector_store_type = st.radio(
        "Backend Database",
        options=["ChromaDB", "FAISS"],
        index=0 if DEFAULT_VECTOR_STORE == "chroma" else 1,
        help="ChromaDB supports SQLite persistent metadata; FAISS provides hyper-optimized C++ similarity math."
    )

    with st.expander("🛠 Advanced Pipeline Hyperparameters"):
        chunk_size = st.slider("Chunk Size (characters)", 200, 2000, DEFAULT_CHUNK_SIZE, step=100)
        chunk_overlap = st.slider("Chunk Overlap (characters)", 0, 500, DEFAULT_CHUNK_OVERLAP, step=50)
        retrieval_strategy = st.selectbox(
            "Retrieval Strategy",
            options=["similarity", "mmr"],
            index=0 if DEFAULT_RETRIEVAL_TYPE == "similarity" else 1,
            help="MMR (Maximal Marginal Relevance) penalizes redundancy to retrieve diverse chunks."
        )
        top_k = st.slider("Top-K Chunks to Retrieve", 1, 10, DEFAULT_TOP_K)
        temperature = st.slider("LLM Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE, step=0.05)

    st.divider()

    # Index management
    st.subheader("3. Database Status")
    num_docs = len(set(d.metadata.get("source", "") for d in st.session_state.indexed_docs))
    num_chunks = len(st.session_state.chunks)
    st.write(f"📁 Indexed Files: **{num_docs}**")
    st.write(f"🧩 Total Chunks: **{num_chunks}**")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Load Samples", use_container_width=True, help="Load pre-bundled corporate policy & financial report"):
            with st.spinner("Indexing sample documents..."):
                sample_docs = []
                for p in SAMPLE_DOCS_DIR.glob("*.txt"):
                    docs = load_document(p, p.name)
                    sample_docs.extend(docs)

                if sample_docs:
                    st.session_state.indexed_docs = sample_docs
                    chunks = chunk_documents(sample_docs, chunk_size, chunk_overlap)
                    st.session_state.chunks = chunks

                    # Initialize vector store
                    provider_code = "openai" if provider_choice == "OpenAI" and api_key_input else (
                        "gemini" if provider_choice == "Google Gemini" and api_key_input else "offline"
                    )
                    emb = get_embedding_provider(provider_code, api_key_input)
                    vsm = VectorStoreManager(
                        store_type=vector_store_type.lower(),
                        embeddings=emb
                    )
                    vsm.initialize_with_documents(chunks)
                    st.session_state.vsm = vsm
                    st.success("Sample documents indexed!")
                    st.rerun()

    with col_btn2:
        if st.button("Reset DB", use_container_width=True, help="Wipe all indexed chunks"):
            if st.session_state.vsm:
                st.session_state.vsm.clear()
            st.session_state.vsm = None
            st.session_state.indexed_docs = []
            st.session_state.chunks = []
            st.session_state.chat_history = []
            st.session_state.last_eval_result = None
            st.success("Database cleared.")
            st.rerun()

# ==============================================================================
# Main View & Tabs
# ==============================================================================
st.markdown('<div class="main-title">AI Document Intelligence Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Context-Aware Question Answering • LangChain • ChromaDB / FAISS • Grounded Citations • RAG Triad Evaluation</div>',
    unsafe_allow_html=True
)

tab_chat, tab_eval, tab_docs, tab_arch = st.tabs([
    "💬 Document Q&A",
    "📊 RAG Evaluation & Hallucinations",
    "📁 Document Explorer",
    "🧠 Architecture & Interview Guide"
])

# ------------------------------------------------------------------------------
# TAB 1: Document Q&A (Chat)
# ------------------------------------------------------------------------------
with tab_chat:
    if not st.session_state.chunks:
        st.warning(
            "⚠️ No documents indexed yet. Upload documents in the **Document Explorer** tab "
            "or click **'Load Samples'** in the sidebar to try sample documents immediately!"
        )

    # Display past conversation
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 Verified Sources ({len(msg['sources'])} chunks cited)"):
                    for i, s in enumerate(msg["sources"], 1):
                        st.markdown(
                            f"**[{i}] {s['source']} (Page {s['page']})** — "
                            f"<span class='badge badge-green'>Similarity: {s['similarity_score']}</span>",
                            unsafe_allow_html=True
                        )
                        st.caption(f"Chunk ID: `{s['chunk_index']}` | Characters: `{s['char_length']}`")
                        st.markdown(f"> *{s['snippet']}*")

            if msg.get("metrics"):
                m = msg["metrics"]
                st.caption(f"⏱ Retrieval: `{m['retrieval_latency']}s` | Generation: `{m['generation_latency']}s`")

    # Chat Input
    user_query = st.chat_input("Ask a question about your indexed documents...")

    if user_query:
        if not st.session_state.vsm or not st.session_state.chunks:
            st.error("Please index at least one document before asking questions.")
        else:
            # Append user message
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # Build RAG Pipeline
            provider_code = "openai" if provider_choice == "OpenAI" and api_key_input else (
                "gemini" if provider_choice == "Google Gemini" and api_key_input else "offline"
            )
            emb = get_embedding_provider(provider_code, api_key_input)
            pipeline = RAGPipeline(
                vector_store_manager=st.session_state.vsm,
                provider=provider_code,
                api_key=api_key_input,
                model_name=selected_model,
                temperature=temperature
            )

            # Assistant response generation
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_answer = ""
                final_sources = []
                final_docs = []
                r_latency = 0.0
                g_latency = 0.0
                context_str = ""

                # Stream response
                for chunk in pipeline.stream_answer_query(
                    query=user_query,
                    k=top_k,
                    retrieval_type=retrieval_strategy
                ):
                    if chunk["type"] == "token":
                        full_answer += chunk["content"]
                        response_placeholder.markdown(full_answer + "▌")
                    elif chunk["type"] == "done":
                        final_sources = chunk["sources"]
                        final_docs = chunk["retrieved_docs"]
                        r_latency = chunk["retrieval_latency"]
                        g_latency = chunk["generation_latency"]
                        context_str = chunk["context_text"]

                response_placeholder.markdown(full_answer)

                if final_sources:
                    with st.expander(f"📚 Verified Sources ({len(final_sources)} chunks cited)"):
                        for i, s in enumerate(final_sources, 1):
                            st.markdown(
                                f"**[{i}] {s['source']} (Page {s['page']})** — "
                                f"<span class='badge badge-green'>Similarity: {s['similarity_score']}</span>",
                                unsafe_allow_html=True
                            )
                            st.caption(f"Chunk ID: `{s['chunk_index']}` | Characters: `{s['char_length']}`")
                            st.markdown(f"> *{s['snippet']}*")

                st.caption(f"⏱ Retrieval: `{r_latency}s` | Generation: `{g_latency}s`")

                # Run live evaluation on this turn
                evaluator = RAGEvaluator(llm_judge=pipeline.llm)
                eval_payload = evaluator.run_full_evaluation(
                    query=user_query,
                    answer=full_answer,
                    retrieved_docs=final_docs,
                    context_text=context_str
                )
                eval_payload["retrieval_latency"] = r_latency
                eval_payload["generation_latency"] = g_latency
                eval_payload["sources"] = final_sources
                eval_payload["query"] = user_query
                eval_payload["answer"] = full_answer
                st.session_state.last_eval_result = eval_payload

                # Store assistant message in history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": full_answer,
                    "sources": final_sources,
                    "metrics": {
                        "retrieval_latency": r_latency,
                        "generation_latency": g_latency
                    }
                })

# ------------------------------------------------------------------------------
# TAB 2: RAG Evaluation & Hallucination Analysis
# ------------------------------------------------------------------------------
with tab_eval:
    st.subheader("🔬 Real-Time RAG Triad Evaluation")
    st.markdown(
        "Evaluates the three essential pillars of RAG: **Context Relevance**, "
        "**Faithfulness (Hallucination Detection)**, and **Answer Relevance**."
    )

    ev = st.session_state.last_eval_result

    if not ev:
        st.info("Run a query in the **Document Q&A** tab to inspect automated evaluation metrics.")
    else:
        st.markdown(f"**Evaluated Query:** *\"{ev['query']}\"*")

        col1, col2, col3, col4 = st.columns(4)

        c_rel = ev["context_relevance"]["score"]
        faith = ev["faithfulness"]["score"]
        a_rel = ev["answer_relevance"]["score"]
        halluc_risk = ev["hallucination_risk"]

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Context Relevance</div>
                <div class="metric-val">{int(c_rel * 100)}%</div>
                <small>{ev['context_relevance']['reason']}</small>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Faithfulness (Grounded)</div>
                <div class="metric-val">{int(faith * 100)}%</div>
                <small>{ev['faithfulness']['reason']}</small>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Answer Relevance</div>
                <div class="metric-val">{int(a_rel * 100)}%</div>
                <small>{ev['answer_relevance']['reason']}</small>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            risk_color = "#166534" if halluc_risk == 0 else "#991B1B"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Hallucination Risk</div>
                <div class="metric-val" style="color: {risk_color}">{int(halluc_risk * 100)}%</div>
                <small>{'⚠️ Risk detected' if ev['hallucination_detected'] else '✅ Fully grounded'}</small>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Hallucination Detail Inspection
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("🛡 Hallucination & Faithfulness Audit")
            if ev["hallucination_detected"]:
                st.error("🚨 Potential Hallucination Identified!")
                st.write("The following statements in the answer could not be verified in the retrieved documents:")
                for s in ev["faithfulness"].get("unsupported_statements", []):
                    st.markdown(f"- ❌ *\"{s}\"*")
            else:
                st.success("✅ **100% Grounded**: Every factual proposition in the generated answer is directly backed by the retrieved context.")

            st.write(f"- Total statements verified: **{ev['faithfulness'].get('total_claims', 0)}**")
            st.write(f"- Supported claims: **{ev['faithfulness'].get('supported_claims', 0)}**")

        with col_right:
            st.subheader("⏱ Latency & Retrieval Quality Breakdown")
            st.write(f"- **Retrieval Latency**: `{ev.get('retrieval_latency', 0)}s`")
            st.write(f"- **Generation Latency**: `{ev.get('generation_latency', 0)}s`")
            st.write(f"- **Retrieved Chunks**: `{len(ev.get('sources', []))}`")

            if ev.get("sources"):
                st.markdown("**Retrieved Chunk Similarity Scores:**")
                for s in ev["sources"]:
                    raw_sim = float(s.get("similarity_score", 0.0))
                    clamped_score = float(min(1.0, max(0.0, raw_sim)))
                    st.progress(
                        clamped_score,
                        text=f"{s['source']} (Page {s['page']}) — Score: {clamped_score:.3f}"
                    )

# ------------------------------------------------------------------------------
# TAB 3: Document Explorer
# ------------------------------------------------------------------------------
with tab_docs:
    st.subheader("📁 Ingest & Inspect Documents")

    uploaded_files = st.file_uploader(
        "Upload Multi-Format Documents (PDF, DOCX, TXT, MD)",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Index Uploaded Documents", type="primary"):
            with st.spinner("Parsing and chunking documents..."):
                all_new_docs = []
                for uf in uploaded_files:
                    bytes_data = uf.read()
                    parsed = load_document(bytes_data, uf.name)
                    all_new_docs.extend(parsed)

                if all_new_docs:
                    st.session_state.indexed_docs.extend(all_new_docs)
                    new_chunks = chunk_documents(all_new_docs, chunk_size, chunk_overlap)
                    st.session_state.chunks.extend(new_chunks)

                    provider_code = "openai" if provider_choice == "OpenAI" and api_key_input else (
                        "gemini" if provider_choice == "Google Gemini" and api_key_input else "offline"
                    )
                    emb = get_embedding_provider(provider_code, api_key_input)
                    vsm = VectorStoreManager(
                        store_type=vector_store_type.lower(),
                        embeddings=emb
                    )
                    vsm.initialize_with_documents(st.session_state.chunks)
                    st.session_state.vsm = vsm

                    st.success(f"Successfully processed {len(uploaded_files)} files into {len(new_chunks)} chunks!")
                    st.rerun()

    st.divider()

    # Document & Chunk Overview
    if st.session_state.chunks:
        st.subheader(f"Current Knowledge Base ({len(st.session_state.chunks)} Total Chunks)")

        doc_stats = {}
        for c in st.session_state.chunks:
            src = c.metadata.get("source", "Unknown")
            doc_stats[src] = doc_stats.get(src, 0) + 1

        cols = st.columns(min(4, len(doc_stats)))
        for idx, (doc_name, count) in enumerate(doc_stats.items()):
            with cols[idx % len(cols)]:
                st.metric(label=doc_name[:20] + ("..." if len(doc_name) > 20 else ""), value=f"{count} chunks")

        st.subheader("🔍 Chunk Inspector")
        chunk_options = [
            f"Chunk #{i} | {c.metadata.get('source')} (Page {c.metadata.get('page')})"
            for i, c in enumerate(st.session_state.chunks)
        ]
        selected_chunk_idx = st.selectbox("Select chunk to view text & metadata", range(len(chunk_options)), format_func=lambda x: chunk_options[x])

        selected_chunk = st.session_state.chunks[selected_chunk_idx]
        st.markdown(f"**Document**: `{selected_chunk.metadata.get('source')}` | **Page**: `{selected_chunk.metadata.get('page')}` | **Tokens**: `~{selected_chunk.metadata.get('token_estimate')}`")
        st.text_area("Chunk Content", value=selected_chunk.page_content, height=180)
    else:
        st.info("No documents currently loaded. Upload files above or load sample documents from the sidebar.")

# ------------------------------------------------------------------------------
# TAB 4: Architecture & Interview Guide
# ------------------------------------------------------------------------------
with tab_arch:
    st.subheader("🧠 System Architecture & Resume Mapping")

    st.markdown("""
    This project fulfills all four resume requirements:
    1. **Context-Aware QA**: Python + LLM API (OpenAI / Gemini) streaming answers strictly conditioned on context.
    2. **RAG Pipeline**: Semantic chunking (`RecursiveCharacterTextSplitter`), high-dimensional embeddings, and vector similarity search.
    3. **Vector Store Integration**: Dynamic toggle between **ChromaDB** (persistent, metadata-rich) and **FAISS** (ultra-fast C++ similarity index) with Maximal Marginal Relevance (MMR).
    4. **Streamlit UI & Evaluation**: Built-in source citations with page numbers and full **RAG Triad** evaluation (Context Relevance, Faithfulness, Answer Relevance).
    """)

    st.divider()

    st.markdown("### 📚 Key Architecture References")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        - **Chunking Strategy**: Recursive character splitting with sliding window overlap (1000 characters, 200 overlap) to preserve multi-sentence semantic relationships without context dilution.
        - **Cosine vs. Euclidean**: Embeddings are unit-normalized so Cosine Similarity $\\equiv$ Dot Product, optimizing inference throughput.
        - **Maximal Marginal Relevance (MMR)**: Balances query relevance with chunk diversity to avoid duplicate retrieved paragraphs.
        """)
    with col_b:
        st.markdown("""
        - **Hallucination Guardrails**: Explicit zero-shot abstention constraints in system prompt + zero temperature + source attribution brackets.
        - **RAG Triad Evaluation**:
          - *Context Relevance*: Precision of retrieved chunks against user query.
          - *Faithfulness*: Groundedness verification of factual statements against source context.
          - *Answer Relevance*: Semantic alignment between user intent and generated response.
        """)

    st.info("💡 For deep architectural details and over 20+ interview questions & answers, refer to `ARCHITECTURE.md`, `EVALUATION_GUIDE.md`, and `INTERVIEW_PREP.md` in the project root.")
