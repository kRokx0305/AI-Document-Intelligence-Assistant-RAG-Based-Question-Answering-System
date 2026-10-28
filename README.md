# AI Document Intelligence Assistant — RAG-Based Question Answering System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.2+-green.svg)](https://python.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-v1.30+-red.svg)](https://streamlit.io/)
[![Vector Stores](https://img.shields.io/badge/VectorStore-ChromaDB%20%7C%20FAISS-purple.svg)](https://github.com/chroma-core/chroma)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade **Document Intelligence & Question Answering System** powered by **Retrieval-Augmented Generation (RAG)**. This system enables users to upload heterogeneous multi-page documents (PDF, DOCX, TXT), index their semantic content into high-performance vector databases (FAISS / ChromaDB), and query them through an interactive conversational interface with source citations and quantitative RAG evaluation metrics.

---

## 📑 Table of Contents
1. [Key Features](#-key-features)
2. [Architecture Overview](#-architecture-overview)
3. [Project Structure](#-project-structure)
4. [Tech Stack](#-tech-stack)
5. [Getting Started](#-getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Environment Configuration](#environment-configuration)
   - [Running the Application](#running-the-application)
6. [Core Technical Components](#-core-technical-components)
   - [Document Processing & Chunking](#document-processing--chunking)
   - [Vector Search & Retrieval (FAISS vs. ChromaDB)](#vector-search--retrieval-faiss-vs-chromadb)
   - [Grounding & Hallucination Mitigation](#grounding--hallucination-mitigation)
   - [RAG Triad Evaluation Engine](#rag-triad-evaluation-engine)
7. [Documentation Links](#-documentation-links)
8. [License](#-license)

---

## 🚀 Key Features

- **Multi-Format Document Ingestion**: Ingests, parses, and cleans multi-page PDFs, Microsoft Word (.docx), and plain text (.txt / .md) files while tracking structural metadata (file name, page number, chunk index).
- **Configurable Chunking Pipeline**: Optimized semantic chunking using `RecursiveCharacterTextSplitter` with customizable chunk size, chunk overlap, and hierarchical separators.
- **Dual Vector Store Architecture**: Seamless toggle between **ChromaDB** (persistent, metadata-rich) and **FAISS** (Facebook AI Similarity Search, in-memory, ultra-fast vector math).
- **Flexible Embedding & LLM Backends**:
  - Supports OpenAI (`gpt-4o`, `gpt-4o-mini`, `text-embedding-3-small`), Google Gemini (`gemini-1.5-pro`, `gemini-1.5-flash`), and local HuggingFace embeddings (`all-MiniLM-L6-v2`).
  - Includes an offline fallback mock mode for instant local testing without API keys.
- **Advanced Retrieval Strategies**:
  - Standard Vector Cosine Similarity Search.
  - **Maximal Marginal Relevance (MMR)** to balance relevance with diversity and eliminate redundant context chunks.
  - Similarity score thresholding to filter out low-confidence context.
- **Source Attribution & Grounding**: Every generated answer includes interactive source citations highlighting exact source documents, page numbers, chunk text, and similarity scores.
- **Built-In RAG Evaluation Suite**:
  - Evaluates **Context Relevance** (signal-to-noise ratio in retrieved documents).
  - Evaluates **Faithfulness / Groundedness** (detection of hallucinations against retrieved context).
  - Evaluates **Answer Relevance** (alignment with the user's original query).

---

## 🏗 Architecture Overview

```
                      ┌──────────────────────────────────────────────┐
                      │              User Input / Query              │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
┌─────────────────────────┐         ┌───────────────────┐
│ Multi-Document Upload   │         │ Embed User Query  │
│ (PDF, DOCX, TXT)        │         │ (Embedding Model) │
└────────────┬────────────┘         └─────────┬─────────┘
             │                                │
             ▼                                │
┌─────────────────────────┐                   │
│ Text Extraction &       │                   │
│ Semantic Chunking       │                   │
└────────────┬────────────┘                   │
             │                                │
             ▼                                │
┌─────────────────────────┐                   │
│ Generate Embeddings     │                   │
│ (OpenAI / HuggingFace)  │                   │
└────────────┬────────────┘                   │
             │                                │
             ▼                                ▼
┌─────────────────────────┐         ┌───────────────────┐
│ Vector Store Index      │ ◄───────┤ Similarity / MMR  │
│ (ChromaDB / FAISS)      │         │ Retrieval (Top-K) │
└─────────────────────────┘         └─────────┬─────────┘
                                              │
                                              ▼
                                    ┌───────────────────┐
                                    │ Relevant Context  │
                                    │ + Source Metadata │
                                    └─────────┬─────────┘
                                              │
                                              ▼
                                    ┌───────────────────┐
                                    │ Grounded Prompt   │
                                    │ Construction      │
                                    └─────────┬─────────┘
                                              │
                                              ▼
                                    ┌───────────────────┐
                                    │ LLM Generation    │
                                    │ (GPT-4o / Gemini) │
                                    └─────────┬─────────┘
                                              │
                                              ▼
                         ┌──────────────────────────────────────────────┐
                         │   Streamlit Interface: Response + Sources    │
                         │   + Automated RAG Triad Evaluation Metrics   │
                         └──────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```bash
AI Document Intelligence Assistant/
├── README.md                 # Project overview, setup, and quickstart (this file)
├── ARCHITECTURE.md           # Deep-dive architecture and design decisions
├── EVALUATION_GUIDE.md       # Metrics explanation (Faithfulness, Relevance, Hallucination)
├── INTERVIEW_PREP.md         # Comprehensive interview Q&A based on resume bullets
├── requirements.txt          # Python dependencies
├── .env.example              # Sample environment variables
├── sample_documents/         # Ready-to-use sample policies and financial reports
│   ├── ACME_Corp_Cloud_Security_Policy.txt
│   └── Q3_Financial_Performance_Report.txt
├── app.py                    # Streamlit web application frontend
└── src/                      # Modular backend package
    ├── __init__.py
    ├── config.py             # Global constants, defaults, and environment parsing
    ├── document_loader.py    # Multi-format document parser (PDF, DOCX, TXT)
    ├── chunker.py            # Recursive character splitting with metadata retention
    ├── embeddings.py         # Embedding providers (OpenAI, HuggingFace, Mock)
    ├── vectorstore.py        # ChromaDB & FAISS integration with persistence
    ├── rag_chain.py          # Grounded QA chain with citations and streaming
    └── evaluator.py          # RAG Triad evaluation engine (Faithfulness, Relevance)
```

---

## 🛠 Tech Stack

| Layer | Technologies |
|---|---|
| **User Interface** | [Streamlit](https://streamlit.io/) |
| **Framework & Orchestration** | [LangChain](https://www.langchain.com/) |
| **Vector Databases** | [FAISS (Meta)](https://github.com/facebookresearch/faiss), [ChromaDB](https://www.trychroma.com/) |
| **Document Processing** | `pypdf`, `python-docx`, `tiktoken` |
| **LLMs Supported** | OpenAI (`gpt-4o`, `gpt-4o-mini`), Google Gemini (`gemini-1.5-flash`), Anthropic |
| **Embedding Models** | `text-embedding-3-small`, `sentence-transformers/all-MiniLM-L6-v2` |
| **Evaluation Metrics** | RAG Triad (Context Relevance, Faithfulness/Hallucination, Answer Relevance) |

---

## ⚡ Getting Started

### Prerequisites
- Python 3.9, 3.10, or 3.11 installed.
- (Optional) An OpenAI API key or Google Gemini API key. *(The system includes an offline test mode if you want to explore the pipeline without an API key!)*

### Installation

1. **Clone the repository / navigate to workspace**:
   ```bash
   cd "AI Document Intelligence Assistant — RAG-Based Question Answering System "
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate    # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Environment Configuration

Create a `.env` file in the root directory by copying the example:
```bash
cp .env.example .env
```

Edit `.env` to supply your API credentials:
```env
OPENAI_API_KEY="your-openai-api-key-here"
# or
GOOGLE_API_KEY="your-google-api-key-here"

# Vector Store configuration
VECTOR_STORE_TYPE="chroma" # or "faiss"
PERSIST_DIRECTORY="./data/chroma_db"
```
*(You can also set your API keys directly inside the Streamlit sidebar during runtime.)*

### Running the Application

Launch the Streamlit interactive dashboard:
```bash
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

---

## 🔍 Core Technical Components

### Document Processing & Chunking
Documents are parsed into uniform text segments while retaining file provenance:
- **Chunk Size**: `1000` characters (default, tunable).
- **Chunk Overlap**: `200` characters (prevents semantic boundary truncation across sentences).
- **Separators**: `["\n\n", "\n", ". ", " ", ""]` ensuring natural paragraph and sentence preservation.

### Vector Search & Retrieval (FAISS vs. ChromaDB)
Users can select between two vector backends:
1. **ChromaDB**: SQLite/DuckDB-backed vector database with persistent disk storage, rich metadata querying, and zero-setup deployment.
2. **FAISS (Facebook AI Similarity Search)**: Extremely fast C++ vector similarity indexing (L2 / Inner Product distance), optimized for high-throughput batch vector operations.

Retrieval options include:
- **Similarity Search**: Direct Top-K cosine similarity.
- **Maximal Marginal Relevance (MMR)**: Penalizes duplicate chunks to ensure diverse perspectives from multi-section documents.

### Grounding & Hallucination Mitigation
The system enforces strict grounding:
1. **Zero-Shot Prompt Constraint**: The system prompt instructs the LLM: *"Answer strictly using only the provided context. If the context does not contain the answer, explicitly state that the information is unavailable."*
2. **Citation Injection**: Every fact is linked back to `[Doc: <filename>, Page: <n>]`.
3. **Low Temperature**: Default generation temperature is set to `0.0` or `0.1` to reduce stochastic extrapolation.

### RAG Triad Evaluation Engine
The application includes a built-in evaluation panel implementing the industry-standard RAG Triad:
1. **Context Relevance**: Did the retriever fetch clean, relevant chunks without irrelevant noise?
2. **Faithfulness (Groundedness)**: Are the claims in the generated answer 100% supported by the retrieved context, or did the LLM hallucinate?
3. **Answer Relevance**: Does the generated answer directly resolve the user's question?

---

## 📚 Documentation Links
- **[ARCHITECTURE.md](file:///Users/krokx/AI%20Document%20Intelligence%20Assistant%20—%20RAG-Based%20Question%20Answering%20System%20/ARCHITECTURE.md)**: Deep dive into chunking trade-offs, vector indices, retrieval math, and hallucination reduction.
- **[EVALUATION_GUIDE.md](file:///Users/krokx/AI%20Document%20Intelligence%20Assistant%20—%20RAG-Based%20Question%20Answering%20System%20/EVALUATION_GUIDE.md)**: Technical breakdown of RAG evaluation metrics, formulas, and LLM-as-a-judge implementation.
- **[INTERVIEW_PREP.md](file:///Users/krokx/AI%20Document%20Intelligence%20Assistant%20—%20RAG-Based%20Question%20Answering%20System%20/INTERVIEW_PREP.md)**: 25+ detailed interview questions, technical trade-offs, and resume talking points.

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
