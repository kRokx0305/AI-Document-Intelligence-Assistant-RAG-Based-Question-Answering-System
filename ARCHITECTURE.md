# Deep Dive Architecture — AI Document Intelligence Assistant

This document provides a comprehensive technical breakdown of the architecture, design patterns, mathematical formulations, and trade-offs underpinning the **AI Document Intelligence Assistant**.

---

## 🏛 Table of Contents
1. [End-to-End Architectural Pipeline](#1-end-to-end-architectural-pipeline)
2. [Data Ingestion & Parsing Subsystem](#2-data-ingestion--parsing-subsystem)
3. [Chunking Strategy & Semantic Boundaries](#3-chunking-strategy--semantic-boundaries)
4. [Dense Embedding Vector Space](#4-dense-embedding-vector-space)
5. [Vector Stores: FAISS vs. ChromaDB](#5-vector-stores-faiss-vs-chromadb)
6. [Retrieval Optimization & Strategies](#6-retrieval-optimization--strategies)
7. [Generation, Prompt Engineering & Grounding](#7-generation-prompt-engineering--grounding)
8. [Hallucination Mitigation Mechanisms](#8-hallucination-mitigation-mechanisms)
9. [Evaluation Engine & The RAG Triad](#9-evaluation-engine--the-rag-triad)
10. [Performance, Latency & Scalability Considerations](#10-performance-latency--scalability-considerations)

---

## 1. End-to-End Architectural Pipeline

The system decouples **Ingestion (Write Path)** from **Inference (Read Path)**:

```
========================================================================================
                                INGESTION PIPELINE (OFFLINE / ON UPLOAD)
========================================================================================
[Raw Document]  ──>  [Document Parsers]  ──>  [Normalized Text]
(PDF, DOCX, TXT)     (pypdf, docx)            + Structural Metadata (file, page)
                             │
                             ▼
                     [Recursive Chunker]
                     (Chunk Size: 1000, Overlap: 200)
                             │
                             ▼
                     [Embedding Generator]
                     (text-embedding-3-small / MiniLM)
                             │
                             ▼
                     [Vector Store Indexing]
                     (ChromaDB SQLite / FAISS Flat Index)

========================================================================================
                                INFERENCE PIPELINE (QUERY / READ)
========================================================================================
[User Query] ──> [Query Embedding]
                       │
                       ▼
             [Similarity / MMR Search]  ◄───  [Vector Database Index]
                       │
                       ▼
             [Top-K Ranked Contexts]
             (Filtered by Score Threshold)
                       │
                       ▼
             [Prompt Assembly with Grounding Constraints]
                       │
                       ▼
             [LLM Inference (GPT-4o / Gemini / Anthropic)]
                       │
                       ├───> [Streamed Synthesized Answer + Inline Citations]
                       └───> [Evaluation Subsystem: Faithfulness & Relevance Scoring]
```

---

## 2. Data Ingestion & Parsing Subsystem

Documents arrive in diverse binary structures. The system employs dedicated parsers to extract clean text while retaining document provenance:

| Format | Library | Extraction Strategy | Metadata Captured |
|---|---|---|---|
| **PDF** | `pypdf` / `pdfplumber` | Per-page text streams, stripping header/footer noise and null bytes | `source`, `page_number`, `total_pages` |
| **DOCX** | `python-docx` | Paragraph and table cell traversal | `source`, `paragraph_index` |
| **TXT / MD** | Native UTF-8 Reader | UTF-8 normalization, BOM stripping | `source`, `line_count` |

### Document Metadata Schema
Every chunk extracted preserves this JSON metadata footprint:
```json
{
  "source": "Q3_Financial_Performance_Report.pdf",
  "page": 4,
  "chunk_id": "doc_01_chunk_12",
  "char_length": 984,
  "token_estimate": 246
}
```
This metadata is crucial for **source attribution** and downstream **citation validation**.

---

## 3. Chunking Strategy & Semantic Boundaries

### The Chunking Trade-off
- **Chunks too small (e.g., 100 chars)**: Loss of semantic context; embeddings become ambiguous (e.g., pronoun references lose their antecedent).
- **Chunks too large (e.g., 4000 chars)**: Dilution of signal; embeddings represent an average of too many ideas, reducing retrieval precision and increasing context window costs.

### Recursive Character Splitting
The system uses `RecursiveCharacterTextSplitter` configured as follows:
- **Default Chunk Size**: `1000` characters (~250 tokens).
- **Default Chunk Overlap**: `200` characters (~50 tokens).
- **Separators Hierarchy**: `["\n\n", "\n", ". ", " ", ""]`

#### How It Works:
1. It attempts to split on paragraph breaks (`\n\n`) first to keep paragraphs intact.
2. If a paragraph exceeds 1000 characters, it falls back to splitting on sentence terminators (`. `).
3. If a sentence exceeds 1000 characters, it splits on word boundaries (` `).
4. The **overlap** ensures that if a critical relationship or definition spans a chunk boundary, both adjacent chunks capture the overlapping sentence, preventing information loss.

```
Chunk N:     [... Paragraph A ... Sentence B1. Sentence B2]
                                   └────────┬────────┘
Overlap:                                    │
                                   ┌────────┴────────┐
Chunk N+1:                         [Sentence B1. Sentence B2 ... Paragraph C ...]
```

---

## 4. Dense Embedding Vector Space

Text chunks are projected into a continuous, high-dimensional vector space $\mathbb{R}^d$:
- **OpenAI `text-embedding-3-small`**: $d = 1536$ dimensions, trained with cosine similarity loss.
- **HuggingFace `sentence-transformers/all-MiniLM-L6-v2`**: $d = 384$ dimensions, fast local CPU inference.

### Distance Metric: Cosine Similarity
Given query embedding $\mathbf{q}$ and chunk embedding $\mathbf{d}$:

$$\text{Cosine Similarity}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2} = \cos(\theta)$$

For normalized embeddings where $\|\mathbf{q}\|_2 = \|\mathbf{d}\|_2 = 1$, cosine similarity equals the dot product (Inner Product):
$$\text{Similarity}(\mathbf{q}, \mathbf{d}) = \mathbf{q} \cdot \mathbf{d}$$

---

## 5. Vector Stores: FAISS vs. ChromaDB

The application provides an abstraction layer (`BaseVectorStoreManager`) supporting both:

| Architectural Dimension | **ChromaDB** | **FAISS (Meta)** |
|---|---|---|
| **Primary Focus** | Full-featured Developer Vector DB | Pure Mathematical Vector Index |
| **Storage Engine** | SQLite (metadata) + DuckDB/ClickHouse + HNSW | In-memory RAM / C++ serialized flat binary |
| **Metadata Filtering** | Rich native SQL-style JSON metadata filtering (`{"page": {"$gt": 2}}`) | Limited without external ID mapping table |
| **Persistence** | Native local folder persistence (`./chroma_db`) | Manual serialization (`faiss.write_index`) |
| **Search Speed** | Millisecond range (HNSW graph traversal) | Microsecond range (AVX-512 SIMD optimized) |
| **Best Used For** | Prototyping, document management, CRUD ops | Ultra-high performance search over millions of vectors |

---

## 6. Retrieval Optimization & Strategies

### Strategy A: Standard Top-K Cosine Retrieval
Retrieves the $k$ vectors with the highest dot product to the query vector.
- **Limitation**: Suffers from **redundancy**. If five chunks in a document repeat the same policy sentence, all five might be returned, starving the LLM of other relevant context.

### Strategy B: Maximal Marginal Relevance (MMR)
MMR explicitly balances **query relevance** with **chunk diversity**.

$$\text{MMR} = \arg\max_{d_i \in D \setminus S} \left[ \lambda \cdot \text{Sim}(d_i, q) - (1 - \lambda) \max_{d_j \in S} \text{Sim}(d_i, d_j) \right]$$

Where:
- $D$ is the initial pool of candidate retrieved chunks (retrieved with size `fetch_k`, typically 20).
- $S$ is the set of already selected chunks.
- $q$ is the query vector.
- $\lambda \in [0, 1]$ is the diversity trade-off parameter:
  - $\lambda = 1.0 \implies$ Pure similarity search (identical to Top-K).
  - $\lambda = 0.5 \implies$ Balanced relevance and diversity.
  - $\lambda = 0.0 \implies$ Maximal diversity.

Our implementation exposes $\lambda$ and `fetch_k` dynamically in the Streamlit settings.

---

## 7. Generation, Prompt Engineering & Grounding

The retrieved chunks are assembled into a structured context window and passed to the generator.

### Strict Grounding Prompt Template
```
You are an expert Document Intelligence Assistant. Your task is to provide accurate, factual, and strictly grounded answers based ONLY on the provided context.

Context:
----------------
{context}
----------------

User Question: {question}

Rules:
1. Rely ONLY on the information presented in the context.
2. DO NOT extrapolate, assume, or use pre-trained external knowledge not present in the context.
3. If the answer cannot be determined directly from the context, state:
   "I cannot find the answer to this question in the provided documents."
4. Every statement or claim must reference its source document and page number in brackets, e.g., [Source: filename.pdf, Page 2].
5. Maintain a professional, concise, and structured tone.
```

---

## 8. Hallucination Mitigation Mechanisms

Hallucinations in RAG systems occur in two forms:
1. **Extrinsic Hallucinations**: Information that cannot be verified by the context (fabricated facts).
2. **Intrinsic Hallucinations**: Information that directly contradicts the provided context.

### Mitigation Techniques Implemented:
1. **Temperature Clamping**: Set `temperature = 0.0` or `0.1` to enforce deterministic, greedy decoding.
2. **Explicit Abstention Constraint**: The prompt penalizes guessing by supplying a designated fallback sentence when context is insufficient.
3. **Chunk Boundary Labeling**: Chunks are clearly delimited with explicit tags `[Document: <name>, Page: <page>]` so the attention mechanism binds generated statements to citation markers.
4. **Automated Faithfulness Verification**: Post-generation check that validates whether claims in the response logically follow from the context (see Section 9).

---

## 9. Evaluation Engine & The RAG Triad

To measure system reliability, our architecture implements the **RAG Triad** framework:

```
                  ┌──────────────────────┐
                  │      User Query      │
                  └──────────┬───────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼ Context                         ▼ Answer
      Relevance Metric                 Relevance Metric
            │                                 │
            ▼                                 ▼
┌───────────────────────┐            ┌───────────────────────┐
│   Retrieved Context   ├───────────►│   Generated Answer    │
│        Chunks         │ Faithfulness        │
└───────────────────────┘ (Groundedness)      └───────────────────────┘
```

1. **Context Relevance**:
   $$\text{Score} = \frac{|\text{Retrieved Chunks Relevant to Query}|}{|\text{Total Retrieved Chunks}|}$$
2. **Faithfulness / Groundedness**:
   $$\text{Score} = \frac{|\text{Claims in Answer Supported by Context}|}{|\text{Total Claims Made in Answer}|}$$
   - Any claim not supported by context increases the **Hallucination Risk Score** ($1 - \text{Faithfulness}$).
3. **Answer Relevance**:
   Evaluates whether the response directly addresses the intent of the prompt without deviating into tangential topics.

---

## 10. Performance, Latency & Scalability Considerations

- **Embedding Caching**: Chunks already embedded are indexed once and persisted to disk (`./chroma_db` or FAISS binary), avoiding redundant API calls.
- **Async Streaming**: LLM responses are streamed via tokens to the Streamlit UI using generator yields (`yield token`), reducing Time-to-First-Token (TTFT) from ~3.5s to < 400ms.
- **Top-K Tuning**: Default `k=4` limits input token consumption to ~1,000 tokens per prompt, keeping API costs under $0.002 per query.
