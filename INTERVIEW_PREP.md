# RAG System Interview Preparation & Technical Breakdown

This document directly maps each bullet point on your resume to technical talking points, design justifications, and over 20 commonly asked engineering interview questions with model answers.

---

## 📌 Resume Bullet Breakdown & Key Talking Points

### Bullet 1:
> *"Developed a document question-answering application using Python and an LLM API to generate context-aware responses from uploaded documents."*
- **What to emphasize**:
  - Modular Python architecture with strict separation of concerns (document parsing, chunking, embedding, retrieval, LLM orchestration, evaluation).
  - Multi-provider support (OpenAI `gpt-4o-mini`, Google Gemini `gemini-1.5-flash`, HuggingFace).
  - Context injection pattern using structured Markdown prompts with strict boundary markers.

### Bullet 2:
> *"Implemented a Retrieval-Augmented Generation (RAG) pipeline using document chunking, embeddings, and vector similarity search for knowledge retrieval."*
- **What to emphasize**:
  - Chunking strategy: Recursive character splitting with sliding window overlap (1000 chunk size, 200 overlap) to preserve multi-sentence semantic relationships without context dilution.
  - Embedding geometry: Dense vector representation in 1536-dimensional or 384-dimensional space using cosine similarity for directional alignment.

### Bullet 3:
> *"Integrated LangChain with FAISS or ChromaDB to retrieve relevant document context and generate grounded responses."*
- **What to emphasize**:
  - Architectural comparison between **ChromaDB** (persistent, metadata filtering, SQLite/DuckDB backing) and **FAISS** (ultra-fast C++ in-memory similarity search).
  - Implementation of **Maximal Marginal Relevance (MMR)** to balance retrieval relevance with chunk diversity, avoiding duplicate context.

### Bullet 4:
> *"Built a Streamlit interface with source references and evaluated response relevance, retrieval quality, and hallucination risks."*
- **What to emphasize**:
  - Interactive UX with streaming responses and expandable source cards showing file name, page number, chunk index, and similarity scores.
  - Implementation of the **RAG Triad** framework: Context Relevance (retrieval precision), Faithfulness/Groundedness (hallucination detection), and Answer Relevance (query satisfaction).

---

## 🧠 Top 20+ Technical Interview Questions & Answers

### 1. Ingestion & Chunking

#### Q1: Why did you choose Recursive Character Splitting instead of fixed-length token chunking?
**Answer**:
> *"Fixed-length chunking cuts text blindly at exact character or token counts, often cutting mid-sentence, splitting numbers from their units, or severing words. Recursive character splitting works with a hierarchy of natural separators: double newlines (`\n\n` for paragraphs), single newlines (`\n` for lines), periods with spaces (`. ` for sentences), and spaces for words. This guarantees that semantic units like paragraphs or complete sentences remain contiguous, giving the embedding model coherent semantic concepts to encode."*

#### Q2: How did you select chunk size (1000) and overlap (200)?
**Answer**:
> *"Chunk size represents a trade-off between semantic specificity and surrounding context. A small chunk size (e.g. 200 characters) captures precise facts but loses relational context (like pronouns or section headings). A large chunk size (e.g. 4000 characters) dilutes the embedding vector across multiple distinct ideas, lowering retrieval precision. 1000 characters (~250 tokens) is optimal for technical and policy documents because standard paragraphs are 150–300 tokens. The 200-character overlap (~50 tokens) prevents edge-boundary clipping, ensuring that sentences straddling a chunk boundary are present in both adjacent chunks."*

---

### 2. Embeddings & Vector Stores

#### Q3: How do you choose between FAISS and ChromaDB in production?
**Answer**:
> *"We integrated both to support different deployment needs:
> - **ChromaDB** is ideal for stateful enterprise document management. It provides out-of-the-box local or client-server persistence, embedded SQLite metadata tracking, and powerful metadata filtering (e.g., querying only pages belonging to a specific department or date).
> - **FAISS** (by Meta) is a low-level, hyper-optimized C++ library for vector indexing. It is preferred when operating at scale with millions of vectors requiring microsecond retrieval latencies, GPU acceleration, or custom quantization techniques like Product Quantization (IVFPQ)."*

#### Q4: Why use Cosine Similarity instead of Euclidean (L2) distance?
**Answer**:
> *"Euclidean distance measures the straight-line distance between two vector endpoints, which is sensitive to vector magnitude (e.g., text length or token count differences). Cosine similarity measures the cosine of the angle between two vectors, focusing strictly on direction/orientation in semantic space regardless of vector length. Furthermore, when embeddings are L2-normalized to unit length ($\|v\|_2 = 1$), cosine similarity is mathematically equivalent to the inner product ($u \cdot v$), which can be computed with extreme computational efficiency via matrix multiplication."*

---

### 3. Retrieval & Ranking

#### Q5: What is Maximal Marginal Relevance (MMR) and why did you implement it?
**Answer**:
> *"Standard Top-K cosine retrieval often returns near-identical chunks if multiple paragraphs in a document repeat similar phrasing or keywords. This wastes precious context window space. MMR solves this by selecting chunks iteratively using a parameterized trade-off:
> $$\text{MMR} = \arg\max_{d_i} \left[ \lambda \cdot \text{Sim}(d_i, q) - (1 - \lambda) \max_{d_j \in S} \text{Sim}(d_i, d_j) \right]$$
> Here, a candidate chunk is penalized if it is too similar to chunks already selected into set $S$. With $\lambda = 0.6$, the system retrieves chunks that are both relevant to the query and diverse from one another, giving the LLM a broader perspective across the document."*

#### Q6: How do you handle low-confidence or out-of-domain queries where no relevant chunks exist?
**Answer**:
> *"We implemented two guardrails:
> 1. **Similarity Score Thresholding**: If the top retrieved chunks have a similarity score below a threshold (e.g. 0.45), the retriever flags the context as low confidence.
> 2. **Explicit Abstention in System Prompt**: The prompt instructs the model: 'If the provided context does not contain sufficient facts to answer the question, state that the information is unavailable.' This eliminates guesswork."*

---

### 4. Grounding & Hallucination Mitigation

#### Q7: What are the two types of hallucinations in RAG and how does your system combat them?
**Answer**:
> *"1. **Extrinsic Hallucinations**: The model introduces facts from its training data that are not mentioned in the context. We combat this using strict system prompts forbidding outside knowledge, zero temperature ($T=0$), and dynamic citation tagging where each fact must map to a `[Source: Doc, Page]` citation.
> 2. **Intrinsic Hallucinations**: The model misinterprets or contradicts facts actually present in the context. We address this using the automated Faithfulness evaluator, which breaks the answer into atomic statements and verifies each against the source text."*

#### Q8: How does inline citation tagging work?
**Answer**:
> *"During chunking, every chunk is assigned a unique metadata header containing the document name and page number. When assembled into the context prompt, each chunk is prefixed with `[Doc: filename, Page: X, Chunk: Y]`. The LLM's instruction set forces it to attribute any statement to the corresponding chunk tag. In the Streamlit UI, these citations are parsed and linked to interactive source cards displaying the exact excerpt."*

---

### 5. Evaluation & The RAG Triad

#### Q9: What is the RAG Triad and why is it superior to traditional NLP metrics?
**Answer**:
> *"Traditional metrics like BLEU or ROUGE measure superficial n-gram overlap against a reference answer, which is brittle and unavailable for real-time user queries. The RAG Triad assesses three orthogonal components:
> 1. **Context Relevance**: Measures retrieval precision (is the retrieved text free of noise?).
> 2. **Faithfulness / Groundedness**: Measures whether claims in the answer are 100% grounded in context (detecting hallucinations).
> 3. **Answer Relevance**: Measures whether the answer satisfies the user's intent without drifting."*

#### Q10: How do you implement Faithfulness evaluation in code?
**Answer**:
> *"We use an LLM-as-a-Judge pattern with a structured JSON output schema. The evaluator prompt takes the retrieved context and generated answer, decomposes the answer into atomic factual propositions, and evaluates each proposition as a binary classification (Supported vs. Unsupported). The Faithfulness score is the ratio of supported claims over total claims. If unsupported claims exist, they are flagged alongside a Hallucination Risk warning."*

---

### 6. System Architecture & Scalability

#### Q11: How would you scale this RAG system to 1,000,000+ documents?
**Answer**:
> *"To scale to million-document scale:
> 1. **Vector Index Quantization**: Switch FAISS or ChromaDB to an Inverted File Index with Product Quantization (IVF-PQ) or HNSW (Hierarchical Navigable Small World) to compress vectors and enable sub-linear graph-based search.
> 2. **Hybrid Search**: Combine BM25 sparse keyword search with dense vector search using Reciprocal Rank Fusion (RRF) to capture exact part numbers/acronyms as well as semantic meaning.
> 3. **Two-Stage Retrieval (Re-ranking)**: Retrieve the top 50 candidates using fast vector search, then use a Cross-Encoder (e.g., `bge-reranker-large` or Cohere Rerank) to re-rank the top 5 chunks before passing them to the LLM.
> 4. **Caching**: Implement semantic caching (e.g., Redis with vector search) to return cached responses for semantically identical questions."*
