# RAG Evaluation Guide: Response Relevance, Retrieval Quality & Hallucination Risks

This guide provides a comprehensive manual for evaluating the **AI Document Intelligence Assistant**. It details the evaluation methodology, mathematical definitions, LLM-as-a-judge prompts, and instructions for using the system's live evaluation dashboard.

---

## 📊 1. Why Evaluate RAG?

Standard LLM evaluation metrics like BLEU or ROUGE fall short for Document Question Answering:
- They require ground-truth reference texts, which rarely exist for dynamic ad-hoc user queries.
- They only evaluate surface-level n-gram overlap, failing to detect factual contradictions or subtle hallucinations.

To systematically validate RAG systems, we implement **Reference-Free Evaluation** based on the **RAG Triad**.

---

## 🎯 2. The RAG Triad Architecture

The RAG Triad decomposes the RAG pipeline into three distinct relationships:

```
                  ┌──────────────────────┐
                  │      User Query      │
                  └──────────┬───────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   [Context Relevance]               [Answer Relevance]
   Is the retrieved context          Does the answer directly
   relevant to the question?         address the question?
            │                                 │
            ▼                                 ▼
┌───────────────────────┐            ┌───────────────────────┐
│   Retrieved Context   ├───────────►│   Generated Answer    │
│        Chunks         │            │                       │
└───────────────────────┘            └───────────────────────┘
                     [Faithfulness]
               Is the answer grounded
              strictly in the context?
```

### 1. Context Relevance (Retrieval Quality)
- **Goal**: Measures how much of the retrieved context is genuinely useful for answering the query vs. irrelevant clutter.
- **Formula**:
  $$\text{Context Relevance} = \frac{\sum_{i=1}^{k} \mathbb{I}(\text{chunk}_i \text{ is relevant to query})}{k}$$
- **Low Context Relevance Causes**:
  - Embedding model mismatch (e.g., keyword search needed instead of dense semantic search).
  - Chunk size too large (embedding dilutes specific facts).
  - Sub-optimal vector distance threshold.

### 2. Faithfulness / Groundedness (Hallucination Risk)
- **Goal**: Measures whether every claim in the generated answer can be mathematically or logically derived from the retrieved context.
- **Formula**:
  $$\text{Faithfulness} = \frac{|\text{Verified Claims in Answer}|}{|\text{Total Claims in Answer}|}$$
- **Hallucination Risk Score**:
  $$\text{Hallucination Risk} = 1.0 - \text{Faithfulness}$$
- **Low Faithfulness Causes**:
  - LLM temperature too high ($> 0.3$).
  - Prompt instructions not strict enough.
  - Incomplete retrieved context prompting the LLM to extrapolate from its pre-trained weights.

### 3. Answer Relevance (Response Quality)
- **Goal**: Measures how well the final response addresses the user's intent. An answer can be completely faithful to context but fail to answer the user's actual question.
- **Calculation**:
  - Calculated using semantic cosine similarity between the original user query and a synthesized question generated from the answer:
    $$\text{Answer Relevance} = \cos\left(E(\text{Original Query}), E(\text{Reverse Generated Query})\right)$$
  - Or via LLM-as-a-judge score on a normalized scale $[0.0, 1.0]$.

---

## 🔬 3. Evaluation Algorithms & Prompts

Our `src/evaluator.py` module evaluates these dimensions using structured LLM evaluations with JSON outputs:

### A. Faithfulness / Hallucination Judge Prompt
```json
{
  "role": "system",
  "content": "You are an automated evaluation judge for a RAG system. Your job is to check for hallucinations by verifying whether claims in an answer are grounded in the provided context."
}
```
**User prompt structure**:
```
Context:
----------------
{context}
----------------

Generated Answer:
----------------
{answer}
----------------

Task:
1. Break down the Generated Answer into atomic factual statements.
2. For each statement, determine if it is directly supported by the Context (TRUE) or unsupported/hallucinated (FALSE).
3. Compute the Faithfulness score: (Supported Statements / Total Statements).

Return JSON only in this schema:
{
  "total_claims": <int>,
  "supported_claims": <int>,
  "faithfulness_score": <float between 0.0 and 1.0>,
  "hallucination_detected": <true or false>,
  "unsupported_statements": [<list of strings>],
  "reasoning": "<concise justification>"
}
```

### B. Context Relevance Judge Prompt
```
User Query: {question}

Retrieved Context Chunks:
{numbered_chunks}

Task:
Assess each retrieved chunk. Is it relevant to answering the query?
Return JSON:
{
  "relevant_chunk_ids": [<list of ints>],
  "context_relevance_score": <float between 0.0 and 1.0>,
  "critique": "<brief notes on noise or missing context>"
}
```

---

## 💻 4. Using the Streamlit Evaluation Dashboard

The assistant includes an interactive **Evaluation View** inside the Streamlit application:

1. **Ask a Query**: Submit any document question in the Chat tab.
2. **Switch to 'RAG Evaluation' Tab**:
   - View the calculated **RAG Triad Radar / Metric Cards**:
     - 🟢 **Context Relevance**: e.g., `0.85`
     - 🟢 **Faithfulness**: e.g., `1.00` (Zero Hallucination)
     - 🟢 **Answer Relevance**: e.g., `0.92`
     - ⏱ **Retrieval Latency**: e.g., `210 ms`
     - ⏱ **Generation Latency**: e.g., `1.15 s`
3. **Inspect Retrieved Chunks & Scores**:
   - See the exact distance/similarity score for each retrieved chunk.
   - Inspect whether MMR or Similarity search yielded superior diversity.
4. **Inspect Unsupported Claims**:
   - If an answer introduced outside knowledge, the evaluation tab flags the exact unsupported sentence in red with a Hallucination warning.

---

## 📈 5. Production Benchmark Metrics

When evaluating retrieval quality across a benchmark test set:

| Metric | Definition | Good Production Benchmark |
|---|---|---|
| **Hit Rate @ K** | Proportion of queries where at least one correct chunk appears in Top-K | $\ge 90\%$ at $k=4$ |
| **Mean Reciprocal Rank (MRR)** | $\frac{1}{|Q|}\sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$ of the first relevant chunk | $\ge 0.75$ |
| **Average Faithfulness** | Groundedness of generated responses across all queries | $\ge 0.95$ ($< 5\%$ hallucination rate) |
| **P95 Retrieval Latency** | Time taken by FAISS/ChromaDB to return Top-K candidates | $< 150 \text{ ms}$ |
