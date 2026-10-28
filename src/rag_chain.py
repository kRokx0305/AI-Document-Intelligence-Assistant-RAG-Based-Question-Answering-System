"""
RAG Orchestration & Generation Chain Subsystem.
Constructs strictly grounded prompts, handles LLM streaming,
formats source attribution citations, and records latency metrics.
"""

import time
from typing import List, Dict, Any, Generator, Tuple, Optional

try:
    from langchain_core.documents import Document
except ImportError:
    from src.document_loader import Document

from src.vectorstore import VectorStoreManager


STRICT_SYSTEM_PROMPT = """You are an expert AI Document Intelligence Assistant. Your role is to provide accurate, concise, and strictly factual answers based ONLY on the provided context.

Context:
----------------
{context}
----------------

User Question: {question}

Strict Instructions:
1. Ground your answer EXCLUSIVELY in the provided context above.
2. DO NOT speculate, assume, or utilize pre-trained knowledge not present in the context.
3. If the answer cannot be found in the context, respond EXACTLY with:
   "I cannot find the answer to this question in the provided documents."
4. Attribute factual statements to their source documents and page numbers using inline citation brackets, e.g.: [Doc: {sample_doc}, Page: {sample_page}].
5. Keep your tone objective, professional, and clear.
"""


def format_context(documents: List[Document]) -> str:
    """Format retrieved document chunks with clear citation metadata tags."""
    context_parts = []
    for i, doc in enumerate(documents):
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", 1)
        chunk_id = doc.metadata.get("chunk_index", i)
        header = f"--- [Doc: {source} | Page: {page} | Chunk: {chunk_id}] ---"
        context_parts.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n".join(context_parts)


class RAGPipeline:
    """
    End-to-End RAG Pipeline managing retrieval, prompt injection, and LLM inference.
    """

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.0
    ):
        self.vsm = vector_store_manager
        self.provider = (provider or "openai").lower()
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._init_llm()

    def _init_llm(self) -> Any:
        """Initialize the requested LLM client."""
        if self.provider == "openai" and self.api_key:
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=self.model_name or "gpt-4o-mini",
                    temperature=self.temperature,
                    openai_api_key=self.api_key,
                    streaming=True
                )
            except Exception as e:
                print(f"[Warning] Failed to initialize ChatOpenAI ({e}).")

        elif self.provider in ["google", "gemini"] and self.api_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=self.model_name or "gemini-1.5-flash",
                    temperature=self.temperature,
                    google_api_key=self.api_key
                )
            except Exception as e:
                print(f"[Warning] Failed to initialize Google Gemini ({e}).")

        return None

    def retrieve(
        self,
        query: str,
        k: int = 4,
        retrieval_type: str = "similarity",
        min_score_threshold: float = 0.0
    ) -> Tuple[List[Document], List[Dict[str, Any]], float]:
        """
        Retrieve relevant context chunks with score and metadata extraction.
        """
        start_time = time.perf_counter()

        if retrieval_type == "mmr":
            raw_docs = self.vsm.max_marginal_relevance_search(query, k=k)
            scored_results = self.vsm.similarity_search_with_score(query, k=k)
            score_map = {doc.page_content: score for doc, score in scored_results}
            docs_with_scores = [(doc, score_map.get(doc.page_content, 0.85)) for doc in raw_docs]
        else:
            docs_with_scores = self.vsm.similarity_search_with_score(query, k=k)

        filtered = [
            (doc, score) for doc, score in docs_with_scores if score >= min_score_threshold
        ]
        retrieval_latency = round(time.perf_counter() - start_time, 4)

        retrieved_docs = [doc for doc, _ in filtered]
        sources = [
            {
                "source": doc.metadata.get("source", "Document"),
                "page": doc.metadata.get("page", 1),
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "similarity_score": float(round(float(score), 3)),
                "char_length": len(doc.page_content),
                "snippet": doc.page_content[:250] + ("..." if len(doc.page_content) > 250 else "")
            }
            for doc, score in filtered
        ]

        return retrieved_docs, sources, retrieval_latency

    def _offline_answer_generator(self, query: str, context_docs: List[Document]) -> str:
        """
        High-fidelity heuristic generator for offline / demo mode.
        Extracts the most relevant sentences and grounds them with citation tags.
        """
        if not context_docs:
            return "I cannot find the answer to this question in the provided documents."

        query_terms = set(query.lower().replace("?", "").split())
        matched_sentences = []

        for doc in context_docs:
            source = doc.metadata.get("source", "Document")
            page = doc.metadata.get("page", 1)
            sentences = doc.page_content.replace("\n", " ").split(". ")

            for s in sentences:
                s_clean = s.strip()
                if not s_clean:
                    continue
                s_lower = s_clean.lower()
                overlap = sum(1 for term in query_terms if term in s_lower and len(term) > 3)
                if overlap > 0:
                    matched_sentences.append((overlap, s_clean, source, page))

        if not matched_sentences:
            first_doc = context_docs[0]
            first_page = first_doc.metadata.get("page", 1)
            first_source = first_doc.metadata.get("source", "Document")
            return (
                f"Based on the indexed document, here is the most relevant section found:\n\n"
                f"> \"{first_doc.page_content[:350]}...\"\n\n"
                f"— [Doc: {first_source}, Page: {first_page}]"
            )

        matched_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = matched_sentences[:4]

        answer_parts = ["Based on the provided documents:"]
        seen_texts = set()
        for _, s_text, source, page in top_sentences:
            if s_text not in seen_texts:
                answer_parts.append(f"- {s_text}. [Doc: {source}, Page: {page}]")
                seen_texts.add(s_text)

        return "\n\n".join(answer_parts)

    def answer_query(
        self,
        query: str,
        k: int = 4,
        retrieval_type: str = "similarity"
    ) -> Dict[str, Any]:
        """
        Execute full RAG generation pipeline and return answer, sources, and latency stats.
        """
        retrieved_docs, sources, retrieval_latency = self.retrieve(
            query, k=k, retrieval_type=retrieval_type
        )

        if not retrieved_docs:
            return {
                "answer": "I cannot find the answer to this question in the provided documents.",
                "sources": [],
                "retrieved_docs": [],
                "retrieval_latency": retrieval_latency,
                "generation_latency": 0.0,
                "context_text": ""
            }

        context_str = format_context(retrieved_docs)
        sample_doc = sources[0]["source"] if sources else "sample.pdf"
        sample_page = sources[0]["page"] if sources else 1

        prompt = STRICT_SYSTEM_PROMPT.format(
            context=context_str,
            question=query,
            sample_doc=sample_doc,
            sample_page=sample_page
        )

        gen_start = time.perf_counter()

        if self.llm is not None:
            try:
                response = self.llm.invoke(prompt)
                answer_text = response.content if hasattr(response, "content") else str(response)
            except Exception as e:
                print(f"[Warning] LLM invocation failed ({e}). Falling back to heuristic generator.")
                answer_text = self._offline_answer_generator(query, retrieved_docs)
        else:
            answer_text = self._offline_answer_generator(query, retrieved_docs)

        generation_latency = round(time.perf_counter() - gen_start, 4)

        return {
            "answer": answer_text,
            "sources": sources,
            "retrieved_docs": retrieved_docs,
            "retrieval_latency": retrieval_latency,
            "generation_latency": generation_latency,
            "context_text": context_str
        }

    def stream_answer_query(
        self,
        query: str,
        k: int = 4,
        retrieval_type: str = "similarity"
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream the answer tokens while yielding metadata upon completion.
        """
        retrieved_docs, sources, retrieval_latency = self.retrieve(
            query, k=k, retrieval_type=retrieval_type
        )

        if not retrieved_docs:
            yield {
                "type": "token",
                "content": "I cannot find the answer to this question in the provided documents."
            }
            yield {
                "type": "done",
                "sources": [],
                "retrieved_docs": [],
                "retrieval_latency": retrieval_latency,
                "generation_latency": 0.0,
                "context_text": ""
            }
            return

        context_str = format_context(retrieved_docs)
        sample_doc = sources[0]["source"] if sources else "sample.pdf"
        sample_page = sources[0]["page"] if sources else 1

        prompt = STRICT_SYSTEM_PROMPT.format(
            context=context_str,
            question=query,
            sample_doc=sample_doc,
            sample_page=sample_page
        )

        gen_start = time.perf_counter()

        if self.llm is not None and hasattr(self.llm, "stream"):
            try:
                for chunk in self.llm.stream(prompt):
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    yield {"type": "token", "content": token}
            except Exception as e:
                print(f"[Warning] LLM stream error ({e}), using offline generator.")
                offline_ans = self._offline_answer_generator(query, retrieved_docs)
                yield {"type": "token", "content": offline_ans}
        else:
            offline_ans = self._offline_answer_generator(query, retrieved_docs)
            for word in offline_ans.split(" "):
                yield {"type": "token", "content": word + " "}
                time.sleep(0.015)

        generation_latency = round(time.perf_counter() - gen_start, 4)

        yield {
            "type": "done",
            "sources": sources,
            "retrieved_docs": retrieved_docs,
            "retrieval_latency": retrieval_latency,
            "generation_latency": generation_latency,
            "context_text": context_str
        }
