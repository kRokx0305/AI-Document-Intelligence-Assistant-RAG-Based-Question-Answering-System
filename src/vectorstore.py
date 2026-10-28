"""
Vector Store Management Subsystem.
Integrates ChromaDB and FAISS with persistence, metadata filtering,
cosine similarity scoring, and Maximal Marginal Relevance (MMR) search.
Includes pure-Python mathematical fallbacks for zero-dependency portability.
"""

import os
import math
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Any

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from langchain_core.documents import Document
except ImportError:
    from src.document_loader import Document

from src.embeddings import Embeddings, OfflineDemoEmbeddings
from src.config import CHROMA_PERSIST_DIR, FAISS_INDEX_PATH


def _dot_product(v1: List[float], v2: List[float]) -> float:
    """Compute dot product between two float vectors."""
    return sum(a * b for a, b in zip(v1, v2))


def _l2_normalize(v: List[float]) -> List[float]:
    """Normalize vector to unit length."""
    norm = math.sqrt(sum(x * x for x in v))
    if norm > 1e-9:
        return [x / norm for x in v]
    return v


class FallbackInMemoryVectorStore:
    """
    Dependency-free in-memory vector store with Cosine Similarity and MMR.
    Guarantees that the assistant functions smoothly even before FAISS/Chroma binaries are compiled.
    """

    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self.docs: List[Document] = []
        self.vectors: List[List[float]] = []

    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return
        texts = [d.page_content for d in documents]
        raw_vecs = self.embeddings.embed_documents(texts)
        for doc, vec in zip(documents, raw_vecs):
            normed = _l2_normalize(vec)
            self.docs.append(doc)
            self.vectors.append(normed)

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        if not self.docs:
            return []
        q_vec = _l2_normalize(self.embeddings.embed_query(query))
        scores = [_dot_product(q_vec, v) for v in self.vectors]

        # Rank indices by descending score
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        top_k_indices = indexed_scores[:k]

        return [(self.docs[i], round(score, 4)) for i, score in top_k_indices]

    def max_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5
    ) -> List[Document]:
        if not self.docs:
            return []
        fetch_k = min(fetch_k, len(self.docs))
        k = min(k, fetch_k)

        q_vec = _l2_normalize(self.embeddings.embed_query(query))
        sims_to_query = [_dot_product(q_vec, v) for v in self.vectors]

        # Sort candidate indices by similarity to query
        candidate_indices = sorted(range(len(sims_to_query)), key=lambda i: sims_to_query[i], reverse=True)[:fetch_k]

        selected_indices: List[int] = []

        while len(selected_indices) < k and candidate_indices:
            if not selected_indices:
                best = candidate_indices.pop(0)
                selected_indices.append(best)
                continue

            best_mmr_score = -float("inf")
            best_idx = None
            best_pos = None

            for pos, c_idx in enumerate(candidate_indices):
                sim_q = sims_to_query[c_idx]
                sim_to_selected = max([_dot_product(self.vectors[c_idx], self.vectors[s_idx]) for s_idx in selected_indices])
                mmr_score = lambda_mult * sim_q - (1 - lambda_mult) * sim_to_selected
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = c_idx
                    best_pos = pos

            if best_idx is not None:
                selected_indices.append(best_idx)
                candidate_indices.pop(best_pos)
            else:
                break

        return [self.docs[i] for i in selected_indices]


class VectorStoreManager:
    """
    Unified manager for ChromaDB and FAISS vector stores with seamless fallbacks.
    """

    def __init__(
        self,
        store_type: str = "chroma",
        persist_dir: Optional[Path] = None,
        embeddings: Optional[Embeddings] = None
    ):
        self.store_type = store_type.lower()
        self.persist_dir = persist_dir or (CHROMA_PERSIST_DIR if self.store_type == "chroma" else FAISS_INDEX_PATH)
        self.embeddings = embeddings or OfflineDemoEmbeddings()
        self.vector_store: Any = None
        self.fallback_store: Optional[FallbackInMemoryVectorStore] = None

    def initialize_with_documents(self, documents: List[Document]) -> Any:
        """Build and populate the chosen vector store from documents."""
        if not documents:
            return None

        if self.store_type == "chroma":
            try:
                from langchain_community.vectorstores import Chroma
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=str(self.persist_dir)
                )
                return self.vector_store
            except Exception as e:
                print(f"[Notice] ChromaDB initialization ({e}). Using native high-speed vector engine.")

        elif self.store_type == "faiss":
            try:
                from langchain_community.vectorstores import FAISS
                self.vector_store = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
                self.persist_dir.mkdir(parents=True, exist_ok=True)
                self.vector_store.save_local(str(self.persist_dir))
                return self.vector_store
            except Exception as e:
                print(f"[Notice] FAISS initialization ({e}). Using native high-speed vector engine.")

        # Self-contained in-memory fallback
        self.fallback_store = FallbackInMemoryVectorStore(self.embeddings)
        self.fallback_store.add_documents(documents)
        return self.fallback_store

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        """Perform cosine similarity search and return documents alongside similarity scores."""
        if self.fallback_store is not None:
            return self.fallback_store.similarity_search_with_score(query, k=k)

        if self.vector_store is not None:
            try:
                results = self.vector_store.similarity_search_with_score(query, k=k)
                normalized_results = []
                for doc, score in results:
                    score_val = float(score)
                    if self.store_type == "chroma":
                        sim = max(0.0, min(1.0, 1.0 - (score_val / 2.0)))
                    elif self.store_type == "faiss":
                        sim = score_val if score_val <= 1.0 else max(0.0, 1.0 / (1.0 + score_val))
                    else:
                        sim = score_val
                    normalized_results.append((doc, float(round(float(sim), 4))))
                return normalized_results
            except Exception as e:
                print(f"[Error] Vector search failed ({e})")
                return []

        return []

    def max_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5
    ) -> List[Document]:
        """Perform MMR search to ensure diverse non-redundant context chunks."""
        if self.fallback_store is not None:
            return self.fallback_store.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
            )

        if self.vector_store is not None:
            try:
                return self.vector_store.max_marginal_relevance_search(
                    query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
                )
            except Exception as e:
                print(f"[Warning] MMR failed ({e}), falling back to standard similarity search.")
                results_with_score = self.similarity_search_with_score(query, k=k)
                return [doc for doc, _ in results_with_score]

        return []

    def clear(self) -> None:
        """Clear the vector store index and remove persisted files."""
        self.vector_store = None
        self.fallback_store = None
        if self.persist_dir and os.path.exists(self.persist_dir):
            try:
                shutil.rmtree(self.persist_dir)
                os.makedirs(self.persist_dir, exist_ok=True)
            except Exception as e:
                print(f"[Error] Failed to clear vector store directory: {e}")
