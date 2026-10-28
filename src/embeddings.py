"""
Embedding Factory & Provider Subsystem.
Supports OpenAI, Google Gemini, HuggingFace Sentence-Transformers,
and a deterministic Offline Demo Embedding engine for local zero-dependency testing.
"""

import math
import hashlib
from typing import List, Optional

try:
    from langchain_core.embeddings import Embeddings
except ImportError:
    class Embeddings:
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            raise NotImplementedError
        def embed_query(self, text: str) -> List[float]:
            raise NotImplementedError


class OfflineDemoEmbeddings(Embeddings):
    """
    Deterministic TF-IDF & Hashing Embedding Engine.
    Generates normalized 384-dimensional dense vectors using character and word n-gram hashing.
    Enables fully functional vector similarity search, FAISS, and ChromaDB pipelines
    even without third-party API keys or internet access.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _hash_token(self, token: str) -> int:
        h = hashlib.sha256(token.encode("utf-8")).hexdigest()
        return int(h, 16) % self.dimension

    def _embed_single(self, text: str) -> List[float]:
        vec = [0.0] * self.dimension
        if not text:
            return vec

        # Tokenize by words and character 3-grams
        words = text.lower().split()
        for w in words:
            idx = self._hash_token(w)
            vec[idx] += 1.0

        # Sub-word ngrams for semantic sub-string matching
        clean = "".join(filter(str.isalnum, text.lower()))
        for i in range(len(clean) - 2):
            gram = clean[i:i + 3]
            idx = self._hash_token(gram)
            vec[idx] += 0.5

        # L2 Normalization so dot-product equals cosine similarity
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-9:
            vec = [x / norm for x in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_single(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_single(text)


def get_embedding_provider(
    provider: str = "openai",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> Embeddings:
    """
    Factory function returning the appropriate Embeddings instance.
    """
    provider = (provider or "openai").lower()

    if provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
            model = model_name or "text-embedding-3-small"
            return OpenAIEmbeddings(
                model=model,
                openai_api_key=api_key or None
            )
        except Exception as e:
            print(f"[Warning] Failed to initialize OpenAI Embeddings ({e}). Falling back to OfflineDemoEmbeddings.")
            return OfflineDemoEmbeddings()

    elif provider in ["google", "gemini"]:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            model = model_name or "models/embedding-001"
            return GoogleGenerativeAIEmbeddings(
                model=model,
                google_api_key=api_key or None
            )
        except Exception as e:
            print(f"[Warning] Failed to initialize Google Embeddings ({e}). Falling back to OfflineDemoEmbeddings.")
            return OfflineDemoEmbeddings()

    elif provider in ["huggingface", "local"]:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            model = model_name or "sentence-transformers/all-MiniLM-L6-v2"
            return HuggingFaceEmbeddings(model_name=model)
        except Exception as e:
            print(f"[Warning] Failed to initialize HuggingFace Embeddings ({e}). Falling back to OfflineDemoEmbeddings.")
            return OfflineDemoEmbeddings()

    else:
        return OfflineDemoEmbeddings()
