"""
Document Chunking Subsystem.
Implements Recursive Character Chunking with sliding-window overlap and metadata preservation.
"""

from typing import List, Optional
try:
    from langchain_core.documents import Document
except ImportError:
    from src.document_loader import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_SPLITTER_AVAILABLE = True
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        LANGCHAIN_SPLITTER_AVAILABLE = True
    except ImportError:
        LANGCHAIN_SPLITTER_AVAILABLE = False


def estimate_tokens(text: str) -> int:
    """Fast approximation of token count (~4 chars per token)."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


class CustomRecursiveTextSplitter:
    """Pure-Python fallback recursive text splitter when LangChain is unavailable."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: Optional[List[str]] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)
        good_splits = []

        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_chunks = self._split(s, new_separators)
                    final_chunks.extend(other_chunks)

        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        docs = []
        current_doc = []
        total = 0

        for s in splits:
            _len = len(s) + (len(separator) if current_doc else 0)
            if total + _len > self.chunk_size and current_doc:
                doc = separator.join(current_doc)
                if doc:
                    docs.append(doc)
                # Keep sliding window overlap
                while total > self.chunk_overlap and current_doc:
                    popped = current_doc.pop(0)
                    total -= len(popped) + len(separator)
            current_doc.append(s)
            total += _len

        if current_doc:
            doc = separator.join(current_doc)
            if doc:
                docs.append(doc)

        return docs


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Chunks a list of documents while retaining document metadata and assigning chunk indices.
    """
    if not documents:
        return []

    separators = ["\n\n", "\n", ". ", " ", ""]

    if LANGCHAIN_SPLITTER_AVAILABLE:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len
        )
    else:
        splitter = CustomRecursiveTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )

    chunked_docs: List[Document] = []
    chunk_counter = 0

    for doc in documents:
        raw_text = doc.page_content.strip()
        if not raw_text:
            continue

        if LANGCHAIN_SPLITTER_AVAILABLE:
            raw_chunks = splitter.split_text(raw_text)
        else:
            raw_chunks = splitter.split_text(raw_text)

        for i, chunk_str in enumerate(raw_chunks):
            chunk_str = chunk_str.strip()
            if not chunk_str:
                continue

            metadata = dict(doc.metadata)
            metadata.update({
                "chunk_index": i,
                "global_chunk_id": chunk_counter,
                "char_length": len(chunk_str),
                "token_estimate": estimate_tokens(chunk_str)
            })

            chunked_docs.append(Document(page_content=chunk_str, metadata=metadata))
            chunk_counter += 1

    return chunked_docs
