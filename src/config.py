"""
Configuration Module for AI Document Intelligence Assistant.
Handles environment variables, default paths, model identifiers, and RAG pipeline parameters.
"""

import os
from pathlib import Path

# Load .env if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index"
SAMPLE_DOCS_DIR = BASE_DIR / "sample_documents"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Model Defaults
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
DEFAULT_OPENAI_EMBEDDING = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
DEFAULT_GOOGLE_MODEL = os.getenv("GOOGLE_MODEL_NAME", "gemini-1.5-flash")

# Default RAG Hyperparameters
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", 1000))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", 200))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 4))
DEFAULT_RETRIEVAL_TYPE = os.getenv("DEFAULT_RETRIEVAL_TYPE", "similarity")  # 'similarity' or 'mmr'
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", 0.0))
DEFAULT_VECTOR_STORE = os.getenv("VECTOR_STORE_TYPE", "chroma").lower()  # 'chroma' or 'faiss'

# Supported Document Extensions
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md"]
