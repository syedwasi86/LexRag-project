"""
config.py — Central configuration for the Legal Contract RAG Pipeline.
All paths, model names, and tuning parameters live here.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent


class Config:
    # ── Data Paths ──────────────────────────────────────────────────────────
    PDF_DIR    = str(BASE_DIR / "full_contract_pdf" / "Part_I")
    DB_PATH    = str(BASE_DIR / "contracts.db")
    CHROMA_DIR = str(BASE_DIR / "chroma_db")

    # ── Embedding Model ─────────────────────────────────────────────────────
    # all-MiniLM-L6-v2: 384-dim, fast on CPU, great for semantic similarity
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # ── Chunking Parameters ─────────────────────────────────────────────────
    CHUNK_SIZE    = 1000   # characters
    CHUNK_OVERLAP = 200    # characters — preserves cross-boundary context

    # ── Retrieval ───────────────────────────────────────────────────────────
    RETRIEVER_K = 5        # final chunks returned to LLM

    # ── LLM (Groq) ──────────────────────────────────────────────────────────
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL   = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")  # Active model on current Groq key

    # ── OCR ─────────────────────────────────────────────────────────────────
    # Minimum characters on a page to consider it digitally native (not scanned)
    OCR_TEXT_THRESHOLD = 50

    # ── ChromaDB Collection Names ────────────────────────────────────────────
    COLLECTION_CHUNKS   = "contract_chunks"    # PDF text chunks
    COLLECTION_METADATA = "contract_metadata"  # SQL structured rows
