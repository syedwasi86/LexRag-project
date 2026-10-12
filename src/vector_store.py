"""
src/vector_store.py
===============================================================================
Stage 3 — Vector Store (ChromaDB)

Builds and persists two separate ChromaDB collections:

  contract_chunks   — Embedded PDF text chunks for semantic search
  contract_metadata — Embedded SQL metadata rows for structured lookup

Embedding model: all-MiniLM-L6-v2 (384-dim, fast CPU inference)
  • Wrapped via HuggingFaceEmbeddings from langchain-huggingface
  • normalize_embeddings=True > cosine similarity works correctly

The vector store is persisted to chroma_db/ so indexing only runs once.
On subsequent runs, load_vector_store() rehydrates from disk.
===============================================================================
"""

import os
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from  langchain_core.documents import Document

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


# --- Embeddings ---------------------------------------------------------------

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load the all-MiniLM-L6-v2 sentence-transformer model.
    First call downloads (~90 MB) and caches to ~/.cache/huggingface.
    """
    return HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# --- Build (first-time indexing) ----------------------------------------------

def build_vector_store(
    pdf_chunks: list[Document],
    metadata_docs: list[Document],
    embeddings: HuggingFaceEmbeddings,
    persist_dir: str = Config.CHROMA_DIR,
) -> tuple:
    """
    Embed all chunks and metadata docs, persist to ChromaDB.

    This is the SLOW step (~20–40 min for 198 PDFs on CPU).
    Run once; subsequent sessions use load_vector_store().

    Returns
    -------
    (pdf_store, meta_store) : tuple[Chroma, Chroma]
    """
    print("-" * 60)
    print("STAGE 3 — Building Vector Store (one-time)")
    print("-" * 60)
    os.makedirs(persist_dir, exist_ok=True)

    print(f"Embedding {len(pdf_chunks)} PDF chunks > collection: {Config.COLLECTION_CHUNKS}")
    pdf_store = Chroma.from_documents(
        documents=pdf_chunks,
        embedding=embeddings,
        collection_name=Config.COLLECTION_CHUNKS,
        persist_directory=persist_dir,
    )

    print(f"Embedding {len(metadata_docs)} metadata docs > collection: {Config.COLLECTION_METADATA}")
    meta_store = Chroma.from_documents(
        documents=metadata_docs,
        embedding=embeddings,
        collection_name=Config.COLLECTION_METADATA,
        persist_directory=persist_dir,
    )

    print(f"\n✅ Vector store persisted to: {persist_dir}")
    print(f"   PDF chunks indexed    : {pdf_store._collection.count()}")
    print(f"   Metadata docs indexed : {meta_store._collection.count()}")
    return pdf_store, meta_store


# --- Load (fast rehydration from disk) ----------------------------------------

def load_vector_store(
    embeddings: HuggingFaceEmbeddings,
    persist_dir: str = Config.CHROMA_DIR,
) -> tuple:
    """
    Load existing ChromaDB collections from disk (fast path).
    Call this on every run after the first build.

    Returns
    -------
    (pdf_store, meta_store) : tuple[Chroma, Chroma]
    """
    print("-" * 60)
    print("STAGE 3 — Loading Vector Store from disk")
    print("-" * 60)

    pdf_store = Chroma(
        collection_name=Config.COLLECTION_CHUNKS,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    meta_store = Chroma(
        collection_name=Config.COLLECTION_METADATA,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    print(f"✅ PDF chunks loaded    : {pdf_store._collection.count()}")
    print(f"✅ Metadata docs loaded : {meta_store._collection.count()}")
    return pdf_store, meta_store


def vector_store_exists(persist_dir: str = Config.CHROMA_DIR) -> bool:
    """Return True if a persisted vector store already exists."""
    chroma_path = Path(persist_dir)
    return chroma_path.exists() and any(chroma_path.iterdir())


# --- Quick smoke test ----------------------------------------------------------
if __name__ == "__main__":
    embeddings = get_embeddings()
    if vector_store_exists():
        pdf_store, meta_store = load_vector_store(embeddings)
    else:
        from src.document_loader import load_all_documents
        from src.text_splitter import split_documents
        pdf_docs, meta_docs = load_all_documents()
        chunks = split_documents(pdf_docs)
        pdf_store, meta_store = build_vector_store(chunks, meta_docs, embeddings)

    results = pdf_store.similarity_search("governing law California", k=2)
    for r in results:
        print(r.metadata.get("filename"), "—", r.page_content[:150])
