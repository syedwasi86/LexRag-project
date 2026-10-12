"""
src/retriever.py
===============================================================================
Stage 4 — Hybrid Retriever

Architecture:
  - Dense leg: ChromaDB MMR retriever (semantic)
  - Sparse leg: BM25 keyword retriever (exact term matching)
  - Fusion: Custom Reciprocal Rank Fusion (RRF) to avoid LangChain import issues

Note: LangChain 1.x moved/removed many retrievers (Ensemble, MultiQuery, etc.).
We implement a robust custom RRF class to guarantee stability.
===============================================================================
"""

from pathlib import Path
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

class CustomHybridRetriever(BaseRetriever):
    """A custom implementation of EnsembleRetriever using Reciprocal Rank Fusion."""
    semantic: BaseRetriever
    bm25: BaseRetriever
    k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Get results from both retrievers
        semantic_docs = self.semantic.invoke(query)
        bm25_docs = self.bm25.invoke(query)
        
        # Reciprocal Rank Fusion
        rrf_score = {}
        
        # Dense weight = 0.6
        for rank, doc in enumerate(semantic_docs, 1):
            key = doc.page_content
            if key not in rrf_score:
                rrf_score[key] = {"doc": doc, "score": 0.0}
            rrf_score[key]["score"] += 0.6 * (1 / (rank + 60))
            
        # Sparse weight = 0.4
        for rank, doc in enumerate(bm25_docs, 1):
            key = doc.page_content
            if key not in rrf_score:
                rrf_score[key] = {"doc": doc, "score": 0.0}
            rrf_score[key]["score"] += 0.4 * (1 / (rank + 60))
            
        # Sort by RRF score
        sorted_docs = sorted(rrf_score.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs][:self.k]


def build_retriever(
    pdf_store: Chroma,
    pdf_chunks: list[Document],
    llm,
    k: int = Config.RETRIEVER_K,
):
    """
    Build the full hybrid retrieval stack.

    Parameters
    ----------
    pdf_store   : ChromaDB collection of PDF text chunks
    pdf_chunks  : The same chunks as Python objects (needed for BM25)
    llm         : ChatGroq instance (unused here, kept for signature compatibility)
    k           : Number of final chunks to return per query
    """

    # -- 1. Dense semantic retriever (ChromaDB MMR) ----------------------------
    semantic_retriever = pdf_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k * 2,
            "fetch_k": k * 5,
            "lambda_mult": 0.7,
        },
    )

    # -- 2. Sparse BM25 retriever (keyword matching) ---------------------------
    bm25_retriever = BM25Retriever.from_documents(pdf_chunks)
    bm25_retriever.k = k * 2

    # -- 3. Custom Ensemble (Reciprocal Rank Fusion) ---------------------------
    ensemble_retriever = CustomHybridRetriever(semantic=semantic_retriever, bm25=bm25_retriever, k=k)

    print("✅ Retriever stack built: Custom BM25 + MMR Hybrid (RRF)")
    return ensemble_retriever, ensemble_retriever
