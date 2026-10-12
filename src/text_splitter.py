"""
src/text_splitter.py
===============================================================================
Stage 2 — Text Splitter

Two-phase chunking strategy designed for legal contracts:

Phase 1 — MarkdownHeaderTextSplitter
    pymupdf4llm outputs Markdown with headers like "# ARTICLE 5" and
    "## Section 5.2 — Late Delivery". Splitting ON these headers keeps
    each chunk legally coherent (one clause per chunk).

Phase 2 — RecursiveCharacterTextSplitter (overflow guard)
    Any chunk still > CHUNK_SIZE after Phase 1 is recursively split
    with 200-char overlap to preserve cross-sentence context.

Metadata propagated per chunk:
    header_1, header_2, header_3, header_4 (from Markdown headers)
    + all original PDF metadata (filename, parties, dates, etc.)
===============================================================================
"""

from pathlib import Path
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


# --- Splitter configuration ---------------------------------------------------

HEADERS_TO_SPLIT_ON = [
    ("#",    "header_1"),
    ("##",   "header_2"),
    ("###",  "header_3"),
    ("####", "header_4"),
]

_md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=HEADERS_TO_SPLIT_ON,
    strip_headers=False,   # Keep headers inside chunk text for LLM context
)

_recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=Config.CHUNK_SIZE,
    chunk_overlap=Config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", ", ", " ", ""],
    length_function=len,
)


# --- Main function ------------------------------------------------------------

def split_documents(pdf_docs: list[Document]) -> list[Document]:
    """
    Apply two-phase chunking to a list of PDF Documents.

    Parameters
    ----------
    pdf_docs : list[Document]
        Output from document_loader.load_all_documents()

    Returns
    -------
    list[Document]
        Chunked documents with enriched metadata.
    """
    print("-" * 60)
    print("STAGE 2 — Text Splitter")
    print("-" * 60)

    all_chunks: list[Document] = []

    for doc in pdf_docs:
        try:
            # -- Phase 1: Markdown header-based split ----------------------
            md_chunks = _md_splitter.split_text(doc.page_content)

            for md_chunk in md_chunks:
                # Merge PDF metadata with header metadata extracted by splitter
                chunk_meta = {**doc.metadata, **md_chunk.metadata}

                if not md_chunk.page_content.strip():
                    continue

                # -- Phase 2: Overflow guard -------------------------------
                if len(md_chunk.page_content) > Config.CHUNK_SIZE:
                    sub_chunks = _recursive_splitter.split_text(md_chunk.page_content)
                    for sub in sub_chunks:
                        if sub.strip():
                            all_chunks.append(
                                Document(page_content=sub, metadata=chunk_meta)
                            )
                else:
                    all_chunks.append(
                        Document(
                            page_content=md_chunk.page_content,
                            metadata=chunk_meta,
                        )
                    )

        except Exception:
            # Fallback: plain recursive split if Markdown parsing fails
            fallback = _recursive_splitter.split_documents([doc])
            all_chunks.extend(fallback)

    print(f"✅ {len(pdf_docs)} PDFs > {len(all_chunks)} chunks")
    print(f"   avg chunk size ~ {sum(len(c.page_content) for c in all_chunks) // max(len(all_chunks), 1)} chars")
    return all_chunks


# --- Quick smoke test ----------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.document_loader import load_all_documents

    pdf_docs, _ = load_all_documents()
    chunks = split_documents(pdf_docs[:5])   # Only test on 5 docs
    print("\nSample chunk:")
    print(chunks[0].page_content[:300])
    print("\nChunk metadata:", chunks[0].metadata)
