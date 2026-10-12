"""
src/rag_pipeline.py
===============================================================================
Stage 5 — RAG Chain

Uses Groq's free API (llama-3.3-70b-versatile) via langchain-groq.
Builds a RetrievalQA chain with a strict anti-hallucination prompt.
return_source_documents=True ensures every answer is citable.

query_rag() is the public interface used by app.py and evaluation.py.
===============================================================================
"""

from pathlib import Path
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


# --- Prompt -------------------------------------------------------------------

LEGAL_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert legal contract analyst with deep knowledge of commercial law.
Your task is to answer questions about legal contracts based SOLELY on the provided contract text.

STRICT RULES — follow these exactly:
1. Answer ONLY using information from the CONTEXT below. Do not use outside knowledge.
2. If the answer is not found in the context, respond with exactly:
   "I could not find this information in the provided contracts."
3. Always mention the contract name or filename your answer is based on.
4. Quote the exact clause text when answering about specific legal provisions.
5. Never infer, speculate, or hallucinate. Stick to what is written.

------------------------------------------------------------------
CONTEXT (retrieved contract clauses):
{context}
------------------------------------------------------------------

QUESTION: {question}

ANSWER (cite the contract name and clause/section where found):""",
)


# --- LLM factory --------------------------------------------------------------

def get_llm(temperature: float = 0) -> ChatGroq:
    """Initialise Groq LLM. Reads GROQ_API_KEY from Config (> .env)."""
    if not Config.GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Copy .env.example > .env and add your key from console.groq.com"
        )
    return ChatGroq(
        model=Config.GROQ_MODEL,
        temperature=temperature,
        api_key=Config.GROQ_API_KEY,
        max_tokens=2048,
    )


# --- Chain builder ------------------------------------------------------------

def build_rag_chain(retriever) -> RetrievalQA:
    """
    Build the full RetrievalQA chain.

    Parameters
    ----------
    retriever : LangChain retriever (output of retriever.build_retriever())

    Returns
    -------
    chain : RetrievalQA with source_documents returned
    """
    print("-" * 60)
    print("STAGE 5 — Building RAG Chain")
    print("-" * 60)

    llm = get_llm()
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",           # Stuff all chunks into one prompt
        retriever=retriever,
        return_source_documents=True, # ← Anti-hallucination key
        chain_type_kwargs={
            "prompt": LEGAL_RAG_PROMPT,
            "verbose": False,
        },
    )
    print(f"✅ RAG chain ready — LLM: {Config.GROQ_MODEL}")
    return chain


# --- Public query interface ----------------------------------------------------

def query_rag(chain: RetrievalQA, question: str) -> dict:
    """
    Run a question through the RAG pipeline.

    Parameters
    ----------
    chain    : Built RetrievalQA chain
    question : Natural language question about the contracts

    Returns
    -------
    dict with keys:
        "answer"    : str — LLM-generated answer
        "citations" : list[dict] — source chunks with metadata
        "sources"   : list[Document] — raw LangChain Documents
    """
    result      = chain.invoke({"query": question})
    answer      = result.get("result", "")
    source_docs = result.get("source_documents", [])

    # De-duplicate citations by filename + content fingerprint
    citations   = []
    seen        = set()
    for doc in source_docs:
        key = (doc.metadata.get("filename", ""), doc.page_content[:80])
        if key in seen:
            continue
        seen.add(key)
        citations.append({
            "filename":      doc.metadata.get("filename", "Unknown"),
            "contract_type": doc.metadata.get("contract_type", ""),
            "parties":       doc.metadata.get("parties", ""),
            "agreement_date":doc.metadata.get("agreement_date", ""),
            "governing_law": doc.metadata.get("governing_law", ""),
            "section":       (
                doc.metadata.get("header_2")
                or doc.metadata.get("header_1")
                or "—"
            ),
            "content":       doc.page_content,
            "source_path":   doc.metadata.get("source", ""),
        })

    return {"answer": answer, "citations": citations, "sources": source_docs}


# --- Quick smoke test ----------------------------------------------------------
if __name__ == "__main__":
    from src.vector_store import get_embeddings, load_vector_store
    from src.retriever import build_retriever

    embeddings = get_embeddings()
    pdf_store, _ = load_vector_store(embeddings)

    # For BM25 we need the raw chunks — load a small sample
    results = pdf_store.similarity_search("governing law", k=50)
    llm = get_llm()
    retriever, _ = build_retriever(pdf_store, results, llm)
    chain = build_rag_chain(retriever)

    out = query_rag(chain, "What is the governing law in the Reynolds Consumer service agreement?")
    print("\nAnswer:\n", out["answer"])
    print(f"\nCitations ({len(out['citations'])}):")
    for c in out["citations"]:
        print(f"  • {c['filename']} | {c['section']}")
