"""
src/evaluation.py
===============================================================================
Stage 6 — RAGAS Evaluation

Builds a ground-truth QA dataset from contracts.db (using the -Answer columns
as verified ground truth) and evaluates the RAG pipeline on four metrics:

  Context Precision  — Are retrieved chunks relevant? (no noise)
  Context Recall     — Are all relevant chunks found?
  Faithfulness       — Does the answer stay within retrieved context?
  Answer Relevancy   — Does the answer address the question?

Results are printed to console and saved to evaluation_results.json.
===============================================================================
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


# --- QA dataset builder -------------------------------------------------------

# Templates: (DB column, question template)
QA_TEMPLATES = [
    ("Governing Law",              "What is the governing law in the {doc} contract?"),
    ("Expiration Date",            "When does the {doc} contract expire?"),
    ("Renewal Term",               "What is the renewal term in the {doc} contract?"),
    ("Termination For Convenience","What are the termination for convenience terms in the {doc} contract?"),
    ("Liquidated Damages",         "Are there liquidated damages clauses in the {doc} contract?"),
    ("Non-Compete",                "Is there a non-compete clause in the {doc} contract?"),
    ("Cap On Liability",           "What is the cap on liability in the {doc} contract?"),
    ("Audit Rights",               "Does the {doc} contract include audit rights?"),
    ("License Grant",              "What license is granted in the {doc} contract?"),
    ("Anti-Assignment",            "What are the anti-assignment provisions in the {doc} contract?"),
]


def build_eval_dataset(db_path: str = Config.DB_PATH, n_samples: int = 5) -> list[dict]:
    """
    Build evaluation QA pairs from contracts.db ground truth.
    Uses only contracts whose PDFs are in Part_I (cross-references file system).

    Returns
    -------
    list of dicts with keys: question, ground_truth, filename
    """
    conn = sqlite3.connect(db_path)
    df   = pd.read_sql("SELECT * FROM contracts", conn)
    conn.close()

    # Only keep rows whose PDF actually exists in Part_I
    pdf_dir   = Path(Config.PDF_DIR)
    existing  = {p.name for p in pdf_dir.rglob("*.pdf")}
    df        = df[df["Filename"].isin(existing)].reset_index(drop=True)

    pairs = []
    for _, row in df.sample(min(n_samples * 3, len(df)), random_state=42).iterrows():
        doc_name = str(row.get("Document_Name_Answer", row.get("Filename", "the contract")))
        doc_name = doc_name.replace(".pdf", "").strip()[:60]

        for col, template in QA_TEMPLATES:
            answer_col = f"{col.replace(' ', '_').replace('-', '_')}_Answer"
            answer     = str(row.get(answer_col, "")).strip()
            if answer and answer not in ("", "nan", "[]", "No", "None"):
                pairs.append({
                    "question":     template.format(doc=doc_name),
                    "ground_truth": answer,
                    "filename":     row.get("Filename", ""),
                })
        if len(pairs) >= n_samples:
            break

    print(f"✅ Built {len(pairs[:n_samples])} evaluation QA pairs from contracts.db")
    return pairs[:n_samples]


# --- RAGAS runner -------------------------------------------------------------

def run_evaluation(
    rag_chain,
    embeddings,
    eval_pairs: list[dict] | None = None,
    output_file: str = "evaluation_results.json",
) -> dict:
    """
    Run RAGAS on the RAG pipeline and return aggregated metric scores.

    Parameters
    ----------
    rag_chain   : Built RetrievalQA chain (from rag_pipeline.py)
    embeddings  : HuggingFaceEmbeddings instance
    eval_pairs  : Pre-built QA pairs; if None, builds from DB automatically
    output_file : Path to save JSON results

    Returns
    -------
    dict of metric_name > float score
    """
    from src.rag_pipeline import get_llm

    if eval_pairs is None:
        eval_pairs = build_eval_dataset()

    print(f"\n{'-'*60}")
    print(f"STAGE 6 — RAGAS Evaluation ({len(eval_pairs)} QA pairs)")
    print("-" * 60)

    data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for i, pair in enumerate(eval_pairs):
        print(f"  [{i+1:02d}/{len(eval_pairs)}] {pair['question'][:70]}", end=" ... ")
        try:
            result      = rag_chain.invoke({"query": pair["question"]})
            answer      = result.get("result", "")
            src_docs    = result.get("source_documents", [])
            contexts    = [d.page_content for d in src_docs] or [""]
            print("OK")
        except Exception as exc:
            print(f"ERROR: {exc}")
            answer, contexts = "", [""]

        data["question"].append(pair["question"])
        data["answer"].append(answer)
        data["contexts"].append(contexts)
        data["ground_truth"].append(pair["ground_truth"])

    dataset = Dataset.from_dict(data)

    # Wrap LLM + embeddings for RAGAS
    llm              = get_llm()
    ragas_llm        = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    results = evaluate(
        dataset=dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    scores = (
        results.to_pandas()[
            ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
        ]
        .mean()
        .to_dict()
    )

    print(f"\n{'='*40}")
    print("📊  RAGAS Evaluation Results")
    print("=" * 40)
    print(f"  Context Precision  : {scores.get('context_precision', 0):.3f}")
    print(f"  Context Recall     : {scores.get('context_recall', 0):.3f}")
    print(f"  Faithfulness       : {scores.get('faithfulness', 0):.3f}")
    print(f"  Answer Relevancy   : {scores.get('answer_relevancy', 0):.3f}")
    print("=" * 40)

    with open(output_file, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\n✅ Scores saved to {output_file}")

    return scores


# --- Run standalone -----------------------------------------------------------
if __name__ == "__main__":
    from src.vector_store import get_embeddings, load_vector_store
    from src.retriever import build_retriever
    from src.rag_pipeline import build_rag_chain, get_llm

    embeddings          = get_embeddings()
    pdf_store, _        = load_vector_store(embeddings)
    sample_chunks       = pdf_store.similarity_search("contract", k=200)
    llm                 = get_llm()
    retriever, _        = build_retriever(pdf_store, sample_chunks, llm)
    chain               = build_rag_chain(retriever)

    run_evaluation(chain, embeddings)
