# LexRAG Legal Contract RAG Pipeline Report

## Overview
The LexRAG pipeline is a specialized Retrieval-Augmented Generation system designed for analyzing and querying legal contracts. It consists of six distinct stages, ranging from initial document loading and intelligent chunking to hybrid retrieval, RAG answer generation, and automated pipeline evaluation.

---

## 1. Pipeline Structure & Working of Each Stage

### Stage 1: Document Loader (`src/document_loader.py`)
- **Working:** Ingests PDF contracts from the filesystem and enriches them with structured metadata from a SQLite database (`contracts.db`).
- **Techniques:**
  - **Native Parsing:** Uses `pymupdf4llm` to extract clean, Markdown-formatted text from digitally native PDFs.
  - **Heuristic OCR Fallback:** Detects scanned/image-only PDFs by checking the character count on the first three pages. If a document is scanned, it falls back to `pytesseract` OCR, rendering pages at 2x resolution (`fitz.Matrix(2.0, 2.0)`) for higher accuracy.
  - **Metadata Enrichment:** Links each PDF document to its corresponding metadata row from the DB, creating distinct text and metadata `Document` objects.

### Stage 2: Text Splitter (`src/text_splitter.py`)
- **Working:** Applies a two-phase chunking strategy specifically tailored to preserve the legal coherence of contract clauses.
- **Techniques:**
  - **Phase 1 (Header-based Split):** Uses LangChain's `MarkdownHeaderTextSplitter` to split the text exactly on Markdown headers (e.g., `# ARTICLE 5`, `## Section 5.2`). This prevents legal clauses from being torn in half.
  - **Phase 2 (Overflow Guard):** If a resulting chunk exceeds the character limit (1000 chars), a `RecursiveCharacterTextSplitter` is applied with a 200-character overlap to retain cross-sentence context.

### Stage 3: Vector Store (`src/vector_store.py`)
- **Working:** Embeds the chunked documents and saves them to disk. It maintains two distinct collections: `contract_chunks` for standard semantic search over the PDF text, and `contract_metadata` for embedded SQL rows.
- **Techniques:**
  - Uses **ChromaDB** for localized, fast vector storage (`chroma_db/`).
  - Automatically caches the embeddings to avoid re-running the expensive embedding process on subsequent application starts.

### Stage 4: Hybrid Retriever (`src/retriever.py`)
- **Working:** Retrieves the most relevant chunks using a combination of dense (semantic) and sparse (keyword) search.
- **Techniques:**
  - **Dense Leg:** Utilizes ChromaDB's Maximal Marginal Relevance (MMR) search to find semantically similar chunks while maintaining diversity (`lambda_mult=0.7`).
  - **Sparse Leg:** Utilizes a `BM25Retriever` to capture exact keyword and terminology matches.
  - **Reciprocal Rank Fusion (RRF):** Implements a custom RRF algorithm to merge the results of both legs, applying a 0.6 weight to semantic results and a 0.4 weight to BM25 results.

### Stage 5: RAG Pipeline (`src/rag_pipeline.py`)
- **Working:** The final generation step where the retrieved context is passed alongside the user's question to the LLM to generate an answer.
- **Techniques:**
  - Uses a `RetrievalQA` chain with `chain_type="stuff"` (stuffing all retrieved chunks into the prompt context).
  - Forces exact citations by returning `source_documents`. The app maps these back to the UI to prove the absence of hallucination.
  - Employs a strict, anti-hallucination prompt.

### Stage 6: RAGAS Evaluation (`src/evaluation.py`)
- **Working:** Automatically tests the performance of the retrieval and generation stack against known ground-truth data.

---

## 2. Models Used
- **Embedding Model:** `all-MiniLM-L6-v2`
  - A fast, 384-dimensional sentence-transformer model optimized for CPU inference. Embeddings are normalized (`normalize_embeddings=True`) for optimal cosine similarity operations.
- **Large Language Model (LLM):** `llama-3.3-70b-versatile`
  - Served via the Groq API, this 70B parameter model powers the core reasoning, question answering, and RAGAS metric evaluations.
- **OCR Model:** Tesseract (`pytesseract`)
  - Used with Page Segmentation Mode 3 (`--psm 3`) for processing non-native PDFs.

---

## 3. The Retrieval Prompt
After the hybrid retriever pulls the most relevant chunks, they are attached to the `{context}` variable of the following strict, zero-tolerance prompt:

```text
You are an expert legal contract analyst with deep knowledge of commercial law.
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

ANSWER (cite the contract name and clause/section where found):
```

---

## 4. How RAGAS is Used
RAGAS is implemented in `src/evaluation.py` to provide empirical metrics on pipeline quality without requiring manual human grading.

1. **Automated Dataset Generation:** It leverages the pre-existing SQLite database (`contracts.db`), matching specific DB columns (e.g., `Governing Law-Answer`) to programmatic question templates (e.g., *"What is the governing law in the {doc} contract?"*). This auto-generates a set of **Question + Ground Truth** pairs.
2. **Execution:** The RAG pipeline is invoked on each generated question, and the resulting `answer` and retrieved `contexts` are collected.
3. **Metric Calculation:** RAGAS uses the Groq LLM as a judge to calculate four critical metrics:
   - **Context Precision:** Are the retrieved chunks strictly relevant without introducing noise?
   - **Context Recall:** Did the retriever manage to find all the information required to match the ground truth?
   - **Faithfulness:** Is the generated answer entirely derived from the retrieved context without hallucination?
   - **Answer Relevancy:** How well does the final answer actually address the user's initial question?
4. **Outputs:** The results are averaged and output to `evaluation_results.json`, which is then read by the Streamlit UI to display live performance metrics on the dashboard sidebar.
