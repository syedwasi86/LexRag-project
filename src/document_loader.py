"""
src/document_loader.py
===============================================================================
Stage 1 — Document Loader

Loads all PDFs from full_contract_pdf/Part_I using pymupdf4llm (outputs clean
Markdown). Falls back to pytesseract OCR for scanned/image-only PDFs.
Joins each PDF document with its structured metadata row from contracts.db.

Returns two lists:
  • pdf_docs      — one Document per PDF (full markdown text)
  • metadata_docs — one Document per contract row in the DB
===============================================================================
"""

import sqlite3
from pathlib import Path

import fitz                 # PyMuPDF — low-level PDF access + OCR rendering
import pymupdf4llm          # Converts native PDFs to clean Markdown
import pytesseract
from PIL import Image
from tqdm import tqdm

from langchain_core.documents import Document

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


# --- Helpers ------------------------------------------------------------------

def _is_scanned(pdf_path: str) -> bool:
    """
    Heuristic: open the PDF and check the first 3 pages.
    If none yield more than OCR_TEXT_THRESHOLD characters, treat as scanned.
    """
    try:
        doc = fitz.open(pdf_path)
        pages_checked = min(3, len(doc))
        text_found = sum(
            1 for i in range(pages_checked)
            if len(doc[i].get_text().strip()) > Config.OCR_TEXT_THRESHOLD
        )
        doc.close()
        return text_found == 0
    except Exception:
        return False


def _load_native(pdf_path: str) -> str:
    """Use pymupdf4llm to extract Markdown-formatted text from a native PDF."""
    return pymupdf4llm.to_markdown(pdf_path)


def _load_ocr(pdf_path: str) -> str:
    """
    Render each page as a 2× image and run Tesseract OCR.
    Falls back to native text extraction per page if available.
    """
    doc = fitz.open(pdf_path)
    pages_text = []
    for i, page in enumerate(doc):
        native = page.get_text().strip()
        if len(native) > Config.OCR_TEXT_THRESHOLD:
            pages_text.append(f"## Page {i + 1}\n\n{native}")
        else:
            mat = fitz.Matrix(2.0, 2.0)          # 2× zoom > better OCR quality
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(img, config="--psm 3")
            pages_text.append(f"## Page {i + 1}\n\n{ocr_text}")
    doc.close()
    return "\n\n".join(pages_text)


# --- Metadata helpers ---------------------------------------------------------

def _load_metadata_lookup(db_path: str) -> dict:
    """
    Read every row from contracts.db and return a dict keyed by Filename.
    Handles the mismatch between 510 DB rows and 198 actual PDFs gracefully.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM contracts")
    rows = cursor.fetchall()
    conn.close()
    return {dict(r)["Filename"]: dict(r) for r in rows if dict(r).get("Filename")}


def _contract_type_from_path(pdf_path: str) -> str:
    return Path(pdf_path).parent.name.replace("_", " ")


def _format_metadata_doc(row: dict, filename: str, contract_type: str) -> str:
    """
    Convert a DB row into a readable text document for embedding.
    Only include fields that have meaningful (non-empty, non-'No') answers.
    """
    FIELDS = [
        ("Document Name",                    "Document Name-Answer"),
        ("Parties",                          "Parties-Answer"),
        ("Agreement Date",                   "Agreement Date-Answer"),
        ("Effective Date",                   "Effective Date-Answer"),
        ("Expiration Date",                  "Expiration Date-Answer"),
        ("Renewal Term",                     "Renewal Term-Answer"),
        ("Notice Period To Terminate Renewal","Notice Period To Terminate Renewal- Answer"),
        ("Governing Law",                    "Governing Law-Answer"),
        ("Most Favored Nation",              "Most Favored Nation-Answer"),
        ("Non-Compete",                      "Non-Compete-Answer"),
        ("Exclusivity",                      "Exclusivity-Answer"),
        ("Termination For Convenience",      "Termination For Convenience-Answer"),
        ("Change Of Control",                "Change Of Control-Answer"),
        ("Anti-Assignment",                  "Anti-Assignment-Answer"),
        ("Revenue/Profit Sharing",           "Revenue/Profit Sharing-Answer"),
        ("License Grant",                    "License Grant-Answer"),
        ("Non-Transferable License",         "Non-Transferable License-Answer"),
        ("Audit Rights",                     "Audit Rights-Answer"),
        ("Uncapped Liability",               "Uncapped Liability-Answer"),
        ("Cap On Liability",                 "Cap On Liability-Answer"),
        ("Liquidated Damages",               "Liquidated Damages-Answer"),
        ("Warranty Duration",                "Warranty Duration-Answer"),
        ("Insurance",                        "Insurance-Answer"),
        ("Covenant Not To Sue",              "Covenant Not To Sue-Answer"),
    ]
    lines = [
        f"# Contract Metadata: {filename}",
        f"**Contract Type:** {contract_type}",
        "",
    ]
    for label, key in FIELDS:
        val = str(row.get(key, "")).strip()
        if val and val not in ("", "nan", "[]", "No", "None"):
            lines.append(f"**{label}:** {val}")
    return "\n".join(lines)


# --- Main loader --------------------------------------------------------------

def load_all_documents(
    pdf_dir: str = Config.PDF_DIR,
    db_path: str = Config.DB_PATH,
) -> tuple:
    """
    Walk Part_I directory, load every PDF, enrich with DB metadata.

    Returns
    -------
    pdf_docs : list[Document]
        Full markdown text of each PDF with metadata fields attached.
    metadata_docs : list[Document]
        One structured-text Document per DB row (for metadata collection).
    """
    print("-" * 60)
    print("STAGE 1 - Document Loader")
    print("-" * 60)

    # 1. Load metadata lookup from DB
    print(f"Loading metadata from {db_path} ...")
    lookup = _load_metadata_lookup(db_path)
    print(f"  > {len(lookup)} rows loaded from contracts.db")

    # -- Apply category filter to keep dataset manageable (~85 PDFs) -----------
    SELECTED_CATEGORIES = {
        "Endorsement", "Distributor", "Joint Venture", "Maintenance", 
        "Affiliate_Agreements", "Franchise", "Manufacturing", 
        "Marketing", "Strategic Alliance", "Reseller"
    }
    
    _all_paths = sorted(Path(pdf_dir).rglob("*.pdf"))
    pdf_paths = [p for p in _all_paths if p.parent.name in SELECTED_CATEGORIES]
    print(f"  > {len(pdf_paths)} PDFs found from 10 selected diverse categories\n")

    pdf_docs: list[Document] = []
    metadata_docs: list[Document] = []

    for pdf_path in tqdm(pdf_paths, desc="Loading PDFs", unit="pdf"):
        filename        = pdf_path.name
        contract_type   = _contract_type_from_path(str(pdf_path))
        row             = lookup.get(filename, {})

        # Base metadata shared by both document types
        base_meta = {
            "source":          str(pdf_path),
            "filename":        filename,
            "contract_type":   contract_type,
            "doc_name":        str(row.get("Document Name-Answer", filename)),
            "parties":         str(row.get("Parties-Answer", "")),
            "agreement_date":  str(row.get("Agreement Date-Answer", "")),
            "effective_date":  str(row.get("Effective Date-Answer", "")),
            "expiration_date": str(row.get("Expiration Date-Answer", "")),
            "governing_law":   str(row.get("Governing Law-Answer", "")),
        }

        # -- PDF Document --------------------------------------------------
        try:
            if _is_scanned(str(pdf_path)):
                text = _load_ocr(str(pdf_path))
                base_meta["loader_type"] = "ocr"
            else:
                text = _load_native(str(pdf_path))
                base_meta["loader_type"] = "native"

            if text.strip():
                pdf_docs.append(Document(page_content=text, metadata=base_meta))

        except Exception as exc:
            tqdm.write(f"  [WARN] Skipping {filename}: {exc}")
            continue

        # -- Metadata Document (only if DB row exists) ---------------------
        if row:
            meta_text = _format_metadata_doc(row, filename, contract_type)
            metadata_docs.append(Document(
                page_content=meta_text,
                metadata={**base_meta, "doc_type": "metadata"},
            ))

    print(f"\n✅ PDF documents loaded   : {len(pdf_docs)}")
    print(f"✅ Metadata documents     : {len(metadata_docs)}")
    return pdf_docs, metadata_docs


# --- Quick smoke test ----------------------------------------------------------
if __name__ == "__main__":
    pdf_docs, meta_docs = load_all_documents()
    print("\nSample PDF doc (first 400 chars):")
    print(pdf_docs[0].page_content[:400])
    print("\nSample metadata doc:")
    print(meta_docs[0].page_content[:400])
