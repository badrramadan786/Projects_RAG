"""
Project RAG Pro — Multi-project RAG web application with dual document pools.

Each project has two independent document pools:
  1. Main Documents — primary bid/project PDFs and Excel files
  2. Clarification Documents — clarification Excel files (and PDFs)

Each pool has its own upload, sync, FAISS index, and chat.

Usage:
    1. pip install flask faiss-cpu numpy openai pypdf openpyxl
    2. export OPENAI_API_KEY="sk-..."
    3. python app.py
    4. Open http://localhost:8080
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import pickle
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import faiss
import numpy as np
from flask import Flask, jsonify, render_template, request, session, redirect, url_for
from openai import OpenAI
from pypdf import PdfReader

# ── Fix macOS OpenMP issue ──────────────────────────────────────────────────
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── Configurable defaults ───────────────────────────────────────────────────

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"
EMBED_DIM = 1536  # dimension for text-embedding-3-small

BASE_DIR = pathlib.Path(__file__).parent.resolve()
PROJECTS_ROOT = BASE_DIR / "projects"

CHUNK_CHARS = 1800
OVERLAP_CHARS = 250
EMBED_BATCH_SIZE = 100

TOP_K_RETRIEVE = 40
TOP_K_FINAL = 20

# The two pool types
POOL_MAIN = "main"
POOL_CLARIFICATION = "clarification"
POOL_TYPES = [POOL_MAIN, POOL_CLARIFICATION]

SYSTEM_PROMPT_MAIN = (
    "You are an Expert Bid Review Engineer with 25+ years of experience analyzing client "
    "tender and bid documents for major EPC projects (oil & gas, subsea, pipeline, offshore, "
    "civil, and industrial). Your mission is ZERO DATA LOSS — every single detail in the "
    "client's bid documents matters and must not be missed.\n\n"

    "═══ CORE PRINCIPLES ═══\n"
    "• EXHAUSTIVE EXTRACTION: Read every single context snippet provided. Treat each snippet "
    "  as potentially containing critical bid requirements. Do not dismiss any snippet.\n"
    "• ZERO ASSUMPTIONS: Never assume, infer, or fabricate any data. If a value, date, spec, "
    "  or requirement is not explicitly stated in the snippets, say so clearly.\n"
    "• COMPLETENESS OVER BREVITY: It is far better to include too much detail than to miss "
    "  a single requirement. When in doubt, include it.\n"
    "• CROSS-REFERENCING: Information about the same topic may be scattered across multiple "
    "  documents and pages. You MUST cross-reference and consolidate all related data.\n\n"

    "═══ ANALYSIS METHODOLOGY ═══\n"
    "Follow this exact process for every question:\n\n"
    "STEP 1 — SCAN: Read all provided snippets end-to-end. Flag every snippet that contains "
    "any information even remotely related to the question.\n\n"
    "STEP 2 — EXTRACT: From each flagged snippet, extract ALL relevant details:\n"
    "  - Technical specifications (dimensions, pressures, temperatures, materials, grades)\n"
    "  - Scope of work items and deliverables\n"
    "  - Design codes, standards, and regulations (API, ASME, DNV, ISO, NACE, etc.)\n"
    "  - Schedule dates, milestones, and durations\n"
    "  - Commercial terms (quantities, rates, payment terms, warranties)\n"
    "  - Client requirements, hold points, witness points, and approval requirements\n"
    "  - Environmental and safety requirements\n"
    "  - Interface requirements and responsibilities (client vs contractor)\n"
    "  - Exclusions, exceptions, and special conditions\n"
    "  - Referenced documents and drawings\n\n"

    "STEP 3 — CONSOLIDATE: Merge extracted data from all snippets into a unified answer. "
    "If two snippets give different values for the same parameter, report BOTH values and "
    "cite both sources — flag it as a potential discrepancy.\n\n"

    "STEP 4 — STRUCTURE: Organize the answer with clear headings, numbered lists, and "
    "tables where appropriate. Group related information together.\n\n"

    "STEP 5 — VERIFY: Before finalizing, re-scan all snippets one more time to ensure "
    "nothing was missed.\n\n"

    "═══ CITATION RULES ═══\n"
    "• Cite EVERY piece of information with: (File: <filename>, Page/Sheet: <page>)\n"
    "• If the same fact appears in multiple sources, cite all of them\n"
    "• Group citations at the end of each paragraph or data point\n\n"

    "═══ RESPONSE FORMAT ═══\n"
    "• Use clear headings and subheadings for complex answers\n"
    "• Use numbered lists for requirements, deliverables, and scope items\n"
    "• Use tables for specifications, parameters, and comparisons\n"
    "• Highlight critical values, deadlines, and mandatory requirements\n"
    "• At the end of your answer, add a section called 'Related Documents Referenced' "
    "  listing all source files and pages used\n\n"

    "═══ CRITICAL RULES ═══\n"
    "• NEVER say 'the document mentions...' vaguely — always give the exact detail\n"
    "• NEVER invent file names, page numbers, specifications, or any facts\n"
    "• NEVER skip a snippet — if it contains relevant data, use it\n"
    "• If the answer is not found in any snippet, reply: 'Not found in provided documents.'\n"
    "• If only partial information is found, provide what is available and clearly state "
    "  what is missing: 'Note: The following details were not found in the provided snippets: ...'\n"
    "• If you detect contradictions between documents, flag them explicitly\n"
)

SYSTEM_PROMPT_CLARIFICATION = (
    "You are an Expert Bid Clarification Analyst with 25+ years of experience reviewing "
    "client clarification documents, technical queries (TQs), and responses for major EPC "
    "projects. Your mission is ZERO DATA LOSS — every clarification, amendment, and revised "
    "requirement must be captured accurately.\n\n"

    "═══ CORE PRINCIPLES ═══\n"
    "• EXHAUSTIVE EXTRACTION: Read every single context snippet. Clarification documents "
    "  often contain critical changes to the original bid requirements.\n"
    "• TRACK CHANGES: Pay special attention to items that modify, supersede, or add to "
    "  the original bid requirements. Highlight what changed and what the new requirement is.\n"
    "• QUESTION-ANSWER PAIRS: Clarifications are often in Q&A format. Preserve the link "
    "  between the question asked and the answer/clarification provided.\n"
    "• ZERO ASSUMPTIONS: Never assume or fabricate data. If something is ambiguous, say so.\n\n"

    "═══ ANALYSIS METHODOLOGY ═══\n"
    "STEP 1 — SCAN: Read all snippets. Flag every snippet related to the question.\n\n"
    "STEP 2 — EXTRACT: From each snippet, extract:\n"
    "  - Clarification number/reference and date\n"
    "  - Original question or query raised\n"
    "  - Client's response or clarification\n"
    "  - Any revised specifications, requirements, or scope changes\n"
    "  - References to original bid document sections being clarified\n"
    "  - Action items, deadlines, or follow-up requirements\n"
    "  - Items marked as 'noted', 'accepted', 'rejected', or 'refer to...'\n\n"

    "STEP 3 — CONSOLIDATE: Group related clarifications together. If multiple "
    "clarifications address the same topic, present them chronologically.\n\n"

    "STEP 4 — STRUCTURE: Organize with clear headings. Use tables for Q&A pairs.\n\n"

    "STEP 5 — VERIFY: Re-scan all snippets to ensure nothing was missed.\n\n"

    "═══ CITATION & FORMAT RULES ═══\n"
    "• Cite every piece of information: (File: <filename>, Sheet: <sheet>)\n"
    "• Use tables for clarification Q&A pairs\n"
    "• Highlight items that CHANGE original requirements\n"
    "• If the answer is not found, reply: 'Not found in provided clarification documents.'\n"
    "• NEVER invent file names, sheet names, or any facts\n"
)

# ── Flask app ───────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB max upload
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(32).hex())

# ── Password protection ──────────────────────────────────────────────────
# Set APP_PASSWORD env var to enable password protection.
# If not set, the app is open (useful for local use).

APP_PASSWORD = os.environ.get("APP_PASSWORD", "").strip()


@app.before_request
def check_auth():
    """Check password before every request (if APP_PASSWORD is set)."""
    if not APP_PASSWORD:
        return  # No password set — open access
    # Allow login page and static assets
    if request.endpoint in ("login_page", "login_submit", "static"):
        return
    if not session.get("authenticated"):
        return redirect(url_for("login_page"))


@app.route("/login", methods=["GET"])
def login_page():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Project RAG Pro — Login</title>
        <style>
            :root {
                --bg: #0c0e14; --surface: #13161f; --surface2: #1a1e2a;
                --border: #2a2f42; --text: #e8eaf0; --text-dim: #7c829a;
                --accent: #5b8def; --accent-hover: #4a7de0; --red: #ef5b5b;
            }
            * { margin:0; padding:0; box-sizing:border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', sans-serif;
                background: var(--bg); color: var(--text);
                height:100vh; display:flex; align-items:center; justify-content:center;
            }
            .login-box {
                background: var(--surface); border:1px solid var(--border);
                border-radius:14px; padding:40px; width:380px; text-align:center;
            }
            .login-box h1 { font-size:22px; font-weight:700; margin-bottom:6px; }
            .login-box p { font-size:13px; color:var(--text-dim); margin-bottom:24px; }
            .login-box input {
                width:100%; padding:12px 16px; background:var(--surface2);
                border:1px solid var(--border); border-radius:10px;
                color:var(--text); font-size:14px; outline:none;
                margin-bottom:16px; text-align:center;
            }
            .login-box input:focus { border-color:var(--accent); }
            .login-box input::placeholder { color:var(--text-dim); }
            .login-box button {
                width:100%; padding:12px; background:var(--accent);
                border:none; border-radius:10px; color:white;
                font-size:14px; font-weight:600; cursor:pointer;
            }
            .login-box button:hover { background:var(--accent-hover); }
            .error { color:var(--red); font-size:12px; margin-bottom:12px; }
        </style>
    </head>
    <body>
        <div class="login-box">
            <h1>Project RAG Pro</h1>
            <p>Enter the team password to continue</p>
            <form method="POST" action="/login">
                <input type="password" name="password" placeholder="Password" autofocus />
                <button type="submit">Sign In</button>
            </form>
        </div>
    </body>
    </html>
    '''


@app.route("/login", methods=["POST"])
def login_submit():
    password = request.form.get("password", "")
    if password == APP_PASSWORD:
        session["authenticated"] = True
        return redirect("/")
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Project RAG Pro — Login</title>
        <style>
            :root {
                --bg: #0c0e14; --surface: #13161f; --surface2: #1a1e2a;
                --border: #2a2f42; --text: #e8eaf0; --text-dim: #7c829a;
                --accent: #5b8def; --accent-hover: #4a7de0; --red: #ef5b5b;
            }
            * { margin:0; padding:0; box-sizing:border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', sans-serif;
                background: var(--bg); color: var(--text);
                height:100vh; display:flex; align-items:center; justify-content:center;
            }
            .login-box {
                background: var(--surface); border:1px solid var(--border);
                border-radius:14px; padding:40px; width:380px; text-align:center;
            }
            .login-box h1 { font-size:22px; font-weight:700; margin-bottom:6px; }
            .login-box p { font-size:13px; color:var(--text-dim); margin-bottom:24px; }
            .login-box input {
                width:100%; padding:12px 16px; background:var(--surface2);
                border:1px solid var(--border); border-radius:10px;
                color:var(--text); font-size:14px; outline:none;
                margin-bottom:16px; text-align:center;
            }
            .login-box input:focus { border-color:var(--accent); }
            .login-box input::placeholder { color:var(--text-dim); }
            .login-box button {
                width:100%; padding:12px; background:var(--accent);
                border:none; border-radius:10px; color:white;
                font-size:14px; font-weight:600; cursor:pointer;
            }
            .login-box button:hover { background:var(--accent-hover); }
            .error { color:var(--red); font-size:12px; margin-bottom:12px; }
        </style>
    </head>
    <body>
        <div class="login-box">
            <h1>Project RAG Pro</h1>
            <p>Enter the team password to continue</p>
            <div class="error">Incorrect password. Please try again.</div>
            <form method="POST" action="/login">
                <input type="password" name="password" placeholder="Password" autofocus />
                <button type="submit">Sign In</button>
            </form>
        </div>
    </body>
    </html>
    '''

# ── OpenAI client ───────────────────────────────────────────────────────────

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            print("\n[ERROR] OPENAI_API_KEY is not set.")
            print('  export OPENAI_API_KEY="sk-..."')
            sys.exit(1)
        _client = OpenAI(api_key=api_key)
    return _client


# ── Helpers ─────────────────────────────────────────────────────────────────


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def slugify(name: str) -> str:
    """Create a safe directory name from a project name."""
    slug = re.sub(r"[^\w\s-]", "", name.lower().strip())
    slug = re.sub(r"[\s_-]+", "_", slug)
    return slug or "project"


# ── Project management ──────────────────────────────────────────────────────


def get_projects_list() -> list[dict[str, Any]]:
    """Return list of all projects with metadata."""
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
    projects = []
    for d in sorted(PROJECTS_ROOT.iterdir()):
        meta_path = d / "project_meta.json"
        if d.is_dir() and meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            # Count docs and chunks per pool
            for pool in POOL_TYPES:
                docs_dir = d / f"documents_{pool}"
                doc_count = 0
                if docs_dir.exists():
                    doc_count = len([
                        p for p in docs_dir.iterdir()
                        if p.suffix.lower() in SUPPORTED_EXTENSIONS
                    ])
                manifest = load_manifest(d, pool)
                chunk_count = sum(len(e.get("chunk_ids", [])) for e in manifest.values())
                meta[f"{pool}_doc_count"] = doc_count
                meta[f"{pool}_chunk_count"] = chunk_count
            projects.append(meta)
    return projects


def create_project(name: str, description: str = "") -> dict[str, Any]:
    """Create a new project directory and metadata."""
    slug = slugify(name)
    project_dir = PROJECTS_ROOT / slug

    # Handle duplicate slugs
    counter = 1
    while project_dir.exists():
        slug = f"{slugify(name)}_{counter}"
        project_dir = PROJECTS_ROOT / slug
        counter += 1

    project_dir.mkdir(parents=True, exist_ok=True)
    for pool in POOL_TYPES:
        (project_dir / f"documents_{pool}").mkdir(exist_ok=True)
        (project_dir / f"faiss_store_{pool}").mkdir(exist_ok=True)

    meta = {
        "id": slug,
        "name": name,
        "description": description,
        "created": datetime.now().isoformat(),
        "last_accessed": datetime.now().isoformat(),
    }
    with open(project_dir / "project_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def get_project_dir(project_id: str) -> pathlib.Path:
    return PROJECTS_ROOT / project_id


def update_last_accessed(project_id: str):
    project_dir = get_project_dir(project_id)
    meta_path = project_dir / "project_meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["last_accessed"] = datetime.now().isoformat()
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


def delete_project(project_id: str) -> bool:
    project_dir = get_project_dir(project_id)
    if project_dir.exists() and project_dir.is_dir():
        # Release stores if cached
        for pool in POOL_TYPES:
            key = f"{project_id}_{pool}"
            if key in _stores:
                del _stores[key]
        shutil.rmtree(project_dir)
        return True
    return False


# ── Manifest (per pool) ───────────────────────────────────────────────────


def manifest_path(project_dir: pathlib.Path, pool: str) -> pathlib.Path:
    return project_dir / f"manifest_{pool}.json"


def load_manifest(project_dir: pathlib.Path, pool: str) -> dict[str, Any]:
    mp = manifest_path(project_dir, pool)
    if mp.exists():
        with open(mp, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(project_dir: pathlib.Path, pool: str, manifest: dict[str, Any]) -> None:
    mp = manifest_path(project_dir, pool)
    mp.parent.mkdir(parents=True, exist_ok=True)
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


# ── PDF extraction ──────────────────────────────────────────────────────────


def extract_pdf_pages(pdf_path: pathlib.Path) -> list[tuple[str, str]]:
    """Return list of (page_label, text) for a PDF."""
    reader = PdfReader(str(pdf_path))
    pages: list[tuple[str, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        text = normalize_whitespace(raw)
        if text.strip():
            pages.append((f"Page {i}", text))
    return pages


# ── Excel extraction ────────────────────────────────────────────────────────


def extract_excel_sheets(excel_path: pathlib.Path) -> list[tuple[str, str]]:
    """Return list of (sheet_label, text) for an Excel file."""
    try:
        import openpyxl
    except ImportError:
        print("[WARN] openpyxl not installed. Run: pip install openpyxl")
        return []

    wb = openpyxl.load_workbook(str(excel_path), read_only=True, data_only=True)
    sheets: list[tuple[str, str]] = []

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows_text: list[str] = []

        # Get header row
        header: list[str] = []
        first_row = True

        for row in ws.iter_rows(values_only=True):
            cell_values = []
            for cell in row:
                if cell is not None:
                    cell_values.append(str(cell).strip())
                else:
                    cell_values.append("")

            if first_row:
                header = cell_values
                first_row = False
                rows_text.append(" | ".join(v for v in cell_values if v))
            else:
                # Create "Header: Value" pairs for non-empty cells
                pairs = []
                for idx, val in enumerate(cell_values):
                    if val:
                        col_name = header[idx] if idx < len(header) and header[idx] else f"Col{idx+1}"
                        pairs.append(f"{col_name}: {val}")
                if pairs:
                    rows_text.append("; ".join(pairs))

        if rows_text:
            text = normalize_whitespace(" . ".join(rows_text))
            sheets.append((f"Sheet: {sheet_name}", text))

    wb.close()
    return sheets


# ── Universal extraction ────────────────────────────────────────────────────


def extract_document(file_path: pathlib.Path) -> list[tuple[str, str]]:
    """Extract text from PDF or Excel. Returns list of (section_label, text)."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_pages(file_path)
    elif suffix in (".xlsx", ".xls"):
        return extract_excel_sheets(file_path)
    return []


# ── Chunking ────────────────────────────────────────────────────────────────


def chunk_text(text: str, chunk_chars: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_chars
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def build_chunks_for_file(file_path: pathlib.Path, file_hash: str) -> list[dict[str, Any]]:
    sections = extract_document(file_path)
    all_chunks: list[dict[str, Any]] = []
    for section_label, section_text in sections:
        parts = chunk_text(section_text)
        for ci, part in enumerate(parts):
            chunk_id = f"{file_hash}_{section_label}_c{ci}"
            all_chunks.append({
                "id": chunk_id,
                "document": part,
                "metadata": {
                    "file": file_path.name,
                    "path": str(file_path.resolve()),
                    "page": section_label,
                    "chunk_index": ci,
                    "sha256": file_hash,
                },
            })
    return all_chunks


# ── OpenAI API helpers ──────────────────────────────────────────────────────


def _embed_batch(batch: list[str], model: str) -> list[list[float]]:
    """Embed a single batch — called in parallel threads. Retries on rate limit."""
    oa = get_client()
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = oa.embeddings.create(model=model, input=batch)
            return [item.embedding for item in response.data]
        except Exception as e:
            err_str = str(e).lower()
            if "rate" in err_str or "429" in err_str:
                wait = 2 ** attempt  # 1, 2, 4, 8, 16 seconds
                print(f"    Rate limit hit, waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                raise
    # Final attempt without catching
    response = oa.embeddings.create(model=model, input=batch)
    return [item.embedding for item in response.data]


def embed_texts(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    if not texts:
        return []
    batches = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batches.append(texts[i: i + EMBED_BATCH_SIZE])
    total = len(batches)
    if total == 1:
        print(f"    Embedding {len(texts)} chunks in 1 batch...")
        return _embed_batch(batches[0], model)
    # Parallel embedding — up to 3 concurrent API calls
    print(f"    Embedding {len(texts)} chunks in {total} batches (parallel)...")
    results: dict[int, list[list[float]]] = {}
    done_count = 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_embed_batch, b, model): idx for idx, b in enumerate(batches)}
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
            done_count += 1
            print(f"    Batch {done_count}/{total} done")
    # Reassemble in order
    all_embeddings: list[list[float]] = []
    for idx in range(total):
        all_embeddings.extend(results[idx])
    return all_embeddings


def chat_completion(messages: list[dict[str, str]], model: str = CHAT_MODEL) -> str:
    oa = get_client()
    response = oa.chat.completions.create(model=model, messages=messages, temperature=0.2)
    return response.choices[0].message.content


# ── FAISS Vector Store ──────────────────────────────────────────────────────


class FaissStore:
    def __init__(self, store_dir: pathlib.Path):
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = store_dir / "faiss.index"
        self.meta_path = store_dir / "metadata.pkl"
        self.index = None
        self.ids: list[str] = []
        self.documents: list[str] = []
        self.metadatas: list[dict[str, Any]] = []
        self._load()

    def _load(self):
        if self.index_path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.meta_path, "rb") as f:
                saved = pickle.load(f)
            self.ids = saved["ids"]
            self.documents = saved["documents"]
            self.metadatas = saved["metadatas"]

    def _save(self):
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "wb") as f:
            pickle.dump({"ids": self.ids, "documents": self.documents, "metadatas": self.metadatas}, f)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms

    def count(self) -> int:
        return len(self.ids)

    def upsert(self, ids, documents, metadatas, embeddings):
        existing_set = set(self.ids)
        remove_ids = [cid for cid in ids if cid in existing_set]
        if remove_ids:
            self.delete(remove_ids)
        new_vecs = self._normalize(np.array(embeddings, dtype=np.float32))
        if self.index is None:
            self.index = faiss.IndexFlatIP(new_vecs.shape[1])
        self.index.add(new_vecs)
        self.ids.extend(ids)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self._save()

    def delete(self, ids_to_remove):
        remove_set = set(ids_to_remove)
        keep_indices = [i for i, cid in enumerate(self.ids) if cid not in remove_set]
        if not keep_indices:
            self.index = None
            self.ids, self.documents, self.metadatas = [], [], []
            self._save()
            return
        kept_vecs = np.array([self.index.reconstruct(i) for i in keep_indices], dtype=np.float32)
        self.ids = [self.ids[i] for i in keep_indices]
        self.documents = [self.documents[i] for i in keep_indices]
        self.metadatas = [self.metadatas[i] for i in keep_indices]
        self.index = faiss.IndexFlatIP(kept_vecs.shape[1])
        self.index.add(kept_vecs)
        self._save()

    def query(self, query_embedding, top_k=TOP_K_RETRIEVE):
        if self.index is None or self.index.ntotal == 0:
            return {"documents": [], "metadatas": [], "distances": []}
        q_vec = self._normalize(np.array([query_embedding], dtype=np.float32))
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_vec, k)
        docs, metas, dists = [], [], []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            docs.append(self.documents[idx])
            metas.append(self.metadatas[idx])
            dists.append(1.0 - float(score))
        return {"documents": docs, "metadatas": metas, "distances": dists}

    def clear(self):
        self.index = None
        self.ids, self.documents, self.metadatas = [], [], []
        for p in [self.index_path, self.meta_path]:
            if p.exists():
                p.unlink()


# ── Store cache (keyed by project_id + pool) ─────────────────────────────

_stores: dict[str, FaissStore] = {}
_sync_lock = threading.Lock()

# ── Background task tracking ──────────────────────────────────────────────

_tasks: dict[str, dict[str, Any]] = {}  # task_id -> {status, progress, message, result}


def _task_key(project_id: str, pool: str) -> str:
    return f"{project_id}_{pool}"


def _run_sync_bg(project_id: str, pool: str, rebuild: bool = False):
    """Run sync/rebuild in background thread and update task status."""
    key = _task_key(project_id, pool)
    try:
        _tasks[key] = {"status": "running", "progress": "Starting...", "result": None}
        with _sync_lock:
            if rebuild:
                _tasks[key]["progress"] = "Clearing old index..."
                project_dir = get_project_dir(project_id)
                store_dir = project_dir / f"faiss_store_{pool}"
                if store_dir.exists():
                    shutil.rmtree(store_dir)
                mpath = manifest_path(project_dir, pool)
                if mpath.exists():
                    mpath.unlink()
                store_dir.mkdir(parents=True, exist_ok=True)
                sk = f"{project_id}_{pool}"
                _stores[sk] = FaissStore(store_dir)
            _tasks[key]["progress"] = "Scanning documents..."
            result = smart_sync(project_id, pool, progress_cb=lambda msg: _tasks.get(key, {}).__setitem__("progress", msg))
        _tasks[key] = {"status": "done", "progress": "Complete", "result": result}
    except Exception as e:
        _tasks[key] = {"status": "error", "progress": str(e), "result": None}


def get_store(project_id: str, pool: str) -> FaissStore:
    key = f"{project_id}_{pool}"
    if key not in _stores:
        store_dir = get_project_dir(project_id) / f"faiss_store_{pool}"
        _stores[key] = FaissStore(store_dir)
    return _stores[key]


# ── Store helpers ───────────────────────────────────────────────────────────


def upsert_chunks(store: FaissStore, chunks: list[dict[str, Any]]) -> None:
    if not chunks:
        return
    texts = [c["document"] for c in chunks]
    embeddings = embed_texts(texts)
    batch = 256
    for i in range(0, len(chunks), batch):
        sl = chunks[i: i + batch]
        store.upsert(
            ids=[c["id"] for c in sl],
            documents=[c["document"] for c in sl],
            metadatas=[c["metadata"] for c in sl],
            embeddings=embeddings[i: i + batch],
        )


# ── Smart sync (per pool) ────────────────────────────────────────────────


SUPPORTED_EXTENSIONS = {".pdf", ".xlsx", ".xls"}


def discover_docs(docs_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    docs: dict[str, pathlib.Path] = {}
    if not docs_dir.exists():
        return docs
    for ext in SUPPORTED_EXTENSIONS:
        for p in sorted(docs_dir.rglob(f"*{ext}")):
            docs[str(p.resolve())] = p
    return docs


def smart_sync(project_id: str, pool: str, progress_cb=None) -> dict:
    """Run smart sync for a project pool and return a status dict."""
    project_dir = get_project_dir(project_id)
    docs_dir = project_dir / f"documents_{pool}"
    store = get_store(project_id, pool)
    manifest = load_manifest(project_dir, pool)
    disk_docs = discover_docs(docs_dir)
    log: list[str] = []

    # Removals
    removed_keys = [k for k in manifest if k not in disk_docs]
    for key in removed_keys:
        entry = manifest.pop(key)
        ids = entry.get("chunk_ids", [])
        if ids:
            store.delete(ids)
            log.append(f"Removed: {entry['filename']} ({len(ids)} chunks)")

    # Additions and updates
    for abs_path, doc_path in disk_docs.items():
        current_hash = sha256_file(doc_path)
        existing = manifest.get(abs_path)

        if existing and existing.get("sha256") == current_hash:
            continue

        if existing:
            old_ids = existing.get("chunk_ids", [])
            if old_ids:
                store.delete(old_ids)
            log.append(f"Re-indexed: {doc_path.name}")
        else:
            log.append(f"Indexed: {doc_path.name}")

        if progress_cb:
            progress_cb(f"Processing: {doc_path.name}")
        chunks = build_chunks_for_file(doc_path, current_hash)
        if progress_cb:
            progress_cb(f"Embedding: {doc_path.name} ({len(chunks)} chunks)")
        upsert_chunks(store, chunks)

        manifest[abs_path] = {
            "path": abs_path,
            "filename": doc_path.name,
            "sha256": current_hash,
            "chunk_ids": [c["id"] for c in chunks],
        }

    save_manifest(project_dir, pool, manifest)
    return {
        "files": len(manifest),
        "chunks": store.count(),
        "log": log,
    }


def full_rebuild(project_id: str, pool: str) -> dict:
    """Full rebuild — delete all embeddings and re-index for a pool."""
    project_dir = get_project_dir(project_id)
    store_dir = project_dir / f"faiss_store_{pool}"
    if store_dir.exists():
        shutil.rmtree(store_dir)
    mpath = manifest_path(project_dir, pool)
    if mpath.exists():
        mpath.unlink()
    store_dir.mkdir(parents=True, exist_ok=True)
    key = f"{project_id}_{pool}"
    _stores[key] = FaissStore(store_dir)
    return smart_sync(project_id, pool)


# ── Prompt builder ──────────────────────────────────────────────────────────


def build_user_message(snippets: list[dict[str, Any]], question: str) -> str:
    parts = ["Context snippets:"]
    for i, s in enumerate(snippets, 1):
        meta = s["metadata"]
        parts.append(f"[{i}] (File: {meta['file']}, {meta['page']})")
        parts.append(s["document"])
        parts.append("")
    parts.append(f"User question:\n{question}")
    return "\n".join(parts)


# ── Answer pipeline ─────────────────────────────────────────────────────────


def answer_question(question: str, project_id: str, pool: str) -> tuple[str, list[dict[str, Any]]]:
    st = get_store(project_id, pool)
    if st.count() == 0:
        pool_label = "main documents" if pool == POOL_MAIN else "clarification documents"
        return f"No {pool_label} indexed yet. Please upload files and click Sync.", []

    q_emb = embed_texts([question])
    raw = st.query(query_embedding=q_emb[0], top_k=TOP_K_RETRIEVE)

    docs = raw["documents"]
    metas = raw["metadatas"]
    dists = raw["distances"]

    if not docs:
        return "Not found in provided documents.", []

    snippets = []
    for doc, meta, dist in zip(docs[:TOP_K_FINAL], metas[:TOP_K_FINAL], dists[:TOP_K_FINAL]):
        snippets.append({"document": doc, "metadata": meta, "distance": dist})

    user_msg = build_user_message(snippets, question)
    system_prompt = SYSTEM_PROMPT_MAIN if pool == POOL_MAIN else SYSTEM_PROMPT_CLARIFICATION
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
    answer = chat_completion(messages)
    sources = []
    seen = set()
    for s in snippets:
        key = (s["metadata"]["file"], s["metadata"]["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"file": s["metadata"]["file"], "page": s["metadata"]["page"]})
    return answer, sources


# ══════════════════════════════════════════════════════════════════════════════
# Flask routes
# ══════════════════════════════════════════════════════════════════════════════


@app.route("/")
def index():
    return render_template("index.html")


# ── Project CRUD ─────────────────────────────────────────────────────────────


@app.route("/api/projects", methods=["GET"])
def list_projects():
    return jsonify({"projects": get_projects_list()})


@app.route("/api/projects", methods=["POST"])
def create_project_route():
    data = request.get_json()
    name = data.get("name", "").strip()
    description = data.get("description", "").strip()
    if not name:
        return jsonify({"error": "Project name is required"}), 400
    meta = create_project(name, description)
    return jsonify(meta)


@app.route("/api/projects/<project_id>", methods=["DELETE"])
def delete_project_route(project_id):
    if delete_project(project_id):
        return jsonify({"deleted": project_id})
    return jsonify({"error": "Project not found"}), 404


@app.route("/api/projects/<project_id>/status", methods=["GET"])
def project_status(project_id):
    project_dir = get_project_dir(project_id)
    if not project_dir.exists():
        return jsonify({"error": "Project not found"}), 404

    update_last_accessed(project_id)

    pool_status = {}
    for pool in POOL_TYPES:
        manifest = load_manifest(project_dir, pool)
        st = get_store(project_id, pool)
        docs_dir = project_dir / f"documents_{pool}"

        files_list = []
        if docs_dir.exists():
            for f in sorted(docs_dir.iterdir()):
                if f.suffix.lower() in SUPPORTED_EXTENSIONS:
                    entry = next((e for e in manifest.values() if e.get("filename") == f.name), None)
                    chunks = len(entry.get("chunk_ids", [])) if entry else 0
                    indexed = entry is not None
                    files_list.append({
                        "filename": f.name,
                        "size": f.stat().st_size,
                        "chunks": chunks,
                        "indexed": indexed,
                        "type": f.suffix.lower().replace(".", "").upper(),
                    })

        pool_status[pool] = {
            "files": len(manifest),
            "chunks": st.count(),
            "files_list": files_list,
        }

    # Load project meta
    meta_path = project_dir / "project_meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return jsonify({
        "project": meta,
        "pools": pool_status,
        "embed_model": EMBED_MODEL,
        "chat_model": CHAT_MODEL,
    })


# ── Document management (per pool) ──────────────────────────────────────────


@app.route("/api/projects/<project_id>/<pool>/upload", methods=["POST"])
def upload_files(project_id, pool):
    if pool not in POOL_TYPES:
        return jsonify({"error": f"Invalid pool: {pool}"}), 400
    project_dir = get_project_dir(project_id)
    if not project_dir.exists():
        return jsonify({"error": "Project not found"}), 404

    docs_dir = project_dir / f"documents_{pool}"
    docs_dir.mkdir(parents=True, exist_ok=True)

    files = request.files.getlist("files")
    uploaded = []
    for f in files:
        if f.filename:
            ext = pathlib.Path(f.filename).suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                save_path = docs_dir / f.filename
                f.save(str(save_path))
                uploaded.append(f.filename)
    return jsonify({"uploaded": uploaded, "count": len(uploaded)})


@app.route("/api/projects/<project_id>/<pool>/delete_file", methods=["POST"])
def delete_file(project_id, pool):
    if pool not in POOL_TYPES:
        return jsonify({"error": f"Invalid pool: {pool}"}), 400
    project_dir = get_project_dir(project_id)
    data = request.get_json()
    filename = data.get("filename", "")
    filepath = project_dir / f"documents_{pool}" / filename
    if filepath.exists() and filepath.suffix.lower() in SUPPORTED_EXTENSIONS:
        filepath.unlink()
        return jsonify({"deleted": filename})
    return jsonify({"error": "File not found"}), 404


# ── Sync (per pool) ─────────────────────────────────────────────────────────


@app.route("/api/projects/<project_id>/<pool>/sync", methods=["POST"])
def sync_route(project_id, pool):
    if pool not in POOL_TYPES:
        return jsonify({"error": f"Invalid pool: {pool}"}), 400
    project_dir = get_project_dir(project_id)
    if not project_dir.exists():
        return jsonify({"error": "Project not found"}), 404
    key = _task_key(project_id, pool)
    existing = _tasks.get(key, {})
    if existing.get("status") == "running":
        return jsonify({"error": "Sync already in progress"}), 409
    t = threading.Thread(target=_run_sync_bg, args=(project_id, pool, False), daemon=True)
    t.start()
    return jsonify({"started": True, "task_key": key})


@app.route("/api/projects/<project_id>/<pool>/rebuild", methods=["POST"])
def rebuild_route(project_id, pool):
    if pool not in POOL_TYPES:
        return jsonify({"error": f"Invalid pool: {pool}"}), 400
    project_dir = get_project_dir(project_id)
    if not project_dir.exists():
        return jsonify({"error": "Project not found"}), 404
    key = _task_key(project_id, pool)
    existing = _tasks.get(key, {})
    if existing.get("status") == "running":
        return jsonify({"error": "Sync already in progress"}), 409
    t = threading.Thread(target=_run_sync_bg, args=(project_id, pool, True), daemon=True)
    t.start()
    return jsonify({"started": True, "task_key": key})


@app.route("/api/projects/<project_id>/<pool>/task_status", methods=["GET"])
def task_status_route(project_id, pool):
    key = _task_key(project_id, pool)
    task = _tasks.get(key, {"status": "idle", "progress": "", "result": None})
    return jsonify(task)


# ── Chat (per pool) ─────────────────────────────────────────────────────────


@app.route("/api/projects/<project_id>/<pool>/chat", methods=["POST"])
def chat_route(project_id, pool):
    if pool not in POOL_TYPES:
        return jsonify({"error": f"Invalid pool: {pool}"}), 400
    project_dir = get_project_dir(project_id)
    if not project_dir.exists():
        return jsonify({"error": "Project not found"}), 404

    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    update_last_accessed(project_id)
    try:
        answer, sources = answer_question(question, project_id, pool)
        return jsonify({"answer": answer, "sources": sources})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    get_client()
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"\n  Project RAG Pro — Dual Pool Edition")
    print(f"  ─────────────────────────────────────")
    print(f"  ✓ OpenAI API key found")
    print(f"  ✓ Embed model: {EMBED_MODEL}")
    print(f"  ✓ Chat model:  {CHAT_MODEL}")
    print(f"  ✓ Projects in: {PROJECTS_ROOT}")
    print(f"  ✓ Pools: Main Documents + Clarification Documents")
    print(f"\n  Open http://localhost:8080 in your browser.\n")
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
