"""
ingest.py
---------
Handles the ingestion (indexing) side of the RAG pipeline:

  PDF(s) -> extract text page-by-page -> split into chunks
         -> embed chunks (OpenAI) -> store in ChromaDB (persisted to disk)

Design notes:
- We keep one Document per PDF *page* first (so we know which page each
  piece of text came from), then split those page-documents into smaller
  overlapping chunks. Because LangChain's text splitter preserves the
  metadata of the document it split, every resulting chunk still carries
  the correct 'source' (filename) and 'page' number.
- Re-running ingestion on the same PDF will not create duplicate chunks:
  each chunk gets a deterministic ID (a hash of its source, page, position,
  and text), and we skip any ID that is already present in the collection.
"""

import os
import hashlib
from typing import List, Dict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# --- Configuration -----------------------------------------------------

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "finance_reports"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

EMBEDDING_MODEL = "text-embedding-3-small"


# --- Vector store helpers ------------------------------------------------

def get_embeddings() -> OpenAIEmbeddings:
    """Create the OpenAI embeddings client used for both indexing and querying."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def get_vectorstore() -> Chroma:
    """
    Get (or create) the persistent Chroma vector store.

    Using a persist_directory means the collection is written to disk under
    ./chroma_db and will still be there the next time the app starts.
    """
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


# --- PDF loading -----------------------------------------------------------

def load_pdf_pages(pdf_path: str) -> List:
    """
    Load a single PDF and return one LangChain Document per page.

    PyPDFLoader extracts selectable text only (no OCR), which matches the
    assumption that the quarterly report PDFs contain real text, not scans.
    """
    filename = os.path.basename(pdf_path)
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()  # one Document per page

    for doc in pages:
        # PyPDFLoader's default page metadata is 0-indexed; we convert to a
        # human-friendly 1-indexed page number and normalize the filename
        # (so uploaded files are identified by name, not by a temp path).
        raw_page = doc.metadata.get("page", 0)
        doc.metadata["source"] = filename
        doc.metadata["page"] = int(raw_page) + 1

    return pages


def chunk_pages(pages: List) -> List:
    """Split page-level documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(pages)


def _make_chunk_id(source: str, page: int, chunk_index: int, text: str) -> str:
    """
    Build a stable, deterministic ID for a chunk.

    Because the ID is derived from the chunk's content and position (not a
    random UUID), re-indexing the same PDF produces the same IDs, which lets
    us detect and skip chunks that are already stored.
    """
    raw = f"{source}|{page}|{chunk_index}|{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# --- Main ingestion entry point --------------------------------------------

def ingest_pdfs(pdf_paths: List[str]) -> Dict:
    """
    Full ingestion pipeline for one or more PDF files.

    Steps:
      1. Read each PDF page by page (preserving filename + page number).
      2. Split pages into 1200-character chunks with 200-character overlap.
      3. Embed chunks using OpenAI text-embedding-3-small.
      4. Store chunks + embeddings in the persistent ChromaDB collection,
         skipping any chunk that is already indexed.

    Returns a summary dict, e.g.:
        {
            "files_processed": 3,
            "total_pages": 42,
            "new_chunks_stored": 214,
            "skipped_files": [],
        }
    """
    if not pdf_paths:
        raise ValueError("No PDF files were provided for ingestion.")

    vectorstore = get_vectorstore()

    # Existing chunk IDs already stored, so we don't index the same content twice.
    existing = vectorstore.get(include=[])
    existing_ids = set(existing.get("ids", []))

    total_pages = 0
    new_chunks = []
    new_ids = []
    files_processed = 0
    skipped_files = []  # list of (filename, reason)

    for pdf_path in pdf_paths:
        source = os.path.basename(pdf_path)

        try:
            pages = load_pdf_pages(pdf_path)
        except Exception as e:
            skipped_files.append((source, f"Could not read PDF ({e})."))
            continue

        if not pages or not any(p.page_content.strip() for p in pages):
            skipped_files.append(
                (source, "No extractable text found (empty file or scanned/image-only PDF).")
            )
            continue

        total_pages += len(pages)
        chunks = chunk_pages(pages)

        # Track how many chunks we've seen per page, for deterministic IDs.
        page_chunk_counters: Dict[int, int] = {}

        for chunk in chunks:
            page = chunk.metadata.get("page", 0)
            idx = page_chunk_counters.get(page, 0)
            page_chunk_counters[page] = idx + 1

            chunk_id = _make_chunk_id(source, page, idx, chunk.page_content)
            if chunk_id in existing_ids:
                continue  # already indexed in a previous run

            new_chunks.append(chunk)
            new_ids.append(chunk_id)
            existing_ids.add(chunk_id)

        files_processed += 1

    if new_chunks:
        vectorstore.add_documents(documents=new_chunks, ids=new_ids)

    return {
        "files_processed": files_processed,
        "total_pages": total_pages,
        "new_chunks_stored": len(new_chunks),
        "skipped_files": skipped_files,
    }
