"""
app.py
------
Streamlit UI for the Finance RAG application.

Sidebar:  upload PDFs -> index them into ChromaDB
Main area: ask a question -> see the answer, its sources, and (optionally)
           the raw retrieved context used to generate it.
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load OPENAI_API_KEY (and any other vars) from a local .env file.
# The key is never hardcoded anywhere in this project.
load_dotenv()

from ingest import ingest_pdfs, get_vectorstore  # noqa: E402
from rag import answer_question  # noqa: E402

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(page_title="Finance RAG", layout="wide")

st.title("Finance RAG — Quarterly Financial Reports")
st.caption("Ask questions about quarterly financial reports using Retrieval-Augmented Generation.")

# --- Startup check: the app cannot do anything useful without an API key ---
if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "`OPENAI_API_KEY` is not set. Create a `.env` file in the project root "
        "(see `.env.example` for the format), add your OpenAI API key, and "
        "restart the app."
    )
    st.stop()


def has_indexed_documents() -> bool:
    """Check whether ChromaDB already contains any indexed chunks."""
    try:
        vectorstore = get_vectorstore()
        existing = vectorstore.get(include=[])
        return len(existing.get("ids", [])) > 0
    except Exception:
        # If ChromaDB itself is unavailable/corrupted, treat as "nothing indexed"
        # so the UI shows a clear message instead of crashing.
        return False


# --- Sidebar: upload + index -------------------------------------------

with st.sidebar:
    st.header("1. Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more quarterly financial report PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    st.header("2. Index Documents")
    index_clicked = st.button("Index Documents", use_container_width=True)

    if index_clicked:
        if not uploaded_files:
            st.warning("Please upload at least one PDF before indexing.")
        else:
            # Save uploaded files into data/ so they persist on disk too.
            saved_paths = []
            for f in uploaded_files:
                dest = Path(DATA_DIR) / f.name
                with open(dest, "wb") as out:
                    out.write(f.getbuffer())
                saved_paths.append(str(dest))

            with st.spinner("Processing PDFs, creating chunks, and generating embeddings..."):
                try:
                    result = ingest_pdfs(saved_paths)
                except Exception as e:
                    st.error(f"Indexing failed: {e}")
                    result = None

            if result:
                st.success(f"✓ {result['files_processed']} files processed")
                st.success(f"✓ {result['new_chunks_stored']} chunks stored")
                if result["skipped_files"]:
                    for fname, reason in result["skipped_files"]:
                        st.warning(f"Skipped '{fname}': {reason}")

    st.divider()
    st.caption(
        "You can also place PDFs directly inside the `data/` folder on disk "
        "and then upload them here to index them."
    )

# --- Main area: ask a question ------------------------------------------

st.header("Ask a Question")

question = st.text_input(
    "Question",
    placeholder="What was the total revenue in the most recent quarter?",
)
ask_clicked = st.button("Ask")

if ask_clicked:
    if not question.strip():
        st.warning("Please enter a question.")
    elif not has_indexed_documents():
        st.warning(
            "No documents have been indexed yet. Upload PDFs and click "
            "'Index Documents' in the sidebar first."
        )
    else:
        with st.spinner("Retrieving relevant context and generating answer..."):
            try:
                result = answer_question(question)
            except Exception as e:
                st.error(f"Something went wrong while generating the answer: {e}")
                result = None

        if result:
            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Sources")
            if result["sources"]:
                for s in result["sources"]:
                    st.markdown(f"- **{s['source']}** — Page {s['page']}")
            else:
                st.markdown("_No sources retrieved._")

            with st.expander("Retrieved Context"):
                st.text(result["context"] if result["context"] else "No context retrieved.")
