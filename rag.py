"""
rag.py
------
Handles the question-answering side of the RAG pipeline:

  question -> embed -> similarity search in ChromaDB (top 4 chunks)
           -> build context -> GPT-4o -> answer + sources

This is intentionally a simple, single-pass RAG pipeline: no agents,
no re-ranking, no hybrid search, no memory. Just retrieve, then answer.
"""

from typing import List, Dict

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from ingest import get_vectorstore

# --- Configuration -----------------------------------------------------

TOP_K = 4
MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.1

# The system prompt is the main guardrail against hallucination: the model
# is told, explicitly, to answer only from the provided context and to say
# so plainly when the answer isn't there.
SYSTEM_PROMPT = """You are a financial research assistant.

Answer the user's question ONLY using the information contained in the provided context from the uploaded quarterly financial reports.

Do not use outside knowledge.

If the answer is not contained in the provided context, clearly say:
"The information is not available in the uploaded documents."

Do not guess, estimate, fabricate, or infer unsupported financial figures.

When answering numerical questions, preserve the units and values from the source documents.

Use the retrieved context as the only source of truth."""


def get_llm() -> ChatGroq:
    """Create the Groq chat client used to generate answers."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=TEMPERATURE
    )

def retrieve_context(question: str, k: int = TOP_K) -> List:
    """Embed the question and run a similarity search against ChromaDB."""
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search(question, k=k)


def build_context_block(chunks: List) -> str:
    """
    Format retrieved chunks into a single context string for the LLM,
    tagging each snippet with its source filename and page number so the
    model's citations line up with what was actually retrieved.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "?")
        parts.append(f"[Chunk {i} | Source: {source} | Page: {page}]\n{chunk.page_content}")
    return "\n\n".join(parts)


def extract_sources(chunks: List) -> List[Dict]:
    """
    Build a de-duplicated (source, page) list from the retrieved chunks'
    ChromaDB metadata. Page numbers are never invented here — they come
    directly from what was stored during ingestion.
    """
    seen = set()
    sources = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "?")
        key = (source, page)
        if key not in seen:
            seen.add(key)
            sources.append({"source": source, "page": page})
    return sources


def answer_question(question: str) -> Dict:
    """
    Full RAG query pipeline for a single question.

    Returns:
        {
            "answer": str,
            "sources": [{"source": str, "page": int}, ...],
            "context": str,   # raw retrieved context, for the debug expander
        }
    """
    chunks = retrieve_context(question)

    if not chunks:
        return {
            "answer": "The information is not available in the uploaded documents.",
            "sources": [],
            "context": "",
        }

    context = build_context_block(chunks)
    sources = extract_sources(chunks)

    llm = get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
    ]

    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "sources": sources,
        "context": context,
    }
