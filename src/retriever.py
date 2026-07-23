"""
ChromaDB vector store and retriever — Mac and Windows compatible.

Why CPU and not MPS:
  torch.backends.mps.is_available() returns True on M-series Macs, but
  langchain-huggingface's HuggingFaceEmbeddings has a known incompatibility
  with the MPS device string on certain versions of sentence-transformers.
  It causes silent failures or crashes that do not occur on Windows (where
  MPS is never available). Forcing CPU gives identical results on all
  platforms and eliminates the Mac-specific bug entirely.
  The embedding step is fast enough on CPU — ~4 min for 1,500 chunks.
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load MiniLM-L6-v2 on CPU.

    CPU is used explicitly because MPS (Apple Silicon GPU) causes
    compatibility issues with langchain-huggingface on some versions.
    First call downloads ~90 MB to ~/.cache/huggingface/ — cached after.
    """
    print("  Device: CPU (cross-platform compatible)")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(chunks: list, embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Embed all chunks and persist to ChromaDB.
    Called once by build_index.py — never by app.py.
    """
    os.makedirs(CHROMA_DIR, exist_ok=True)

    print(f"  Embedding {len(chunks)} chunks → {CHROMA_DIR}/")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    count = vector_store._collection.count()
    print(f"  ChromaDB ready: {count} vectors stored")
    return vector_store


def get_retriever(k: int = 6):
    """
    Load ChromaDB from disk and return a configured retriever.
    Called by src/chain.py on every app session start.

    Raises FileNotFoundError if build_index.py has not been run yet.
    chroma_db/ is in .gitignore — it does not exist on a fresh clone.
    You must run build_index.py once on every new machine.
    """
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            "\n\n  chroma_db/ not found. This directory is not in git.\n"
            "  Run this once to build the index:\n\n"
            "    python build_index.py\n\n"
            "  Expected runtime: 4–8 minutes.\n"
        )

    embeddings = get_embeddings()

    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    return vector_store.as_retriever(search_kwargs={"k": k})
