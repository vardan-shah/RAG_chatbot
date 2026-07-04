"""
ChromaDB vector store and retriever — with Apple Silicon MPS support.

On M5 MacBook: uses the built-in GPU (MPS backend) for embedding,
making build_index.py ~3x faster than CPU.
Falls back to CPU automatically if MPS is unavailable.
"""

import os
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR      = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _get_device() -> str:
    """
    Detect the best available compute device.

    Priority: MPS (Apple Silicon GPU) > CPU
    MPS is available on M1/M2/M3/M4/M5 Macs with PyTorch >= 1.13.
    On any non-Mac or older PyTorch, falls back to CPU silently.
    """
    if torch.backends.mps.is_available():
        print("  Using Apple MPS (GPU) for embeddings")
        return "mps"
    print("  Using CPU for embeddings")
    return "cpu"


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load MiniLM-L6-v2 embedding model on the best available device.

    On M5: MPS backend gives ~3x speedup during build_index.py.
    At query time (app.py): single-query embedding is fast on either device.
    First call downloads ~90 MB model — cached in ~/.cache/huggingface/ after.
    """
    device = _get_device()
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(chunks: list, embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Embed all chunks and persist to ChromaDB.
    Called once by build_index.py — never called by app.py.
    """
    os.makedirs(CHROMA_DIR, exist_ok=True)

    print(f"  Embedding {len(chunks)} chunks → {CHROMA_DIR}/")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    count = vector_store._collection.count()
    print(f"  ChromaDB contains {count} vectors")
    return vector_store


def get_retriever(k: int = 6):
    """
    Load existing ChromaDB from disk and return a configured retriever.
    Called by src/chain.py on every app session start.

    Raises FileNotFoundError if build_index.py has not been run yet.
    """
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"Vector store not found at '{CHROMA_DIR}/'. "
            "Run 'python build_index.py' first."
        )

    embeddings = get_embeddings()

    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    return vector_store.as_retriever(search_kwargs={"k": k})
