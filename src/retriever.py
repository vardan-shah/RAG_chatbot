import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR     = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load the sentence-transformers embedding model.
    Runs locally on CPU — no API key, no cost.
    First call downloads ~90 MB and caches it.
    Called by both build_index.py (to embed chunks) and
    get_retriever() (to embed queries at search time).
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(chunks: list, embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Embed all document chunks and persist them to ChromaDB on disk.
    Called once by build_index.py.

    Args:
        chunks:     Output of ingest.chunk_documents()
        embeddings: Output of get_embeddings()
    """
    os.makedirs(CHROMA_DIR, exist_ok=True)

    print(f"  Embedding {len(chunks)} chunks → {CHROMA_DIR}/")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    count = vector_store._collection.count()
    print(f"  ✅ ChromaDB contains {count} vectors")
    return vector_store


def get_retriever(k: int = 5):
    """
    Load an existing ChromaDB store from disk and return a retriever.
    Called by chain.py on every app session start.

    Raises FileNotFoundError if build_index.py has not been run yet.
    """
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"Vector store not found at '{CHROMA_DIR}/'. "
            "Run 'python build_index.py' first to build the index."
        )

    embeddings = get_embeddings()

    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    return vector_store.as_retriever(search_kwargs={"k": k})
