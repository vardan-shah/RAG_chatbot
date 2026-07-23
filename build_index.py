# ── Mac SQLite compatibility patch ────────────────────────────────────────
# Must be the very first lines — before any other import.
# ChromaDB requires SQLite >= 3.35. Some macOS Python installs use the
# system SQLite (3.30) which is too old. This swaps in pysqlite3-binary's
# modern SQLite. Safe on all platforms — silently skipped if not installed.
try:
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import sys
import time

from src.ingest import load_all_documents, chunk_documents
from src.retriever import get_embeddings, build_vector_store


def main() -> None:
    total_start = time.time()

    print("=" * 58)
    print("  🌍  WORLD CUP RAG — Building Knowledge Index")
    print("=" * 58)

    print("\n📄 Step 1 of 4 — Scraping Wikipedia pages")
    print("  (polite 1.2 s delay between requests)")
    documents = load_all_documents()

    if not documents:
        print("\n❌ No documents loaded. Check your internet connection.")
        sys.exit(1)

    print("\n✂️  Step 2 of 4 — Splitting into chunks")
    chunks = chunk_documents(documents)

    print("\n🔢 Step 3 of 4 — Loading embedding model")
    print("  sentence-transformers/all-MiniLM-L6-v2")
    print("  (first run downloads ~90 MB — cached after)")
    embeddings = get_embeddings()

    print("\n💾 Step 4 of 4 — Embedding chunks and writing to ChromaDB")
    build_vector_store(chunks, embeddings)

    elapsed = time.time() - total_start
    minutes, seconds = divmod(int(elapsed), 60)

    print(f"\n{'='*58}")
    print(f"  ✅  Index built in {minutes}m {seconds}s")
    print(f"  📦  {len(chunks)} chunks stored in chroma_db/")
    print(f"\n  Next step: streamlit run app.py")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
