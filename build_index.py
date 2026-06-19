"""
Build the World Cup knowledge index — run this ONCE before launching the app.

What this script does:
  1. Scrapes 26 Wikipedia pages (tournaments 1930–2022 + key players)
  2. Cleans and splits the text into 512-token chunks
  3. Embeds every chunk using MiniLM-L6-v2 (runs locally, no API cost)
  4. Persists the vectors to ChromaDB at ./chroma_db/

Expected runtime: 4–8 minutes total (scraping ~45 sec, embedding ~3–5 min)
After this runs successfully, just do: streamlit run app.py

Usage:
    python build_index.py

Re-running will overwrite the existing index. If you add new pages to
src/ingest.WIKIPEDIA_PAGES, re-run this script to include them.
"""

import sys
import time

from src.ingest import load_all_documents, chunk_documents
from src.retriever import get_embeddings, build_vector_store


def main() -> None:
    total_start = time.time()

    print("=" * 58)
    print("  🌍  WORLD CUP RAG — Building Knowledge Index")
    print("=" * 58)

    # ── Step 1: Scrape Wikipedia ─────────────────────────────────
    print("\n📄 Step 1 of 4 — Scraping Wikipedia pages")
    print("  (polite 1.2 s delay between requests)")
    documents = load_all_documents()

    if not documents:
        print("\n❌ No documents loaded. Check internet connection.")
        sys.exit(1)

    # ── Step 2: Chunk ────────────────────────────────────────────
    print("\n✂️  Step 2 of 4 — Splitting into chunks")
    chunks = chunk_documents(documents)

    # ── Step 3: Load embedding model ─────────────────────────────
    print("\n🔢 Step 3 of 4 — Loading embedding model")
    print("  sentence-transformers/all-MiniLM-L6-v2")
    print("  (first run downloads ~90 MB — cached after)")
    embeddings = get_embeddings()

    # ── Step 4: Build ChromaDB ───────────────────────────────────
    print("\n💾 Step 4 of 4 — Embedding chunks and writing to ChromaDB")
    build_vector_store(chunks, embeddings)

    elapsed = time.time() - total_start
    minutes, seconds = divmod(int(elapsed), 60)

    print(f"\n{'='*58}")
    print(f"  ✅  Index built in {minutes}m {seconds}s")
    print(f"  📦  {len(chunks)} chunks stored in chroma_db/")
    print(f"\n  Next step:")
    print(f"    streamlit run app.py")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
