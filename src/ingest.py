"""
Data ingestion pipeline for the World Cup RAG Chatbot.

Responsibilities:
  - Scrape Wikipedia pages for each World Cup tournament and key players
  - Convert raw HTML into clean LangChain Document objects
  - Split documents into fixed-size chunks for embedding

Run order: ingest.py is called by build_index.py — never run directly by app.py.
"""

import re
import time
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ── Wikipedia pages to index ───────────────────────────────────────────────
# Each entry: {url, topic}. Topic is stored as metadata for filtered search.
# Keep this list curated — quality > quantity. 20 pages → ~1,500 chunks.

WIKIPEDIA_PAGES = [
    # Overview and records
    {"url": "https://en.wikipedia.org/wiki/FIFA_World_Cup",
     "topic": "overview"},
    {"url": "https://en.wikipedia.org/wiki/FIFA_World_Cup_records_and_statistics",
     "topic": "records"},
    {"url": "https://en.wikipedia.org/wiki/List_of_FIFA_World_Cup_top_scorers",
     "topic": "top_scorers"},

    # Tournaments — every edition included for completeness
    {"url": "https://en.wikipedia.org/wiki/1930_FIFA_World_Cup", "topic": "tournament_1930"},
    {"url": "https://en.wikipedia.org/wiki/1950_FIFA_World_Cup", "topic": "tournament_1950"},
    {"url": "https://en.wikipedia.org/wiki/1958_FIFA_World_Cup", "topic": "tournament_1958"},
    {"url": "https://en.wikipedia.org/wiki/1966_FIFA_World_Cup", "topic": "tournament_1966"},
    {"url": "https://en.wikipedia.org/wiki/1970_FIFA_World_Cup", "topic": "tournament_1970"},
    {"url": "https://en.wikipedia.org/wiki/1978_FIFA_World_Cup", "topic": "tournament_1978"},
    {"url": "https://en.wikipedia.org/wiki/1982_FIFA_World_Cup", "topic": "tournament_1982"},
    {"url": "https://en.wikipedia.org/wiki/1986_FIFA_World_Cup", "topic": "tournament_1986"},
    {"url": "https://en.wikipedia.org/wiki/1990_FIFA_World_Cup", "topic": "tournament_1990"},
    {"url": "https://en.wikipedia.org/wiki/1994_FIFA_World_Cup", "topic": "tournament_1994"},
    {"url": "https://en.wikipedia.org/wiki/1998_FIFA_World_Cup", "topic": "tournament_1998"},
    {"url": "https://en.wikipedia.org/wiki/2002_FIFA_World_Cup", "topic": "tournament_2002"},
    {"url": "https://en.wikipedia.org/wiki/2006_FIFA_World_Cup", "topic": "tournament_2006"},
    {"url": "https://en.wikipedia.org/wiki/2010_FIFA_World_Cup", "topic": "tournament_2010"},
    {"url": "https://en.wikipedia.org/wiki/2014_FIFA_World_Cup", "topic": "tournament_2014"},
    {"url": "https://en.wikipedia.org/wiki/2018_FIFA_World_Cup", "topic": "tournament_2018"},
    {"url": "https://en.wikipedia.org/wiki/2022_FIFA_World_Cup", "topic": "tournament_2022"},

    # Key players — World Cup legends
    {"url": "https://en.wikipedia.org/wiki/Pel%C3%A9",                    "topic": "player_pele"},
    {"url": "https://en.wikipedia.org/wiki/Diego_Maradona",               "topic": "player_maradona"},
    {"url": "https://en.wikipedia.org/wiki/Ronaldo_(Brazilian_footballer)", "topic": "player_ronaldo"},
    {"url": "https://en.wikipedia.org/wiki/Miroslav_Klose",               "topic": "player_klose"},
    {"url": "https://en.wikipedia.org/wiki/Lionel_Messi",                 "topic": "player_messi"},
    {"url": "https://en.wikipedia.org/wiki/Kylian_Mbapp%C3%A9",           "topic": "player_mbappe"},
    {"url": "https://en.wikipedia.org/wiki/Zinedine_Zidane",              "topic": "player_zidane"},
    {"url": "https://en.wikipedia.org/wiki/Ronaldo_(Portuguese_footballer)", "topic": "player_ronaldo_portugal"},
]

# HTTP headers — identify ourselves politely to Wikipedia
_HEADERS = {
    "User-Agent": (
        "WorldCupRAGBot/1.0 (educational project; "
        "https://github.com/vardan-shah/rag_chatbot)"
    )
}

# Sections to skip — Wikipedia boilerplate that adds noise, not knowledge
_SKIP_SECTIONS = {
    "see also", "references", "external links", "notes", "further reading",
    "bibliography", "sources", "citations", "footnotes", "contents",
    "navigation menu", "retrieved from",
}


def _clean_text(text: str) -> str:
    """Strip Wikipedia artefacts from extracted text."""
    text = re.sub(r"\[\d+\]", "", text)        # Citation numbers [1], [23]
    text = re.sub(r"\[edit\]", "", text)        # Section edit links
    text = re.sub(r"\n{3,}", "\n\n", text)      # Collapse blank lines
    text = re.sub(r" {2,}", " ", text)          # Collapse spaces
    return text.strip()


def scrape_wikipedia(url: str, topic: str) -> list[Document]:
    """
    Scrape one Wikipedia page and return a list of Documents, one per section.

    Each Document's page_content is prefixed with the page title and section
    header so that even out-of-context chunks carry enough information for
    the retriever to match them correctly.

    Returns an empty list on network failure (build_index handles logging).
    """
    try:
        response = requests.get(url, headers=_HEADERS, timeout=20)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"    ⚠ Skipping {url}: {exc}")
        return []

    soup = BeautifulSoup(response.content, "html.parser")

    # Extract page title
    h1 = soup.find("h1", {"id": "firstHeading"})
    page_title = h1.get_text(strip=True) if h1 else "Unknown"

    # Main content area — skip sidebars, infoboxes, navigation
    content = soup.find("div", {"class": "mw-parser-output"})
    if content is None:
        return []

    documents: list[Document] = []
    current_section = "Introduction"
    current_paragraphs: list[str] = []

    def _flush() -> None:
        """Save accumulated paragraphs as one Document."""
        if not current_paragraphs:
            return
        body = _clean_text("\n".join(current_paragraphs))
        if len(body) < 80:          # Too short to be useful
            return
        documents.append(Document(
            page_content=f"[{page_title}] {current_section}\n\n{body}",
            metadata={
                "source":       url,
                "topic":        topic,
                "page_title":   page_title,
                "section":      current_section,
            },
        ))

    for element in content.find_all(["h2", "h3", "h4", "p", "li"], recursive=True):
        if element.name in ("h2", "h3", "h4"):
            _flush()
            current_paragraphs = []
            section_name = element.get_text(strip=True)
            # Skip noisy boilerplate sections
            if section_name.lower() in _SKIP_SECTIONS:
                current_section = "__skip__"
            else:
                current_section = section_name

        elif element.name in ("p", "li"):
            if current_section == "__skip__":
                continue
            text = element.get_text(strip=True)
            if text:
                current_paragraphs.append(text)

    _flush()  # Don't forget the final section
    return documents


def load_all_documents(pages: list[dict] | None = None,
                       delay: float = 1.2) -> list[Document]:
    """
    Scrape all configured Wikipedia pages with a polite crawl delay.

    Args:
        pages:  Override the default WIKIPEDIA_PAGES list.
        delay:  Seconds to wait between HTTP requests (be kind to Wikipedia).

    Returns:
        Flat list of LangChain Documents across all pages.
    """
    if pages is None:
        pages = WIKIPEDIA_PAGES

    all_docs: list[Document] = []

    for i, page in enumerate(tqdm(pages, desc="  Scraping Wikipedia")):
        docs = scrape_wikipedia(page["url"], page["topic"])
        all_docs.extend(docs)

        if i < len(pages) - 1:
            time.sleep(delay)

    print(f"\n  Loaded {len(all_docs)} document sections from {len(pages)} pages")
    return all_docs


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into smaller chunks suitable for embedding and retrieval.

    Optimized Values:
      chunk_size=1200   — Gives the retriever large, contextual paragraphs of data.
      chunk_overlap=200 — Provides a deep 30-40 word overlap buffer to prevent 
                          critical sentences from being split in half.
      separators        — Splits on double newline first (paragraph), then 
                          single newline, then sentence, then word.

    Returns:
        List of chunk Documents (same metadata as parent, page_content trimmed).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    avg_len = (
        sum(len(c.page_content) for c in chunks) // len(chunks)
        if chunks else 0
    )
    print(f"  Split into {len(chunks)} chunks (avg {avg_len} chars each)")
    return chunks