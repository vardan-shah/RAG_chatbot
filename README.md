# 🏆 World Cup RAG Chatbot

> **A full Retrieval-Augmented Generation (RAG) pipeline** that answers natural language questions about FIFA World Cup history (1930–2022) using a local vector knowledge base, history-aware retrieval, and a free LLM via Groq.


---

## What this project demonstrates

| Skill Area | What was built |
|---|---|
| **LLM Engineering** | Full RAG pipeline using LangChain LCEL — not just an API wrapper |
| **Vector Databases** | ChromaDB with local MiniLM-L6-v2 embeddings (no paid embedding API) |
| **Multi-turn Conversation** | History-aware retrieval that rewrites follow-up questions before searching |
| **Data Engineering** | Web scraping pipeline across 26 Wikipedia pages with clean chunking |
| **Prompt Engineering** | Anti-hallucination system prompts tuned for Llama's instruction-following behaviour |
| **Full-Stack** | Streamlit app with source citations, sidebar Q&A, conversation memory |

---

## Architecture

```
─── BUILD INDEX (run once) ────────────────────────────────────────────

  Wikipedia (26 pages)
      │ BeautifulSoup scraper · 1.2s polite delay
      ▼
  Text sections → RecursiveCharacterTextSplitter (512 tok / 50 overlap)
      │
      ▼
  MiniLM-L6-v2 embeddings (sentence-transformers · runs locally on CPU)
      │
      ▼
  ChromaDB  ──── persisted to chroma_db/ ────────────────────────────

─── QUERY PIPELINE (every conversation turn) ──────────────────────────

  User question + chat history
      │
      ▼
  create_history_aware_retriever
  (Llama rewrites "How old was he?" → "How old was Miroslav Klose in 2006?")
      │
      ▼
  ChromaDB similarity search (top-6 chunks)
      │
      ▼
  create_stuff_documents_chain
  (retrieved context injected into QA prompt)
      │
      ▼
  Llama-3.3-70b via Groq API  →  Answer + source citations
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | `llama-3.3-70b-versatile` via Groq free tier |
| **Orchestration** | LangChain v0.3 (LCEL) — `create_history_aware_retriever`, `create_retrieval_chain` |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (local, CPU, no API cost) |
| **Vector Store** | ChromaDB with persistent local storage |
| **Data Ingestion** | `requests` + `BeautifulSoup4` web scraper |
| **Text Splitting** | `RecursiveCharacterTextSplitter` (512 tokens, 50 overlap) |
| **Frontend** | Streamlit with `st.chat_message`, `st.cache_resource` |
| **Environment** | `python-dotenv` |

---

## Knowledge Base

The `chroma_db/` vector store indexes **26 Wikipedia pages** (~1,500 chunks):

- **3 overview pages** — FIFA World Cup, Records & Statistics, All-Time Top Scorers
- **18 tournament pages** — every World Cup from 1930 to 2022
- **8 player pages** — Pelé, Maradona, Ronaldo (Brazil), Klose, Messi, Mbappé, Zidane, Ronaldo (Portugal)

---

## Project Structure

```
rag_chatbot/
├── src/
│   ├── __init__.py
│   ├── ingest.py         # Wikipedia scraper → LangChain Documents → chunks
│   ├── retriever.py      # ChromaDB build + load + retriever configuration
│   ├── prompts.py        # System prompts (QA + question condensation)
│   └── chain.py          # Full LangChain LCEL RAG pipeline
├── build_index.py        # One-time script: scrape → embed → persist ChromaDB
├── app.py                # Streamlit chat interface
├── requirements.txt
├── .env.example
└── .gitignore            # Excludes .env, chroma_db/, venv/
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- A free Groq API key from [console.groq.com/keys](https://console.groq.com/keys)

### 1. Clone and install

```bash
git clone https://github.com/vardan-shah/RAG_chatbot.git
cd RAG_chatbot
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and add your Groq key:

```
GROQ_API_KEY=gsk_your_key_here
```

### 3. Build the knowledge index

**Run this once.** It scrapes 26 Wikipedia pages, embeds ~1,500 chunks using MiniLM-L6-v2, and persists the vector store to `chroma_db/`.

```bash
python build_index.py
```

Expected runtime: **4–8 minutes** (scraping ~45 sec + embedding ~4 min on CPU).

### 4. Launch the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Example Questions

```
"What happened in the 1970 World Cup final?"
"Tell me about Maradona's Hand of God goal"
"Which countries have won the World Cup more than once?"
"How many World Cups did Pelé win, and how many goals did he score?"
"What records did Miroslav Klose set?"
"Describe the 2022 World Cup final between Argentina and France"
```

The chatbot handles **multi-turn conversations**. After asking "Who scored in the 1986 final?", you can follow up with "How old was he at the time?" and the system correctly rewrites the second question into a standalone query before searching the vector store.

---

## Design Decisions

**Why local embeddings instead of OpenAI?**
`sentence-transformers/all-MiniLM-L6-v2` produces 384-dimensional vectors, runs on CPU in under a second per batch, costs nothing, and achieves excellent retrieval precision for factual document matching. The quality difference from paid embedding APIs is negligible for this use case.

**Why `create_history_aware_retriever`?**
The naive approach — sending the raw user question directly to ChromaDB — breaks for follow-up questions that use pronouns ("he", "that tournament", "they"). The history-aware retriever calls the LLM first to rewrite the question into a self-contained string before retrieval. This is the difference between a search box and a real conversational system.

**Why strict prohibition language in the system prompt?**
`llama-3.3-70b-versatile` treats soft instructions ("say so if you don't know") as suggestions. Under thin retrieval context it defaults to training data, producing confident but incorrect answers. Explicit NEVER/MUST language enforces the context boundary correctly.

**Why `temperature=0.0`?**
Factual historical Q&A benefits from maximum determinism. Variability introduces hallucination risk on specific statistics like scorelines and goal tallies.

---

## Known Limitations

- Knowledge cutoff: Wikipedia pages were scraped at index build time. No live data.
- 26 pages do not cover every group stage match or every player — some specific queries will return "I don't have that information."
- Groq free tier: 500 requests/day, 14,400 tokens/minute.
- Llama-3.3 occasionally supplements thin context with training knowledge despite strict prompting — source citations reveal when this happens.

---

## Part of a larger portfolio

This is **Project 4** in a FIFA World Cup analytics series:

| # | Project | Key Technologies |
|---|---|---|
| 1 | Match Outcome Predictor | Logistic Regression, XGBoost, FastAPI |
| 2 | xG Engine & AI Scout | StatsBomb, XGBoost, SHAP, Anthropic API, Streamlit |
| 3 | Penalty Shootout Predictor | PyTorch, D3.js, Flask |
| **4** | **World Cup RAG Chatbot** | **LangChain, ChromaDB, Groq, Llama-3.3** |

---

## Author

**Vardan Shah** — BTech Computer Science, DAIICT  
[GitHub](https://github.com/vardan-shah)
