"""
World Cup RAG Chatbot — Streamlit interface.

Key changes from previous version:
  - Loading spinner added for cold-start chain initialization
  - Chain now returns a dict {"answer": ..., "context": [...]} not a string
  - Source citations shown in an expander below each answer
  - Uses @st.cache_resource for efficient chain loading
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="World Cup RAG Chatbot",
    page_icon="🏆",
    layout="centered",
)

# ── Guard rails ─────────────────────────────────────────────────────────────
if not os.getenv("GROQ_API_KEY"):
    st.error(
        "**GROQ_API_KEY not found.** "
        "Get a free key at [console.groq.com/keys](https://console.groq.com/keys) "
        "and add it to a `.env` file."
    )
    st.stop()

if not os.path.exists("chroma_db"):
    st.error(
        "**Knowledge base not found.** "
        "Run `python build_index.py` first to scrape Wikipedia and build the index."
    )
    st.code("python build_index.py")
    st.stop()


# ── Chain loading (cached across reruns for the session) ───────────────────
# @st.cache_resource runs once per app session and reuses the chain for all
# subsequent interactions — avoids reloading the embedding model on every rerun.
@st.cache_resource(show_spinner=False)
def load_chain():
    from src.chain import create_rag_chain
    return create_rag_chain()


with st.spinner("Loading knowledge base and RAG chain (first load ~30 seconds)..."):
    rag_chain = load_chain()


# ── Session state ────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # List of HumanMessage / AIMessage objects

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []  # List of {"role", "content", "sources"}


# ── Header ───────────────────────────────────────────────────────────────────
st.title("🏆 World Cup RAG Chatbot")
st.markdown(
    "Ask anything about FIFA World Cup history **(1930–2022)**. "
    "Answers are grounded in Wikipedia — sources shown below each response."
)

if not st.session_state.display_messages:
    st.info(
        "Try: *'Who scored the most goals in a single World Cup?'* "
        "or *'Describe the 1986 final.'*",
        icon="💬",
    )

# ── Render conversation ───────────────────────────────────────────────────────
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show source citations for assistant turns
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 Sources retrieved from knowledge base"):
                for i, src in enumerate(msg["sources"], 1):
                    label = src.split("/wiki/")[-1].replace("_", " ")
                    st.markdown(f"{i}. [{label}]({src})")


# ── Handle input ─────────────────────────────────────────────────────────────
if user_query := st.chat_input("Ask about World Cup history..."):

    # Display user message
    st.session_state.display_messages.append(
        {"role": "user", "content": user_query}
    )
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                result = rag_chain.invoke({
                    "input": user_query,
                    "chat_history": st.session_state.chat_history,
                })

                answer = result["answer"]

                # Extract unique source URLs from retrieved documents
                source_docs = result.get("context", [])
                seen: set[str] = set()
                sources: list[str] = []
                for doc in source_docs:
                    url = doc.metadata.get("source", "")
                    if url and url not in seen:
                        seen.add(url)
                        sources.append(url)

                st.markdown(answer)

                if sources:
                    with st.expander("📚 Sources retrieved from knowledge base"):
                        for i, src in enumerate(sources, 1):
                            label = src.split("/wiki/")[-1].replace("_", " ")
                            st.markdown(f"{i}. [{label}]({src})")

                # Persist to display history
                st.session_state.display_messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

                # Update LangChain conversation memory
                st.session_state.chat_history.extend([
                    HumanMessage(content=user_query),
                    AIMessage(content=answer),
                ])

            except Exception as exc:
                err = str(exc).lower()
                if "authentication" in err or "api_key" in err:
                    st.error("Invalid API key. Verify at console.groq.com/keys")
                elif "rate" in err or "limit" in err:
                    st.error("Rate limit hit. Wait 60 seconds and try again.")
                elif "connection" in err:
                    st.error("Connection error. Check your internet connection.")
                else:
                    st.error(f"Error: {exc}")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🏆 World Cup RAG")
    st.caption("ChromaDB · MiniLM-L6-v2 · Llama-3.3-70b · Groq")

    st.divider()
    st.subheader("💡 Example questions")

    EXAMPLES = [
        "What happened in the 1970 World Cup final?",
        "Tell me about Maradona's Hand of God goal",
        "Which team won the 2010 World Cup and where was it hosted?",
        "How many World Cups did Pelé win?",
        "What records did Miroslav Klose set?",
        "Describe the 2022 World Cup final",
        "Which nations have won the World Cup more than once?",
    ]

    for q in EXAMPLES:
        if st.button(q, use_container_width=True, key=f"ex_{q[:20]}"):
            st.session_state._pending_question = q
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.display_messages = []
        st.rerun()

# Handle sidebar button injection
if hasattr(st.session_state, "_pending_question"):
    pending = st.session_state._pending_question
    del st.session_state._pending_question

    st.session_state.display_messages.append(
        {"role": "user", "content": pending}
    )
    with st.spinner("Searching knowledge base..."):
        try:
            result = rag_chain.invoke({
                "input": pending,
                "chat_history": st.session_state.chat_history,
            })
            answer = result["answer"]
            source_docs = result.get("context", [])
            seen: set[str] = set()
            sources: list[str] = []
            for doc in source_docs:
                url = doc.metadata.get("source", "")
                if url and url not in seen:
                    seen.add(url)
                    sources.append(url)

            st.session_state.display_messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })
            st.session_state.chat_history.extend([
                HumanMessage(content=pending),
                AIMessage(content=answer),
            ])
            st.rerun()
        except Exception as exc:
            st.error(f"Error: {exc}")
