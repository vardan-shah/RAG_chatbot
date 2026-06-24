"""
RAG chain for the World Cup Chatbot.

Pipeline (in order of execution on each user question):

  1. create_history_aware_retriever
       Takes the raw user question + full chat history → calls the LLM to
       rewrite ambiguous follow-ups ("How old was he?") into standalone
       questions ("How old was Miroslav Klose during the 2006 World Cup?")
       → passes that rewritten question to ChromaDB for retrieval.

  2. create_stuff_documents_chain
       Injects the retrieved chunks into {context} → calls the LLM to
       generate an answer grounded in those chunks.

  3. create_retrieval_chain
       Wires steps 1 and 2 together and returns:
         {"answer": str, "context": [Document, ...], "input": str, ...}

This architecture replaces the old LCEL dict pipeline which sent raw user
input directly to the retriever with no history context.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

from src.retriever import get_retriever
from src.prompts import QA_SYSTEM_PROMPT, CONDENSE_SYSTEM_PROMPT

load_dotenv()


def create_rag_chain():
    """
    Build and return the full conversational RAG chain.

    Called once at app startup and stored in st.session_state.
    The returned chain accepts:
        {"input": str, "chat_history": list[HumanMessage | AIMessage]}
    and returns:
        {"answer": str, "context": list[Document], ...}
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Get a free key at console.groq.com/keys and add it to your .env file."
        )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0.0,     # 0 = most deterministic; critical for factual Q&A
        max_tokens=1024,
    )

    # Retrieve more chunks for complex questions (scorelines, goalscorers, etc.)
    retriever = get_retriever(k=6)

    # ── Step 1: History-aware retriever ────────────────────────────────────
    # The condense prompt rewrites the user's question using chat history
    # BEFORE the question reaches ChromaDB. This solves pronoun ambiguity in
    # follow-up questions.
    condense_prompt = ChatPromptTemplate.from_messages([
        ("system", CONDENSE_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, condense_prompt
    )

    # ── Step 2: QA chain ───────────────────────────────────────────────────
    # create_stuff_documents_chain automatically formats the retrieved
    # documents into {context} inside QA_SYSTEM_PROMPT.
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", QA_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    # ── Step 3: Full pipeline ──────────────────────────────────────────────
    # create_retrieval_chain connects history_aware_retriever → qa_chain.
    # Output keys: "input", "chat_history", "context", "answer"
    return create_retrieval_chain(history_aware_retriever, qa_chain)
