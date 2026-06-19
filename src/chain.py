import os
from operator import itemgetter
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from src.retriever import get_retriever

load_dotenv()

def create_rag_chain():
    # Retrieve the Groq API key securely from environment variables
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables. Please check your .env file.")

    # Initialize the Groq LLM client using the stable llama-3.3 model
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0.1,
    )

   # WIDEN THE NET: Ask the database for the top 15 chunks instead of just 5
    retriever = get_retriever(k=15)

    # RELAX THE RULES: Allow Groq to use its internal 2025 knowledge as a backup
    system_prompt = (
        "You are an expert assistant specialised in FIFA World Cup history.\n"
        "Try to use the following retrieved context to answer the question first.\n"
        "If the retrieved context does not contain the answer, you are allowed to "
        "use your own internal knowledge to answer it, but politely mention to the user "
        "that you are answering from general knowledge rather than the database.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # The pipeline routes the user's string input directly to the retriever
    rag_chain = (
        {
            "context":      itemgetter("input") | retriever | format_docs,
            "input":        itemgetter("input"),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain