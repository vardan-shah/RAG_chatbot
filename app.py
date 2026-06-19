import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.chain import create_rag_chain

st.set_page_config(page_title="World Cup RAG Chatbot", page_icon="⚽")
st.title("⚽ World Cup RAG Chatbot")

# Initialize the RAG chain within the user session state
if "rag_chain" not in st.session_state:
    try:
        st.session_state.rag_chain = create_rag_chain()
    except Exception as e:
        st.error(f"Failed to initialize RAG chain: {e}")
        st.stop()

# Initialize conversational chat history list
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Render existing chat logs
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

# Accept user input queries
if user_query := st.chat_input("Ask me anything about the World Cup:"):
    with st.chat_message("user"):
        st.write(user_query)
        
    with st.chat_message("assistant"):
        with st.spinner("Searching database and thinking..."):
            # Execute modern LCEL syntax chain invocation
            answer = st.session_state.rag_chain.invoke({
                "input": user_query,
                "chat_history": st.session_state.chat_history
            })
            
            st.write(answer)

    # Append interaction logs back into memory list
    st.session_state.chat_history.extend([
        HumanMessage(content=user_query),
        AIMessage(content=answer)
    ])