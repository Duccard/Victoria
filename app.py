import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# 1. Page Configuration
st.set_page_config(page_title="Victoria: Victorian Historian", page_icon="📜")
st.title("📜 Victoria: RAG Historian")

load_dotenv()

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/scroll.png")
    st.title("Victoria's Archive")
    st.info(
        "Victoria uses RAG to analyze the Sadler Report and Industrial Revolution documents."
    )
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.chat_history = ""
        st.rerun()


# 2. Initialize the Brain
@st.cache_resource
def load_victoria():
    db = Chroma(persist_directory="chroma_db", embedding_function=OpenAIEmbeddings())
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # Note: RetrievalQA expects the variable name 'context' and 'question'
    # but we added 'history' to the mix.
    template = """You are Victoria, an expert Victorian Historian. 
    Below is the chat history and some context from the archives. 
    Use the context to answer the latest question.
    
    History: {history}
    Context: {context}
    Question: {question}
    
    Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["history", "context", "question"], template=template
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 6}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )
    return qa_chain


victoria_brain = load_victoria()

# 3. Session State Setup
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ""

# Display chat history in UI
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. The Logic
if prompt := st.chat_input("Ask about the factory conditions..."):
    # Add User message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Assistant message
    with st.chat_message("assistant"):
        with st.spinner("Searching archives..."):
            # We use 'query' for the text search and 'history' for the prompt
            # If RetrievalQA is still being stubborn, we use this precise format:
            inputs = {"query": prompt, "history": st.session_state.chat_history}

            response = victoria_brain.invoke(inputs)

            answer = response["result"]
            st.markdown(answer)

            # Update history string for the next turn
            st.session_state.chat_history += f"\nUser: {prompt}\nVictoria: {answer}\n"

            with st.expander("View Evidence"):
                for doc in response["source_documents"]:
                    st.write(f"- {doc.metadata.get('source', 'Unknown PDF')}")

    # Save Assistant message to UI history
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Save Assistant message to UI history
    st.session_state.messages.append({"role": "assistant", "content": answer})
