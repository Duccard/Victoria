import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain  # <--- New Chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# 1. Page Configuration
st.set_page_config(page_title="Victoria: Victorian Historian", page_icon="📜")
st.title("📜 Victoria: RAG Historian")
st.markdown("---")

with st.sidebar:
    st.image("https://img.icons8.com/color/96/scroll.png")  # A little icon
    st.title("About Victoria")
    st.info(
        """
    Victoria is an AI historian powered by RAG. 
    She is currently reading from:
    - The Sadler Report (1832)
    - Industrial Revolution Analysis
    - Academic Papers on Labor
    """
    )
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

load_dotenv()

# 2. Load function


@st.cache_resource
def load_victoria():
    db = Chroma(persist_directory="chroma_db", embedding_function=OpenAIEmbeddings())
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    template = """You are Victoria, a specialized Victorian Era Historian. 
    Use the following pieces of context to answer the user's question. 
    If you don't know the answer, say you don't know.
    
    Context: {context}
    Question: {question}
    Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"], template=template
    )

    # Use ConversationalRetrievalChain for Day 3 memory!
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 6}),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
    )
    return qa_chain


# 3. Single call to start the brain
victoria_brain = load_victoria()
# 4. Chat History Setup (Persistent storage for the browser session)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. The Chat Input & Logic
# Create a place to store memory for the AI (different from the UI messages)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if prompt := st.chat_input("Ask me about the Victorian factory conditions..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consulting the archives..."):
            # We pass BOTH the prompt and the chat_history to the brain
            result = victoria_brain.invoke(
                {"question": prompt, "chat_history": st.session_state.chat_history}
            )

            answer = result["answer"]
            st.markdown(answer)

            # Update memory so she remembers this for the next question
            st.session_state.chat_history.append((prompt, answer))

            with st.expander("View Evidence"):
                for doc in result["source_documents"]:
                    st.write(f"- {doc.metadata.get('source')}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
