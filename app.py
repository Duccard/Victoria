import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# --- IMPORT FROM YOUR CORE FOLDER ---
from core.retriever import get_retriever

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Victoria", page_icon="👑")

# Title and Subtitle
st.title("👑 Victoria")
st.subheader("Victorian Era Histographer")

load_dotenv()

# 2. SESSION STATE INITIALIZATION
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. SIDEBAR
with st.sidebar:
    st.image("https://img.icons8.com/color/96/scroll.png")
    st.title("Victoria's Archive")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# 4. INITIALIZE THE BRAIN
@st.cache_resource
def load_victoria():
    victoria_retriever = get_retriever()
    
    # Lower temperature for better grammar
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3) 

    template = """You are Victoria, a professional Victorian Era Histographer. 
    Use the provided archives to answer the user's question. 
    If the context doesn't have the answer, state that the archives are silent on the matter, 
    but provide a brief response based on general historical knowledge.

    Context: {context}
    Question: {question}
    
    Formal Historical Response:"""

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=victoria_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

victoria_brain = load_victoria()

# 5. DISPLAY UI CHAT HISTORY
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. REFINED CHAT LOGIC (Includes Greeting Filter)
if prompt := st.chat_input("Ask about the Victorian Era..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Check for simple greetings
        greetings = ["hello", "hi", "greetings", "good morning", "how are you"]
        
        if prompt.lower().strip() in greetings:
            answer = "Good day to you! I am Victoria. How may I assist your historical research into the Victorian Era today?"
            st.markdown(answer)
        else:
            with st.spinner("Searching the Royal Archives..."):
                try:
                    response = victoria_brain