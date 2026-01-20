import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from core.retriever import get_retriever

# 1. PAGE CONFIG
st.set_page_config(page_title="Victoria", page_icon="👑")
st.title("👑 Victoria: Victorian Historian")
load_dotenv()

# 2. SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. SIDEBAR (Simplified)
with st.sidebar:
    st.image("https://img.icons8.com/color/96/scroll.png")
    st.title("Victoria's Archive")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# 4. INITIALIZE THE BRAIN
@st.cache_resource
def load_victoria():
    # Still using your Multi-Query logic from core/retriever.py
    victoria_retriever = get_retriever()
    llm = ChatOpenAI(
        model_name="gpt-4o-mini", temperature=0.7
    )  # Slightly higher temp for more natural flow

    # A much warmer, more helpful prompt
    template = """You are Victoria, a specialized historian of the Victorian Era. 
    Use the provided archives to answer the user's question in an engaging, historical tone.
    If the context doesn't have the exact answer, use your general knowledge of the 19th century to be helpful, 
    but mention that you are drawing from general history.

    Context: {context}
    Question: {question}
    
    Historical Response:"""

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"], template=template
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=victoria_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )


victoria_brain = load_victoria()

# 5. DISPLAY CHAT
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. CHAT LOGIC (Simplified)
if prompt := st.chat_input("Ask about the Victorian Era..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = victoria_brain.invoke({"query": prompt})
            answer = response["result"]
            st.markdown(answer)

            with st.expander("📜 Historical Sources"):
                for doc in response["source_documents"]:
                    source = os.path.basename(doc.metadata.get("source", "Archive"))
                    st.write(f"- {source}")
        except:
            st.error("The archives are currently unavailable.")
            answer = "My apologies, I cannot reach the records."

    st.session_state.messages.append({"role": "assistant", "content": answer})
