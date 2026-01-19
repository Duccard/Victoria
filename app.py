import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Victoria: Victorian Historian", page_icon="📜")
st.title("📜 Victoria: RAG Historian")

load_dotenv()

# 2. SESSION STATE INITIALIZATION (Must be at the top)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ""

# 3. SIDEBAR
with st.sidebar:
    st.image("https://img.icons8.com/color/96/scroll.png")
    st.title("Victoria's Archive")
    st.info(
        "Victoria uses RAG to analyze the Sadler Report and Industrial Revolution documents."
    )

    if st.button("🗑️ Clear Archive Memory"):
        st.session_state.messages = []
        st.session_state.chat_history = ""
        st.toast("Memory wiped. Victoria is ready for a new topic.")
        st.rerun()


# 4. INITIALIZE THE BRAIN
@st.cache_resource
def load_victoria():
    db = Chroma(persist_directory="chroma_db", embedding_function=OpenAIEmbeddings())
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # Simplified Template for stability with RetrievalQA
    template = """You are Victoria, an expert Victorian Historian. 
    Below is the chat history and some context from the archives. 
    Use the context to answer the latest question.
    
    Context: {context}
    Question: {question}
    
    Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"], template=template
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 6}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )


victoria_brain = load_victoria()

# 5. DISPLAY UI CHAT HISTORY
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. THE CHAT LOGIC
if prompt := st.chat_input("Ask about the factory conditions..."):
    # Add User message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Consulting the archives..."):
            # Bake the history into the query to give the LLM context
            combined_query = f"Recent History: {st.session_state.chat_history}\n\nUser Question: {prompt}"

            # Use the 'query' key as expected by RetrievalQA
            response = victoria_brain.invoke({"query": combined_query})

            answer = response["result"]
            st.markdown(answer)

            # --- OPTION A: THE BIBLIOGRAPHER ---
            with st.expander("📜 View Historical Citations"):
                st.write("Victoria found evidence in the following documents:")
                unique_sources = set()
                for doc in response["source_documents"]:
                    source_name = doc.metadata.get("source", "Unknown Archive")
                    unique_sources.add(os.path.basename(source_name))

                for source in unique_sources:
                    st.markdown(f"* **Source Document:** _{source}_")

    # 7. UPDATE MEMORY & UI HISTORY
    st.session_state.chat_history += f"User: {prompt} AI: {answer} | "
    st.session_state.messages.append({"role": "assistant", "content": answer})
