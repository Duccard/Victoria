import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# --- NEW IMPORT FROM YOUR RETRIEVER FILE ---
from retriever import get_retriever

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Victoria: Victorian Historian", page_icon="👑")
st.title("👑 Victoria: Industrial Revolution Historian")

load_dotenv()

# 2. SESSION STATE INITIALIZATION
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ""

# 3. SIDEBAR
with st.sidebar:
    st.image("https://img.icons8.com/color/96/scroll.png")
    st.title("Victoria's Archive")
    st.info(
        "Victoria uses Multi-Query RAG to analyze the Industrial Revolution archives."
    )

    if st.button("🗑️ Clear Archive Memory"):
        st.session_state.messages = []
        st.session_state.chat_history = ""
        st.toast("Memory wiped. Victoria is ready for a new topic.")
        st.rerun()


# 4. INITIALIZE THE BRAIN (Now using your modular retriever)
@st.cache_resource
def load_victoria():
    # Load your high-precision retriever from retriever.py
    victoria_retriever = get_retriever()

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    template = """You are Victoria, a specialized Victorian Era Historian (1837–1901).
    Your knowledge is strictly limited to the provided context from the archives.
    
    RULES:
    1. Only answer questions related to the Victorian Era or the Industrial Revolution.
    2. If the question is about a different time period, politely decline.
    3. If the answer is not in the context, say you do not know.
    
    Context: {context}
    Question: {question}
    
    Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"], template=template
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=victoria_retriever,  # Using the Multi-Query brain here
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
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consulting the archives..."):
            try:
                # Victoria now generates 3 variations of your question internally
                response = victoria_brain.invoke({"query": prompt})
                answer = response["result"]
                st.markdown(answer)

                # --- EVIDENCE UI ---
                with st.expander("📜 View Historical Citations"):
                    citations = set()
                    for doc in response["source_documents"]:
                        source_name = os.path.basename(
                            doc.metadata.get("source", "Unknown")
                        )
                        page_num = doc.metadata.get("page", "Unknown")
                        display_page = (
                            (page_num + 1) if isinstance(page_num, int) else page_num
                        )
                        citations.add(f"_{source_name}_ (Page {display_page})")

                    for citation in sorted(citations):
                        st.markdown(f"* **Source:** {citation}")

            except Exception as e:
                st.error("Heavens! The telegraph lines to the archive are down.")
                answer = "I cannot reach the records at this moment."

    # 7. UPDATE MEMORY
    st.session_state.chat_history += f"User: {prompt} AI: {answer} | "
    st.session_state.messages.append({"role": "assistant", "content": answer})
