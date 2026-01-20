import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from core.retriever import get_retriever

# 1. PAGE CONFIG
st.set_page_config(page_title="Victoria", page_icon="👑")

# Updated Title and Subtitle
st.title("👑 Victoria")
st.subheader("Victorian Era Histographer")

load_dotenv()

# 2. SESSION STATE
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

    # Lower temperature (0.3) reduces grammar "hallucinations" and mistakes
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)

    # Refined prompt for perfect grammar and formal tone
    template = """You are Victoria, a professional Victorian Era Histographer. 
    You write with impeccable formal grammar and the scholarly precision of a 19th-century academic.
    
    Use the provided archives to answer the user's question. 
    If the context doesn't have the answer, state that the archives are silent on the matter, 
    but provide a brief response based on general historical consensus.

    Context: {context}
    Question: {question}
    
    Formal Historical Response:"""

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

# 6. CHAT LOGIC
if prompt := st.chat_input("Ask about the Victorian Era..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Custom loading message
        with st.spinner("Searching the Royal Archives..."):
            try:
                response = victoria_brain.invoke({"query": prompt})
                answer = response["result"]
                st.markdown(answer)

                # RE-ADDED: Detailed Evidence UI with Page References
                with st.expander("📜 View Historical Evidence"):
                    citations = set()
                    for doc in response["source_documents"]:
                        source_name = os.path.basename(
                            doc.metadata.get("source", "Unknown Archive")
                        )
                        # Retrieve the page number from metadata
                        page = doc.metadata.get("page", "N/A")

                        # Adjust page number if it's an integer (starting from 0)
                        if isinstance(page, int):
                            page = page + 1

                        citations.add(f"**Source:** {source_name} (Page {page})")

                    for citation in sorted(citations):
                        st.markdown(f"* {citation}")

            except Exception as e:
                st.error("The telegraph lines are down.")
                answer = "My apologies, I cannot reach the records."

    st.session_state.messages.append({"role": "assistant", "content": answer})
