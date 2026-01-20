import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# --- IMPORT FROM YOUR CORE FOLDER ---
from core.retriever import get_retriever

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

    # HOUR 4: Structured Retrieval (Temporal Context)
    st.header("🕰️ Archive Filters")
    year_range = st.slider("Historical Focus (Year):", 1800, 1900, (1830, 1860))
    st.caption(f"Searching records focused on {year_range[0]}–{year_range[1]}")

    if st.button("🗑️ Clear Archive Memory"):
        st.session_state.messages = []
        st.session_state.chat_history = ""
        st.rerun()


# 4. INITIALIZE THE BRAIN
@st.cache_resource
def load_victoria():
    victoria_retriever = get_retriever()
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # HOUR 1: Step-Back Prompting (Added "broader historical context" instruction)
    template = """You are Victoria, a specialized Victorian Era Historian (1837–1901).
    Before providing a specific answer, briefly consider the broader historical context or societal shifts of the period.
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
        retriever=victoria_retriever,
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

    # HOUR 5: Security (Basic Input Validation)
    forbidden = ["ignore previous", "system prompt", "developer mode"]
    if any(term in prompt.lower() for term in forbidden):
        st.warning(
            "My apologies, but a lady does not discuss such technical subversions."
        )
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Consulting the archives..."):
                try:
                    # HOUR 4: Apply temporal context to the query
                    enhanced_query = (
                        f"In the period {year_range[0]}-{year_range[1]}: {prompt}"
                    )

                    response = victoria_brain.invoke({"query": enhanced_query})
                    answer = response["result"]
                    st.markdown(answer)

                    with st.expander("📜 View Historical Citations"):
                        citations = set()
                        for doc in response["source_documents"]:
                            source_name = os.path.basename(
                                doc.metadata.get("source", "Unknown")
                            )
                            page_num = doc.metadata.get("page", "Unknown")
                            display_page = (
                                (page_num + 1)
                                if isinstance(page_num, int)
                                else page_num
                            )
                            citations.add(f"_{source_name}_ (Page {display_page})")

                        for citation in sorted(citations):
                            st.markdown(f"* **Source:** {citation}")

                except Exception as e:
                    # HOUR 3: Error Handling
                    st.error("Heavens! The telegraph lines to the archive are down.")
                    answer = "I cannot reach the records at this moment."

        # 7. UPDATE MEMORY
        st.session_state.chat_history += f"User: {prompt} AI: {answer} | "
        st.session_state.messages.append({"role": "assistant", "content": answer})
