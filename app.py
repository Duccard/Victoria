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

    # Adding a bit of flair
    st.markdown("---")

    if st.button("🗑️ Clear Archive Memory"):
        # 1. Clear the visible chat history
        st.session_state.messages = []

        # 2. Clear the AI's internal memory string
        st.session_state.chat_history = ""

        # 3. Provide feedback
        st.toast("Memory wiped. Victoria is ready for a new topic.")

        # 4. Refresh the app
        st.rerun()

    # 2. Generate Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Consulting the archives..."):
            combined_query = (
                f"HISTORY: {st.session_state.chat_history}\n\nLATEST QUESTION: {prompt}"
            )

            response = victoria_brain.invoke(
                {"query": combined_query, "history": st.session_state.chat_history}
            )

            answer = response["result"]
            st.markdown(answer)

            # --- IMPROVED BIBLIOGRAPHER SECTION ---
            with st.expander("📜 View Historical Citations"):
                st.write("Victoria found the following evidence in the archives:")

                # Use a set to avoid showing the same file name multiple times
                unique_sources = set()
                for doc in response["source_documents"]:
                    source_name = doc.metadata.get("source", "Unknown Archive")
                    # Clean up the path to show just the filename
                    clean_name = os.path.basename(source_name)
                    unique_sources.add(clean_name)

                for source in unique_sources:
                    st.markdown(f"* **Source Document:** _{source}_")

                st.caption(
                    "Note: These documents were retrieved using semantic search (k=6) from the local vector database."
                )

# 3. Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ""

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. The Logic
if prompt := st.chat_input("Ask about the factory conditions..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consulting the archives..."):
            # We "bake" the history into the query ourselves
            # This satisfies the chain because it only sees the 'query' key
            full_query = f"Recent History: {st.session_state.chat_history}\n\nUser Question: {prompt}"

            response = victoria_brain.invoke({"query": full_query})

            answer = response["result"]
            st.markdown(answer)

            with st.expander("View Evidence"):
                for doc in response["source_documents"]:
                    st.write(f"- {doc.metadata.get('source', 'Unknown PDF')}")

    # Update state
    st.session_state.chat_history += f"User: {prompt} AI: {answer} | "
    st.session_state.messages.append({"role": "assistant", "content": answer})
