import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv
import os

# 1. Page Configuration
st.set_page_config(page_title="Victoria: Victorian Historian", page_icon="📜")
st.title("📜 Victoria: RAG Historian")
st.markdown("---")

load_dotenv()


# 2. Initialize the "Brain" (Cached so it stays fast)
@st.cache_resource
def load_victoria():
    # 1. Connect to the same DB you've been using
    db = Chroma(persist_directory="chroma_db", embedding_function=OpenAIEmbeddings())
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # 2. Define the personality (The System Prompt)
    template = """You are Victoria, a specialized Victorian Era Historian. 
    Use the following pieces of context to answer the user's question. 
    If you don't know the answer based on the context, say that you don't know, do not try to make up an answer.
    Keep the tone academic yet accessible.

    Context: {context}
    Question: {question}
    Helpful Answer:"""

    victoria_brain = load_victoria()

    # 3. Create the Prompt Template object
    from langchain.prompts import PromptTemplate

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    # 4. Build the chain with the new "Instructions" (chain_type_kwargs)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 6}),
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": QA_CHAIN_PROMPT
        },  # This tells the LLM how to behave
    )
    return qa_chain


# 3. Chat History Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. The Chat Input
if prompt := st.chat_input("Ask me about the Victorian factory conditions..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Searching the archives..."):
            response = victoria_brain.invoke(prompt)
            answer = response["result"]
            sources = response["source_documents"]

            st.markdown(answer)

            # Show sources in an expandable section
            with st.expander("View Evidence (Sources)"):
                for doc in sources:
                    st.write(f"- {doc.metadata.get('source', 'Unknown source')}")

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
