import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools.retriever import create_retriever_tool
from core.tools import (
    victorian_currency_converter,
    industry_stats_calculator,
    get_era_latency_check,
)

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
def load_victoria_agent():
    retriever = get_retriever()

    # 1. Turn your RAG into a Tool
    retriever_tool = create_retriever_tool(
        retriever,
        "search_royal_archives",
        "Search for historical facts, social conditions, and industrial data in the archives.",
    )

    # 2. Combine all tools
    tools = [
        retriever_tool,
        victorian_currency_converter,
        industry_stats_calculator,
        get_era_latency_check,
    ]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # 3. Use a standard Agent Prompt
    from langchain import hub

    prompt = hub.pull("hwchase17/openai-tools-agent")

    # 4. Create the Agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )


victoria_agent = load_victoria_agent()


victoria_brain = load_victoria()

# 5. DISPLAY UI CHAT HISTORY
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. REFINED CHAT LOGIC
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
                    # Run the RAG chain
                    response = victoria_brain.invoke({"query": prompt})
                    answer = response["result"]
                    st.markdown(answer)

                    # Show evidence only if documents were found
                    if response.get("source_documents"):
                        with st.expander("📜 View Historical Evidence"):
                            citations = set()
                            for doc in response["source_documents"]:
                                source_name = os.path.basename(
                                    doc.metadata.get("source", "Archive")
                                )
                                # FIXED LINE: Added closing parenthesis
                                page = doc.metadata.get("page", "N/A")

                                if isinstance(page, int):
                                    page += 1
                                citations.add(
                                    f"**Source:** {source_name} (Page {page})"
                                )

                            for citation in sorted(citations):
                                st.markdown(f"* {citation}")

                except Exception as e:
                    st.error(f"The telegraph lines are down.")
                    answer = "My apologies, I cannot reach the records at this moment."

    # Update session state with the assistant's answer
    st.session_state.messages.append({"role": "assistant", "content": answer})
