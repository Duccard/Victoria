import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool

# 1. PAGE SETUP
st.set_page_config(page_title="Victoria", page_icon="👑", layout="wide")
load_dotenv()

# 2. STATE INITIALIZATION
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Good day to you, seeker of knowledge. I am Victoria, your humble Histographer. What historical curiosities shall we explore together this fine day?",
        }
    ]
if "last_evidence" not in st.session_state:
    st.session_state.last_evidence = []

# --- 3. THE INSTANT SIDEBAR FIX (CALLBACK) ---
def handle_input():
    if st.session_state.user_text:
        # Update session state BEFORE the script reruns
        new_prompt = st.session_state.user_text
        st.session_state.messages.append({"role": "user", "content": new_prompt})
        st.session_state.pending_input = new_prompt
        # Clear the input box immediately
        st.session_state.user_text = ""

# 4. SIDEBAR RENDERING
with st.sidebar:
    st.image("https://img.icons8.com/color/96/scroll.png")
    st.title("📜 Research Log")
    st.caption("A record of our scholarly correspondence.")
    st.divider()
    
    # Display every user message in the log (now perfectly synced)
    user_queries = [
        m["content"] for m in st.session_state.messages if m["role"] == "user"
    ]
    if user_queries:
        for i, query in enumerate(user_queries):
            st.caption(f"{i+1}. {query[:40]}...")
    else:
        st.info("The log is empty.")

    st.divider()
    if st.button("🗑️ Reset Archive"):
        st.session_state.messages = [{"role": "assistant", "content": "Archives cleared. How may I assist?"}]
        st.session_state.last_evidence = []
        st.rerun()

# 5. DEFINE ARCHIVE TOOL
from core.retriever import get_retriever

@tool
def search_royal_archives(query: str):
    """MANDATORY: Consult this for any factual historical claims or evidence from the Victorian era."""
    retriever = get_retriever()
    docs = retriever.invoke(query)
    
    # Store evidence globally so the UI can catch it after the tool call finishes
    st.session_state.last_evidence = docs

    results = []
    for d in docs:
        source = d.metadata.get("source", "Unknown Archive")
        page = d.metadata.get("page", "N/A")
        results.append(f"SOURCE: {source} (Page {page})\nCONTENT: {d.page_content}")
    return "\n\n".join(results)

# 6. AGENT SETUP
@st.cache_resource
def load_victoria():
    from core.tools import victorian_currency_converter, industry_stats_calculator

    tools = [
        search_royal_archives,
        victorian_currency_converter,
        industry_stats_calculator,
    ]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are Victoria, a formal British Histographer. 
                You MUST use search_royal_archives for any historical inquiry to provide evidence. 
                Always cite your sources with (Source Name, Page Number).""",
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True