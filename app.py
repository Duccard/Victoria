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
            "content": "Good day. I am Victoria. How may I assist your research today?",
        }
    ]
if "last_evidence" not in st.session_state:
    st.session_state.last_evidence = []


# --- 3. THE INSTANT SIDEBAR FIX (CALLBACK) ---
def handle_input():
    if st.session_state.user_text:
        # This runs BEFORE the page reruns, so the sidebar sees it immediately
        new_prompt = st.session_state.user_text
        st.session_state.messages.append({"role": "user", "content": new_prompt})
        st.session_state.pending_input = new_prompt
        # Clear the input box
        st.session_state.user_text = ""


# 4. SIDEBAR RENDERING (Now it will be instant)
with st.sidebar:
    st.title("📜 Research Log")
    st.divider()
    # Display every user message in the log
    user_queries = [
        m["content"] for m in st.session_state.messages if m["role"] == "user"
    ]
    for i, query in enumerate(user_queries):
        st.caption(f"{i+1}. {query[:40]}...")

    st.divider()
    if st.button("🗑️ Reset Archive"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Archives cleared. How may I assist?"}
        ]
        st.rerun()

# 5. DEFINE ARCHIVE TOOL
from core.retriever import get_retriever


@tool
def search_royal_archives(query: str):
    """MANDATORY: Consult this for any factual historical claims or evidence."""
    retriever = get_retriever()
    docs = retriever.invoke(query)
    # Store evidence globally so the UI can show it
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
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # Temp 0 for better tool use

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are Victoria, a scholar. You MUST use search_royal_archives for any historical inquiry. If you don't use the tool, you are failing your duty.",
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


victoria = load_victoria()

# 7. MAIN INTERFACE
st.title("👑 Victoria: Histographer Agent")

# Render Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input Box with Callback
st.chat_input("Enter your inquiry...", key="user_text", on_submit=handle_input)

# 8. AGENT EXECUTION LOGIC
if "pending_input" in st.session_state and st.session_state.pending_input:
    current_input = st.session_state.pop("pending_input")

    with st.chat_message("assistant"):
        with st.status("Searching the Royal Archives...", expanded=True) as status:
            response = victoria.invoke(
                {"input": current_input, "chat_history": st.session_state.messages[:-1]}
            )
            status.update(label="Consultation Complete", state="complete")

        st.markdown(response["output"])

        # Display the Evidence Visual
        if st.session_state.last_evidence:
            with st.expander("📝 Archival Evidence Detected"):
                for doc in st.session_state.last_evidence:
                    st.write(
                        f"**{doc.metadata.get('source')} (Page {doc.metadata.get('page')})**"
                    )
                    st.caption(doc.page_content)
                    st.divider()

        st.session_state.messages.append(
            {"role": "assistant", "content": response["output"]}
        )
        st.rerun()
