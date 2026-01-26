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
            "content": "Good day to you, seeker of knowledge. I am Victoria, your humble Histographer. How may I assist your research today?",
        }
    ]
if "last_evidence" not in st.session_state:
    st.session_state.last_evidence = []


# --- 3. THE INSTANT SIDEBAR FIX ---
def handle_input():
    if st.session_state.user_text:
        new_prompt = st.session_state.user_text
        st.session_state.messages.append({"role": "user", "content": new_prompt})
        st.session_state.pending_input = new_prompt
        st.session_state.user_text = ""


# 4. SIDEBAR RENDERING
with st.sidebar:
    st.image("https://img.icons8.com/color/96/scroll.png")
    st.title("📜 Research Log")
    st.divider()
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
        st.session_state.last_evidence = []
        st.rerun()

# 5. DEFINE ARCHIVE TOOL
from core.retriever import get_retriever


@tool
def search_royal_archives(query: str):
    """MANDATORY: Consult this for any factual historical claims or evidence."""
    retriever = get_retriever()
    docs = retriever.invoke(query)
    # CRITICAL: Store evidence in session state for the UI to see
    st.session_state.last_evidence = docs

    results = []
    for d in docs:
        source = os.path.basename(d.metadata.get("source", "Unknown Archive"))
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
                """You are Victoria, an impeccably refined British Lady and Histographer.
        RULES:
        1. For historical topics (Steam engines, Mines Act, etc.), you MUST use 'search_royal_archives'.
        2. Always cite your sources in the text as (Source, Page).
        3. Maintain a formal, scholarly Victorian tone.""",
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )


victoria = load_victoria()

# 7. MAIN INTERFACE
st.title("👑 Victoria: Histographer Agent")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

st.chat_input("Enter your inquiry...", key="user_text", on_submit=handle_input)

# 8. AGENT EXECUTION LOGIC
if "pending_input" in st.session_state and st.session_state.pending_input:
    current_input = st.session_state.pop("pending_input")

    # Check for basic interactions first
    greetings = ["hello", "hi", "greetings", "good day", "how are you"]
    if current_input.lower().strip() in greetings:
        answer = "Good day to you! I am delighted by your presence. Pray, what specific historical curiosity shall we explore from our grand archives?"
        st.session_state.last_evidence = []  # No evidence for greetings
    else:
        with st.chat_message("assistant"):
            st.session_state.last_evidence = []  # Clear old evidence
            with st.status("Consulting the Royal Archives...", expanded=True) as status:
                # Force the agent to use the tool
                response = victoria.invoke(
                    {
                        "input": f"{current_input}. Search the archives for details.",
                        "chat_history": st.session_state.messages[:-1],
                    }
                )
                answer = response["output"]
                status.update(label="Consultation Complete", state="complete")

            st.markdown(answer)

            # THE EVIDENCE BAR (Restored and Fixed)
            if st.session_state.last_evidence:
                with st.expander("📝 Archival Evidence Detected", expanded=True):
                    for doc in st.session_state.last_evidence:
                        src = os.path.basename(doc.metadata.get("source", "Archive"))
                        pg = doc.metadata.get("page", "N/A")
                        st.write(f"📖 **{src}** — *Page {pg}*")
                        st.caption(f'"{doc.page_content[:400]}..."')
                        st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
