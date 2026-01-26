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

# --- SOURCE TITLES DICTIONARY ---
SOURCE_TITLES = {
    "20-Industrial-Rev.pdf": "The Industrial Revolution Archives (Vol. 20)",
    "Chapter-8-The-Industrial-Revolution.pdf": "British Industrial History, Chapter VIII",
    "sadler-report.pdf": "The Michael Sadler Report on Factory Labor (1832)",
    "mines-act-1842.pdf": "Royal Commission on Children's Employment in Mines",
    "2020_Kelly_Mokyr_Mechanics_Ind_Rev.pdf": "Mechanics of the Industrial Revolution (Kelly & Mokyr)",
}

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


# --- 3. SIDEBAR CALLBACK ---
def handle_input():
    if st.session_state.user_text:
        new_prompt = st.session_state.user_text
        st.session_state.messages.append({"role": "user", "content": new_prompt})
        st.session_state.pending_input = new_prompt
        st.session_state.last_evidence = []  # Clear table for new search
        st.session_state.user_text = ""


# 4. SIDEBAR
with st.sidebar:
    st.image("https://img.icons8.com/color/96/scroll.png")
    st.title("📜 Research Log")
    st.divider()
    user_queries = [
        m["content"] for m in st.session_state.messages if m["role"] == "user"
    ]
    for i, query in enumerate(user_queries):
        st.caption(f"{i+1}. {query[:40]}...")

    if st.button("🗑️ Reset Archive"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Archives cleared."}
        ]
        st.session_state.last_evidence = []
        st.rerun()

# 5. ARCHIVE TOOL
from core.retriever import get_retriever


@tool
def search_royal_archives(query: str):
    """MANDATORY: Consult this for any factual historical claims or evidence."""
    retriever = get_retriever()
    docs = retriever.invoke(query)

    evidence_list = []
    seen = set()
    for d in docs:
        fname = os.path.basename(d.metadata.get("source", ""))
        title = SOURCE_TITLES.get(fname, fname)
        page = d.metadata.get("page", "N/A")

        ref = f"{title}-{page}"
        if ref not in seen:
            evidence_list.append({"Source Title": title, "Page": page})
            seen.add(ref)

    st.session_state.last_evidence = evidence_list
    return "\n\n".join(
        [f"Source: {e['Source Title']} (Page {e['Page']})" for e in evidence_list]
    )


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
                "You are Victoria, a formal British Histographer. Use search_royal_archives for history.",
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

# Render History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 8. MINIMAL EVIDENCE TABLE ---
if st.session_state.last_evidence:
    with st.expander("📝 ARCHIVAL CITATIONS", expanded=True):
        st.table(st.session_state.last_evidence)

# Input Box
st.chat_input("Enter your inquiry...", key="user_text", on_submit=handle_input)

# 9. LOGIC
if "pending_input" in st.session_state and st.session_state.pending_input:
    current_input = st.session_state.pop("pending_input")

    greetings = ["hello", "hi", "greetings", "good day"]
    if current_input.lower().strip() in greetings:
        answer = "Good day! How may I assist your research into our glorious era?"
    else:
        with st.chat_message("assistant"):
            with st.status("Searching the Royal Archives...", expanded=True) as status:
                response = victoria.invoke(
                    {
                        "input": f"{current_input}. Cite archival sources.",
                        "chat_history": st.session_state.messages[:-1],
                    }
                )
                answer = response["output"]
                status.update(label="Consultation Complete", state="complete")
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
