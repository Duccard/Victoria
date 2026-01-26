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

# --- CUSTOM THEME (CSS) ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f4f1ea; /* Parchment Background */
    }
    [data-testid="stSidebar"] {
        background-color: #2e3b4e !important;
    }
    [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] p {
        color: #d1d1d1 !important;
    }
    /* Style the subtitle */
    .subtitle {
        font-style: italic;
        color: #5d5d5d;
        margin-top: -20px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- SOURCE TITLES DICTIONARY ---
SOURCE_TITLES = {
    "20-Industrial-Rev.pdf": "The Industrial Revolution Archives (Vol. 20)",
    "Chapter-8-The-Industrial-Revolution.pdf": "British Industrial History, Chapter VIII",
    "sadler-report.pdf": "The Michael Sadler Report (1832)",
    "2020_Kelly_Mokyr_Mechanics_Ind_Rev.pdf": "Mechanics of the Industrial Revolution",
}

# 2. STATE INITIALIZATION
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Good day. I am Victoria. How may I assist your research today?",
            "evidence": None,
            "theme": "Greeting",
        }
    ]
if "temp_evidence" not in st.session_state:
    st.session_state.temp_evidence = []


# --- 3. THEME IDENTIFIER ---
def identify_theme(text):
    if not text or len(text) < 5:
        return "General"
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        response = llm.invoke(
            f"Summarize this historical query into 2-3 words. Query: {text}"
        )
        return response.content.strip().replace('"', "")
    except:
        return "Inquiry"


# --- 4. INPUT CALLBACK ---
def handle_input():
    if st.session_state.user_text:
        new_prompt = st.session_state.user_text
        theme = identify_theme(new_prompt)
        st.session_state.messages.append(
            {"role": "user", "content": new_prompt, "evidence": None, "theme": theme}
        )
        st.session_state.pending_input = new_prompt
        st.session_state.temp_evidence = []
        st.session_state.user_text = ""


# 5. SIDEBAR (Your Enquiries History)
with st.sidebar:
    st.title("📜 Your Enquiries History")
    st.divider()

    user_themes = [
        m.get("theme") for m in st.session_state.messages if m["role"] == "user"
    ]
    for theme in reversed(user_themes):
        st.write(f"📂 {theme}")

    st.divider()
    if st.button("🗑️ Reset Archive"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Archives cleared.",
                "evidence": None,
                "theme": "Greeting",
            }
        ]
        st.rerun()

# 6. ARCHIVE TOOL
from core.retriever import get_retriever


@tool
def search_royal_archives(query: str):
    """Consult this for historical facts."""
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
    st.session_state.temp_evidence = evidence_list
    return str(evidence_list)


# 7. AGENT SETUP
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
                "You are Victoria, a British Histographer. Use search_royal_archives. Do NOT list sources in text.",
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


victoria = load_victoria()

# 8. MAIN INTERFACE (Royal Header)
st.title("Victoria 👑")
st.markdown(
    '<p class="subtitle">Victorian Era Histographer Agent</p>', unsafe_allow_html=True
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("evidence"):
            with st.expander("📝 ARCHIVAL CITATIONS", expanded=True):
                st.table(msg["evidence"])

st.chat_input("Enter your inquiry...", key="user_text", on_submit=handle_input)

# 9. EXECUTION
if "pending_input" in st.session_state and st.session_state.pending_input:
    current_input = st.session_state.pop("pending_input")

    with st.chat_message("assistant"):
        with st.status("Consulting Archives...", expanded=True) as status:
            response = victoria.invoke(
                {"input": current_input, "chat_history": st.session_state.messages[:-1]}
            )
            answer = response["output"]
            status.update(label="Complete", state="complete")
        st.markdown(answer)

        curr_ev = (
            st.session_state.temp_evidence if st.session_state.temp_evidence else None
        )
        if curr_ev:
            with st.expander("📝 ARCHIVAL CITATIONS", expanded=True):
                st.table(curr_ev)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "evidence": curr_ev}
        )
    st.rerun()
