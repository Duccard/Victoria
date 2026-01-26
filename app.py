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
    /* Main Chat Background */
    .stApp {
        background-color: #f4f1ea; /* Light Parchment Color */
    }
    
    /* Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: #2e3b4e !important;
    }
    
    /* Sidebar Text Color */
    [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] p {
        color: #d1d1d1 !important;
    }

    /* Input Box Styling */
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_value=True,
)

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
            "evidence": None,
            "theme": "Greeting",
        }
    ]
if "temp_evidence" not in st.session_state:
    st.session_state.temp_evidence = []
if "focus_theme" not in st.session_state:
    st.session_state.focus_theme = None


# --- 3. THEME IDENTIFIER (Mini-LLM) ---
def identify_theme(text):
    if not text or len(text) < 5:
        return "General"
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(
        f"Summarize this historical query into 2-3 words (e.g., 'Steam Engine' or 'Child Labor'). Query: {text}"
    )
    return response.content.strip().replace('"', "")


# --- 4. SIDEBAR CALLBACK ---
def handle_input():
    if st.session_state.user_text:
        new_prompt = st.session_state.user_text
        theme = identify_theme(new_prompt)
        st.session_state.messages.append(
            {"role": "user", "content": new_prompt, "evidence": None, "theme": theme}
        )
        st.session_state.pending_input = new_prompt
        st.session_state.temp_evidence = []
        st.session_state.focus_theme = None
        st.session_state.user_text = ""


# 5. SIDEBAR (Your Enquiries History)
with st.sidebar:
    st.title("📜 Your Enquiries History")
    st.caption("Select a theme to focus your research view.")
    st.divider()

    # Show unique themes from history
    all_themes = [
        m.get("theme")
        for m in st.session_state.messages
        if m.get("theme") and m["role"] == "user"
    ]

    if st.button("👁️ Show Full Correspondence", use_container_width=True):
        st.session_state.focus_theme = None

    for theme in reversed(all_themes):
        if st.button(f"📂 {theme}", use_container_width=True):
            st.session_state.focus_theme = theme

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
        st.session_state.focus_theme = None
        st.rerun()

# 6. ARCHIVE TOOL
from core.retriever import get_retriever


@tool
def search_royal_archives(query: str):
    """MANDATORY: Use this for any factual historical query."""
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
            evidence_list.append({"Source": title, "Page": page})
            seen.add(ref)
    st.session_state.temp_evidence = evidence_list
    return "\n".join(
        [f"Found in: {e['Source']} Page {e['Page']}" for e in evidence_list]
    )


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
                "You are Victoria, a formal British Histographer. Use search_royal_archives. Do NOT list sources in text.",
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


victoria = load_victoria()

# 8. MAIN INTERFACE
st.title("👑 Victoria: Histographer Agent")

# Focus Mode Logic
display_messages = st.session_state.messages
if st.session_state.focus_theme:
    st.warning(f"Now viewing: **{st.session_state.focus_theme}** records only.")
    idx = next(
        i
        for i, m in enumerate(st.session_state.messages)
        if m.get("theme") == st.session_state.focus_theme
    )
    display_messages = st.session_state.messages[idx : idx + 2]

# RENDER MESSAGES
for msg in display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("evidence"):
            with st.expander("📝 ARCHIVAL CITATIONS", expanded=True):
                st.table(msg["evidence"])

# Input
st.chat_input("Enter your inquiry...", key="user_text", on_submit=handle_input)

# 9. EXECUTION
if "pending_input" in st.session_state and st.session_state.pending_input:
    current_input = st.session_state.pop("pending_input")

    if len(current_input.split()) < 3 and "hello" in current_input.lower():
        answer = "Good day! How may I assist your research?"
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "evidence": None,
                "theme": "Greeting",
            }
        )
    else:
        with st.chat_message("assistant"):
            with st.status("Consulting Archives...", expanded=True) as status:
                response = victoria.invoke(
                    {
                        "input": current_input,
                        "chat_history": st.session_state.messages[:-1],
                    }
                )
                answer = response["output"]
                status.update(label="Complete", state="complete")
            st.markdown(answer)

            curr_ev = (
                st.session_state.temp_evidence
                if st.session_state.temp_evidence
                else None
            )
            if curr_ev:
                with st.expander("📝 ARCHIVAL CITATIONS", expanded=True):
                    st.table(curr_ev)

            last_theme = next(
                m["theme"]
                for m in reversed(st.session_state.messages)
                if m["role"] == "user"
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "evidence": curr_ev,
                    "theme": last_theme,
                }
            )
    st.rerun()
