# ==========================================
# 0. IMPORTS
# ==========================================

import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool

# ==========================================
# 1. CORE CONFIGURATION & CONSTANTS
# ==========================================
st.set_page_config(page_title="Victoria", page_icon="👑", layout="wide")
load_dotenv()

SOURCE_TITLES = {
    "20-Industrial-Rev.pdf": "The Industrial Revolution Archives (Vol. 20)",
    "Chapter-8-The-Industrial-Revolution.pdf": "British Industrial History, Chapter VIII",
    "sadler-report.pdf": "The Michael Sadler Report on Factory Labor (1832)",
    "mines-act-1842.pdf": "Royal Commission on Children's Employment in Mines",
    "2020_Kelly_Mokyr_Mechanics_Ind_Rev.pdf": "Mechanics of the Industrial Revolution (Kelly & Mokyr)",
    "1851-Great-Exhibition-Malta.pdf": "Official Catalog: Works of Industry of All Nations (1851)",
    "Getty-practical-treatise-on-weaving.pdf": "A Practical Treatise on Weaving and Designing (Ashenhurst, 1885)",
    "MPRA_paper_96644.pdf": "The First Industrial Revolution: Creation of a New Global Era",
    "WHP-6526-Read--Innovations-and-Inventions.pdf": "Innovations and Innovators of the Industrial Revolution (OER Project)",
}

STYLE_AVATARS = {
    "Queen Victoria": "👑",
    "Oscar Wilde": "🎭",
    "Jack the Ripper": "🔪",
    "Isambard Kingdom Brunel": "⚙️",
}


# ==========================================
# 2. STATE MANAGEMENT
# ==========================================
def initialize_state():
    """Ensures all session state variables are present."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "We are pleased to receive you. How may We assist your research into Our Empire today?",
                "evidence": None,
                "theme": "Greeting",
                "avatar": "👑",
            }
        ]
    if "temp_evidence" not in st.session_state:
        st.session_state.temp_evidence = []
    if "focus_theme" not in st.session_state:
        st.session_state.focus_theme = None
    if "current_style" not in st.session_state:
        st.session_state.current_style = "Queen Victoria"


initialize_state()


# ==========================================
# 3. HELPER FUNCTIONS & TOOLS
# ==========================================
def identify_theme(text):
    """Summarizes query for history categorization."""
    if not text or len(text) < 5:
        return "General"
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(
        f"Summarize this historical query into 2-3 words. Query: {text}"
    )
    return response.content.strip().replace('"', "")


@tool
def search_royal_archives(query: str):
    """MANDATORY: Use this for any factual historical query."""
    from core.retriever import get_retriever  # Localized import for cleaner flow

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
    return "\n".join(
        [f"Found in: {e['Source Title']} Page {e['Page']}" for e in evidence_list]
    )


# ==========================================
# 4. AGENT LOGIC
# ==========================================
@st.cache_resource
def load_victoria(style, strength):
    """Creates the LangChain agent based on persona and strength."""
    from core.tools import victorian_currency_converter, industry_stats_calculator

    tools = [
        search_royal_archives,
        victorian_currency_converter,
        industry_stats_calculator,
    ]

    prompts = {
        "Queen Victoria": "Queen Victoria. Royal We. Sovereign and moral.",
        "Oscar Wilde": "Oscar Wilde. Aesthetic, witty, and paradoxical.",
        "Jack the Ripper": "Jack the Ripper. Menacing cockney shadows.",
        "Isambard Kingdom Brunel": "Brunel. Passionate engineering and progress.",
    }

    modifiers = {
        1: "Professional and subtle.",
        2: "Distinct personality and phrases.",
        3: "Extreme method acting and theatricality.",
    }

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"You are Victoria. Persona: {prompts[style]}. Strength: {modifiers[strength]}. Use search_royal_archives.",
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    temp = {1: 0.1, 2: 0.5, 3: 0.9}[strength]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=temp)
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )


# ==========================================
# 5. SIDEBAR & NAVIGATION
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/scroll.png")
    st.title("Correspondence Archives")

    st.divider()
    st.subheader("Historical Persona")
    st.session_state.current_style = st.selectbox(
        "Select Correspondent:", list(STYLE_AVATARS.keys()), index=0
    )
    st.session_state.char_strength = st.slider("Character Strength:", 1, 3, 2)

    st.divider()
    st.subheader("Inquiry History")

    all_themes = [
        m.get("theme")
        for m in st.session_state.messages
        if m.get("theme") and m["role"] == "user"
    ]
    if st.button("👁️ Show All History"):
        st.session_state.focus_theme = None

    for i, theme in enumerate(reversed(all_themes)):
        if st.button(f"📜 {theme}", key=f"hist_{i}_{theme}", use_container_width=True):
            st.session_state.focus_theme = theme

    if st.button("🗑️ Reset Archive", type="secondary"):
        st.session_state.clear()
        st.rerun()

# ==========================================
# 6. MAIN CHAT INTERFACE
# ==========================================
st.title("Victoria 👑")
st.caption("### Victorian Era Histographer")

display_messages = st.session_state.messages
if st.session_state.focus_theme:
    st.info(f"Viewing records related to: **{st.session_state.focus_theme}**")
    idx = next(
        i
        for i, m in enumerate(st.session_state.messages)
        if m.get("theme") == st.session_state.focus_theme
    )
    display_messages = st.session_state.messages[idx : idx + 2]

for msg in display_messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])
        if msg.get("evidence"):
            with st.expander("📝 ARCHIVAL CITATIONS"):
                st.table(msg["evidence"])


# ==========================================
# 7. CHAT INPUT & EXECUTION
# ==========================================
def handle_input():
    if st.session_state.user_text:
        theme = identify_theme(st.session_state.user_text)
        st.session_state.messages.append(
            {
                "role": "user",
                "content": st.session_state.user_text,
                "theme": theme,
                "avatar": None,
            }
        )
        st.session_state.pending_input = st.session_state.user_text
        st.session_state.user_text = ""


st.chat_input("Enter your inquiry...", key="user_text", on_submit=handle_input)

if "pending_input" in st.session_state and st.session_state.pending_input:
    current_input = st.session_state.pop("pending_input")
    active_avatar = STYLE_AVATARS.get(st.session_state.current_style, "🤖")
    victoria = load_victoria(
        st.session_state.current_style, st.session_state.char_strength
    )

    with st.chat_message("assistant", avatar=active_avatar):
        with st.status("Searching Royal Archives...", expanded=True) as status:
            response = victoria.invoke(
                {"input": current_input, "chat_history": st.session_state.messages[:-1]}
            )
            status.update(label="Archives Consulted", state="complete")

        st.markdown(response["output"])

        curr_ev = (
            st.session_state.temp_evidence if st.session_state.temp_evidence else None
        )
        if curr_ev:
            with st.expander("📝 ARCHIVAL CITATIONS"):
                st.table(curr_ev)

        last_theme = next(
            m["theme"]
            for m in reversed(st.session_state.messages)
            if m["role"] == "user"
        )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response["output"],
                "evidence": curr_ev,
                "theme": last_theme,
                "avatar": active_avatar,
            }
        )
    st.rerun()
