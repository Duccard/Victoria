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
# 1. PAGE SETUP
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

# ==========================================
# 2. STATE INITIALIZATION
# ==========================================
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


# ==========================================
# 3. THEME IDENTIFIER
# ==========================================
def identify_theme(text):
    if not text or len(text) < 5:
        return "General"
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(
        f"Summarize this historical query into 2-3 words. Query: {text}"
    )
    return response.content.strip().replace('"', "")


# ==========================================
# 4. SIDEBAR CALLBACK
# ==========================================
def handle_input():
    if st.session_state.user_text:
        new_prompt = st.session_state.user_text
        theme = identify_theme(new_prompt)
        st.session_state.messages.append(
            {
                "role": "user",
                "content": new_prompt,
                "evidence": None,
                "theme": theme,
                "avatar": "👤",
            }
        )
        st.session_state.pending_input = new_prompt
        st.session_state.temp_evidence = []
        st.session_state.focus_theme = None


# ==========================================
# 5. SETTINGS SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/settings.png")  # Clog/Settings Logo
    st.title("Settings")

    st.divider()
    st.subheader("Historical Persona")
    style_choice = st.selectbox(
        "Select Correspondent:",
        ["Queen Victoria", "Oscar Wilde", "Jack the Ripper", "Isambard Kingdom Brunel"],
        index=0,
    )
    st.session_state.current_style = style_choice

    st.divider()
    st.subheader("Inquiry History")

    all_themes = [
        m.get("theme")
        for m in st.session_state.messages
        if m.get("theme") and m["role"] == "user"
    ]

    if st.button("👁️ Show All Records"):
        st.session_state.focus_theme = None

    for i, theme in enumerate(reversed(all_themes)):
        if st.button(f"📜 {theme}", key=f"hist_{i}_{theme}", use_container_width=True):
            st.session_state.focus_theme = theme

    st.divider()
    if st.button("🗑️ Reset Archive"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Archives cleared.",
                "evidence": None,
                "theme": "Greeting",
                "avatar": "👑",
            }
        ]
        st.session_state.focus_theme = None
        st.rerun()

# ==========================================
# 6. ARCHIVE TOOL
# ==========================================
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
            evidence_list.append({"Source Title": title, "Page": page})
            seen.add(ref)
    st.session_state.temp_evidence = evidence_list
    return "\n".join(
        [f"Found in: {e['Source Title']} Page {e['Page']}" for e in evidence_list]
    )


# ==========================================
# 7. AGENT SETUP
# ==========================================
@st.cache_resource
def load_victoria(style):
    from core.tools import victorian_currency_converter, industry_stats_calculator

    tools = [
        search_royal_archives,
        victorian_currency_converter,
        industry_stats_calculator,
    ]

    style_prompts = {
        "Queen Victoria": "You are Her Majesty Queen Victoria. Speak with absolute charisma and royal authority. Use the 'Royal We'.",
        "Oscar Wilde": "You are Oscar Wilde. Be charismatic, witty, and flamboyant. Every word should be aesthetic and sharp.",
        "Jack the Ripper": "Speak in a dark, charismatic, yet menacing cockney whisper. Speak from the London fog.",
        "Isambard Kingdom Brunel": "You are Brunel. Speak with charismatic passion for engineering and progress.",
    }

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"{style_prompts.get(style)} \n\nMANDATORY: Search archives for every query. Never mention filenames.",
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )


victoria = load_victoria(st.session_state.get("current_style", "Queen Victoria"))

# ==========================================
# 8. MAIN INTERFACE
# ==========================================
st.title("Victoria 👑")
st.caption("Victorian Era Histographer")

display_messages = st.session_state.messages
if st.session_state.focus_theme:
    st.info(f"Viewing records related to: **{st.session_state.focus_theme}**")
    display_messages = [
        m
        for m in st.session_state.messages
        if m.get("theme") == st.session_state.focus_theme
    ]

AVATARS = {
    "Queen Victoria": "👑",
    "Oscar Wilde": "🎭",
    "Jack the Ripper": "🔪",
    "Isambard Kingdom Brunel": "🎩",
    "user": "👤",
}

for msg in display_messages:
    icon = msg.get("avatar", "👑")
    with st.chat_message(msg["role"], avatar=icon):
        st.markdown(msg["content"])
        if msg.get("evidence"):
            with st.expander("📝 ARCHIVAL CITATIONS", expanded=True):
                st.table(msg["evidence"])

st.chat_input("Enter your inquiry...", key="user_text", on_submit=handle_input)

# ==========================================
# 9. EXECUTION
# ==========================================
if "pending_input" in st.session_state and st.session_state.pending_input:
    current_input = st.session_state.pop("pending_input")
    current_persona = st.session_state.current_style

    with st.chat_message("assistant", avatar=AVATARS.get(current_persona, "👑")):
        with st.status("Searching Royal Archives...") as status:
            response = victoria.invoke(
                {"input": current_input, "chat_history": st.session_state.messages[:-1]}
            )
            answer = response["output"]
            status.update(label="Archives Consulted", state="complete")

        st.markdown(answer)
        curr_ev = (
            st.session_state.temp_evidence if st.session_state.temp_evidence else None
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
                "avatar": AVATARS.get(current_persona, "👑"),
            }
        )
    st.rerun()
