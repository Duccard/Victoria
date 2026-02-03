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
# 1. PAGE SETUP & DATA
# ==========================================
st.set_page_config(page_title="Victoria", page_icon="👑", layout="wide")
load_dotenv()

SOURCE_TITLES = {
    "20-Industrial-Rev.pdf": "The Industrial Revolution and its Impact on European Society",
    "1851_GreatExhibition_Cap...gue.pdf": "Official Descriptive and Illustrated Catalogue of the Great Exhibition (1851)",
    "2020_Kelly_Mokyr_Mech..._Rev.pdf": "The Mechanics of the Industrial Revolution",
    "Chapter-8-The-Industri...ution.pdf": "The Industrial Revolution: British History Chapter VIII",
    "Getty_Research_Institute...w_0).pdf": "The Construction of the Power Loom and the Art of Weaving",
    "key_inventions.csv": "Chronological Register of Key Victorian Inventions",
    "MPRA_paper_96644.pdf": "The First Industrial Revolution: Creation of a New Global Era",
    "The_Sadler_Report_Repo...abor.pdf": "The Sadler Report: Royal Commission on Factory Children's Labour",
    "WHP 6526 Read Innovati...30L.pdf": "Innovations and Innovators of the Industrial Revolution (WHP Project)",
}

# ==========================================
# 2. STATE INITIALIZATION
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "We are pleased to receive you. How may We assist your research into Our Empire today?",
            "avatar": "👑",
            "theme": "Greeting",
            "evidence": None,
        }
    ]
if "temp_evidence" not in st.session_state:
    st.session_state.temp_evidence = []
if "focus_theme" not in st.session_state:
    st.session_state.focus_theme = None


# ==========================================
# 3. UTILITIES
# ==========================================
def identify_theme(text):
    if not text or len(text) < 2:
        return "Inquiry"
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        res = llm.invoke(f"Summarize into 2 words: {text}")
        return res.content.strip().replace('"', "")
    except:
        return "Inquiry"


def handle_input():
    if st.session_state.user_text:
        new_prompt = st.session_state.user_text
        theme = identify_theme(new_prompt)
        st.session_state.messages.append(
            {
                "role": "user",
                "content": new_prompt,
                "theme": theme,
                "avatar": "🎩",
                "evidence": None,
            }
        )
        st.session_state.pending_input = new_prompt
        st.session_state.temp_evidence = []


# ==========================================
# 4. ARCHIVE TOOL
# ==========================================
from core.retriever import get_retriever


@tool
def search_royal_archives(query: str):
    """MANDATORY: Consult the archives for every historical fact."""
    retriever = get_retriever()
    docs = retriever.invoke(query)
    evidence_list = []
    seen = set()
    for d in docs:
        fname = os.path.basename(d.metadata.get("source", ""))
        title = SOURCE_TITLES.get(fname, fname)
        page = d.metadata.get("page", "N/A")
        if f"{title}-{page}" not in seen:
            evidence_list.append({"Source Title": title, "Page": page})
            seen.add(f"{title}-{page}")
    st.session_state.temp_evidence = evidence_list
    return "\n".join(
        [f"Source: {e['Source Title']} (Page {e['Page']})" for e in evidence_list]
    )


# ==========================================
# 5. MAIN INTERFACE
# ==========================================
st.title("Victoria 👑")
st.markdown("#### Victorian Era Histographer")  # Medium-Small Subtitle

AVATARS = {
    "Queen Victoria": "👑",
    "Oscar Wilde": "🎭",
    "Jack the Ripper": "🔪",
    "Isambard Kingdom Brunel": "⚙️",  # Brunel is a Cog
    "user": "🎩",
}

with st.sidebar:
    st.image("https://img.icons8.com/color/96/settings.png")
    st.title("Settings")

    style_choice = st.selectbox("Select Correspondent:", list(AVATARS.keys())[:-1])
    st.session_state.current_style = style_choice

    st.divider()
    st.subheader("Inquiry History")
    all_themes = [
        m.get("theme")
        for m in st.session_state.messages
        if m.get("theme") and m["role"] == "user"
    ]

    if st.button("👁️ Show All Records", use_container_width=True):
        st.session_state.focus_theme = None

    for i, theme in enumerate(reversed(all_themes)):
        if st.button(f"📜 {theme}", key=f"hist_{i}", use_container_width=True):
            st.session_state.focus_theme = theme

    st.divider()
    if st.button("🗑️ Reset Archive", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Archives cleared.",
                "avatar": "👑",
                "theme": "Greeting",
            }
        ]
        st.rerun()

display_messages = st.session_state.messages
if st.session_state.focus_theme:
    st.info(f"Viewing records: **{st.session_state.focus_theme}**")
    display_messages = [
        m
        for m in st.session_state.messages
        if m.get("theme") == st.session_state.focus_theme or m["role"] == "assistant"
    ]

for msg in display_messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])
        if msg.get("evidence"):
            with st.expander("📝 ARCHIVAL EVIDENCE", expanded=True):
                st.table(msg["evidence"])

st.chat_input("Enter your inquiry...", key="user_text", on_submit=handle_input)

# ================= 6. EXECUTION  =================
if "pending_input" in st.session_state and st.session_state.pending_input:
    current_input = st.session_state.pop("pending_input")

    from core.tools import victorian_currency_converter, industry_stats_calculator

    tools = [
        search_royal_archives,
        victorian_currency_converter,
        industry_stats_calculator,
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"You are {st.session_state.current_style}. Be charismatic. MANDATORY: Search archives for ALL facts.",
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    agent = create_openai_tools_agent(llm, tools, prompt)
    vic_agent = AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )

    with st.chat_message("assistant", avatar=AVATARS[st.session_state.current_style]):
        with st.status("Consulting Archives...") as status:
            response = vic_agent.invoke(
                {"input": current_input, "chat_history": st.session_state.messages[:-1]}
            )
            status.update(label="Complete", state="complete")

        st.markdown(response["output"])

        evidence = st.session_state.temp_evidence
        if evidence:
            with st.expander("📝 ARCHIVAL EVIDENCE", expanded=True):
                st.table(evidence)

        last_theme = next(
            (
                m["theme"]
                for m in reversed(st.session_state.messages)
                if m["role"] == "user"
            ),
            "Inquiry",
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response["output"],
                "evidence": evidence,
                "theme": last_theme,
                "avatar": AVATARS[st.session_state.current_style],
            }
        )
    st.rerun()
