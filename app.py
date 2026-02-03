# ==========================================
# 0. IMPORTS
# ==========================================
import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from core.retriever import get_retriever

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
            "content": "Good day. How may I assist your research today?",
            "avatar": "👑",
            "theme": "Greeting",
            "evidence": None,
        }
    ]
if "focus_theme" not in st.session_state:
    st.session_state.focus_theme = None
if "current_evidence" not in st.session_state:
    st.session_state.current_evidence = []


# ==========================================
# 3. UTILITIES
# ==========================================
def identify_theme(text):
    if not text or len(text) < 2:
        return "Inquiry"
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        res = llm.invoke(f"Summarize this historical query into 2 words: {text}")
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
        st.session_state.current_evidence = []
        st.session_state.focus_theme = None


# ==========================================
# 4. ARCHIVE TOOL (FIXED EVIDENCE OUTPUT)
# ==========================================
@tool
def search_royal_archives(query: str):
    """MANDATORY: Use this to retrieve documents and page numbers for the evidence table."""
    retriever = get_retriever()
    docs = retriever.invoke(query)
    evidence_list = []
    seen = set()
    for d in docs:
        fname = os.path.basename(d.metadata.get("source", ""))
        title = SOURCE_TITLES.get(fname, fname)
        page = d.metadata.get("page", "N/A")
        if f"{title}-{page}" not in seen:
            evidence_list.append({"Source Title": title, "Reference/Page": page})
            seen.add(f"{title}-{page}")

    # FORCED INJECTION: Ensuring evidence is saved even if the agent loop is complex
    st.session_state.current_evidence = evidence_list
    return f"Documents found: {str(evidence_list)}"


# ==========================================
# 5. SIDEBAR & INTERFACE
# ==========================================
AVATARS = {
    "Queen Victoria": "👑",
    "Oscar Wilde": "🎭",
    "Jack the Ripper": "🔪",
    "Isambard Kingdom Brunel": "⚙️",
    "user": "🎩",
}

with st.sidebar:
    st.title("Correspondent")
    st.session_state.current_style = st.selectbox(
        "Select Character:", list(AVATARS.keys())[:-1]
    )

    st.divider()
    st.subheader("📜 Inquiry History")  # RESTORED SECTION
    all_themes = [
        m.get("theme")
        for m in st.session_state.messages
        if m.get("theme") and m["role"] == "user"
    ]

    if st.button("👁️ Show All Records", use_container_width=True):
        st.session_state.focus_theme = None

    for i, theme in enumerate(reversed(list(dict.fromkeys(all_themes)))):
        if st.button(f"📜 {theme}", key=f"hist_{i}", use_container_width=True):
            st.session_state.focus_theme = theme

    st.divider()
    if st.button("🗑️ Reset Archive", use_container_width=True):
        st.session_state.messages = [st.session_state.messages[0]]
        st.rerun()

st.title("Victoria 👑")
st.markdown("#### Histographer Agent")

# Filter logic for history
display_messages = st.session_state.messages
if st.session_state.focus_theme:
    st.info(f"Viewing records for: **{st.session_state.focus_theme}**")
    display_messages = [
        m
        for m in st.session_state.messages
        if m.get("theme") == st.session_state.focus_theme or m["role"] == "assistant"
    ]

for msg in display_messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])
        if msg.get("evidence"):
            st.info("📂 **VERIFIED SOURCES FROM ARCHIVES**")
            st.table(msg["evidence"])

st.chat_input("Enter your inquiry...", key="user_text", on_submit=handle_input)

# ==========================================
# 6. EXECUTION
# ==========================================
if "pending_input" in st.session_state and st.session_state.pending_input:
    current_input = st.session_state.pop("pending_input")

    persona = {
        "Queen Victoria": "You are Her Majesty Queen Victoria. Speak with absolute royal authority.",
        "Oscar Wilde": "You are Oscar Wilde. Speak with wit and flamboyant elegance.",
        "Jack the Ripper": "Speak in a dark, menacing whisper.",
        "Isambard Kingdom Brunel": "Speak with engineering passion.",
    }[st.session_state.current_style]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [search_royal_archives]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"{persona}\n\n"
                "MANDATORY: You MUST use 'search_royal_archives' for historical facts. "
                "DO NOT write the sources in your response. "
                "If it is outside Victorian history, use general knowledge but mention it is outside the royal records.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    with st.chat_message("assistant", avatar=AVATARS[st.session_state.current_style]):
        with st.status("Accessing Royal Vaults...") as status:
            response = executor.invoke(
                {"input": current_input, "chat_history": st.session_state.messages[:-1]}
            )
            status.update(label="Evidence Retrieved", state="complete")

        st.markdown(response["output"])

        # DISPLAY AND SAVE EVIDENCE
        final_ev = st.session_state.current_evidence
        if final_ev:
            st.info("📂 **VERIFIED SOURCES FROM ARCHIVES**")
            st.table(final_ev)

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
                "avatar": AVATARS[st.session_state.current_style],
                "evidence": final_ev,
                "theme": last_theme,
            }
        )
    st.rerun()
