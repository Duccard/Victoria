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

# Complete list of documents based on your archive directory
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
    if not text or len(text) < 5:
        return "General"
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        response = llm.invoke(
            f"Summarize this historical query into 2 words. Query: {text}"
        )
        return response.content.strip().replace('"', "")
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
                "avatar": "🎩",  # Cylinder Hat user icon
            }
        )
        st.session_state.pending_input = new_prompt
        st.session_state.temp_evidence = []


# ==========================================
# 4. SETTINGS SIDEBAR (Clog Logo)
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/settings.png")
    st.title("Settings")
    st.divider()

    style_choice = st.selectbox(
        "Select Correspondent:",
        ["Queen Victoria", "Oscar Wilde", "Jack the Ripper", "Isambard Kingdom Brunel"],
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
        if st.button(f"📜 {theme}", key=f"hist_{i}", use_container_width=True):
            st.session_state.focus_theme = theme

# ==========================================
# 5. ARCHIVE TOOL
# ==========================================
from core.retriever import get_retriever


@tool
def search_royal_archives(query: str):
    """MANDATORY: Consult the archives for all historical facts."""
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
        [f"Document: {e['Source Title']} (Page {e['Page']})" for e in evidence_list]
    )


# ==========================================
# 6. AGENT SETUP
# ==========================================
@st.cache_resource
def load_persona(style):
    from core.tools import victorian_currency_converter, industry_stats_calculator

    tools = [
        search_royal_archives,
        victorian_currency_converter,
        industry_stats_calculator,
    ]

    prompts = {
        "Queen Victoria": "You are Her Majesty Queen Victoria. Speak with absolute royal charisma and authority. Use 'The Royal We'. You are the custodian of history. Cite your pages with pride.",
        "Oscar Wilde": "You are Oscar Wilde. Be devastatingly charismatic, flamboyant, and witty. Charm the user with aesthetic brilliance and sharp epigrams while citing the records.",
        "Jack the Ripper": "Speak in a dark, charismatic, terrifying cockney whisper. You are the shadow in the Whitechapel fog. Cite the evidence of your era with a chilling flair.",
        "Isambard Kingdom Brunel": "You are the charismatic titan of engineering, Brunel. Speak with passion for iron, steam, and progress. Use the archives to support your visionary claims.",
    }

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"{prompts.get(style)} \n\nMANDATORY: You MUST search the archives for names or dates. Reference specific documents and page numbers in your charismatic response.",
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )


agent_executor = load_persona(st.session_state.get("current_style", "Queen Victoria"))

# ==========================================
# 7. MAIN INTERFACE
# ==========================================
st.title("Victoria 👑")
st.markdown("## **Victorian Era Histographer**")  # Bold, larger subtitle

AVATARS = {
    "Queen Victoria": "👑",
    "Oscar Wilde": "🎭",
    "Jack the Ripper": "🔪",
    "Isambard Kingdom Brunel": "🎩",
    "user": "🎩",
}

display_messages = st.session_state.messages
if st.session_state.focus_theme:
    st.info(f"Viewing records: **{st.session_state.focus_theme}**")
    display_messages = [
        m
        for m in st.session_state.messages
        if m.get("theme") == st.session_state.focus_theme
    ]

for msg in display_messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])
        if msg.get("evidence"):
            with st.expander("📝 ARCHIVAL EVIDENCE", expanded=True):
                st.table(msg["evidence"])

st.chat_input("Enter your inquiry...", key="user_text", on_submit=handle_input)

# ==========================================
# 8. EXECUTION
# ==========================================
if "pending_input" in st.session_state and st.session_state.pending_input:
    current_input = st.session_state.pop("pending_input")
    current_persona = st.session_state.current_style

    with st.chat_message("assistant", avatar=AVATARS.get(current_persona, "👑")):
        with st.status("Consulting the Royal Archives...") as status:
            response = agent_executor.invoke(
                {"input": current_input, "chat_history": st.session_state.messages[:-1]}
            )
            answer = response["output"]
            status.update(label="Archives Consulted", state="complete")

        st.markdown(answer)
        curr_ev = (
            st.session_state.temp_evidence if st.session_state.temp_evidence else None
        )

        if curr_ev:
            with st.expander("📝 ARCHIVAL EVIDENCE", expanded=True):
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
