import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from core.retriever import get_retriever

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
            {"role": "user", "content": new_prompt, "evidence": None, "theme": theme}
        )
        st.session_state.pending_input = new_prompt
        st.session_state.temp_evidence = []
        st.session_state.user_text = ""


# ==========================================
# 5. SIDEBAR (INQUIRY HISTORY)
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/scroll.png")
    st.title("Correspondence Archives")
    st.divider()
    style_choice = st.selectbox(
        "Select Correspondent:",
        ["Queen Victoria", "Oscar Wilde", "Jack the Ripper", "Isambard Kingdom Brunel"],
    )
    st.session_state.current_style = style_choice
    st.divider()
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


# ==========================================
# 6. ARCHIVE TOOL
# ==========================================
@tool
def search_royal_archives(query: str):
    """MANDATORY: Use this for factual queries. Searches documents for years, inventors, and patent data."""
    retriever = get_retriever()
    docs = retriever.invoke(query)
    evidence_list, seen, full_content = [], set(), ""
    if not docs:
        st.session_state.temp_evidence = [
            {"Source Title": "No matching records found", "Page": "N/A"}
        ]
        return "No records found."
    for d in docs:
        fname = os.path.basename(d.metadata.get("source", ""))
        title, page = SOURCE_TITLES.get(fname, fname), d.metadata.get("page", "N/A")
        if f"{title}-{page}" not in seen:
            evidence_list.append({"Source Title": title, "Page": page})
            seen.add(f"{title}-{page}")
        full_content += f"\n[Doc: {title} pg {page}]: {d.page_content}\n"
    st.session_state.temp_evidence = evidence_list
    return full_content


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
        "Queen Victoria": "You are Her Majesty Queen Victoria. Use 'Royal We'. Be dignified.",
        "Oscar Wilde": "You are Oscar Wilde. Be witty and aesthetic. Use sharp dry humor.",
        "Jack the Ripper": "Speak in a dark, mysterious whisper. Use cockney slang.",
        "Isambard Kingdom Brunel": "You are the great engineer. Speak with passion for iron and steam.",
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"{style_prompts.get(style)} \n\nMANDATORY: Search archives for every query. Include dates/patent numbers (e.g. GB913) if found. Never mention filenames in dialogue.",
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    temp = 0.7 if style in ["Oscar Wilde", "Jack the Ripper"] else 0.1
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=temp)
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )


# ==========================================
# 8. MAIN INTERFACE (RESTORED FILTER)
# ==========================================
victoria = load_victoria(st.session_state.get("current_style", "Queen Victoria"))
st.title("Victoria 👑")
display_messages = st.session_state.messages
if st.session_state.focus_theme:
    st.info(f"Viewing records related to: **{st.session_state.focus_theme}**")
    display_messages = [
        m
        for m in st.session_state.messages
        if m.get("theme") == st.session_state.focus_theme
    ]

for msg in display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("evidence"):
            with st.expander("📝 ARCHIVAL CITATIONS", expanded=True):
                st.table(msg["evidence"])

st.chat_input("Enter your inquiry...", key="user_text", on_submit=handle_input)

# ==========================================
# 9. EXECUTION (THEME TRACKING)
# ==========================================
if "pending_input" in st.session_state and st.session_state.pending_input:
    current_input = st.session_state.pop("pending_input")
    current_theme = identify_theme(current_input)
    with st.chat_message("assistant"):
        with st.status("Searching the Royal Archives...") as status:
            response = victoria.invoke(
                {"input": current_input, "chat_history": st.session_state.messages[:-1]}
            )
            status.update(label="Complete", state="complete")
        st.markdown(response["output"])
        curr_ev = (
            st.session_state.temp_evidence
            if st.session_state.temp_evidence
            else [{"Source Title": "No evidence found", "Page": "N/A"}]
        )
        with st.expander("📝 ARCHIVAL CITATIONS", expanded=True):
            st.table(curr_ev)
        if (
            st.session_state.messages
            and st.session_state.messages[-1]["role"] == "user"
        ):
            st.session_state.messages[-1]["theme"] = current_theme
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response["output"],
                "evidence": curr_ev,
                "theme": current_theme,
            }
        )
    st.rerun()
