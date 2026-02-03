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
# 1. PAGE SETUP
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
            "evidence": None,
        }
    ]


# ==========================================
# 3. ARCHIVE TOOL (UPDATED LOGIC)
# ==========================================
@tool
def search_royal_archives(query: str):
    """MANDATORY: Use this for any historical query. Returns document citations."""
    retriever = get_retriever()
    docs = retriever.invoke(query)

    evidence_list = []
    seen = set()
    for d in docs:
        fname = os.path.basename(d.metadata.get("source", ""))
        title = SOURCE_TITLES.get(fname, fname)
        page = d.metadata.get("page", "N/A")
        if f"{title}-{page}" not in seen:
            evidence_list.append({"Source": title, "Reference/Page": page})
            seen.add(f"{title}-{page}")

    # Store evidence directly in a specific execution key to prevent loss
    st.session_state["current_evidence"] = evidence_list

    if not evidence_list:
        return "No specific documents found in the archives."

    return f"I have found information in the following documents: {str(evidence_list)}"


# ==========================================
# 4. UI COMPONENTS
# ==========================================
st.title("Victoria 👑")
st.markdown("#### Histographer Agent")  # Removed years as requested

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
        "Choose Persona:", list(AVATARS.keys())[:-1]
    )
    if st.button("Clear History"):
        st.session_state.messages = [st.session_state.messages[0]]
        st.rerun()

# RENDER CHAT HISTORY
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])
        if msg.get("evidence"):
            st.warning("📜 **ARCHIVAL PROOF / SOURCES USED:**")
            st.dataframe(
                pd.DataFrame(msg["evidence"]), hide_index=True, use_container_width=True
            )

# ==========================================
# 5. EXECUTION
# ==========================================
user_input = st.chat_input("Inquire about history...")

if user_input:
    # Add user message to state
    st.session_state.messages.append(
        {"role": "user", "content": user_input, "avatar": "🎩", "evidence": None}
    )

    # Display user message
    with st.chat_message("user", avatar="🎩"):
        st.markdown(user_input)

    # Agent Processing
    with st.chat_message("assistant", avatar=AVATARS[st.session_state.current_style]):
        # Reset current evidence tracker for this specific turn
        st.session_state["current_evidence"] = []

        persona = {
            "Queen Victoria": "You are Queen Victoria. Use 'The Royal We'.",
            "Oscar Wilde": "You are Oscar Wilde. Be witty and flamboyant.",
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
                    "You are a reliable histographer. "
                    "1. ALWAYS call 'search_royal_archives' for ANY fact. "
                    "2. If someone asks about Rome or non-Victorian topics, give a brief historical answer based on general knowledge but prioritize Victorian topics. "
                    "3. DO NOT list sources in your text. The system handles the table.",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_openai_tools_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        with st.status("Consulting Records...") as status:
            response = executor.invoke(
                {"input": user_input, "chat_history": st.session_state.messages[:-1]}
            )
            status.update(label="Evidence Located", state="complete")

        # Output the answer
        st.markdown(response["output"])

        # DISPLAY THE TABLE (This is the "Completely Other Method")
        final_evidence = st.session_state.get("current_evidence", [])
        if final_evidence:
            st.warning("📜 **ARCHIVAL PROOF / SOURCES USED:**")
            st.dataframe(
                pd.DataFrame(final_evidence), hide_index=True, use_container_width=True
            )

        # Save to history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response["output"],
                "avatar": AVATARS[st.session_state.current_style],
                "evidence": final_evidence,
            }
        )
