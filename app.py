import streamlit as st
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain.tools import tool

# --- 0. TECHNICAL IMPLEMENTATION: LOGGING & MONITORING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("victoria_research.log"), logging.StreamHandler()],
)
logger = logging.getLogger("VictoriaAgent")

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
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        response = llm.invoke(
            f"Summarize this historical query into 2-3 words (e.g., 'Steam Engine' or 'Child Labor'). Query: {text}"
        )
        return response.content.strip().replace('"', "")
    except Exception as e:
        logger.error(f"Theme identification failed: {e}")
        return "General Inquiry"


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


# 5. SIDEBAR
with st.sidebar:
    st.image("https://img.icons8.com/color/96/scroll.png")
    st.title("Your Enquiries History")
    st.divider()

    all_themes = [
        m.get("theme")
        for m in st.session_state.messages
        if m.get("theme") and m["role"] == "user"
    ]

    if st.button("👁️ Show All History"):
        st.session_state.focus_theme = None

    for theme in reversed(all_themes):
        if st.button(f"📜 {theme}", use_container_width=True):
            st.session_state.focus_theme = theme

    st.divider()
    if st.button("🗑️ Reset Archive"):
        logger.info("Session state cleared by user.")
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

# 6. ARCHIVE TOOL (With Error Handling)
from core.retriever import get_retriever


@tool
def search_royal_archives(query: str):
    """MANDATORY: Use this for any factual historical query."""
    try:
        logger.info(f"Retrieval initiated for query: {query}")
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
    except Exception as e:
        logger.error(f"Error in search_royal_archives: {e}")
        return "I encountered a difficulty accessing the physical archives. Please rephrase."


# 7. AGENT SETUP (With Rate Limiting)
@st.cache_resource
def load_victoria():
    from core.tools import victorian_currency_converter, industry_stats_calculator

    # --- TECHNICAL IMPLEMENTATION: RATE LIMITING ---
    # Limits the agent to 1 request every 2 seconds to avoid API ban
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=0.5, check_every_n_seconds=0.1, max_bucket_size=10
    )

    tools = [
        search_royal_archives,
        victorian_currency_converter,
        industry_stats_calculator,
    ]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, rate_limiter=rate_limiter)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are Victoria, a formal British Histographer. Use search_royal_archives for history. Do NOT list sources in text.",
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )


victoria = load_victoria()

# 8. MAIN INTERFACE
st.title("Victoria 👑")
st.subheader("Victorian Era Histographer Agent")

display_messages = st.session_state.messages
if st.session_state.focus_theme:
    st.info(f"Viewing records related to: **{st.session_state.focus_theme}**")
    idx = next(
        i
        for i, m in enumerate(st.session_state.messages)
        if m.get("theme") == st.session_state.focus_theme
    )
    display_messages = st.session_state.messages[idx : idx + 2]

# RENDER
for msg in display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("evidence"):
            with st.expander("📝 ARCHIVAL CITATIONS", expanded=True):
                st.table(msg["evidence"])

# Input
st.chat_input("Enter your inquiry...", key="user_text", on_submit=handle_input)

# 9. EXECUTION (With Validation & Logging)
if "pending_input" in st.session_state and st.session_state.pending_input:
    current_input = st.session_state.pop("pending_input")

    # Basic Validation
    if not current_input.strip():
        st.warning("Please enter a valid historical inquiry.")
    elif len(current_input.split()) < 3 and "hello" in current_input.lower():
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
                try:
                    logger.info(f"Agent executing for input: {current_input}")
                    response = victoria.invoke(
                        {
                            "input": current_input,
                            "chat_history": st.session_state.messages[:-1],
                        }
                    )
                    answer = response["output"]
                    status.update(label="Complete", state="complete")
                except Exception as e:
                    logger.error(f"Agent Execution Error: {e}")
                    answer = "I apologize, but my research was interrupted by a technical fault."
                    status.update(label="Error", state="error")

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
