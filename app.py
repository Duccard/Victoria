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
        }
    ]
# We keep a temporary list for the current tool call
if "temp_evidence" not in st.session_state:
    st.session_state.temp_evidence = []


# --- 3. SIDEBAR CALLBACK ---
def handle_input():
    if st.session_state.user_text:
        new_prompt = st.session_state.user_text
        # Append user message with no evidence
        st.session_state.messages.append(
            {"role": "user", "content": new_prompt, "evidence": None}
        )
        st.session_state.pending_input = new_prompt
        st.session_state.temp_evidence = []
        st.session_state.user_text = ""


# 4. SIDEBAR
with st.sidebar:
    st.image("https://img.icons8.com/color/96/scroll.png")
    st.title("📜 Research Log")
    st.divider()
    user_queries = [
        m["content"] for m in st.session_state.messages if m["role"] == "user"
    ]
    for i, query in enumerate(user_queries):
        st.caption(f"{i+1}. {query[:40]}...")

    if st.button("🗑️ Reset Archive"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Archives cleared.", "evidence": None}
        ]
        st.session_state.temp_evidence = []
        st.rerun()

# 5. ARCHIVE TOOL
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

    # Save to temp area to be picked up by the message loop
    st.session_state.temp_evidence = evidence_list
    return "\n".join(
        [f"Found in: {e['Source Title']} Page {e['Page']}" for e in evidence_list]
    )


# 6. AGENT SETUP
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
                """You are Victoria, a formal British Histographer. 
        When asked about history, use 'search_royal_archives'. 
        IMPORTANT: Do NOT list your sources or page numbers in your text. 
        Narrate the history elegantly.""",
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


victoria = load_victoria()

# 7. MAIN INTERFACE
st.title("👑 Victoria: Histographer Agent")

# RENDER MESSAGES + PERMANENT EVIDENCE
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If this message has evidence attached, show it right here!
        if msg.get("evidence"):
            with st.expander("📝 ARCHIVAL CITATIONS", expanded=False):
                st.table(msg["evidence"])

# Input Box
st.chat_input("Enter your inquiry...", key="user_text", on_submit=handle_input)

# 9. LOGIC
if "pending_input" in st.session_state and st.session_state.pending_input:
    current_input = st.session_state.pop("pending_input")

    greetings = ["hello", "hi", "greetings", "good day"]
    if current_input.lower().strip() in greetings:
        answer = "Good day! How may I assist your research into our glorious era?"
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "evidence": None}
        )
    else:
        with st.chat_message("assistant"):
            with st.status("Searching the Royal Archives...", expanded=True) as status:
                response = victoria.invoke(
                    {
                        "input": current_input,
                        "chat_history": st.session_state.messages[:-1],
                    }
                )
                answer = response["output"]
                status.update(label="Consultation Complete", state="complete")
            st.markdown(answer)

            # Attach the evidence to this specific assistant message
            current_evidence = (
                st.session_state.temp_evidence
                if st.session_state.temp_evidence
                else None
            )
            if current_evidence:
                with st.expander("📝 ARCHIVAL CITATIONS", expanded=True):
                    st.table(current_evidence)

            # Save the message AND its specific evidence together
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "evidence": current_evidence}
            )

    st.rerun()
