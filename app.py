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

# --- SOURCE MAPPING DICTIONARY ---
# Map your PDF filenames to formal titles for the UI
SOURCE_TITLES = {
    "20-Industrial-Rev.pdf": "The Industrial Revolution Archives (Vol. 20)",
    "Chapter-8-The-Industrial-Revolution.pdf": "British Industrial History, Chapter VIII",
    "sadler-report.pdf": "The Michael Sadler Report on Factory Labor (1832)",
    "mines-act-1842.pdf": "Royal Commission on Children's Employment in Mines",
}

# 2. STATE INITIALIZATION
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Good day. I am Victoria. How may I assist your research today?",
        }
    ]
if "last_evidence" not in st.session_state:
    st.session_state.last_evidence = []


# --- 3. THE INSTANT SIDEBAR FIX ---
def handle_input():
    if st.session_state.user_text:
        new_prompt = st.session_state.user_text
        st.session_state.messages.append({"role": "user", "content": new_prompt})
        st.session_state.pending_input = new_prompt
        st.session_state.user_text = ""


# 4. SIDEBAR RENDERING
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
            {"role": "assistant", "content": "Archives cleared."}
        ]
        st.session_state.last_evidence = []
        st.rerun()

# 5. DEFINE ARCHIVE TOOL
from core.retriever import get_retriever


@tool
def search_royal_archives(query: str):
    """MANDATORY: Consult this for any factual historical claims or evidence."""
    retriever = get_retriever()
    docs = retriever.invoke(query)
    st.session_state.last_evidence = docs  # Store for the Table

    results = []
    for d in docs:
        fname = os.path.basename(d.metadata.get("source", ""))
        source_name = SOURCE_TITLES.get(fname, fname)  # Use pretty name if exists
        page = d.metadata.get("page", "N/A")
        results.append(
            f"SOURCE: {source_name} (Page {page})\nCONTENT: {d.page_content}"
        )
    return "\n\n".join(results)


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
                "You are Victoria, a formal British Histographer. Use search_royal_archives for history. Speak refined British English.",
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
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

st.chat_input("Enter your inquiry...", key="user_text", on_submit=handle_input)

# 8. AGENT EXECUTION LOGIC
if "pending_input" in st.session_state and st.session_state.pending_input:
    current_input = st.session_state.pop("pending_input")

    # 1. Check for Greetings
    greetings = ["hello", "hi", "greetings", "good day"]
    if current_input.lower().strip() in greetings:
        answer = "Good day! How may I assist your research into our glorious era?"
        st.session_state.last_evidence = []
    else:
        # 2. Process Historical Question
        with st.chat_message("assistant"):
            st.session_state.last_evidence = []
            with st.status("Searching the Royal Archives...", expanded=True) as status:
                response = victoria.invoke(
                    {
                        "input": f"{current_input}. Cite sources.",
                        "chat_history": st.session_state.messages[:-1],
                    }
                )
                answer = response["output"]
                status.update(label="Consultation Complete", state="complete")

            st.markdown(answer)

            # 3. RENDER EVIDENCE TABLE
            if st.session_state.last_evidence:
                with st.expander("📝 VIEW ARCHIVAL EVIDENCE", expanded=True):
                    # Prepare data for the table
                    table_data = []
                    for doc in st.session_state.last_evidence:
                        raw_name = os.path.basename(doc.metadata.get("source", ""))
                        table_data.append(
                            {
                                "Source Title": SOURCE_TITLES.get(raw_name, raw_name),
                                "Page": doc.metadata.get("page", "N/A"),
                                "Excerpt": f"{doc.page_content[:200]}...",
                            }
                        )
                    st.table(table_data)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
