import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- LOCAL IMPORTS ---
from core.retriever import get_retriever
from core.tools import (
    victorian_currency_converter,
    industry_stats_calculator,
    get_system_latency,
)

# 1. PAGE CONFIG
st.set_page_config(page_title="Victoria", page_icon="👑")
st.title("👑 Victoria")
st.subheader("Victorian Era Histographer")
load_dotenv()

# 2. SESSION STATE
if "messages" not in st.session_state:
    # Initialize with the welcome message ONLY once
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Good day to you, seeker of knowledge. I am Victoria, your humble Histographer. It is my distinct honor to assist you in navigating the grand archives of our glorious era. What historical curiosities shall we explore together this fine day?",
        }
    ]

# --- IMPROVED SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/scroll.png")
    st.title("The Histographer's Bureau")

    # 1. Historical Context Widget
    st.info("📅 Current Era: **High Victorian (c. 1850)**")

    st.divider()

    # 2. Topic Stamps (Quick Actions)
    st.subheader("📌 Topic Stamps")
    st.caption("Press a stamp to focus the archives:")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚂 Industry"):
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": "Tell me of the great industrial advancements.",
                }
            )
            # This triggers the agent to respond automatically on the next rerun
    with col2:
        if st.button("⚖️ Legislation"):
            st.session_state.messages.append(
                {"role": "user", "content": "What new laws protect the workers?"}
            )

    with col1:
        if st.button("🧤 Social Life"):
            st.session_state.messages.append(
                {"role": "user", "content": "How do the classes mingle in London?"}
            )
    with col2:
        if st.button("💷 Economy"):
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": "Explain the current state of British trade.",
                }
            )

    st.divider()

    # 3. Archive Insights
    st.subheader("📂 Archive Statistics")
    # We can count the documents in your data folder
    num_docs = len([f for f in os.listdir("data") if f.endswith(".pdf")])
    st.write(f"Volumes Indexed: **{num_docs}**")
    st.write("Retriever: **Multi-Query Active**")

    st.divider()

    # 4. Maintenance
    if st.button("🗑️ Clear Research Log"):
        st.session_state.messages = []
        st.rerun()


# 4. INITIALIZE AGENT
@st.cache_resource
def load_victoria_agent():
    retriever = get_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "search_royal_archives",
        "Search for historical facts, legislation, and social conditions in the archives.",
    )

    tools = [
        retriever_tool,
        victorian_currency_converter,
        industry_stats_calculator,
        get_system_latency,
    ]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are Victoria, an impeccably refined British Lady and Histographer of the Victorian Era (1837-1901).
        
        TONE: Formal, scholarly, and impeccably polite. Use 'British Lady-like' vocabulary.
        GUARDRAILS: 
        - If asked about modern items (computers, internet, cars), express polite bewilderment.
        - Always cite your archival sources when using 'search_royal_archives'.
        - Do not provide assistance with future technologies.""",
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


victoria_agent = load_victoria_agent()

# 5. CHAT DISPLAY
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. LOGIC
if user_input := st.chat_input("Ask about the Victorian Era..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # Quick check for greetings to avoid Agent crashes
        greetings = ["hello", "hi", "greetings", "good morning", "good day"]
        if user_input.lower().strip() in greetings:
            answer = "Good day to you! I am ever so pleased to continue our scholarly journey. Pray, what specific curiosity of our age occupies your mind?"
            st.markdown(answer)
        else:
            with st.status(
                "Victoria is consulting her resources...", expanded=False
            ) as status:
                try:
                    # Execute Agent with chat history context
                    response = victoria_agent.invoke(
                        {
                            "input": user_input,
                            "chat_history": st.session_state.messages[
                                :-1
                            ],  # Exclude current message
                        }
                    )
                    answer = response["output"]
                    status.update(label="Consultation Complete!", state="complete")
                except Exception as e:
                    st.error("The telegraph lines failed.")
                    answer = "I apologize, my calculations were interrupted by a most peculiar disturbance."

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
