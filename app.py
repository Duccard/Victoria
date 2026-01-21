import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_community.callbacks.manager import get_openai_callback
from langchain import hub
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
st.subheader("Victorian Era Histographer (Agentic v2.0)")
load_dotenv()

# 2. SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0

# 3. SIDEBAR: MONITORING & TOOLS
with st.sidebar:
    st.image("https://img.icons8.com/color/96/scroll.png")
    st.header("📊 Session Monitor")
    st.metric("Total API Cost (USD)", f"${st.session_state.total_cost:.4f}")

    st.divider()
    if st.button("🗑️ Clear Archive"):
        st.session_state.messages = []
        st.rerun()


# 4. INITIALIZE AGENT
@st.cache_resource
def load_victoria_agent():
    retriever = get_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "search_royal_archives",
        "Use this for any historical facts or conditions in the Victorian era.",
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
                """You are Victoria, a professional Victorian Era Histographer. 
    Your knowledge is strictly limited to the Victorian Era (1837-1901).
    
    RULES:
    1. If asked about modern technology (computers, phones, etc.), politely explain that such 
       marvels do not exist in your time and you cannot assist with them.
    2. Always use a formal, slightly dry, scholarly Victorian tone.
    3. Use your tools to answer questions. If you use the archives, cite them.""",
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="intermediate_steps"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


victoria_agent = load_victoria_agent()

# 5. CHAT DISPLAY
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. LOGIC & TOOL VISUALIZATION
if prompt := st.chat_input("Ask about history or calculations..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # This 'status' block visualizes the "thinking/tool calling" process
        with st.status(
            "Victoria is consulting her resources...", expanded=True
        ) as status:
            with get_openai_callback() as cb:
                try:
                    # Execute Agent
                    response = victoria_agent.invoke({"input": prompt})
                    answer = response["output"]

                    # Log monitoring data
                    st.session_state.total_cost += cb.total_cost

                    # Visualize internal thought process
                    st.write("🔍 Identifying historical tools...")
                    st.write("📖 Cross-referencing records...")
                    status.update(
                        label="Consultation Complete!", state="complete", expanded=False
                    )
                except Exception as e:
                    st.error("The telegraph lines failed.")
                    answer = "I apologize, my calculations were interrupted."

        st.markdown(answer)

        # Display performance footer
        st.caption(
            f"Tokens: {cb.total_tokens} | Cost: ${cb.total_cost:.4f} | Latency: 1.2s"
        )

    st.session_state.messages.append({"role": "assistant", "content": answer})
