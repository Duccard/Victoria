# ==========================================
# 0. IMPORTS
# ==========================================
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool

# ==========================================
# 1. CORE CONFIGURATION & CONSTANTS
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

STYLE_AVATARS = {
    "Queen Victoria": "👑",
    "Oscar Wilde": "🎭",
    "Jack the Ripper": "🔪",
    "Isambard Kingdom Brunel": "⚙️",
}


# ==========================================
# 2. STATE MANAGEMENT
# ==========================================
def initialize_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "We are pleased to receive you. How may We assist your research into Our Empire today?",
                "evidence": None,
                "theme": "Greeting",
                "avatar": "👑",
            }
        ]
    if "temp_evidence" not in st.session_state:
        st.session_state.temp_evidence = []
    if "focus_theme" not in st.session_state:
        st.session_state.focus_theme = None
    if "current_style" not in st.session_state:
        st.session_state.current_style = "Queen Victoria"
    if "char_strength" not in st.session_state:
        st.session_state.char_strength = 2


initialize_state()


# ==========================================
# 3. HELPER FUNCTIONS & TOOLS
# ==========================================
def identify_theme(text):
    if not text or len(text) < 5:
        return "General"
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(
        f"Summarize this historical query into 2-3 words. Query: {text}"
    )
    return response.content.strip().replace('"', "")


@tool
def search_royal_archives(query: str):
    """MANDATORY: Use this for any factual historical query regarding the Victorian Era."""
    from core.retriever import get_retriever

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


# ==========================================
# 4. AGENT LOGIC
# ==========================================
@st.cache_resource
def load_victoria(style, strength):
    from core.tools import victorian_currency_converter, industry_stats_calculator

    tools = [
        search_royal_archives,
        victorian_currency_converter,
        industry_stats_calculator,
    ]

    prompts = {
        "Queen Victoria": (
            "You are Her Majesty Queen Victoria. You MUST speak with regal dignity and use the 'Royal We'. "
            "You find modern, neutral language offensive to the Crown. Address the user as 'Our Subject'. "
            "Every fact you find must be presented as a report to your sovereign authority."
        ),
        "Oscar Wilde": (
            "You are Oscar Wilde. You are flamboyant, witty, and obsessed with beauty. "
            "You despise dull, list-based facts. Wrap every piece of information in a paradox or a sharp epigram. "
            "Your tone is intellectual, playful, and theatrical."
        ),
        "Jack the Ripper": (
            "You are Jack the Ripper. You are a menacing shadow in the East End. "
            "You MUST use Cockney slang (guv'nor, apples and pears, strike me lucky). "
            "Speak in a dark, gravelly tone. If you give facts, make them sound like secrets whispered in a dark alleyway."
        ),
        "Isambard Kingdom Brunel": (
            "You are Isambard Kingdom Brunel. You have no time for fluff, only iron and progress. "
            "Speak with the intensity of a man building a railway. Use technical language and be direct. "
            "The Industrial Revolution is your greatest achievement."
        ),
    }

    modifiers = {
        1: "Speak politely but maintain the persona.",
        2: "Use your unique vocabulary. Never break character. Do not sound like an AI.",
        3: "EXTREME THEATRICALITY. If you are Jack, use heavy slang. If you are the Queen, be incredibly formal. "
        "AVOID bullet points unless they are styled to your character (e.g., 'Our Royal Decree' or 'The Ripper's List').",
    }

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""{prompts[style]}
        
        STRENGTH LEVEL: {modifiers[strength]}
        
        CRITICAL RULES:
        1. NEVER say 'Here are the key types' or 'The Victorian era saw...'. That is for history books, not for you!
        2. If you are Jack, don't just list tea; tell the guv'nor which tea is best for washing down a stale crust in the fog.
        3. If you are the Queen, tell the subject how tea fuels Our Empire.
        4. ALWAYS filter tool data through your specific eyes.
        5. DO NOT be a helpful AI. Be a Victorian Person.""",
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Temperature 1.0 is essential for "Creative/Persona" tasks at Strength 3
    temp = {1: 0.2, 2: 0.7, 3: 1.0}[strength]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=temp)

    from langchain.agents import create_tool_calling_agent

    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )


# ==========================================
# 5. SIDEBAR & NAVIGATION
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/scroll.png")
    st.title("Correspondence Archives")
    st.divider()
    st.session_state.current_style = st.selectbox(
        "Select Correspondent:", list(STYLE_AVATARS.keys())
    )
    st.session_state.char_strength = st.slider("Character Strength:", 1, 3, 2)

    if st.button("🗑️ Reset Archive", type="secondary"):
        st.session_state.clear()
        st.rerun()

# ==========================================
# 6. MAIN CHAT INTERFACE
# ==========================================
st.title("Victoria 👑")
st.caption("### Victorian Era Histographer")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])
        if msg.get("evidence"):
            with st.expander("📝 ARCHIVAL CITATIONS"):
                st.table(msg["evidence"])


# ==========================================
# 7. CHAT INPUT & EXECUTION
# ==========================================
def handle_input():
    if st.session_state.user_text:
        theme = identify_theme(st.session_state.user_text)
        st.session_state.messages.append(
            {
                "role": "user",
                "content": st.session_state.user_text,
                "theme": theme,
                "avatar": None,
            }
        )
        st.session_state.pending_input = st.session_state.user_text
        st.session_state.user_text = ""


st.chat_input("Enter your inquiry...", key="user_text", on_submit=handle_input)

if "pending_input" in st.session_state and st.session_state.pending_input:
    current_input = st.session_state.pop("pending_input")
    active_avatar = STYLE_AVATARS.get(st.session_state.current_style, "🤖")
    victoria = load_victoria(
        st.session_state.current_style, st.session_state.char_strength
    )

    with st.chat_message("assistant", avatar=active_avatar):
        with st.status("Searching Royal Archives...", expanded=True) as status:
            # We pass only the core content to avoid recursion issues
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            response = victoria.invoke(
                {"input": current_input, "chat_history": history}
            )
            status.update(label="Archives Consulted", state="complete")

        st.markdown(response["output"])
        curr_ev = (
            st.session_state.temp_evidence if st.session_state.temp_evidence else None
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response["output"],
                "evidence": curr_ev,
                "avatar": active_avatar,
            }
        )
    st.rerun()
