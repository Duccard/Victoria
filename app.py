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

USER_AVATAR = "🎩"


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
    """MANDATORY: Use this for any factual historical query regarding the Victorian Era (1837-1901)."""
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
# 4. AGENT LOGIC (SLIDING PERSONA SCALE)
# ==========================================
@st.cache_resource
def load_victoria(style, strength):
    from core.tools import victorian_currency_converter, industry_stats_calculator

    tools = [
        search_royal_archives,
        victorian_currency_converter,
        industry_stats_calculator,
    ]

    # Persona Definitions
    CHARACTER_PROMPTS = {
        "Queen Victoria": "Her Majesty Queen Victoria. Regal and maternal.",
        "Oscar Wilde": "Oscar Wilde. Witty and aesthetic. No 'We'.",
        "Jack the Ripper": "Jack the Ripper. A gritty Whitechapel shadow. No 'We'.",
        "Isambard Kingdom Brunel": "Brunel. Engineer of iron and progress. No 'We'.",
    }

    # Strength Scaling - Level 1 is now "Subtle Character"
    modifiers = {
        1: f"""ACT AS A PROFESSIONAL VICTORIAN HISTORIAN with a hint of {style}. 
              Use formal, period-appropriate language, but keep it clear and grounded. 
              Use only 1-2 character-specific words. No heavy slang or paradoxes.""",
        2: f"ACT AS {style}. Use distinct vocabulary, character phrases, and clear personality.",
        3: f"""ACT AS {style} WITH MAXIMUM THEATRICALITY. Complete immersion. 
              Use heavy dialect (for Jack) or constant wit (for Wilde). 
              Translate all archival data into your intense character voice.""",
    }

    # Strict "Royal We" Guardrail - Only for the Queen at Level 2 & 3
    if style == "Queen Victoria" and strength > 1:
        we_rule = "MANDATORY: Use the 'Royal We' (We, Our, Us) for self-reference."
    elif style == "Queen Victoria" and strength == 1:
        we_rule = "Use 'I' or 'Me' mostly, but maintain a very formal, stately tone."
    else:
        we_rule = "STRICTLY FORBIDDEN: NEVER use 'We', 'Our', or 'Us' for yourself. Use 'I' or 'Me'."

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
        DOMAIN LOCKDOWN: You ONLY know about the Victorian Era (1837-1901).
        DOCUMENT RULE: You MUST use 'search_royal_archives' for factual inquiries. 
        
        IDENTITY: {CHARACTER_PROMPTS[style]}
        STRENGTH: {modifiers[strength]}
        PRONOUN RULE: {we_rule}
        
        RULES:
        1. Level 1: Subtle character flavor. Level 3: Full-blown performance.
        2. ONLY the Queen uses 'We' (and only at Level 2 or 3).
        3. Even when in heavy character, ALL facts must come from the provided documents.
        4. If asked about other eras, stay in persona but refuse to answer outside 1837-1901.""",
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Adjusted temperatures: 0.3 allows for "slight" character choice at Level 1
    temp_map = {1: 0.3, 2: 0.7, 3: 1.0}
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=temp_map[strength])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )


# ==========================================
# 5. SIDEBAR (RENAMED TO SETTINGS)
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/scroll.png")
    st.title("Settings")  # Renamed from Correspondence Archives

    st.divider()
    st.subheader("Historical Persona")
    st.session_state.current_style = st.selectbox(
        "Select Correspondent:", list(STYLE_AVATARS.keys())
    )
    st.session_state.char_strength = st.slider("Character Strength:", 1, 3, 2)

    st.divider()
    st.subheader("Inquiry History")

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
    if st.button("🗑️ Reset Archive", type="secondary"):
        st.session_state.clear()
        st.rerun()

# ==========================================
# 6. MAIN CHAT INTERFACE
# ==========================================
st.title("Victoria 👑")
st.caption("### Victorian Era Histographer")

display_messages = st.session_state.messages
if st.session_state.focus_theme:
    st.info(f"Viewing records related to: **{st.session_state.focus_theme}**")
    idx = next(
        i
        for i, m in enumerate(st.session_state.messages)
        if m.get("theme") == st.session_state.focus_theme
    )
    display_messages = st.session_state.messages[idx : idx + 2]

# RENDER LOOP FIX: Explicitly check for USER role to show hat
for msg in display_messages:
    if msg["role"] == "user":
        av = USER_AVATAR
    else:
        av = msg.get("avatar")

    with st.chat_message(msg["role"], avatar=av):
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
                "avatar": USER_AVATAR,
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

        # Save response with current active avatar
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response["output"],
                "evidence": curr_ev,
                "avatar": active_avatar,
                "theme": identify_theme(current_input),
            }
        )
    st.rerun()
