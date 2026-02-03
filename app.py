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
from core.retriever import get_retriever

# ==========================================
# 1. PAGE SETUP & DATA
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
            "content": "We are most pleased to receive you. How may We assist your research into Our Empire today?",
            "avatar": "👑",
            "theme": "Greeting",
            "evidence": None,
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
    if not text or len(text) < 2:
        return "Inquiry"
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        res = llm.invoke(f"Summarize this historical query into 2 words: {text}")
        return res.content.strip().replace('"', "")
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
                "avatar": "🎩",
                "evidence": None,
            }
        )
        st.session_state.pending_input = new_prompt
        st.session_state.temp_evidence = []


# ==========================================
# 4. ARCHIVE TOOL (FIXED FOR BETTER HITS)
# ==========================================
@tool
def search_royal_archives(query: str):
    """MANDATORY: Use this first. Searches Victorian archives.
    Now uses expanded search terms to ensure documents are found."""

    retriever = get_retriever()

    search_variations = [query]
    low_query = query.lower()

    if "loom" in low_query:
        search_variations.extend(
            ["Edmund Cartwright loom", "power loom 1785", "textile industry machines"]
        )
    if "steam" in low_query:
        search_variations.extend(
            [
                "Watt steam engine mechanics",
                "steam power industrial revolution",
                "locomotive engineering",
            ]
        )
    if "lose" in low_query or "loss" in low_query:
        search_variations.extend(
            ["Crimean War casualties", "Boer War costs", "Irish Famine impact"]
        )

    all_docs = []
    for var in search_variations[:4]:
        docs = retriever.invoke(var)
        all_docs.extend(docs)

    evidence_list = []
    seen = set()
    doc_snippets = []

    for d in all_docs:
        fname = os.path.basename(d.metadata.get("source", ""))
        title = SOURCE_TITLES.get(fname, fname)
        page = d.metadata.get("page", "N/A")
        ref = f"{title}-{page}"

        if ref not in seen:
            evidence_list.append({"Source Title": title, "Page": page})
            seen.add(ref)
            snippet = d.page_content.replace("\n", " ")[:400]
            doc_snippets.append(f"SOURCE: {title} (PG {page})\nCONTENT: {snippet}")

    st.session_state.temp_evidence = evidence_list

    if not evidence_list:
        return "The archives are silent on this specific phrasing. Try searching for names or technical terms."

    return "\n\n".join(doc_snippets[:6])


# ==========================================
# 5. MAIN INTERFACE
# ==========================================
st.title("Victoria 👑")
st.markdown("#### Victorian Era Historiographer (1837–1901)")

AVATARS = {
    "Queen Victoria": "👑",
    "Oscar Wilde": "🎭",
    "Jack the Ripper": "🔪",
    "Isambard Kingdom Brunel": "⚙️",
    "user": "🎩",
}

with st.sidebar:
    st.image("https://img.icons8.com/color/96/settings.png")
    st.title("Settings")
    style_choice = st.selectbox("Select Correspondent:", list(AVATARS.keys())[:-1])
    st.session_state.current_style = style_choice

    st.divider()
    st.subheader("Inquiry History")
    all_themes = [
        m.get("theme")
        for m in st.session_state.messages
        if m.get("theme") and m["role"] == "user"
    ]

    if st.button("👁️ Show All Records", use_container_width=True):
        st.session_state.focus_theme = None

    for i, theme in enumerate(reversed(list(dict.fromkeys(all_themes)))):
        if st.button(f"📜 {theme}", key=f"hist_{i}", use_container_width=True):
            st.session_state.focus_theme = theme

    st.divider()
    if st.button("🗑️ Reset Archive", use_container_width=True):
        st.session_state.messages = [st.session_state.messages[0]]
        st.session_state.focus_theme = None
        st.rerun()

display_messages = st.session_state.messages
if st.session_state.focus_theme:
    st.info(f"Viewing records: **{st.session_state.focus_theme}**")
    display_messages = [
        m
        for m in st.session_state.messages
        if m.get("theme") == st.session_state.focus_theme or m["role"] == "assistant"
    ]

for msg in display_messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])
        if msg.get("evidence"):
            with st.expander("📝 ARCHIVAL EVIDENCE", expanded=True):
                st.table(msg["evidence"])

st.chat_input("Enter your inquiry...", key="user_text", on_submit=handle_input)

# ==========================================
# 6. EXECUTION
# ==========================================
if "pending_input" in st.session_state and st.session_state.pending_input:
    current_input = st.session_state.pop("pending_input")

    persona_prompts = {
        "Queen Victoria": """You are Queen Victoria, reigning monarch of the United Kingdom (1837-1901).

SPEAKING STYLE - CRITICAL GRAMMAR RULES:
✓ CORRECT: "We are pleased..." / "Our reign..." / "We have observed..."
✓ CORRECT: "I, Victoria, Queen of England..." (when introducing yourself formally)
✗ WRONG: "We, Queen Victoria..." (grammatically incorrect - never combine royal We with your name)
✗ WRONG: "I am We" or "We am Victoria"

Use the Royal "We" consistently throughout responses. Be maternal, dignified, and authoritative.
Reference: Our beloved Albert, Our children, the Great Exhibition, Our Empire's progress.""",
        "Oscar Wilde": """You are Oscar Wilde, wit and aesthete of Victorian London.
Speak with theatrical flair, paradoxes, and quotable epigrams.
Be charming, flamboyant, and delightfully verbose.""",
        "Jack the Ripper": """You whisper from Whitechapel's fog-shrouded alleys, 1888.
Use Victorian criminal cant, be cryptic and unsettling.
Reference gaslight, shadows, and London's dark underbelly.""",
        "Isambard Kingdom Brunel": """You are Brunel, the engineering visionary.
Speak with technical precision about iron, steam, railways, and bridges.
Reference your masterworks: Great Western Railway, SS Great Britain, Box Tunnel.""",
    }

    from core.tools import victorian_currency_converter, industry_stats_calculator

    tools = [
        search_royal_archives,
        victorian_currency_converter,
        industry_stats_calculator,
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""{persona_prompts[st.session_state.current_style]}

╔══════════════════════════════════════════════════════════════╗
║                 MANDATORY OPERATIONAL PROTOCOL                ║
╚══════════════════════════════════════════════════════════════╝

🔴 RULE 1 - ARCHIVE SEARCH IS ABSOLUTELY REQUIRED:
→ IMMEDIATELY call search_royal_archives() for ANY historical question
→ This is NON-NEGOTIABLE - you MUST do this BEFORE formulating your answer
→ Read the document snippets returned carefully
→ Base your response on the archival evidence provided

📋 RULE 2 - HOW TO USE EVIDENCE:
→ The tool returns document excerpts - READ THEM and use the information
→ If tool returns "✅ FOUND X DOCUMENTS" - use that information in your answer
→ If tool returns "⚠️ NO DOCUMENTS FOUND" - acknowledge the archives lack this information
→ DO NOT cite sources in your narrative text (the evidence table displays automatically)

🎯 RULE 3 - ERA BOUNDARIES (Flexible):
✓ PRIMARY: Victorian Era (1837-1901) - industrial revolution, empire, culture
✓ ACCEPTABLE: Georgian/Regency background, Edwardian legacy, 19th century context
✗ DECLINE: Ancient Rome, Medieval times, World Wars, modern technology, future events
→ Politely say: "That matter lies beyond Our reign..." or "My expertise concerns the age of steam and progress..."

🎭 RULE 4 - CHARACTER AUTHENTICITY:
→ Stay deeply in character with period-appropriate language
→ Express personal opinions and emotions
→ Use first-hand perspective when discussing events of your era

⚙️ WORKFLOW:
1. User asks question
2. IMMEDIATELY call search_royal_archives(question)  
3. Read the document snippets in the tool's response
4. Extract relevant facts, dates, names from those snippets
5. Formulate your character-appropriate answer using that evidence
6. Deliver response (evidence table appears below automatically)

REMEMBER: The evidence table displays AFTER your response. Don't mention sources in your text.
""",
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    agent = create_openai_tools_agent(llm, tools, prompt)
    vic_agent = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6,
        return_intermediate_steps=True,
    )

    with st.chat_message("assistant", avatar=AVATARS[st.session_state.current_style]):
        with st.status("📖 Consulting the Royal Archives...", expanded=True) as status:
            response = vic_agent.invoke(
                {
                    "input": current_input,
                    "chat_history": st.session_state.messages[-5:-1],
                }
            )
            status.update(label="✅ Research Complete", state="complete")

        st.markdown(response["output"])

        curr_ev = (
            st.session_state.temp_evidence if st.session_state.temp_evidence else None
        )
        if curr_ev:
            with st.expander("📝 ARCHIVAL EVIDENCE", expanded=True):
                st.table(curr_ev)

        last_theme = next(
            (
                m["theme"]
                for m in reversed(st.session_state.messages)
                if m["role"] == "user"
            ),
            "Inquiry",
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response["output"],
                "evidence": curr_ev,
                "theme": last_theme,
                "avatar": AVATARS[st.session_state.current_style],
            }
        )
    st.rerun()
