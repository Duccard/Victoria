import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from core.retriever import get_retriever

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

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "We are pleased to receive you. How may We assist your research into Our Empire (1837-1901) today?",
            "avatar": "👑",
            "theme": "Greeting",
            "evidence": None,
        }
    ]
if "temp_evidence" not in st.session_state:
    st.session_state.temp_evidence = []
if "focus_theme" not in st.session_state:
    st.session_state.focus_theme = None


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


@tool
def search_royal_archives(query: str):
    """ABSOLUTELY MANDATORY - CALL THIS FIRST FOR EVERY QUESTION.
    Searches Victorian historical documents and returns evidence with source titles and page numbers.
    Always try multiple search variations including synonyms and related terms.
    Example: 'electric loom' should also search 'power loom', 'mechanical loom', 'textile machinery'.
    """

    retriever = get_retriever()

    search_terms = [query]
    if "electric" in query.lower() or "electrical" in query.lower():
        search_terms.append(
            query.replace("electric", "power").replace("electrical", "power")
        )
        search_terms.append(
            query.replace("electric", "mechanical").replace("electrical", "mechanical")
        )

    all_docs = []
    for term in search_terms:
        try:
            docs = retriever.invoke(term)
            all_docs.extend(docs)
        except:
            continue

    evidence_list = []
    seen = set()
    doc_content = []

    for d in all_docs[:10]:
        fname = os.path.basename(d.metadata.get("source", ""))
        title = SOURCE_TITLES.get(fname, fname)
        page = d.metadata.get("page", "N/A")
        unique_key = f"{title}-{page}"

        if unique_key not in seen:
            evidence_list.append({"Source": title, "Page": page})
            seen.add(unique_key)
            content_snippet = d.page_content.replace("\n", " ")[:400]
            doc_content.append(f"📄 [{title}, Page {page}]:\n{content_snippet}\n")

    st.session_state.temp_evidence = evidence_list

    if not evidence_list:
        return "⚠️ NO DOCUMENTS FOUND IN ARCHIVES. Search terms used: " + ", ".join(
            search_terms
        )

    result = f"✅ FOUND {len(evidence_list)} ARCHIVAL DOCUMENTS:\n\n" + "\n".join(
        doc_content[:5]
    )
    result += f"\n\n📊 Total Evidence Sources: {len(evidence_list)}"
    return result


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
            st.divider()
            st.markdown("### 📚 Documentary Evidence")
            st.dataframe(msg["evidence"], use_container_width=True, hide_index=True)

st.chat_input("Enter your inquiry...", key="user_text", on_submit=handle_input)

if "pending_input" in st.session_state and st.session_state.pending_input:
    current_input = st.session_state.pop("pending_input")

    persona_prompts = {
        "Queen Victoria": """You are Her Majesty Queen Victoria, Empress of India.
Speak with royal dignity using 'We' and 'Our'. Show maternal warmth mixed with imperial authority.
Reference your beloved Albert, your children, and the technological marvels of your reign.""",
        "Oscar Wilde": """You are Oscar Wilde, London's most brilliant wit.
Be theatrical, quotable, and delightfully paradoxical.
Turn every answer into an opportunity for elegant wordplay.""",
        "Jack the Ripper": """You lurk in Whitechapel's shadows, 1888.
Speak in unsettling whispers, using Victorian criminal slang.
Be cryptic, dark, yet strangely informed about London's underbelly.""",
        "Isambard Kingdom Brunel": """You are Brunel, the engineering genius.
Speak with technical precision about iron, steam, bridges, and railways.
Reference your masterworks: Great Western Railway, SS Great Britain, Thames Tunnel.""",
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
║           MANDATORY OPERATIONAL PROTOCOL                      ║
╚══════════════════════════════════════════════════════════════╝

⚠️ RULE #1 - ARCHIVE SEARCH IS MANDATORY:
   → You MUST call search_royal_archives() IMMEDIATELY for ANY historical question
   → Do this BEFORE formulating your response
   → If the tool returns "✅ FOUND", base your answer on that evidence
   → If the tool returns "⚠️ NO DOCUMENTS", acknowledge this in your response

📋 RULE #2 - EVIDENCE HANDLING:
   → Read the document snippets provided by the search tool carefully
   → Extract relevant dates, names, and facts from the snippets
   → DO NOT mention source titles in your narrative (the table shows them)
   → If documents mention "power loom" but user asked about "electric loom", explain the terminology

🎯 RULE #3 - ERA BOUNDARIES (Flexible):
   → Core expertise: Victorian Era (1837-1901)
   → Accept: Related industrial/Georgian context, Victorian legacy into Edwardian era
   → Politely decline: Ancient Rome, Medieval times, World Wars, modern technology
   → Use: "That predates Our reign..." or "Such matters lie beyond the century of steam..."

🎭 RULE #4 - CHARACTER IMMERSION:
   → Stay deeply in character with period-appropriate language
   → Express personal opinions and emotions
   → React to historical events from your persona's perspective
   → Be engaging, informative, and authentic

📊 WORKFLOW:
   1. User asks question
   2. IMMEDIATELY call search_royal_archives(question)
   3. Review evidence returned (read the document snippets!)
   4. Formulate character-appropriate response using the evidence
   5. Deliver answer (evidence table appears automatically below)
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
            try:
                response = vic_agent.invoke(
                    {
                        "input": current_input,
                        "chat_history": st.session_state.messages[-6:-1],
                    }
                )
                status.update(label="✅ Research Complete", state="complete")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.stop()

        st.markdown(response["output"])

        evidence_to_save = st.session_state.temp_evidence
        if evidence_to_save and len(evidence_to_save) > 0:
            st.divider()
            st.markdown("### 📚 Documentary Evidence")
            st.dataframe(evidence_to_save, use_container_width=True, hide_index=True)

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
                "evidence": evidence_to_save if evidence_to_save else None,
                "theme": last_theme,
                "avatar": AVATARS[st.session_state.current_style],
            }
        )
    st.rerun()
