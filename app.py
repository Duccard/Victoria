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

# --- CUSTOM THEME (CSS) ---
st.markdown("""
    <style>
    .stApp {
        background-color: #f4f1ea; /* Parchment Background */
    }
    [data-testid="stSidebar"] {
        background-color: #2e3b4e !important;
    }
    [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] p {
        color: #d1d1d1 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SOURCE TITLES DICTIONARY ---
SOURCE_TITLES = {
    "20-Industrial-Rev.pdf": "The Industrial Revolution Archives (Vol. 20)",
    "Chapter-8-The-Industrial-Revolution.pdf": "British Industrial History, Chapter VIII",
    "sadler-report.pdf": "The Michael Sadler Report (1832)",
    "2020_Kelly_Mokyr_Mechanics_Ind_Rev.pdf": "Mechanics of the Industrial Revolution"
}

# 2. STATE INITIALIZATION
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Good day. I am Victoria. How may I assist your research today?", "evidence": None, "theme": "Greeting"}]
if "temp_evidence" not in st.session_state:
    st.session_state.temp_evidence = []

# --- 3. THEME IDENTIFIER ---
def identify_theme(text):
    if not text or len(text) < 5: return "General"
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(f"Summarize this historical query into 2-3 words. Query: {text}")
    return response.content.strip().replace('"', '')

# --- 4. INPUT CALLBACK ---
def handle_input():
    if st.session_state.user_text:
        new_prompt = st.session_state.user_text
        theme = identify_theme(new_prompt)
        # Store theme only on the user message
        st.session_state.messages.append({"role": "user", "content": new_prompt, "evidence": None, "theme": theme})
        st.session_state.pending_input = new_prompt
        st.session_state.temp_evidence = [] 
        st.session_state.user_text = ""

# 5. SIDEBAR (Your Enquiries History)
with st.sidebar:
    st.title("📜 Your Enquiries History")
    st.divider()
    
    # Show themes from previous user questions
    user_themes = [m.get("theme") for m in st.session_state.messages if m["role"] == "user"]
    for theme in reversed(user_themes):
        st.write(f"📂 {theme}")
            
    st.divider()
    if st.button("🗑️ Reset Archive"):
        st.session_state.messages = [{"role": "assistant", "content": "Archives cleared.", "evidence": None, "theme": "Greeting"}]
        st.rerun()

# 6. ARCHIVE TOOL
from core.retriever import get_retriever

@tool
def search_royal_archives(query: str):
    """Consult this for historical facts."""
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
    return str(evidence_list)

# 7. AGENT SETUP
@st.cache_resource
def load_victoria():
    from core.tools import victorian_currency_converter, industry_stats_calculator
    tools = [search_royal_archives, victorian_currency_converter, industry_stats_calculator