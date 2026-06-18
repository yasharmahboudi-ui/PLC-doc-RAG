import os
import streamlit as st
from dotenv import load_dotenv

# Ensure local source directory is in python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from query import query_rag

# Load existing environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="S7-1200 PLC RAG Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for Siemens-like theme
st.markdown("""
<style>
    .main-title {
        color: #006494;
        font-family: 'Outfit', 'Segoe UI', sans-serif;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .subtitle {
        color: #555;
        font-family: 'Segoe UI', sans-serif;
        margin-top: 0px;
        margin-bottom: 25px;
    }
    .answer-card {
        background-color: #f8f9fa;
        border-left: 5px solid #006494;
        padding: 20px;
        border-radius: 4px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .source-card {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    .source-header {
        font-weight: 600;
        color: #495057;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar UI
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("### API Credentials")
    st.markdown("Configure your LLM API keys to enable live generation. If left empty, the assistant will run in Offline Demo Mode.")
    
    gemini_key_input = st.text_input(
        "Google Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Get a free key from Google AI Studio"
    )
    
    openai_key_input = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Provide your OpenAI API key"
    )
    
    # Inject input keys into environment variables
    if gemini_key_input:
        os.environ["GEMINI_API_KEY"] = gemini_key_input
    elif "GEMINI_API_KEY" in os.environ and not gemini_key_input:
        del os.environ["GEMINI_API_KEY"]
        
    if openai_key_input:
        os.environ["OPENAI_API_KEY"] = openai_key_input
    elif "OPENAI_API_KEY" in os.environ and not openai_key_input:
        del os.environ["OPENAI_API_KEY"]

    st.markdown("---")
    st.markdown("### 📘 System Info")
    st.markdown("""
    * **RAG Pipeline**: LangChain + FAISS
    * **Embedding Model**: `all-MiniLM-L6-v2` (Local CPU)
    * **Document**: Siemens S7-1200 Easy Book
    """)

# Main Content UI
st.markdown("<h1 class='main-title'>⚡ Siemens SIMATIC S7-1200 RAG Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Semantic search and QA assistant grounded in official Siemens documentation</p>", unsafe_allow_html=True)

# Status indicators
has_gemini = bool(os.getenv("GEMINI_API_KEY"))
has_openai = bool(os.getenv("OPENAI_API_KEY"))

if has_gemini or has_openai:
    active_llm = "Google Gemini (gemini-1.5-flash)" if has_gemini else "OpenAI (gpt-4o-mini)"
    st.success(f"🟢 **Live GenAI Mode Active** — Responses are generated in real-time by {active_llm}.", icon="✅")
else:
    st.warning("⚠️ **Offline Demo Mode Active** — No LLM API key detected. Custom questions will return pre-computed fallback answers. Provide an API key in the sidebar to enable live query generation.", icon="⚠️")

# Suggested questions section
st.markdown("### ❓ Quick Sample Questions")
col1, col2, col3 = st.columns(3)
q1 = col1.button("What does error code 8090 mean?")
q2 = col2.button("What do the LEDs on the CPU indicate?")
q3 = col3.button("Maximum number of signal modules?")

# Chat input
user_query = st.chat_input("Type your configuration or troubleshooting question here...")

# Handle sample button clicks
selected_query = None
if q1:
    selected_query = "What does error code 8090 mean?"
elif q2:
    selected_query = "What do the LEDs on the CPU indicate?"
elif q3:
    selected_query = "What is the maximum number of signal modules that can be added to an S7-1200 CPU?"
elif user_query:
    selected_query = user_query

# Run QA pipeline
if selected_query:
    st.markdown(f"### 🔍 Question: *{selected_query}*")
    
    with st.spinner("Retrieving relevant pages and synthesizing answer..."):
        result = query_rag(selected_query)
        
    if result:
        st.markdown("#### 🤖 Assistant Answer:")
        st.markdown(f"<div class='answer-card'>{result['answer']}</div>", unsafe_allow_html=True)
        
        st.markdown("#### 📖 Referenced Document Contexts:")
        for idx, doc in enumerate(result['context'], 1):
            source_file = os.path.basename(doc.metadata.get('source', 's71200_easy_book.pdf'))
            page_num = doc.metadata.get('page', 'unknown')
            page_str = f"Page {page_num + 1}" if isinstance(page_num, int) else page_num
            
            with st.expander(f"Context [{idx}]: {source_file} ({page_str})"):
                st.markdown(f"```text\n{doc.page_content.strip()}\n```")
    else:
        st.error("Error executing query. Please verify that the local vector database is built.")
