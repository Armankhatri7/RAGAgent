import streamlit as st
import time
from main import app  # Import your compiled LangGraph workflow

# --- UI Configuration ---
st.set_page_config(
    page_title="Agentic RAG Explorer",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS for high-quality "Source" badges
st.markdown("""
    <style>
    .pdf-tag {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 11px;
        font-weight: bold;
        border: 1px solid #bbf7d0;
    }
    .web-tag {
        background-color: #dbeafe;
        color: #1e40af;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 11px;
        font-weight: bold;
        border: 1px solid #bfdbfe;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 Agentic RAG Explorer")
st.caption("AI Agent | IIT Bombay | Gemini + Supabase + Tavily")

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Conversation ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# --- Chat Input ---
if prompt := st.chat_input("Ask about the technical PDF or the general web..."):
    # Store and display user input
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Agent Routing & Response ---
    with st.chat_message("assistant"):
        ui_status = st.empty()
        ui_status.markdown("🧠 *Deciding route and processing...*")
        
        try:
            # Run the LangGraph agent
            # It will choose between PDF Retrieval and Web Search automatically
            result = app.invoke({"query": prompt})
            
            final_answer = result["answer"]
            source_type = result.get("source", "UNKNOWN")
            
            # Create the styled badge for the UI
            badge_class = "pdf-tag" if source_type == "PDF" else "web-tag"
            source_html = f'<span class="{badge_class}">{source_type} SOURCE</span>'
            
            display_response = f"{source_html}\n\n{final_answer}"
            
            # Update the status with the actual answer
            ui_status.markdown(display_response, unsafe_allow_html=True)
            
            # Save to history
            st.session_state.messages.append({"role": "assistant", "content": display_response})
            
        except Exception as e:
            ui_status.error(f"Execution Error: {str(e)}")