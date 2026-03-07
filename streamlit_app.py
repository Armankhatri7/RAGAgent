import streamlit as st
import os
import uuid
import tempfile
from supabase.client import create_client
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from main import graph  # We will keep your graph logic in main.py

# --- 1. Setup & Configuration ---
st.set_page_config(page_title="Personal RAG Agent", layout="wide")
st.title("🤖 Multi-Source AI Agent")

# Load credentials from Streamlit Secrets (for hosting) or .env (local)
# For local testing, ensure your .env is loaded
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase = create_client(supabase_url, supabase_key)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Generate a unique session ID per browser session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# --- 2. Sidebar: Document Ingestion ---
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if st.button("Ingest PDF") and uploaded_file:
        with st.spinner("Processing PDF..."):
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            # Load and split
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(pages)
            
            # Tag every chunk with the current session_id
            for chunk in chunks:
                chunk.metadata["session_id"] = st.session_state.session_id
            
            # Upload to Supabase
            SupabaseVectorStore.from_documents(
                chunks,
                embeddings,
                client=supabase,
                table_name="documents",
                query_name="match_documents"
            )
            
            # Generate a one-liner summary and store it (scoped to session)
            summary_model = ChatGoogleGenerativeAI(model="gemma-3-4b-it")
            sample_text = "\n".join([c.page_content for c in chunks[:5]])
            summary_response = summary_model.invoke(
                f"Summarize the following document content in ONE short sentence (max 30 words). "
                f"Focus on what the document is about:\n\n{sample_text}"
            )
            summary = summary_response.content.strip()
            supabase.table("document_summaries").insert({
                "filename": uploaded_file.name,
                "summary": summary,
                "session_id": st.session_state.session_id
            }).execute()
            
            os.remove(tmp_path) # Clean up temp file
            st.success(f"✅ '{uploaded_file.name}' ingested successfully!\n\n📄 Summary: {summary}")

# --- 3. Main Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about your PDF or the web..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Build chat history from session state (exclude the current message just added)
            chat_history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]  # everything except the latest user msg
            ]
            
            # Execute your LangGraph workflow (scoped to this session, with memory)
            response = graph.invoke({
                "query": prompt,
                "session_id": st.session_state.session_id,
                "chat_history": chat_history
            })
            
            full_response = f"**Source:** {response['source']}\n\n{response['answer']}"
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})