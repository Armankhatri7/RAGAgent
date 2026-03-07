import os
import uuid
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client

load_dotenv()

def run_ingestion(pdf_path, session_id=None):
    # Generate a session_id if not provided (CLI usage)
    if session_id is None:
        session_id = str(uuid.uuid4())
    print(f"--- Session ID: {session_id} ---")
    
    # 1. Setup Supabase
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
    
    # 2. Process PDF
    print(f"--- Loading {pdf_path} ---")
    loader = PyPDFLoader(pdf_path)
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    print(f"--- Split into {len(chunks)} chunks ---")
    
    # 3. Tag every chunk with the session_id
    for chunk in chunks:
        chunk.metadata["session_id"] = session_id
    
    # 4. Embed and Upload using Gemini
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    print("--- Uploading to Supabase... ---")
    SupabaseVectorStore.from_documents(
        chunks, 
        embeddings, 
        client=supabase, 
        table_name="documents",
        query_name="match_documents"
    )
    
    # 5. Generate a one-liner summary and store it (scoped to session)
    print("--- Generating document summary... ---")
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    sample_text = "\n".join([c.page_content for c in chunks[:5]])
    summary_response = model.invoke(
        f"Summarize the following document content in ONE short sentence (max 30 words). "
        f"Focus on what the document is about:\n\n{sample_text}"
    )
    summary = summary_response.content.strip()
    
    filename = os.path.basename(pdf_path)
    supabase.table("document_summaries").insert({
        "filename": filename,
        "summary": summary,
        "session_id": session_id
    }).execute()
    print(f"--- Summary: {summary} ---")
    print("✅ Ingestion Complete!")
    return session_id

if __name__ == "__main__":
    if os.path.exists("data.pdf"):
        run_ingestion("data.pdf")
    else:
        print("❌ Error: 'data.pdf' not found. Please add a PDF and rename it.")