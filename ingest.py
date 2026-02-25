import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client

load_dotenv()

def run_ingestion(pdf_path):
    # 1. Setup Supabase
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
    
    # 2. Process PDF
    print(f"--- Loading {pdf_path} ---")
    loader = PyPDFLoader(pdf_path)
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    print(f"--- Split into {len(chunks)} chunks ---")
    
    # 3. Embed and Upload using Gemini
    # Note: text-embedding-004 is the standard Google embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    print("--- Uploading to Supabase... ---")
    SupabaseVectorStore.from_documents(
        chunks, 
        embeddings, 
        client=supabase, 
        table_name="documents",
        query_name="match_documents"
    )
    print("✅ Ingestion Complete!")

if __name__ == "__main__":
    if os.path.exists("data.pdf"):
        run_ingestion("data.pdf")
    else:
        print("❌ Error: 'data.pdf' not found. Please add a PDF and rename it.")