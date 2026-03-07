import os
from typing import TypedDict, Optional, List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END
from supabase.client import create_client

load_dotenv()

# 1. Define the State
class AgentState(TypedDict):
    query: str
    answer: str
    source: str
    session_id: str
    chat_history: List[dict]  # [{"role": "user"/"assistant", "content": "..."}]

# 2. Setup Tools
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = SupabaseVectorStore(client=supabase, embedding=embeddings, table_name="documents", query_name="match_documents")
search_tool = TavilySearch(k=3)
model = ChatGoogleGenerativeAI(model="gemma-3-4b-it")

# 3. Define Nodes
def format_chat_history(chat_history: list, max_turns: int = 10) -> str:
    """Format recent chat history into a readable string for the LLM"""
    if not chat_history:
        return ""
    recent = chat_history[-(max_turns * 2):]  # Keep last N turns (user+assistant pairs)
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)

def get_document_summaries(session_id: str):
    """Fetch document summaries for the current session only"""
    response = (
        supabase.table("document_summaries")
        .select("filename, summary")
        .eq("session_id", session_id)
        .execute()
    )
    return response.data or []

def router(state: AgentState):
    """Decides whether to search the PDF or the Web using document summaries as context"""
    summaries = get_document_summaries(state['session_id'])
    
    if not summaries:
        # No documents ingested in this session, go straight to web
        return {"source": "WEB"}
    
    summary_text = "\n".join(
        [f"- {s['filename']}: {s['summary']}" for s in summaries]
    )
    
    history_text = format_chat_history(state.get('chat_history', []))
    history_section = f"\nRecent conversation:\n{history_text}\n" if history_text else ""
    
    prompt = (
        f"You are a routing assistant. Below are summaries of documents the user has uploaded:\n\n"
        f"{summary_text}\n"
        f"{history_section}\n"
        f"User query: '{state['query']}'\n\n"
        f"Based on the document summaries above, is this query likely answerable from the uploaded documents? "
        f"Answer with ONLY 'PDF' or 'WEB'."
    )
    decision = model.invoke(prompt).content.strip()
    
    # Ensure we only get a valid routing value
    if "PDF" in decision.upper():
        return {"source": "PDF"}
    return {"source": "WEB"}

def retrieve_pdf(state: AgentState):
    """RAG tool for the uploaded PDF using direct RPC, scoped to the current session"""
    # 1. Generate the embedding for the user query
    query_embedding = embeddings.embed_query(state['query'])
    
    # 2. Call the 'match_documents_by_session' function which filters by session_id
    rpc_response = supabase.rpc("match_documents_by_session", {
        "query_embedding": query_embedding,
        "match_threshold": 0.5,
        "match_count": 3,
        "p_session_id": state['session_id']
    }).execute()
    
    # 3. Process the results
    docs = rpc_response.data
    if not docs:
        return {"answer": "I couldn't find any relevant information in the PDF.", "source": "PDF"}
        
    context = "\n".join([d['content'] for d in docs])
    
    # 4. Build conversation history for context
    history_text = format_chat_history(state.get('chat_history', []))
    history_section = f"\nPrevious conversation:\n{history_text}\n" if history_text else ""
    
    # 5. Generate answer using Gemini with conversation memory
    response = model.invoke(
        f"You are a helpful assistant. Use the document context and conversation history to answer the user's question.\n"
        f"{history_section}\n"
        f"Document context:\n{context}\n\n"
        f"Question: {state['query']}"
    )
    return {"answer": response.content}

def web_search(state: AgentState):
    """General Web Search tool using Tavily"""
    # .run() typically returns a formatted string of the top results
    results_text = search_tool.run(state['query'])
    
    # Build conversation history for context
    history_text = format_chat_history(state.get('chat_history', []))
    history_section = f"\nPrevious conversation:\n{history_text}\n" if history_text else ""
    
    # Generate answer using Gemini with conversation memory
    response = model.invoke(
        f"You are a helpful assistant. Use the web results and conversation history to answer the user's question.\n"
        f"{history_section}\n"
        f"Web results:\n{results_text}\n\n"
        f"Question: {state['query']}"
    )
    return {"answer": response.content}

# 4. Define the Graph
workflow = StateGraph(AgentState)

workflow.add_node("router", router)
workflow.add_node("retrieve", retrieve_pdf)
workflow.add_node("web_search", web_search)

workflow.set_entry_point("router")

# Conditional logic
workflow.add_conditional_edges(
    "router",
    lambda x: x["source"],
    {"PDF": "retrieve", "WEB": "web_search"}
)

workflow.add_edge("retrieve", END)
workflow.add_edge("web_search", END)

graph = workflow.compile()

# 5. Execute
if __name__ == "__main__":
    import uuid
    session_id = str(uuid.uuid4())
    chat_history = []
    print(f"Session ID: {session_id}")
    while True:
        query = input("\nHi! Ask me something (or type 'quit'): ")
        if query.lower() in ('quit', 'exit', 'q'):
            break
        output = graph.invoke({"query": query, "session_id": session_id, "chat_history": chat_history})
        print(f"\n[Source: {output['source']}]\nAnswer: {output['answer']}")
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": output['answer']})