import os
from typing import TypedDict
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

# 2. Setup Tools
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = SupabaseVectorStore(client=supabase, embedding=embeddings, table_name="documents", query_name="match_documents")
search_tool = TavilySearch(k=3)
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 3. Define Nodes
def router(state: AgentState):
    """Decides whether to search the PDF or the Web"""
    prompt = f"Analyze this query: '{state['query']}'. Is this a question likely answered in a specific uploaded document? Answer with ONLY 'PDF' or 'WEB'."
    decision = model.invoke(prompt).content.strip()
    return {"source": decision}

def retrieve_pdf(state: AgentState):
    """RAG tool for the uploaded PDF using direct RPC to avoid LangChain bugs"""
    # 1. Generate the embedding for the user query
    query_embedding = embeddings.embed_query(state['query'])
    
    # 2. Call the 'match_documents' function directly via Supabase client
    # This bypasses the AttributeError in langchain-community
    rpc_response = supabase.rpc("match_documents", {
        "query_embedding": query_embedding,
        "match_threshold": 0.5, # Adjust this based on how strict you want the search
        "match_count": 3
    }).execute()
    
    # 3. Process the results
    docs = rpc_response.data
    if not docs:
        return {"answer": "I couldn't find any relevant information in the PDF.", "source": "PDF"}
        
    context = "\n".join([d['content'] for d in docs])
    
    # 4. Generate answer using Gemini
    response = model.invoke(f"Answer using this context:\n{context}\n\nQuestion: {state['query']}")
    return {"answer": response.content}

def web_search(state: AgentState):
    """General Web Search tool using Tavily"""
    results = search_tool.invoke({"query": state['query']})
    # Process results into a simple answer
    content = "\n".join([r['content'] for r in results])
    response = model.invoke(f"Based on these web results, answer the question: {state['query']}\n\nResults: {content}")
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
    query = input("\nHi! Ask me something about your PDF or the world: ")
    output = graph.invoke({"query": query})
    print(f"\n[Source: {output['source']}]\nAnswer: {output['answer']}")