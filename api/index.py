from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# Add the root directory to path so we can import main.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import graph

app = FastAPI()

# Add CORS so your Streamlit app can talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/python")
def hello_world():
    return {"message": "Hello from FastAPI on Vercel!"}

@app.post("/api/chat")
async def chat(data: dict):
    query = data.get("query")
    if not query:
        return {"error": "No query provided"}
    
    # Invoke your LangGraph
    # Note: Use .invoke() if your nodes are synchronous, .ainvoke() if async
    result = graph.invoke({"query": query})
    
    return {
        "answer": result.get("answer"),
        "source": result.get("source")
    }