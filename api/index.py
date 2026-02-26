from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

# Link to your main logic in the root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import graph

app = FastAPI()

class ChatRequest(BaseModel):
    query: str

@app.post("/api/chat")
async def chat(request: ChatRequest):
    # This invokes your LangGraph workflow
    result = graph.invoke({"query": request.query})
    return result