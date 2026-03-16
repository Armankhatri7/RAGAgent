# RAG-Agent

A multi-source Retrieval-Augmented Generation (RAG) assistant that combines:
- **Session-scoped PDF retrieval** from Supabase vectors
- **Web fallback** via Tavily when your documents are not enough
- **Conversational memory** to keep answers context-aware across turns

Built with **LangGraph + Streamlit + Supabase + Gemini models**.

## Live Demo

Deployed app: https://ragagent-xu5zm4s4v9jsn9gyvbz5f5.streamlit.app/

---

## Why this project exists

Most “chat with PDF” apps fail in one of two ways:
1. They over-trust local documents and hallucinate when data is missing.
2. They overuse web search and ignore user-provided context.

This project is designed to balance both paths intelligently:
- Route to **PDF retrieval** when uploaded docs can answer the query.
- Route to **web search** when docs are not relevant.
- Keep all retrieval **isolated per session** so users do not leak context into each other’s results.

---

## Key features (and the thinking behind them)

### 1) Intelligent source routing (PDF vs Web)
**What it does:** A router node inspects uploaded document summaries + user query and decides whether to use PDF retrieval or web search.

**Why it matters:** This avoids a common RAG failure mode: forcing vector retrieval even when no relevant chunk exists. Routing improves factuality and user trust.

---

### 2) Session-scoped retrieval isolation
**What it does:** Every ingested chunk is tagged with a unique `session_id`, and retrieval uses the Supabase RPC `match_documents_by_session` to filter results.

**Why it matters:** Prevents cross-session contamination and keeps each user/session private and context-pure.

---

### 3) Document summary memory for better routing
**What it does:** During ingestion, the app creates a one-line summary per document and stores it in `document_summaries`.

**Why it matters:** The router does not need to scan all chunks every time. Summaries provide a lightweight semantic index that improves route quality with lower latency.

---

### 4) Conversation-aware answering
**What it does:** The last turns of chat history are injected into both PDF and web answer prompts.

**Why it matters:** Follow-up questions like “can you simplify that?” or “what about section 2?” remain coherent without forcing users to restate context.

---

### 5) Hybrid interface: UI + CLI flows
**What it does:**
- `streamlit_app.py` for interactive end-user chat and upload
- `ingest.py` for scriptable ingestion
- `main.py` for graph execution + terminal chat mode

**Why it matters:** Supports both fast prototyping and automation workflows.

---

## Tech stack

- **Orchestration:** LangGraph (`StateGraph`)
- **LLM + Embeddings:** Google Gemini via `langchain-google-genai`
- **Vector DB:** Supabase + pgvector (`documents` table)
- **Web search:** Tavily
- **UI:** Streamlit

---

## Project structure

```text
RAG-Agent/
├── streamlit_app.py     # Web UI + upload + chat
├── main.py              # LangGraph workflow (router/retrieve/web)
├── ingest.py            # PDF ingestion pipeline
├── requirements.txt
├── vercel.json
└── README.md
```

---

## How the pipeline works

1. **Upload PDF** in Streamlit sidebar.
2. PDF is chunked and embedded.
3. Chunks are stored in Supabase with `session_id` metadata.
4. A short document summary is generated and saved.
5. User asks a question.
6. Router chooses `PDF` or `WEB`.
7. Selected node builds answer using context + recent chat history.

---

## Requirements

- Python 3.10+
- Supabase project with vector search setup
- Google AI API key
- Tavily API key (for web search path)

---

## Environment variables

Create a `.env` file in the project root:

```env
SUPABASE_URL=...
SUPABASE_SERVICE_KEY=...
GOOGLE_API_KEY=...
TAVILY_API_KEY=...
```

---

## Supabase expectations

This app expects:
- `documents` table for vectorized chunks
- `document_summaries` table with `filename`, `summary`, `session_id`
- RPC function `match_documents_by_session` that filters chunks by `session_id`

If your schema uses different names, update `main.py`, `ingest.py`, and `streamlit_app.py` accordingly.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run the app

### Streamlit UI
```bash
streamlit run streamlit_app.py
```

### CLI ingestion (optional)
```bash
python ingest.py
```

### Terminal chat (optional)
```bash
python main.py
```

---

## Current design choices

- **Router-first architecture** keeps retrieval selective and reduces noisy context.
- **Chunking at 1000/100** is a practical balance between semantic cohesion and retrieval granularity.
- **Top-k retrieval (`match_count=3`)** limits prompt bloat while keeping enough evidence.
- **Recent-turn memory window** preserves conversation intent without unbounded token growth.

---

## Future improvements

- Add citation snippets (chunk-level source attribution)
- Add confidence scoring for route decisions
- Add multi-document filtering by filename/topic
- Add telemetry for retrieval hit rate and route accuracy

---

## License

Add a license file if you plan to open-source or distribute this project.