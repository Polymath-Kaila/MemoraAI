
from fastapi import FastAPI, HTTPException
from .settings import get_settings
from .models import IngestRequest, AskRequest, AskResponse
from .elastic_memory import upsert_chunk, hybrid_search
from .retriever import mmr
from .utils import chunk_text, approx_token_count
from .vertex_ai import get_embedding, generate_text
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")


S = get_settings()
app = FastAPI(title=S.app_name)

@app.get("/health")
def health():
    # minimal liveness check
    return {"status": "ok", "app": S.app_name, "elastic_index": S.elastic_index, "location": S.gcp_location}

@app.post("/ingest")
def ingest(req: IngestRequest):
    chunks = chunk_text(req.text)
    for ch in chunks:
        emb = get_embedding(ch)
        upsert_chunk(req.project_id, ch, emb, req.tags or [])
    return {"ingested_chunks": len(chunks)}




@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Handle user questions with persistent memory retrieval via Elasticsearch + VertexAI."""
    q_vec = get_embedding(req.query)

    # 🔹 Retrieve a larger memory set for richer recall
    hits = hybrid_search(req.project_id, req.query, q_vec, k=max(15, req.k))
    docs = [h["text"] for h in hits]

    if not docs:
        context = ""
        used = 0
        selected_docs = []
    else:
        # 🔹 Compute embeddings for each document to use in MMR (diversity re-ranking)
        doc_vecs = [get_embedding(d) for d in docs]
        sel_idx = mmr(q_vec, docs, doc_vecs, k=min(10, len(docs)))  # keep top diverse docs
        selected_docs = [docs[i] for i in sel_idx]
        used = len(selected_docs)

        # 🔹 Build contextual memory within token limit
        budget = S.token_budget
        context_parts = []
        for d in selected_docs:
            if approx_token_count("\n".join(context_parts)) + approx_token_count(d) < budget:
                context_parts.append(d)
            else:
                break
        context = "\n\n".join(context_parts)

    # 🔍 Debug output for what was actually retrieved
    print("\n🔎 Retrieved context snippets:")
    if selected_docs:
        for i, d in enumerate(selected_docs):
            print(f"{i+1}. {d[:120]}...")
    else:
        print("No relevant memories retrieved.")

    # Prompt design: encourage Gemini to use multiple memories
    prompt = f"""
{S.system_preamble}

You are MemoraAI, an assistant that remembers user facts across sessions.
Below are pieces of information you've previously stored in your memory.
Use *all relevant ones* to answer the question.

Stored memory snippets:
{context if context else '[No prior memory found yet]'}

User question: {req.query}

If multiple memories are relevant, combine them naturally.
If something seems unrelated, ignore it politely.
"""

    # Generate answer using VertexAI (Gemini)
    try:
        text = generate_text(prompt).strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    # 🧾 Return structured response
    return AskResponse(
        response=text,
        used_snippets=used,
        tokens_estimate=approx_token_count(prompt),
    )

