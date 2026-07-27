from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.embeddings import EmbeddingModel
from app.llm import LLMClient
from app.routes import ingest, query
from app.vectorstore import VectorStore

#long the errors 
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once on startup: load the heavy/stateful resources exactly once
    # and stash them on app.state, so route handlers never re-create them
    # per-request.
    app.state.embedder = EmbeddingModel()
    app.state.vectorstore = VectorStore()
    app.state.llm = LLMClient()
    yield
    # Runs once on shutdown: close the DB connection cleanly.
    app.state.vectorstore.close()


app = FastAPI(title="Agentic RAG API", version="0.1.0", lifespan=lifespan)

# Mounts the route handlers defined in routes/ingest.py and routes/query.py
# onto this app, grouped by tag for the /docs Swagger UI.
app.include_router(ingest.router, tags=["ingest"])
app.include_router(query.router, tags=["query"])


@app.get("/health")
async def health():
    # Simple liveness check — no dependencies, just confirms the server
    # process is up and responding.
    return {"status": "ok"}