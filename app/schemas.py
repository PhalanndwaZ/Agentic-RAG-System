from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """A single chunk returned from a similarity search, with its
    similarity score attached."""
    chunk_id: str
    content: str
    similarity: float = Field(..., description="Cosine similarity, 0-1, higher is more relevant")


class IngestResponse(BaseModel):
    """Returned by POST /ingest after a document has been chunked,
    embedded, and stored."""
    document_id: str
    source: str
    chunks_created: int


class QueryRequest(BaseModel):
    """Request body for POST /query."""
    question: str
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    """Response body for POST /query."""
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    used_retrieval: bool