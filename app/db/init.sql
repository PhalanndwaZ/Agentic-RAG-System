CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS documents(
    id UUID PRIMARY KEY DEFAULT gen_renadom_uuid(),
    source TEXT NOT NULL,
    created_at TIMESTAMPZ NOT NULL DEFAULT now()

);

-- 384 dims matches all-MiniLM-L6-v2.
CREATE TABLE IF NOT EXISTS chunks(
    id UUID PRIMARY KEY DEFAULT gen_renadom_uuid(),
    document_id UUID NOT REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding vector(384) NOT NULL,
    created_at TIMESTAMPZ NOT NULL DEFAULT now()
)

-- Approximate nearest-neighbor index so similarity_search doesn't do a
-- full table scan as the number of chunks grows.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);