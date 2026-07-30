import psycopg
from pgvector.psycopg import register_vector
from app.schemas import RetrievedChunk

from app.config import get_settings

class VectorStore:
    """Wraps a psycopg connection with pgvector support. One connection is
    opened at startup (via FastAPI's lifespan) and reused across requests,
    rather than opening a new connection per request."""

    def __init__(self):
        settings = get_settings()
         # autocommit=True: each statement commits immediately, so we don't
        # need to manually manage transactions for these simple inserts/queries.
        self._conn = psycopg.connect(settings.database_url,autocommit = True)
        # Registers the vector type on this connection so psycopg knows how
        # to adapt Python lists <-> Postgres's `vector` column type.
        register_vector(self._conn)



    def add_document(self, source: str) -> str:
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (source) VALUES (%s) RETURNING id",
                (source,),
            )
            row = cur.fetchone()
            assert row is not None
            return str(row[0])

    def add_chunks(self, document_id: str, chunks: list[str], embeddings: list[list[float]]) -> int:
            # Inserts one row per chunk, pairing each chunk's text with its
            # embedding. Returns how many chunks were stored, for confirmation
            # in the API response.
            with self._conn.cursor() as cur:
                for i,(content,embbeding) in enumerate(zip(chunks,embeddings)):
                    cur.execute(
                        """
                        INSERT INTO chunks (document_id, content, chunk_index, embedding)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (document_id,content,i,embbeding),
                    )
            return len(chunks)

    def similarity_search(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        with self._conn.cursor() as cur:
            # Join chunks to documents so we can return which paper each
            # chunk came from, alongside its content and similarity score.
            cur.execute(
                """
                SELECT chunks.id, chunks.content, 1 - (chunks.embedding <=> %s::vector) AS similarity, documents.source
                FROM chunks
                JOIN documents ON chunks.document_id = documents.id
                ORDER BY chunks.embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, top_k),
            )
            rows = cur.fetchall()

        return [
            RetrievedChunk(chunk_id=str(row[0]), content=row[1], similarity=float(row[2]), source=row[3])
            for row in rows
        ]

    def close(self):
        # Closes the Postgres connection cleanly on app shutdown.
        self._conn.close()


    