from sentence_transformers import SentenceTransformer 
from app.config import get_settings

class EmbeddingModel:
    """Wraps sentence-transformers so the rest of the app never imports it
    directly. Load once (in FastAPI's lifespan, not per-request) — reloading
    a transformer model on every call is the most common perf bug in RAG
    demos."""

    def __init__(self,model_name:str | None = None):
        # Fall back to the model name in settings/.env if none is passed in,
        # so callers can override for tests without touching config.
        settings = get_settings()
        self._model = SentenceTransformer(model_name or settings.embedding_model)
    

    def embed(self,texts: list[str]) -> list[list[float]]:
        # normalize_embeddings=True makes each vector unit-length, so cosine
        # similarity later reduces to a simple dot product / distance calc.
        vectors = self._model.encode(texts, normalize_embeddings = True)
        return vectors.tolist()
    

    def embed_one(self, text:str)-> list[float]:
        # Convenience wrapper: query time usually means "embed this one
        # question," not a batch — reuse embed() so there's one code path.
        return self.embed([text])[0]
    
    
    
