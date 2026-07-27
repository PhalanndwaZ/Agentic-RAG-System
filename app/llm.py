from groq import Groq
from app.config import get_settings

# Kept at module level so it's easy to find/tweak without digging into
# the class body.
SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the
    provided context. If the context does not contain the answer, say so clearly instead
    of guessing. Do not fabricate information that isn't in the context."""

class LLMClient:

    def __init__(self):
        settings = get_settings()
        # Groq client picks up the API key from settings (which reads it
        # from .env)

        self._client = Groq(api_key = settings.groq_api_key)
        self._model = settings.groq_model


    def generate_answer(self, question: str, context_chunks: list[str]) -> str:
        context = "\n\n---\n\n".join(context_chunks) if context_chunks else "(no context retrieved)"
        user_prompt = f"Context: \n{context}\n\nQuestion: {question}"

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Groq returned an empty response")
        return content