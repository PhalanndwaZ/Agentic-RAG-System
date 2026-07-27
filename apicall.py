from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq

app = FastAPI()

# hardcoded Groq client
client = Groq(api_key="your-actual-api-key-here")


# request body shape: just a single message from the user
class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(request: ChatRequest):
    """Take a user message, send it to Groq, and return the reply."""
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": request.message}
        ]
    )
    reply = response.choices[0].message.content
    return {"reply": reply}