import json
from app.schemas import FaithfulnessResult

EVALUATION_SYSTEM_PROMPT = """You are a strict evaluator of answer faithfulness. You are
given a question, the context that was retrieved to answer it, and the generated answer.
Score how well the answer is grounded in the provided context ONLY — not whether the
answer is true in general, just whether it's actually supported by this context.

Respond with ONLY valid JSON, no other text:
{
  "score": <1-5, where 1 = not grounded at all / contradicts context, 5 = fully grounded>,
  "reasoning": "<one sentence explaining the score>"
}
"""


def evaluate_faithfulness(
    question: str,
    context_chunks: list[str],
    answer: str,
    groq_client,
    model: str,
    low_confidence_threshold: int = 3,
) -> FaithfulnessResult:
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else "(no context was retrieved)"
    user_prompt = f"Question: {question}\n\nRetrieved context:\n{context}\n\nGenerated answer:\n{answer}"

    response = groq_client.chat.completions.create(
        model = model,
        messages = [
            {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature  = 0.0 ,
    )

    raw = response.choices[0].message.content
    parsed = json.loads(raw)

    return FaithfulnessResult(
        score = parsed["score"],
        reasoning = parsed["reasoning"],
        is_low_confidence = parsed["score"] < low_confidence_threshold,
    )