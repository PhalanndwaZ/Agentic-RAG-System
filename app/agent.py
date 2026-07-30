import json
from typing import Any

from groq import APIStatusError

SIMILARITY_THRESHOLD = 0.3


# This wraps the existing retrieval pipeline (embed -> similarity_search)
# as a single callable the agent can invoke. It's the same logic /query
# already uses, just exposed as a standalone function instead of being
# inlined in the route.
def search_docs(question: str, embedder, store, top_k: int = 5):
    query_embedding = embedder.embed_one(question)
    return store.similarity_search(query_embedding, top_k=top_k)


# Groq's tool-calling API expects tools described in this JSON-schema
# shape. This describes search_docs to the model — its name, what it's
# for, and what argument(s) it accepts — so the LLM can decide when to
# call it and with what input, without us hardcoding that decision.
SEARCH_DOCS_TOOL = {
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": (
            "Search the knowledge base for chunks of text relevant to a query. "
            "Use this when you need information to answer the user's question. "
            "You can call this multiple times with reformulated queries if the "
            "first search doesn't return relevant results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up in the knowledge base.",
                }
            },
            "required": ["query"],
        },
    },
}


def execute_tool_call(tool_call, embedder, store) -> dict:
    args = json.loads(tool_call.function.arguments)

    if tool_call.function.name == "search_docs":
        results = search_docs(args["query"], embedder, store)
        has_relevant = bool(results) and results[0].similarity >= SIMILARITY_THRESHOLD

        if not results:
            content = "No relevant results found."
        else:
            # Tag each chunk with its source filename so the model can
            # cite which paper a fact came from, without needing the
            # full document — just the short identifier.
            content = "\n\n---\n\n".join(
                f"[source: {chunk.source}] [similarity: {chunk.similarity:.2f}] {chunk.content}"
                for chunk in results
            )
        return {"content": content, "has_relevant": has_relevant, "results": results}

    return {"content": f"Unknown tool: {tool_call.function.name}", "has_relevant": False, "results": []}


def run_agent(
    question: str,
    embedder,
    store,
    groq_client,
    model: str,
    max_steps: int = 4,
    max_retries: int = 6,
) -> dict:
    """Runs the ReAct-style agent loop: the model reasons about whether it
    needs to search, calls search_docs if so, reads the result, and either
    searches again (e.g. with a reformulated query) or answers directly.
    Returns the final answer plus a trace of every tool call made, plus the
    actual retrieved context (for downstream faithfulness evaluation). The
    final answer is only trusted if genuinely relevant context was found
    at some point — otherwise a refusal is enforced in code, regardless of
    what the model itself tries to answer. Malformed tool-call recoveries
    (retries) are tracked separately from real reasoning steps, so Groq's
    occasional flakiness doesn't eat into the model's actual max_steps budget."""

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to a search_docs tool "
                "that searches a knowledge base. For ANY question that references "
                "specific facts, names, numbers, findings, or details that could "
                "plausibly come from documents in a knowledge base, you MUST call "
                "search_docs first — do not answer from memory. Only skip search_docs "
                "for pure greetings, small talk, or basic arithmetic. If your first "
                "search returns weak or irrelevant results, reformulate the query and "
                "try again — but once you have relevant results, answer using them "
                "rather than continuing to search. "
                "Answer only using information retrieved via search_docs. Each "
                "retrieved chunk is tagged with its source document, like "
                "[source: filename.pdf]. When you answer, cite which document each "
                "fact came from, especially if your answer draws on more than one "
                "document — do not blend facts from different papers into one "
                "statement without attributing them separately."
            ),
        },
        {"role": "user", "content": question},
    ]

    tool_calls_made = []
    all_retrieved_chunks = []  # accumulates every relevant chunk seen, across all calls
    found_relevant_context = False  # code-enforced, not just prompt-trusted
    retries_used = 0  # counts malformed-tool-call recoveries separately from real steps

    step = 0
    while step < max_steps:
        try:
            response = groq_client.chat.completions.create(
                model=model,
                messages=messages,
                tools=[SEARCH_DOCS_TOOL],
                temperature=0.2,
            )
        except APIStatusError:
            # Some models occasionally generate a malformed tool call
            # (invalid JSON/syntax) that Groq rejects outright. Tracked
            # with its own retries_used counter, separate from step, so
            # this flakiness never eats into the model's real reasoning
            # budget.
            retries_used += 1
            if retries_used > max_retries:
                break
            # Deliberately do NOT offer "answer directly instead" here —
            # the model already decided to search; this message should
            # only push it to fix the malformed call, not give it an
            # excuse to abandon the search attempt entirely.
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous tool call was malformed and could not be "
                        "processed. Please retry the exact same tool call with "
                        "correctly formatted JSON arguments."
                    ),
                }
            )
            continue  # retry doesn't increment step — doesn't cost a real turn

        message = response.choices[0].message

        # No tool call means the model decided it's ready to answer.
        if not message.tool_calls:
            if not found_relevant_context:
                return {
                    "answer": "I can only answer questions about the knowledge base, and I don't have relevant information for this one.",
                    "tool_calls": tool_calls_made,
                    "context_chunks": all_retrieved_chunks,
                }
            return {
                "answer": message.content,
                "tool_calls": tool_calls_made,
                "context_chunks": all_retrieved_chunks,
            }

        # The model wants to call a tool. Append its own turn to the
        # conversation first (required by the API) — reconstructed as a
        # plain dict, since Groq's API rejects the raw SDK object type
        # if we pass it back in directly.
        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ],
            }
        )

        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call, embedder, store)
            if result["has_relevant"]:
                found_relevant_context = True
                # Only keep chunks from calls that actually cleared the
                # relevance bar — no point evaluating faithfulness against
                # noise the agent itself ignored.
                all_retrieved_chunks.extend(chunk.content for chunk in result["results"])

            tool_calls_made.append(
                {"tool": tool_call.function.name, "arguments": tool_call.function.arguments}
            )
            # Feed the tool's result back into the conversation as a
            # "tool" role message, tagged with the same call ID so the
            # model knows which call this result answers.
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result["content"],
                }
            )

        step += 1  # a real reasoning turn happened — this is what counts toward max_steps

    # Safety valve: max_steps (or max_retries) reached without a settled answer.
    if not found_relevant_context:
        return {
            "answer": "I can only answer questions about the knowledge base, and I don't have relevant information for this one.",
            "tool_calls": tool_calls_made,
            "context_chunks": all_retrieved_chunks,
        }
    return {
        "answer": "I wasn't able to settle on an answer in time.",
        "tool_calls": tool_calls_made,
        "context_chunks": all_retrieved_chunks,
    }