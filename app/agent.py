import json

from groq import BadRequestError

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
    # When Groq decides to call a tool, it returns the tool name and
    # arguments as a JSON string (not a Python dict) — parse that and
    # dispatch to the real Python function.
    args = json.loads(tool_call.function.arguments)

    if tool_call.function.name == "search_docs":
        results = search_docs(args["query"], embedder, store)
        # Only count results as "relevant" if the best match clears our
        # similarity threshold — same bar /query already uses, so both
        # endpoints agree on what counts as a genuine match vs. noise.
        has_relevant = bool(results) and results[0].similarity >= SIMILARITY_THRESHOLD

        if not results:
            content = "No relevant results found."
        else:
            content = "\n\n---\n\n".join(
                f"[similarity: {chunk.similarity:.2f}] {chunk.content}" for chunk in results
            )
        return {"content": content, "has_relevant": has_relevant}

    return {"content": f"Unknown tool: {tool_call.function.name}", "has_relevant": False}


def run_agent(question: str, embedder, store, groq_client, model: str, max_steps: int = 4) -> dict:
    """Runs the ReAct-style agent loop: the model reasons about whether it
    needs to search, calls search_docs if so, reads the result, and either
    searches again (e.g. with a reformulated query) or answers directly.
    Returns the final answer plus a trace of every tool call made. The
    final answer is only trusted if genuinely relevant context was found
    at some point — otherwise a refusal is enforced in code, regardless
    of what the model itself tries to answer."""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to a search_docs tool "
                "that searches a knowledge base. Decide whether a question needs "
                "a search. If your first search returns weak or irrelevant "
                "results, try again with a reformulated query before giving up. "
                "Answer only using information retrieved via search_docs."
            ),
        },
        {"role": "user", "content": question},
    ]

    tool_calls_made = []
    found_relevant_context = False  # code-enforced, not just prompt-trusted

    for _ in range(max_steps):
        try:
            response = groq_client.chat.completions.create(
                model=model,
                messages=messages,
                tools=[SEARCH_DOCS_TOOL],
                temperature=0.2,
            )
        except BadRequestError:
            # Some models occasionally generate a malformed tool call
            # (invalid JSON/syntax) that Groq rejects outright. Rather
            # than crashing the whole request, nudge the model to try
            # again and let the loop continue.
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous attempt to call a tool was malformed. "
                        "Please try again, calling the tool correctly, or "
                        "answer directly if no tool is needed."
                    ),
                }
            )
            continue

        message = response.choices[0].message

        # No tool call means the model decided it's ready to answer.
        if not message.tool_calls:
            if not found_relevant_context:
                return {
                    "answer": "I can only answer questions about the knowledge base, and I don't have relevant information for this one.",
                    "tool_calls": tool_calls_made,
                }
            return {"answer": message.content, "tool_calls": tool_calls_made}

        # The model wants to call a tool. Append its own turn to the
        # conversation first (required by the API), then execute each
        # requested call.
        messages.append(message)

        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call, embedder, store)
            if result["has_relevant"]:
                found_relevant_context = True

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

    # Safety valve: max_steps reached without a settled answer.
    if not found_relevant_context:
        return {
            "answer": "I can only answer questions about the knowledge base, and I don't have relevant information for this one.",
            "tool_calls": tool_calls_made,
        }
    return {"answer": "I wasn't able to settle on an answer in time.", "tool_calls": tool_calls_made}