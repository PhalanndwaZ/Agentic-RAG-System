# Agentic RAG API

A retrieval-augmented generation system that goes beyond retrieve-and-answer-once:
the agent decides for itself whether a question needs a knowledge-base search,
reformulates and retries on weak results, and every answer is scored for how
faithfully it's grounded in what was actually retrieved.

## Architecture

**Stack:** sentence-transformers (local, free embeddings) + pgvector (hosted on
Neon) + Groq (llama-3.3-70b-versatile) + FastAPI.


## Endpoints

- **`GET /health`** — liveness check.
- **`POST /ingest`** — single file upload (`.pdf`, `.txt`, `.md`). Chunks,
  embeds, and stores it.
- **`POST /ingest-folder`** — batch-ingests every supported file in a local
  folder path. Failures on individual files are logged and skipped rather
  than crashing the whole batch.
- **`POST /query`** — Stage 1/2 pipeline: always embeds the question,
  retrieves top-k chunks, and asks the LLM to answer grounded in them. If the
  best match is below a similarity threshold, it declines rather than
  answering off weak context.
- **`POST /agentic-query`** — the full agentic pipeline (see below).

## The agentic loop (`/agentic-query`)

Given a question, the model is handed a `search_docs` tool and a system
prompt instructing it to:

1. Decide whether the question needs a knowledge-base search at all (skips
   for greetings/small talk/arithmetic).
2. Call `search_docs` with a query it chooses.
3. If results look weak, reformulate and search again.
4. Once it has relevant results, answer — citing which source document each
   fact came from (chunks are tagged `[source: filename.pdf]`), so answers
   that draw on multiple papers don't silently blend them into one
   unattributed claim.

**Code-enforced, not just prompted:** the loop tracks whether any tool call
actually returned results above the similarity threshold. If none ever did —
regardless of what the model's own answer text says — the response is
overridden with a flat refusal. This closes a real gap where a model could
otherwise ignore its instructions and answer from general knowledge anyway.

**Resilience to malformed tool calls:** Groq's model occasionally generates
invalid tool-call syntax, which the API rejects outright. These are caught
and retried with a corrective nudge — tracked with a separate retry counter
so this flakiness never eats into the model's real reasoning budget
(`max_steps`).

## Self-evaluation (`evaluation.py`)

After the agent settles on an answer, a second, separate LLM call scores it
1-5 on *faithfulness* — whether the answer is actually supported by the
context that was retrieved, not whether it's true in some general sense.
Low scores (currently <3) are flagged as low-confidence. This is the same
rubric-style pattern used in real RAG evaluation research (see the
sensor-interpretation paper in the test set) — grounding, not just
correctness, is what's being measured.

## Known limitations / rough edges

- One ingested PDF produced garbled `uniXXXXXXXX` text due to a font-encoding
  quirk `pypdf` couldn't handle — it's chunked and stored, and can
  occasionally surface as noise on low-relevance queries. Worth re-checking
  which file this was and re-extracting or excluding it.
- The `ivfflat` similarity index was dropped after it caused false "no
  results" failures at small scale (too many clusters for too few rows). No
  index is currently in place — fine at hundreds of chunks, but should be
  re-added (sized to roughly `rows / 1000`) before scaling to thousands.
- Faithfulness scoring assumes the evaluator model returns clean JSON with
  no surrounding text — not yet hardened against malformed responses.
- No `search_web` fallback tool yet — the agent can only search the local knowledge base.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in DATABASE_URL (Neon) and GROQ_API_KEY
uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000/docs` to try it via Swagger.

