from fastapi import APIRouter, Request
from app.schemas import QueryRequest,QueryResponse

router = APIRouter()

SIMILARITY_THRESHOLD = 0.3


@router.post("/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    embedder = request.app.state.embedder
    store = request.app.state.vectorstore
    llm = request.app.state.llm

    query_embedding = embedder.embed_one(body.question)
    retrieved = store.similarity_search(query_embedding, top_k=body.top_k)

    # If nothing retrieved is actually relevant, don't call the LLM at all
    # and don't leak any chunk content back to the caller.
    if not retrieved or retrieved[0].similarity < SIMILARITY_THRESHOLD:
        return QueryResponse(
            answer="I don't have information about that in my knowledge base.",
            retrieved_chunks=[],
            used_retrieval=False,
        )

    answer = llm.generate_answer(
        question=body.question,
        context_chunks=[chunk.content for chunk in retrieved],
    )

    return QueryResponse(
        answer=answer,
        retrieved_chunks=retrieved,
        used_retrieval=True,
    )