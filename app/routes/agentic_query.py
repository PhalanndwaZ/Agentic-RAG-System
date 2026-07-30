from fastapi import APIRouter, Request

from app.agent import run_agent
from app.schemas import AgenticQueryRequest,AgenticQueryResponse
from app.evaluation import evaluate_faithfulness

router = APIRouter()



@router.post("/agentic-query", response_model=AgenticQueryResponse)
async def agentic_query(request: Request, body: AgenticQueryRequest) -> AgenticQueryResponse:
    embedder = request.app.state.embedder
    store = request.app.state.vectorstore
    groq_client = request.app.state.llm._client
    model = request.app.state.llm._model

    result = run_agent(
        question=body.question,
        embedder=embedder,
        store=store,
        groq_client=groq_client,
        model=model,
    )

    faithfulness = evaluate_faithfulness(
        question=body.question,
        context_chunks=result["context_chunks"],
        answer=result["answer"],
        groq_client=groq_client,
        model=model,
    )

    return AgenticQueryResponse(
        answer=result["answer"],
        tool_calls=result["tool_calls"],
        faithfulness=faithfulness,
    )