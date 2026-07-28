from fastapi import APIRouter, Request

from app.agent import run_agent
from app.schemas import AgenticQueryRequest,AgenticQueryResponse


router = APIRouter()



@router.post("/agentic-query", response_model=AgenticQueryResponse)
async def agentic_query(request: Request, body: AgenticQueryRequest):
    embedder = request.app.state.embedder
    store = request.app.state.vectorstore
    groq_client = request.app.state.llm._client
    model = request.app.state.llm._model

    result = run_agent(
        question=body.question,
        embedder= embedder,
        store= store,
        groq_client= groq_client,
        model= model,
    )
    return AgenticQueryResponse(
        answer= result["answer"],
        tool_calls= result["tool_calls"],
    )