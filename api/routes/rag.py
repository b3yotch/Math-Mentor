"""RAG retrieval endpoints."""

from fastapi import APIRouter, Depends
from api.schemas import RetrieveRequest, RetrieveResponse
from api.dependencies import get_components, ComponentManager

router = APIRouter(prefix="/retrieve", tags=["RAG"])


@router.post("", response_model=RetrieveResponse)
def retrieve_context(
    request: RetrieveRequest,
    components: ComponentManager = Depends(get_components),
):
    """Retrieve relevant math knowledge from RAG."""
    retriever = components.retriever

    results = retriever.retrieve(
        request.query, top_k=request.top_k, min_score=0.1, rerank=True
    )
    results = results or []

    if request.topic_filter:
        results = [
            r for r in results
            if request.topic_filter.lower() in str(r.get("topic", "")).lower()
        ]

    formatted = []
    for i, r in enumerate(results):
        formatted.append({
            "rank": i + 1,
            "topic": r.get("topic", ""),
            "subtopic": r.get("subtopic", ""),
            "content": r.get("content", ""),
            "score": float(r.get("score", 0)),
        })

    return RetrieveResponse(
        query=request.query,
        results_count=len(formatted),
        topic_filter=request.topic_filter,
        results=formatted,
    )