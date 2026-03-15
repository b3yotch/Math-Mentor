"""Memory and feedback endpoints."""

from fastapi import APIRouter, Depends, Query
from api.schemas import (
    FeedbackRequest, FeedbackResponse,
    HistoryResponse, HistoryItem,
)
from api.dependencies import get_components, ComponentManager

router = APIRouter(prefix="/memory", tags=["Memory"])


@router.get("/history", response_model=HistoryResponse)
def get_history(
    limit: int = Query(default=10, ge=1, le=50),
    topic: str = Query(default=""),
    components: ComponentManager = Depends(get_components),
):
    """Get problem history."""
    memory = components.memory
    history = memory.get_problem_history(limit=limit) or []

    if topic:
        history = [
            p for p in history
            if topic.lower() in str(p.get("topic", "")).lower()
        ]

    items = [
        HistoryItem(
            id=p.get("id", ""),
            input_type=p.get("input_type", ""),
            question=p.get("extracted_text", ""),
            topic=p.get("topic", ""),
            solution_preview=str(p.get("solution", ""))[:300],
            confidence=float(p.get("confidence_score", 0)),
            was_human_edited=p.get("was_human_edited", False),
            created_at=str(p.get("created_at", "")),
        )
        for p in history
    ]

    return HistoryResponse(total=len(items), problems=items)


@router.post("/feedback", response_model=FeedbackResponse)
def record_feedback(
    request: FeedbackRequest,
    components: ComponentManager = Depends(get_components),
):
    """Record feedback on a solved problem."""
    memory = components.memory
    memory.record_feedback(
        request.problem_id,
        is_correct=request.is_correct,
        comment=request.comment,
        corrected_solution=request.corrected_solution,
    )
    return FeedbackResponse(
        problem_id=request.problem_id,
        is_correct=request.is_correct,
    )


@router.get("/stats")
def get_stats(components: ComponentManager = Depends(get_components)):
    """Get memory statistics."""
    memory = components.memory
    return memory.get_statistics()