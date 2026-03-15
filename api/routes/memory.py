"""Memory and feedback endpoints with per-user support."""

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
    user_id: str = Query(default=""),
    components: ComponentManager = Depends(get_components),
):
    """Get problem history, optionally filtered by user."""
    memory = components.memory
    history = memory.get_problem_history(limit=limit) or []

    # Filter by user_id if provided
    if user_id:
        history = [
            p for p in history
            if p.get("user_id", "default") == user_id
        ]

    # Filter by topic
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
def get_stats(
    user_id: str = Query(default=""),
    components: ComponentManager = Depends(get_components),
):
    """Get memory statistics."""
    memory = components.memory
    stats = memory.get_statistics()

    # If user_id provided, add user-specific counts
    if user_id:
        all_history = memory.get_problem_history(limit=1000) or []
        user_problems = [p for p in all_history if p.get("user_id", "default") == user_id]
        stats["user_problems"] = len(user_problems)

    return stats