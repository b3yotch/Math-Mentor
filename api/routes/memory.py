"""Memory and feedback endpoints with per-user support."""

import logging
from fastapi import APIRouter, Depends, Query
from api.schemas import (
    FeedbackRequest, FeedbackResponse,
    HistoryResponse, HistoryItem,
)
from api.dependencies import get_components, ComponentManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/memory", tags=["Memory"])


@router.get("/history", response_model=HistoryResponse)
def get_history(
    limit: int = Query(default=10, ge=1, le=50),
    topic: str = Query(default=""),
    user_id: str = Query(default=""),
    components: ComponentManager = Depends(get_components),
):
    """
    Get problem history with full solutions.

    User-id filtering uses soft match:
      - Records whose user_id matches → included
      - Records with NO user_id / "default" / empty → also included
        (legacy records saved before per-user tracking)
    """
    memory = components.memory

    fetch_limit = min(limit * 3, 150) if (user_id or topic) else limit
    history = memory.get_problem_history(limit=fetch_limit) or []

    # Soft user-id filter
    if user_id:
        filtered = []
        for p in history:
            record_uid = (p.get("user_id") or "").strip()
            if record_uid == "" or record_uid == "default" or record_uid == user_id:
                filtered.append(p)
        history = filtered

    # Topic filter
    if topic:
        history = [
            p for p in history
            if topic.lower() in str(p.get("topic", "")).lower()
        ]

    # Apply final limit
    history = history[:limit]

    items = []
    for p in history:
        # Robust field extraction
        question = (
            p.get("extracted_text")
            or p.get("question")
            or p.get("original_input")
            or "Unknown problem"
        )

        full_solution = str(
            p.get("solution")
            or p.get("solution_text")
            or ""
        )

        explanation = str(p.get("explanation", ""))
        verification = str(p.get("verification_result", p.get("verification", "")))

        confidence_raw = p.get("confidence_score", p.get("confidence", 0))
        try:
            confidence_val = float(confidence_raw)
        except (TypeError, ValueError):
            confidence_val = 0.0

        record_id = str(
            p.get("id")
            or p.get("_id")
            or p.get("problem_id")
            or f"prob_{len(items)}"
        )

        # Preview: first 200 chars for collapsed view
        preview = full_solution[:200]
        if len(full_solution) > 200:
            preview += "..."

        items.append(
            HistoryItem(
                id=record_id,
                input_type=p.get("input_type", "text"),
                question=question,
                topic=p.get("topic", ""),
                solution_preview=preview,
                solution=full_solution,
                explanation=explanation,
                verification=verification,
                confidence=confidence_val,
                was_human_edited=bool(p.get("was_human_edited", False)),
                created_at=str(p.get("created_at", "")),
            )
        )

    return HistoryResponse(total=len(items), problems=items)


@router.post("/feedback", response_model=FeedbackResponse)
def record_feedback(
    request: FeedbackRequest,
    components: ComponentManager = Depends(get_components),
):
    """Record feedback on a solved problem."""
    memory = components.memory
    try:
        memory.record_feedback(
            request.problem_id,
            is_correct=request.is_correct,
            comment=request.comment,
            corrected_solution=request.corrected_solution,
        )
    except Exception as e:
        logger.error("Feedback save failed: %s", e)

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

    if user_id:
        try:
            all_history = memory.get_problem_history(limit=1000) or []
            user_problems = [
                p for p in all_history
                if (p.get("user_id") or "").strip() in ("", "default", user_id)
            ]
            stats["user_problems"] = len(user_problems)
        except Exception:
            stats["user_problems"] = stats.get("total_problems", 0)

    return stats