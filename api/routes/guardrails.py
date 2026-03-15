"""Guardrails endpoints."""

from fastapi import APIRouter, Depends
from api.schemas import GuardrailsRequest, GuardrailsResponse
from api.dependencies import get_components, ComponentManager

router = APIRouter(prefix="/guardrails", tags=["Guardrails"])


@router.post("/check", response_model=GuardrailsResponse)
def check_input(
    request: GuardrailsRequest,
    components: ComponentManager = Depends(get_components),
):
    """Check if input passes safety guardrails."""
    guardrails = components.guardrails
    report = guardrails.check_input(request.text)

    result = GuardrailsResponse(
        passed=report.passed,
        status=report.status.value,
        category=report.metadata.get("category", "unknown"),
        math_confidence=float(report.metadata.get("math_confidence", 0)),
        blocked_reason=report.blocked_reason,
        warnings=report.warnings,
    )

    if not report.passed:
        result.suggestions = guardrails.get_suggestions(report)
        result.rejection_message = guardrails.get_rejection_message(report)

    return result