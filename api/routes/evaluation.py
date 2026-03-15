"""Evaluation endpoints for batch testing and per-query evaluation."""

import logging
import time
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from api.schemas import EvaluateRequest, BatchEvalRequest
from api.dependencies import get_components, ComponentManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/evaluate", tags=["Evaluation"])


def _deep_serialize(obj):
    """
    Recursively convert ALL non-JSON-native types to Python natives.
    Handles numpy, dataclasses, custom objects, etc.
    """
    try:
        import numpy as np
        numpy_available = True
    except ImportError:
        numpy_available = False

    if obj is None:
        return None

    # Primitive types — pass through
    if isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int, float)):
        return obj

    # Numpy types
    if numpy_available:
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return [_deep_serialize(x) for x in obj.tolist()]

    # Dict
    if isinstance(obj, dict):
        return {str(k): _deep_serialize(v) for k, v in obj.items()}

    # List/tuple
    if isinstance(obj, (list, tuple)):
        return [_deep_serialize(item) for item in obj]

    # Objects with __dict__ (dataclasses, custom classes)
    if hasattr(obj, '__dict__'):
        return {str(k): _deep_serialize(v) for k, v in vars(obj).items()
                if not k.startswith('_')}

    # Objects with to_dict method
    if hasattr(obj, 'to_dict'):
        return _deep_serialize(obj.to_dict())

    # Fallback: try to convert to float/int/str
    try:
        f = float(obj)
        if f == int(f):
            return int(f)
        return f
    except (ValueError, TypeError):
        return str(obj)


@router.post("/solution")
def evaluate_solution(
    request: EvaluateRequest,
    components: ComponentManager = Depends(get_components),
):
    """Evaluate a single solution's quality."""
    try:
        evaluator = components.evaluator

        eval_result = evaluator.evaluate_query(
            question=request.question,
            solution=request.solution,
            explanation=request.explanation or "",
            verification="",
            rag_context=[],
            input_type="text",
            confidence=1.0,
            was_human_edited=False,
            latency_ms=0,
            topic="",
            query_id="",
        )

        response_data = {
            "overall_score": round(float(eval_result.overall_score), 3),
            "grade": str(eval_result.grade),
            "passed": bool(eval_result.passed),
            "rag_relevance": round(float(eval_result.rag_relevance_score), 3),
            "solution_quality": round(float(eval_result.solution_quality_score), 3),
            "explanation_clarity": round(float(eval_result.explanation_clarity_score), 3),
            "issues": list(eval_result.issues_found) if eval_result.issues_found else [],
            "suggestions": list(eval_result.suggestions) if eval_result.suggestions else [],
            "solution_assessment": str(eval_result.solution_assessment or ""),
            "explanation_assessment": str(eval_result.explanation_assessment or ""),
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error("Solution evaluation failed: %s", e, exc_info=True)
        raise HTTPException(500, f"Evaluation failed: {str(e)}")


@router.post("/batch")
def run_batch_evaluation(
    request: BatchEvalRequest,
    components: ComponentManager = Depends(get_components),
):
    """Run batch evaluation against test dataset."""
    start_time = time.time()

    try:
        from src.evaluation.eval_manager import get_eval_manager
        eval_manager = get_eval_manager()

        report = eval_manager.run_full_evaluation(
            include_rag=request.include_rag,
            include_solution=request.include_solutions,
            include_guardrails=request.include_guardrails,
            include_memory=False,
            topic_filter=request.topic if request.topic else None,
            max_cases=request.max_cases,
        )

        # Build component results with deep serialization
        component_results = {}
        for name, comp_report in report.component_reports.items():
            metrics = []
            if hasattr(comp_report, 'metrics') and comp_report.metrics:
                for m in comp_report.metrics:
                    # Extract metric fields safely
                    m_name = str(getattr(m, 'name', 'Unknown'))
                    m_percentage = float(getattr(m, 'percentage', 0))
                    m_details = str(getattr(m, 'details', ''))

                    # Try to get score — might be named differently
                    m_score = 0.0
                    for score_attr in ('score', 'value', 'raw_score'):
                        if hasattr(m, score_attr):
                            try:
                                m_score = float(getattr(m, score_attr))
                            except (TypeError, ValueError):
                                pass
                            break

                    metrics.append({
                        "name": m_name,
                        "score": round(m_score, 3),
                        "percentage": round(m_percentage, 1),
                        "details": m_details,
                    })

            # Extract errors safely
            errors = []
            if hasattr(comp_report, 'errors') and comp_report.errors:
                errors = [str(e) for e in comp_report.errors[:10]]

            # Extract details safely — deep serialize to avoid numpy issues
            details = []
            if hasattr(comp_report, 'details') and comp_report.details:
                raw_details = comp_report.details[:10]
                details = _deep_serialize(raw_details)

            # Get overall score safely
            comp_score = 0.0
            if hasattr(comp_report, 'overall_score'):
                try:
                    comp_score = float(comp_report.overall_score)
                except (TypeError, ValueError):
                    comp_score = 0.0

            component_results[str(name)] = {
                "overall_score": round(comp_score, 1),
                "metrics": metrics,
                "errors": errors,
                "details": details,
            }

        # Get overall report scores safely
        overall_score = 0.0
        overall_grade = "N/A"
        try:
            overall_score = float(report.overall_score)
        except (TypeError, ValueError, AttributeError):
            pass
        try:
            overall_grade = str(report.overall_grade)
        except (TypeError, AttributeError):
            pass

        response_data = {
            "status": "success",
            "overall_score": round(overall_score, 1),
            "overall_grade": overall_grade,
            "components": component_results,
            "total_time_ms": round((time.time() - start_time) * 1000, 1),
            "test_cases": request.max_cases,
            "topic_filter": request.topic or None,
        }

        # Final safety net: deep serialize the entire response
        safe_response = _deep_serialize(response_data)

        return JSONResponse(content=safe_response)

    except ImportError as e:
        logger.error("Eval manager import failed: %s", e, exc_info=True)
        raise HTTPException(
            500,
            "Evaluation module not available. Check src/evaluation/ setup."
        )
    except Exception as e:
        logger.error("Batch evaluation failed: %s", e, exc_info=True)
        raise HTTPException(500, f"Batch evaluation failed: {str(e)}")