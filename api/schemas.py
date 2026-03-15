"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum


# ============================================================
# Enums
# ============================================================
class InputType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"


class SolveStatus(str, Enum):
    SUCCESS = "success"
    BLOCKED = "blocked"
    ERROR = "error"
    OUTPUT_FILTERED = "output_filtered"


# ============================================================
# Requests
# ============================================================
class SolveRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="Math problem to solve")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of RAG documents")
    include_evaluation: bool = Field(default=True, description="Run quality evaluation")

    model_config = {"json_schema_extra": {"examples": [{"question": "Solve x^2 - 5x + 6 = 0", "top_k": 3, "include_evaluation": True}]}}


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)
    topic_filter: Optional[str] = None


class GuardrailsRequest(BaseModel):
    text: str = Field(..., min_length=0, max_length=5000)


class FeedbackRequest(BaseModel):
    problem_id: str = Field(..., min_length=1)
    is_correct: bool
    comment: Optional[str] = ""
    corrected_solution: Optional[str] = ""


class EvaluateRequest(BaseModel):
    question: str
    solution: str
    explanation: Optional[str] = ""


class BatchEvalRequest(BaseModel):
    topic: Optional[str] = None
    max_cases: int = Field(default=10, ge=1, le=26)
    include_rag: bool = True
    include_solutions: bool = True
    include_guardrails: bool = True


class NormalizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


# ============================================================
# Responses
# ============================================================
class RAGSource(BaseModel):
    topic: str = ""
    subtopic: str = ""
    score: float = 0.0
    content_preview: str = ""


class EvaluationResult(BaseModel):
    overall_score: float = 0.0
    grade: str = "F"
    passed: bool = False
    rag_relevance: float = 0.0
    solution_quality: float = 0.0
    explanation_clarity: float = 0.0
    issues: List[str] = []
    suggestions: List[str] = []
    solution_assessment: str = ""
    explanation_assessment: str = ""


class SolveResponse(BaseModel):
    status: SolveStatus
    question: str
    detected_topic: Optional[str] = None
    normalized_question: Optional[str] = None
    solution: Optional[str] = None
    final_answer: Optional[str] = None
    verification: Optional[str] = None
    explanation: Optional[str] = None
    parsed_problem: Optional[Dict[str, Any]] = None
    rag_results_count: int = 0
    rag_sources: List[RAGSource] = []
    similar_problems_found: int = 0
    problem_id: Optional[str] = None
    evaluation: Optional[EvaluationResult] = None
    pipeline_steps: List[str] = []
    latency_ms: float = 0.0
    error: Optional[str] = None
    blocked_reason: Optional[str] = None
    suggestions: Optional[List[str]] = None


class GuardrailsResponse(BaseModel):
    passed: bool
    status: str
    category: str = "unknown"
    math_confidence: float = 0.0
    blocked_reason: Optional[str] = None
    warnings: List[str] = []
    suggestions: Optional[List[str]] = None
    rejection_message: Optional[str] = None


class RetrieveResponse(BaseModel):
    query: str
    results_count: int
    topic_filter: Optional[str] = None
    results: List[Dict[str, Any]] = []


class HistoryItem(BaseModel):
    id: str = ""
    input_type: str = ""
    question: str = ""
    topic: str = ""
    solution_preview: str = ""
    confidence: float = 0.0
    was_human_edited: bool = False
    created_at: str = ""


class HistoryResponse(BaseModel):
    total: int
    problems: List[HistoryItem] = []


class FeedbackResponse(BaseModel):
    status: str = "recorded"
    problem_id: str
    is_correct: bool


class NormalizeResponse(BaseModel):
    original: str
    normalized: str
    changed: bool


class HealthResponse(BaseModel):
    status: str = "healthy"
    groq_api: bool = False
    rag_available: bool = False
    guardrails_active: bool = True
    memory_available: bool = False
    version: str = "1.0.0"