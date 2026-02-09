# src/input_processing/schemas.py
"""
Schemas for input processing with enhanced HITL support.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict, Any
from datetime import datetime
import uuid


@dataclass
class HITLInfo:
    """Information about HITL status for this input."""
    needs_review: bool = False
    request_id: Optional[str] = None
    trigger_reason: str = ""
    confidence_threshold: float = 0.0
    suggestions: List[str] = field(default_factory=list)
    was_reviewed: bool = False
    review_action: str = ""  # "approved", "edited", "rejected", "skipped"
    review_feedback: str = ""


@dataclass
class CanonicalInput:
    """
    Canonical input format with enhanced HITL support.
    
    Represents processed input from any source (text, image, audio)
    in a standardized format with HITL tracking.
    """
    input_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_type: Literal["text", "image", "audio"] = "text"
    raw_file_path: Optional[str] = None
    extracted_text: str = ""
    confidence_score: float = 1.0
    was_human_edited: bool = False
    original_extraction: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)
    
    # Enhanced HITL fields
    hitl_info: HITLInfo = field(default_factory=HITLInfo)
    
    # Configurable thresholds
    _ocr_threshold: float = 0.70
    _asr_threshold: float = 0.65
    
    def needs_hitl(self) -> bool:
        """
        Determine if Human-in-the-Loop review is needed.
        
        Rules:
        - Text input: NEVER needs HITL (user typed it directly)
        - Image input: HITL if confidence < 0.70 or extraction failed
        - Audio input: HITL if confidence < 0.65 or extraction failed
        """
        # Text input never needs HITL
        if self.input_type == "text":
            self.hitl_info.needs_review = False
            return False
        
        # Check for extraction error
        if self.metadata.get("error"):
            self.hitl_info.needs_review = True
            self.hitl_info.trigger_reason = f"Extraction error: {self.metadata['error']}"
            self._set_hitl_suggestions()
            return True
        
        # Check for empty or very short extraction
        if not self.extracted_text or len(self.extracted_text.strip()) < 3:
            self.hitl_info.needs_review = True
            self.hitl_info.trigger_reason = "Extraction returned empty or very short text"
            self._set_hitl_suggestions()
            return True
        
        # Determine threshold
        if self.input_type == "image":
            threshold = self._ocr_threshold
        else:
            threshold = self._asr_threshold
        
        self.hitl_info.confidence_threshold = threshold
        
        # Check confidence
        if self.confidence_score < threshold:
            self.hitl_info.needs_review = True
            self.hitl_info.trigger_reason = (
                f"Confidence ({self.confidence_score:.0%}) below "
                f"threshold ({threshold:.0%})"
            )
            self._set_hitl_suggestions()
            return True
        
        self.hitl_info.needs_review = False
        return False
    
    def _set_hitl_suggestions(self):
        """Set HITL suggestions based on input type."""
        if self.input_type == "image":
            self.hitl_info.suggestions = [
                "Review the extracted text for OCR errors",
                "Check numbers: 0 vs O, 1 vs l, 5 vs S",
                "Verify math symbols: ², √, ∫, π, θ",
                "Check fractions and exponents",
            ]
        elif self.input_type == "audio":
            self.hitl_info.suggestions = [
                "Review transcript for speech errors",
                "Check math terms: 'squared', 'root', 'integral'",
                "Verify numbers and variables",
                "Check homophones: two/to, four/for",
            ]
    
    def get_hitl_display(self) -> Dict[str, Any]:
        """Get HITL information for UI display."""
        needs = self.needs_hitl()
        
        # Confidence indicator
        conf = self.confidence_score
        if conf >= 0.8:
            indicator = "🟢"
        elif conf >= 0.5:
            indicator = "🟡"
        else:
            indicator = "🔴"
        
        return {
            "needs_review": needs,
            "input_type": self.input_type,
            "confidence": self.confidence_score,
            "confidence_display": f"{indicator} {conf:.0%}",
            "trigger_reason": self.hitl_info.trigger_reason,
            "suggestions": self.hitl_info.suggestions,
            "was_reviewed": self.hitl_info.was_reviewed,
            "was_edited": self.was_human_edited,
            "original_text": self.original_extraction,
            "current_text": self.extracted_text,
        }
    
    def mark_reviewed(
        self, 
        action: str, 
        edited_text: str = None,
        feedback: str = ""
    ):
        """
        Mark this input as reviewed by human.
        
        Args:
            action: "approved", "edited", "rejected", or "skipped"
            edited_text: New text if action is "edited"
            feedback: Optional feedback from reviewer
        """
        self.hitl_info.was_reviewed = True
        self.hitl_info.review_action = action
        self.hitl_info.review_feedback = feedback
        
        if action == "edited" and edited_text is not None:
            if edited_text != self.extracted_text:
                self.original_extraction = self.extracted_text
                self.extracted_text = edited_text
                self.was_human_edited = True
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "input_id": self.input_id,
            "input_type": self.input_type,
            "extracted_text": self.extracted_text,
            "confidence_score": float(self.confidence_score),
            "was_human_edited": self.was_human_edited,
            "original_extraction": self.original_extraction,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "hitl": {
                "needs_review": self.hitl_info.needs_review,
                "trigger_reason": self.hitl_info.trigger_reason,
                "was_reviewed": self.hitl_info.was_reviewed,
                "review_action": self.hitl_info.review_action,
            }
        }


@dataclass
class ParsedProblem:
    """
    Structured representation of a parsed math problem.
    Output of the Parser Agent with HITL support.
    """
    problem_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Core parsed information
    problem_text: str = ""
    topic: str = ""
    subtopic: str = ""
    
    # Extracted elements
    variables: List[str] = field(default_factory=list)
    constants: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    goal: str = ""
    problem_type: str = ""
    
    # HITL support
    needs_clarification: bool = False
    clarification_reason: str = ""
    ambiguities: List[str] = field(default_factory=list)
    missing_info: List[str] = field(default_factory=list)
    confidence: float = 1.0
    
    def needs_hitl(self, threshold: float = 0.60) -> bool:
        """Check if HITL is needed for this parsed problem."""
        return (
            self.needs_clarification or
            len(self.ambiguities) > 0 or
            len(self.missing_info) > 0 or
            self.confidence < threshold
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "problem_id": self.problem_id,
            "problem_text": self.problem_text,
            "topic": self.topic,
            "subtopic": self.subtopic,
            "variables": self.variables,
            "constraints": self.constraints,
            "goal": self.goal,
            "problem_type": self.problem_type,
            "needs_clarification": self.needs_clarification,
            "clarification_reason": self.clarification_reason,
            "ambiguities": self.ambiguities,
            "missing_info": self.missing_info,
            "confidence": float(self.confidence),
        }


@dataclass
class VerificationResult:
    """
    Result of solution verification with HITL support.
    Output of the Verifier Agent.
    """
    is_correct: bool = True
    confidence: float = 1.0
    
    # Verification details
    verification_steps: List[str] = field(default_factory=list)
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    
    # Issues and warnings
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    edge_cases: List[str] = field(default_factory=list)
    
    def needs_hitl(self, threshold: float = 0.70) -> bool:
        """Check if HITL is needed based on verification result."""
        if not self.is_correct:
            return True
        if self.checks_failed:
            return True
        if self.issues:
            return True
        if self.confidence < threshold:
            return True
        if self.edge_cases and self.confidence < 0.85:
            return True
        return False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "is_correct": self.is_correct,
            "confidence": float(self.confidence),
            "verification_steps": self.verification_steps,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "issues": self.issues,
            "warnings": self.warnings,
            "edge_cases": self.edge_cases,
        }