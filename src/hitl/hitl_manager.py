# src/hitl/hitl_manager.py
"""
Human-in-the-Loop (HITL) Manager for Math Mentor.

Centralized system for managing all HITL workflows:
1. OCR confidence low → show extracted text for review
2. ASR confidence low → show transcript for review  
3. Parser detects ambiguity → ask for clarification
4. Verifier not confident → request human verification
5. User requests re-check → allow manual trigger
6. Guardrail warning → confirm with user

Each HITL interaction is logged for learning.
"""

import uuid
import json
import logging
from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class HITLTrigger(Enum):
    """Reasons why HITL was triggered."""
    OCR_LOW_CONFIDENCE = "ocr_low_confidence"
    ASR_LOW_CONFIDENCE = "asr_low_confidence"
    EXTRACTION_FAILED = "extraction_failed"
    PARSER_AMBIGUITY = "parser_ambiguity"
    PARSER_MISSING_INFO = "parser_missing_info"
    PARSER_LOW_CONFIDENCE = "parser_low_confidence"
    VERIFIER_LOW_CONFIDENCE = "verifier_low_confidence"
    VERIFIER_INCORRECT = "verifier_incorrect"
    VERIFIER_EDGE_CASE = "verifier_edge_case"
    USER_REQUESTED = "user_requested"
    GUARDRAIL_WARNING = "guardrail_warning"
    OUTPUT_WARNING = "output_warning"


class HITLStatus(Enum):
    """Status of HITL request."""
    PENDING = "pending"
    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class HITLStage(Enum):
    """Stage where HITL was triggered."""
    EXTRACTION = "extraction"
    PARSING = "parsing"
    SOLVING = "solving"
    VERIFICATION = "verification"
    OUTPUT = "output"


@dataclass
class HITLRequest:
    """
    A request for human review.
    
    Contains all information needed for the human to make a decision.
    """
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trigger: HITLTrigger = HITLTrigger.USER_REQUESTED
    stage: HITLStage = HITLStage.EXTRACTION
    
    # What needs review
    original_content: str = ""
    suggested_content: str = ""
    
    # Context for decision
    confidence_score: float = 0.0
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    
    # UI configuration
    allow_edit: bool = True
    allow_reject: bool = True
    show_diff: bool = False
    
    # Status tracking
    status: HITLStatus = HITLStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    
    # Resolution
    resolution: Optional['HITLResolution'] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "request_id": self.request_id,
            "trigger": self.trigger.value,
            "stage": self.stage.value,
            "original_content": self.original_content,
            "suggested_content": self.suggested_content,
            "confidence_score": float(self.confidence_score),
            "reason": self.reason,
            "details": self.details,
            "suggestions": self.suggestions,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


@dataclass
class HITLResolution:
    """Human's resolution of a HITL request."""
    action: HITLStatus
    final_content: str = ""
    human_feedback: str = ""
    corrections_made: Dict[str, str] = field(default_factory=dict)
    time_spent_seconds: float = 0.0
    resolved_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "action": self.action.value,
            "final_content": self.final_content,
            "human_feedback": self.human_feedback,
            "corrections_made": self.corrections_made,
            "time_spent_seconds": self.time_spent_seconds,
            "resolved_at": self.resolved_at.isoformat(),
        }


@dataclass
class HITLConfig:
    """Configuration for HITL thresholds and behavior."""
    
    # Extraction confidence thresholds
    OCR_CONFIDENCE_THRESHOLD: float = 0.70
    ASR_CONFIDENCE_THRESHOLD: float = 0.65
    
    # Parser thresholds
    PARSER_CONFIDENCE_THRESHOLD: float = 0.60
    
    # Verifier thresholds
    VERIFIER_CONFIDENCE_THRESHOLD: float = 0.70
    VERIFIER_EDGE_CASE_THRESHOLD: float = 0.85
    
    # Auto-actions
    AUTO_APPROVE_THRESHOLD: float = 0.95
    AUTO_SKIP_AFTER_SECONDS: float = 300.0  # 5 minutes
    
    # Guardrail integration
    TRIGGER_ON_GUARDRAIL_WARNING: bool = True
    
    def to_dict(self) -> Dict:
        return {
            "ocr_threshold": self.OCR_CONFIDENCE_THRESHOLD,
            "asr_threshold": self.ASR_CONFIDENCE_THRESHOLD,
            "parser_threshold": self.PARSER_CONFIDENCE_THRESHOLD,
            "verifier_threshold": self.VERIFIER_CONFIDENCE_THRESHOLD,
        }


class HITLManager:
    """
    Central manager for all HITL workflows.
    
    Responsibilities:
    1. Evaluate if HITL is needed at each stage
    2. Create HITL requests with context
    3. Process human resolutions
    4. Store learning signals from corrections
    5. Track HITL statistics
    """
    
    def __init__(
        self, 
        memory_manager=None, 
        config: HITLConfig = None,
        guardrails_manager=None
    ):
        """
        Initialize HITL Manager.
        
        Args:
            memory_manager: Optional MemoryManager for storing learning signals
            config: Optional HITLConfig for custom thresholds
            guardrails_manager: Optional GuardrailsManager for validation
        """
        self.config = config or HITLConfig()
        self._memory_manager = memory_manager
        self._guardrails_manager = guardrails_manager
        
        # Track pending requests (in-memory for session)
        self.pending_requests: Dict[str, HITLRequest] = {}
        
        # Track resolved requests for session statistics
        self.resolved_requests: List[HITLRequest] = []
        
        # Learning signal buffer
        self.learning_signals: List[Dict] = []
        
        logger.info("HITL Manager initialized")
    
    @property
    def memory_manager(self):
        """Lazy load memory manager."""
        if self._memory_manager is None:
            try:
                from src.memory.memory_manager import MemoryManager
                self._memory_manager = MemoryManager()
            except Exception as e:
                logger.warning(f"Could not load MemoryManager: {e}")
        return self._memory_manager
    
    @property
    def guardrails_manager(self):
        """Lazy load guardrails manager."""
        if self._guardrails_manager is None:
            try:
                from src.guardrails.guardrails_manager import GuardrailsManager
                self._guardrails_manager = GuardrailsManager()
            except Exception as e:
                logger.warning(f"Could not load GuardrailsManager: {e}")
        return self._guardrails_manager
    
    # ==================== EXTRACTION HITL ====================
    
    def check_extraction_hitl(
        self,
        input_type: str,
        extracted_text: str,
        confidence: float,
        metadata: Dict = None
    ) -> Optional[HITLRequest]:
        """
        Check if HITL is needed for OCR/ASR extraction.
        
        Args:
            input_type: "text", "image", or "audio"
            extracted_text: Extracted text
            confidence: Confidence score (0-1)
            metadata: Additional extraction metadata
            
        Returns:
            HITLRequest if review needed, None otherwise
        """
        metadata = metadata or {}
        
        # Text input never needs extraction HITL
        if input_type == "text":
            return None
        
        # Check for extraction error
        if metadata.get("error"):
            return self._create_extraction_request(
                input_type=input_type,
                extracted_text=extracted_text or "",
                confidence=0.0,
                trigger=HITLTrigger.EXTRACTION_FAILED,
                reason=f"Extraction failed: {metadata['error']}",
                metadata=metadata
            )
        
        # Check for empty or very short extraction
        if not extracted_text or len(extracted_text.strip()) < 3:
            return self._create_extraction_request(
                input_type=input_type,
                extracted_text=extracted_text or "",
                confidence=0.0,
                trigger=HITLTrigger.EXTRACTION_FAILED,
                reason="Extraction returned empty or very short text",
                metadata=metadata
            )
        
        # Determine threshold based on input type
        if input_type == "image":
            threshold = self.config.OCR_CONFIDENCE_THRESHOLD
            trigger = HITLTrigger.OCR_LOW_CONFIDENCE
        else:  # audio
            threshold = self.config.ASR_CONFIDENCE_THRESHOLD
            trigger = HITLTrigger.ASR_LOW_CONFIDENCE
        
        # Check confidence
        if confidence >= threshold:
            # High confidence - no HITL needed
            if confidence >= self.config.AUTO_APPROVE_THRESHOLD:
                logger.debug(f"Auto-approved extraction (confidence: {confidence:.0%})")
            return None
        
        # Create HITL request
        return self._create_extraction_request(
            input_type=input_type,
            extracted_text=extracted_text,
            confidence=confidence,
            trigger=trigger,
            reason=f"Confidence ({confidence:.0%}) below threshold ({threshold:.0%})",
            metadata=metadata
        )
    
    def _create_extraction_request(
        self,
        input_type: str,
        extracted_text: str,
        confidence: float,
        trigger: HITLTrigger,
        reason: str,
        metadata: Dict
    ) -> HITLRequest:
        """Create HITL request for extraction review."""
        
        if input_type == "image":
            suggestions = [
                "Review the extracted text for OCR errors",
                "Check for misread numbers (0 vs O, 1 vs l, 5 vs S)",
                "Verify mathematical symbols (², √, ∫, π, etc.)",
                "Check fractions and exponents",
                "Edit any incorrectly recognized characters",
            ]
        else:  # audio
            suggestions = [
                "Review the transcript for speech recognition errors",
                "Check math terms: 'squared', 'cubed', 'root', 'integral'",
                "Verify numbers and variable names (x, y, z)",
                "Check for homophones: 'two/to/too', 'four/for'",
                "Edit any misheard mathematical expressions",
            ]
        
        request = HITLRequest(
            trigger=trigger,
            stage=HITLStage.EXTRACTION,
            original_content=extracted_text,
            suggested_content=extracted_text,
            confidence_score=confidence,
            reason=reason,
            details={
                "input_type": input_type,
                "threshold": self.config.OCR_CONFIDENCE_THRESHOLD if input_type == "image" else self.config.ASR_CONFIDENCE_THRESHOLD,
                **metadata
            },
            suggestions=suggestions,
            allow_edit=True,
            allow_reject=False,  # Don't reject extractions, just edit
        )
        
        self.pending_requests[request.request_id] = request
        logger.info(f"HITL triggered: {trigger.value} (confidence: {confidence:.0%})")
        
        return request
    
    # ==================== PARSER HITL ====================
    
    def check_parser_hitl(
        self,
        parsed_result: Dict,
        original_text: str
    ) -> Optional[HITLRequest]:
        """
        Check if HITL is needed based on parser output.
        
        Expected parsed_result format:
        {
            "problem_text": "cleaned problem",
            "topic": "algebra",
            "variables": ["x", "y"],
            "constraints": ["x > 0"],
            "goal": "find x",
            "needs_clarification": false,
            "clarification_reason": "",
            "ambiguities": [],
            "missing_info": [],
            "confidence": 0.9
        }
        """
        needs_clarification = parsed_result.get("needs_clarification", False)
        clarification_reason = parsed_result.get("clarification_reason", "")
        ambiguities = parsed_result.get("ambiguities", [])
        missing_info = parsed_result.get("missing_info", [])
        confidence = parsed_result.get("confidence", 1.0)
        
        # Determine if HITL needed
        if needs_clarification:
            trigger = HITLTrigger.PARSER_AMBIGUITY
        elif missing_info:
            trigger = HITLTrigger.PARSER_MISSING_INFO
        elif ambiguities:
            trigger = HITLTrigger.PARSER_AMBIGUITY
        elif confidence < self.config.PARSER_CONFIDENCE_THRESHOLD:
            trigger = HITLTrigger.PARSER_LOW_CONFIDENCE
        else:
            return None  # No HITL needed
        
        # Build reason
        reasons = []
        if clarification_reason:
            reasons.append(clarification_reason)
        if ambiguities:
            reasons.append(f"Ambiguous: {', '.join(ambiguities[:3])}")
        if missing_info:
            reasons.append(f"Missing: {', '.join(missing_info[:3])}")
        if confidence < self.config.PARSER_CONFIDENCE_THRESHOLD:
            reasons.append(f"Low confidence ({confidence:.0%})")
        
        reason = "; ".join(reasons) if reasons else "Parser needs clarification"
        
        # Build suggestions
        suggestions = ["Review the problem statement for clarity"]
        
        if missing_info:
            for info in missing_info[:3]:
                suggestions.append(f"Please provide: {info}")
        
        if ambiguities:
            for amb in ambiguities[:3]:
                suggestions.append(f"Please clarify: {amb}")
        
        suggestions.append("Edit the problem text if needed")
        
        request = HITLRequest(
            trigger=trigger,
            stage=HITLStage.PARSING,
            original_content=original_text,
            suggested_content=parsed_result.get("problem_text", original_text),
            confidence_score=confidence,
            reason=reason,
            details={
                "parsed_result": parsed_result,
                "ambiguities": ambiguities,
                "missing_info": missing_info,
                "topic": parsed_result.get("topic", "unknown"),
            },
            suggestions=suggestions,
            allow_edit=True,
            allow_reject=True,
        )
        
        self.pending_requests[request.request_id] = request
        logger.info(f"HITL triggered: {trigger.value} - {reason}")
        
        return request
    
    # ==================== VERIFIER HITL ====================
    
    def check_verifier_hitl(
        self,
        verification_result: Dict,
        solution: str,
        problem_text: str
    ) -> Optional[HITLRequest]:
        """
        Check if HITL is needed based on verifier output.
        
        Expected verification_result format:
        {
            "is_correct": true,
            "confidence": 0.85,
            "verification_steps": ["Step 1 checked", "Step 2 checked"],
            "issues": [],
            "warnings": [],
            "edge_cases": []
        }
        """
        is_correct = verification_result.get("is_correct", True)
        confidence = verification_result.get("confidence", 1.0)
        issues = verification_result.get("issues", [])
        warnings = verification_result.get("warnings", [])
        edge_cases = verification_result.get("edge_cases", [])
        checks_failed = verification_result.get("checks_failed", [])
        
        # Determine if HITL needed and trigger type
        needs_hitl = False
        trigger = HITLTrigger.VERIFIER_LOW_CONFIDENCE
        
        if not is_correct:
            needs_hitl = True
            trigger = HITLTrigger.VERIFIER_INCORRECT
        elif checks_failed:
            needs_hitl = True
            trigger = HITLTrigger.VERIFIER_INCORRECT
        elif issues:
            needs_hitl = True
            trigger = HITLTrigger.VERIFIER_LOW_CONFIDENCE
        elif confidence < self.config.VERIFIER_CONFIDENCE_THRESHOLD:
            needs_hitl = True
            trigger = HITLTrigger.VERIFIER_LOW_CONFIDENCE
        elif edge_cases and confidence < self.config.VERIFIER_EDGE_CASE_THRESHOLD:
            needs_hitl = True
            trigger = HITLTrigger.VERIFIER_EDGE_CASE
        
        if not needs_hitl:
            return None
        
        # Build reason
        reasons = []
        if not is_correct:
            reasons.append("Verifier detected errors in the solution")
        if checks_failed:
            reasons.append(f"Failed checks: {', '.join(checks_failed[:3])}")
        if issues:
            reasons.append(f"Issues: {', '.join(issues[:3])}")
        if confidence < self.config.VERIFIER_CONFIDENCE_THRESHOLD:
            reasons.append(f"Low confidence ({confidence:.0%})")
        if edge_cases:
            reasons.append(f"Edge cases: {', '.join(edge_cases[:2])}")
        
        reason = "; ".join(reasons) if reasons else "Verification uncertain"
        
        # Build suggestions
        suggestions = [
            "Review the solution for correctness",
            "Verify the final answer",
            "Check the mathematical steps",
        ]
        
        if checks_failed:
            for check in checks_failed[:2]:
                suggestions.append(f"Failed: {check}")
        
        if issues:
            for issue in issues[:2]:
                suggestions.append(f"Check: {issue}")
        
        if edge_cases:
            suggestions.append(f"Consider edge cases: {', '.join(edge_cases[:2])}")
        
        request = HITLRequest(
            trigger=trigger,
            stage=HITLStage.VERIFICATION,
            original_content=solution,
            suggested_content=solution,
            confidence_score=confidence,
            reason=reason,
            details={
                "verification_result": verification_result,
                "problem_text": problem_text,
                "is_correct": is_correct,
                "issues": issues,
                "edge_cases": edge_cases,
                "checks_failed": checks_failed,
            },
            suggestions=suggestions,
            allow_edit=False,  # Can't edit solution here, just approve/reject
            allow_reject=True,
        )
        
        self.pending_requests[request.request_id] = request
        logger.info(f"HITL triggered: {trigger.value} - {reason}")
        
        return request
    
    # ==================== GUARDRAIL HITL ====================
    
    def check_guardrail_hitl(
        self,
        guardrail_report: Any,  # GuardrailsReport
        content: str,
        stage: str = "input"
    ) -> Optional[HITLRequest]:
        """
        Check if HITL is needed based on guardrail warnings.
        
        Only triggers HITL for warnings, not blocks.
        """
        if not self.config.TRIGGER_ON_GUARDRAIL_WARNING:
            return None
        
        # Import here to avoid circular imports
        from src.guardrails.guardrails_manager import CheckStatus
        
        if guardrail_report.status != CheckStatus.WARNING:
            return None
        
        if not guardrail_report.warnings:
            return None
        
        reason = "; ".join(guardrail_report.warnings[:3])
        confidence = guardrail_report.metadata.get("math_confidence", 0.5)
        
        suggestions = guardrail_report.suggestions or [
            "Review the content",
            "Make sure it's a valid math question",
        ]
        
        request = HITLRequest(
            trigger=HITLTrigger.GUARDRAIL_WARNING,
            stage=HITLStage.EXTRACTION if stage == "input" else HITLStage.OUTPUT,
            original_content=content,
            suggested_content=content,
            confidence_score=confidence,
            reason=f"Guardrail warning: {reason}",
            details={
                "warnings": guardrail_report.warnings,
                "metadata": guardrail_report.metadata,
            },
            suggestions=suggestions,
            allow_edit=True,
            allow_reject=True,
        )
        
        self.pending_requests[request.request_id] = request
        logger.info(f"HITL triggered: guardrail_warning - {reason}")
        
        return request
    
    # ==================== USER-REQUESTED HITL ====================
    
    def create_user_recheck_request(
        self,
        content: str,
        stage: str,
        reason: str = "User requested verification"
    ) -> HITLRequest:
        """
        Create HITL request when user explicitly requests re-check.
        
        Args:
            content: Content to review
            stage: Which stage ("extraction", "solution", "verification")
            reason: Why user requested re-check
        """
        stage_map = {
            "extraction": HITLStage.EXTRACTION,
            "parsing": HITLStage.PARSING,
            "solving": HITLStage.SOLVING,
            "solution": HITLStage.SOLVING,
            "verification": HITLStage.VERIFICATION,
            "output": HITLStage.OUTPUT,
        }
        
        request = HITLRequest(
            trigger=HITLTrigger.USER_REQUESTED,
            stage=stage_map.get(stage, HITLStage.SOLVING),
            original_content=content,
            suggested_content=content,
            confidence_score=0.5,  # Unknown confidence
            reason=reason,
            suggestions=[
                "Review the content carefully",
                "Click 'Approve' if correct",
                "Click 'Has Issues' to report problems",
                "Add feedback in the text box",
            ],
            allow_edit=False,
            allow_reject=True,
        )
        
        self.pending_requests[request.request_id] = request
        logger.info(f"HITL triggered: user_requested for {stage}")
        
        return request
    
    # ==================== RESOLUTION ====================
    
    def resolve_request(
        self,
        request_id: str,
        action: HITLStatus,
        final_content: str = "",
        human_feedback: str = "",
        corrections: Dict[str, str] = None,
        time_spent: float = 0.0
    ) -> HITLResolution:
        """
        Resolve a pending HITL request.
        
        Args:
            request_id: ID of the request to resolve
            action: Resolution action (approved, edited, rejected)
            final_content: Final content after human review
            human_feedback: Optional feedback from human
            corrections: Dictionary of original -> corrected mappings
            time_spent: Time spent on review in seconds
        """
        corrections = corrections or {}
        
        # Get the request
        request = self.pending_requests.get(request_id)
        if not request:
            logger.warning(f"HITL request not found: {request_id}")
            return HITLResolution(action=action, final_content=final_content)
        
        # Create resolution
        resolution = HITLResolution(
            action=action,
            final_content=final_content if final_content else request.suggested_content,
            human_feedback=human_feedback,
            corrections_made=corrections,
            time_spent_seconds=time_spent,
        )
        
        # Update request
        request.status = action
        request.resolved_at = datetime.now()
        request.resolution = resolution
        
        # Move to resolved
        self.resolved_requests.append(request)
        del self.pending_requests[request_id]
        
        # Store learning signal
        self._store_learning_signal(request, resolution)
        
        logger.info(f"HITL resolved: {request_id} -> {action.value}")
        
        return resolution
    
    def approve_request(
        self, 
        request_id: str, 
        feedback: str = ""
    ) -> HITLResolution:
        """Approve a HITL request without changes."""
        request = self.pending_requests.get(request_id)
        content = request.suggested_content if request else ""
        return self.resolve_request(
            request_id,
            HITLStatus.APPROVED,
            final_content=content,
            human_feedback=feedback
        )
    
    def edit_request(
        self,
        request_id: str,
        edited_content: str,
        feedback: str = ""
    ) -> HITLResolution:
        """Edit and approve a HITL request."""
        request = self.pending_requests.get(request_id)
        corrections = {}
        if request and edited_content != request.original_content:
            corrections = {request.original_content: edited_content}
        
        return self.resolve_request(
            request_id,
            HITLStatus.EDITED,
            final_content=edited_content,
            human_feedback=feedback,
            corrections=corrections
        )
    
    def reject_request(
        self, 
        request_id: str, 
        reason: str = ""
    ) -> HITLResolution:
        """Reject a HITL request."""
        return self.resolve_request(
            request_id,
            HITLStatus.REJECTED,
            human_feedback=reason
        )
    
    def skip_request(self, request_id: str) -> HITLResolution:
        """Skip HITL review (use suggested content)."""
        request = self.pending_requests.get(request_id)
        content = request.suggested_content if request else ""
        return self.resolve_request(
            request_id,
            HITLStatus.SKIPPED,
            final_content=content
        )
    
    # ==================== LEARNING ====================
    
    def _store_learning_signal(self, request: HITLRequest, resolution: HITLResolution):
        """Store learning signal from HITL resolution."""
        signal = {
            "timestamp": datetime.now().isoformat(),
            "trigger": request.trigger.value,
            "stage": request.stage.value,
            "action": resolution.action.value,
            "original": request.original_content[:500],  # Limit size
            "final": resolution.final_content[:500],
            "was_edited": resolution.action == HITLStatus.EDITED,
            "confidence_before": request.confidence_score,
            "feedback": resolution.human_feedback,
        }
        
        self.learning_signals.append(signal)
        
        # Store extraction corrections in memory
        if self.memory_manager and resolution.action == HITLStatus.EDITED:
            if request.stage == HITLStage.EXTRACTION:
                input_type = request.details.get("input_type", "image")
                if input_type in ["image", "audio"]:
                    try:
                        self.memory_manager.learn_extraction_correction(
                            input_type=input_type,
                            original_text=request.original_content,
                            corrected_text=resolution.final_content
                        )
                        logger.info(f"Stored extraction correction for {input_type}")
                    except Exception as e:
                        logger.warning(f"Could not store correction: {e}")
    
    # ==================== UTILITIES ====================
    
    def get_pending_count(self) -> int:
        """Get number of pending HITL requests."""
        return len(self.pending_requests)
    
    def get_pending_requests(self) -> List[HITLRequest]:
        """Get all pending HITL requests."""
        return list(self.pending_requests.values())
    
    def has_pending(self, request_id: str) -> bool:
        """Check if a request is pending."""
        return request_id in self.pending_requests
    
    def get_request(self, request_id: str) -> Optional[HITLRequest]:
        """Get a specific request."""
        return self.pending_requests.get(request_id)
    
    def clear_pending(self):
        """Clear all pending requests (use with caution)."""
        count = len(self.pending_requests)
        self.pending_requests.clear()
        logger.info(f"Cleared {count} pending HITL requests")
    
    def get_statistics(self) -> Dict:
        """Get HITL statistics for current session."""
        total = len(self.resolved_requests)
        
        if total == 0:
            return {
                "total": 0,
                "pending": len(self.pending_requests),
                "by_action": {},
                "by_trigger": {},
                "by_stage": {},
                "edit_rate": 0.0,
                "approval_rate": 0.0,
                "rejection_rate": 0.0,
                "learning_signals": 0,
            }
        
        # Count by action
        by_action = {}
        for req in self.resolved_requests:
            action = req.status.value
            by_action[action] = by_action.get(action, 0) + 1
        
        # Count by trigger
        by_trigger = {}
        for req in self.resolved_requests:
            trigger = req.trigger.value
            by_trigger[trigger] = by_trigger.get(trigger, 0) + 1
        
        # Count by stage
        by_stage = {}
        for req in self.resolved_requests:
            stage = req.stage.value
            by_stage[stage] = by_stage.get(stage, 0) + 1
        
        # Calculate rates
        edited = by_action.get("edited", 0)
        approved = by_action.get("approved", 0)
        rejected = by_action.get("rejected", 0)
        
        return {
            "total": total,
            "pending": len(self.pending_requests),
            "by_action": by_action,
            "by_trigger": by_trigger,
            "by_stage": by_stage,
            "edit_rate": edited / total if total > 0 else 0.0,
            "approval_rate": approved / total if total > 0 else 0.0,
            "rejection_rate": rejected / total if total > 0 else 0.0,
            "learning_signals": len(self.learning_signals),
        }
    
    def get_display_info(self, request: HITLRequest) -> Dict:
        """Get display-friendly information for HITL UI."""
        trigger_icons = {
            HITLTrigger.OCR_LOW_CONFIDENCE: "📷",
            HITLTrigger.ASR_LOW_CONFIDENCE: "🎤",
            HITLTrigger.EXTRACTION_FAILED: "❌",
            HITLTrigger.PARSER_AMBIGUITY: "🔍",
            HITLTrigger.PARSER_MISSING_INFO: "❓",
            HITLTrigger.PARSER_LOW_CONFIDENCE: "📝",
            HITLTrigger.VERIFIER_LOW_CONFIDENCE: "⚠️",
            HITLTrigger.VERIFIER_INCORRECT: "❌",
            HITLTrigger.VERIFIER_EDGE_CASE: "🔶",
            HITLTrigger.USER_REQUESTED: "👤",
            HITLTrigger.GUARDRAIL_WARNING: "🛡️",
            HITLTrigger.OUTPUT_WARNING: "📤",
        }
        
        trigger_titles = {
            HITLTrigger.OCR_LOW_CONFIDENCE: "OCR Needs Verification",
            HITLTrigger.ASR_LOW_CONFIDENCE: "Transcript Needs Verification",
            HITLTrigger.EXTRACTION_FAILED: "Extraction Failed",
            HITLTrigger.PARSER_AMBIGUITY: "Problem May Be Ambiguous",
            HITLTrigger.PARSER_MISSING_INFO: "Missing Information",
            HITLTrigger.PARSER_LOW_CONFIDENCE: "Parser Uncertain",
            HITLTrigger.VERIFIER_LOW_CONFIDENCE: "Solution Needs Verification",
            HITLTrigger.VERIFIER_INCORRECT: "Solution May Be Incorrect",
            HITLTrigger.VERIFIER_EDGE_CASE: "Edge Case Detected",
            HITLTrigger.USER_REQUESTED: "Manual Review Requested",
            HITLTrigger.GUARDRAIL_WARNING: "Guardrail Warning",
            HITLTrigger.OUTPUT_WARNING: "Output Warning",
        }
        
        # Confidence indicator
        conf = request.confidence_score
        if conf >= 0.8:
            conf_indicator = "🟢"
            conf_label = "High"
        elif conf >= 0.5:
            conf_indicator = "🟡"
            conf_label = "Medium"
        else:
            conf_indicator = "🔴"
            conf_label = "Low"
        
        icon = trigger_icons.get(request.trigger, "❓")
        title = trigger_titles.get(request.trigger, "Review Needed")
        
        return {
            "request_id": request.request_id,
            "icon": icon,
            "title": f"{icon} {title}",
            "reason": request.reason,
            "confidence": f"{conf_indicator} {conf:.0%} ({conf_label})",
            "confidence_score": conf,
            "stage": request.stage.value,
            "trigger": request.trigger.value,
            "suggestions": request.suggestions,
            "content": request.suggested_content,
            "original_content": request.original_content,
            "allow_edit": request.allow_edit,
            "allow_reject": request.allow_reject,
            "details": request.details,
        }


# ==================== SINGLETON ====================

_hitl_manager: Optional[HITLManager] = None


def get_hitl_manager(
    memory_manager=None, 
    config: HITLConfig = None
) -> HITLManager:
    """Get or create HITL manager singleton."""
    global _hitl_manager
    
    if _hitl_manager is None:
        _hitl_manager = HITLManager(
            memory_manager=memory_manager,
            config=config
        )
    
    return _hitl_manager


def reset_hitl_manager():
    """Reset the HITL manager singleton (for testing)."""
    global _hitl_manager
    _hitl_manager = None