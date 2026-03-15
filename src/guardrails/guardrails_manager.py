# src/guardrails/guardrails_manager.py
"""
Unified Guardrails Manager for Math Mentor - IMPROVED VERSION
Provides a single interface for all input/output validation.
Sanitizes dangerous content (XSS, injection) while preserving math.
"""

import re
import logging
from typing import Dict, Optional, Union, List, Any
from dataclasses import dataclass, field
from enum import Enum

from .input_guardrails import (
    InputGuardrails,
    GuardrailResult,
    ValidationSeverity,
)
from .output_guardrails import OutputGuardrails, OutputValidation
from .content_filter import ContentFilter, SafetyCheck, SafetyLevel

logger = logging.getLogger(__name__)


class CheckStatus(Enum):
    """Overall status of guardrails check."""
    PASSED = "passed"
    WARNING = "warning"
    BLOCKED = "blocked"
    ERROR = "error"


@dataclass
class GuardrailsReport:
    """Complete guardrails report with all validation results."""
    status: CheckStatus
    passed: bool
    message: str = ""

    # Individual check results
    input_check: Optional[GuardrailResult] = None
    safety_check: Optional[SafetyCheck] = None
    output_check: Optional[OutputValidation] = None

    # Aggregated info
    blocked_reason: str = ""
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    @property
    def is_blocked(self) -> bool:
        return self.status == CheckStatus.BLOCKED

    def to_dict(self) -> Dict:
        return {
            "status": self.status.value,
            "passed": self.passed,
            "message": self.message,
            "blocked_reason": self.blocked_reason,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
            "metadata": self.metadata,
        }


class GuardrailsManager:
    """
    Unified manager for all guardrails — IMPROVED VERSION.

    Key improvement: Deep sanitization strips XSS, prompt injection,
    and SQL injection patterns BEFORE the injection checker runs,
    so legitimate math mixed with dangerous content is handled
    gracefully (sanitize → check clean text) instead of blocking outright.
    """

    def __init__(
        self,
        strict_mode: bool = False,
        enable_output_validation: bool = True,
        min_input_length: int = 2,
        max_input_length: int = 2000,
    ):
        self.strict_mode = strict_mode
        self.enable_output_validation = enable_output_validation

        self.input_guardrails = InputGuardrails(
            min_length=min_input_length,
            max_length=max_input_length,
            strict_injection_check=strict_mode,
        )
        self.content_filter = ContentFilter(strict_mode=strict_mode)
        self._output_guardrails = None

    @property
    def output_guardrails(self) -> OutputGuardrails:
        if self._output_guardrails is None:
            self._output_guardrails = OutputGuardrails()
        return self._output_guardrails

    # ============================================================
    # DEEP SANITIZATION — strips dangerous patterns, keeps math
    # ============================================================

    @staticmethod
    def _deep_sanitize(text: str) -> tuple:
        """
        Strip dangerous content patterns while preserving math.

        Returns:
            (cleaned_text, list_of_threats_found)

        This runs AFTER basic sanitize() but BEFORE injection checks,
        so the injection checker only sees clean math text.
        """
        if not text:
            return text, []

        threats_found = []
        original = text

        # ── 1. Null bytes and control characters ──────────────────
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        if cleaned != text:
            threats_found.append("control_characters")
        text = cleaned

        # ── 2. HTML / Script tags (XSS) ──────────────────────────
        # Remove <script>...</script> blocks entirely
        text_before = text
        text = re.sub(
            r'<script[^>]*>.*?</script>',
            '', text, flags=re.IGNORECASE | re.DOTALL
        )
        # Remove any remaining HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        if text != text_before:
            threats_found.append("xss_html_tags")

        # ── 3. JavaScript protocol ───────────────────────────────
        text_before = text
        text = re.sub(r'javascript\s*:', '', text, flags=re.IGNORECASE)
        if text != text_before:
            threats_found.append("javascript_protocol")

        # ── 4. JSON prompt injection ─────────────────────────────
        #    {"role": "system", "content": "..."}
        text_before = text
        text = re.sub(
            r'\{\s*["\']role["\']\s*:\s*["\'](?:system|assistant|user)["\']\s*,\s*["\']content["\']\s*:\s*["\'][^"\']*["\']\s*\}',
            '', text, flags=re.IGNORECASE
        )
        if text != text_before:
            threats_found.append("json_prompt_injection")

        # ── 5. Instruction override phrases ──────────────────────
        text_before = text
        override_patterns = [
            r'(?:ignore|disregard|forget|override)\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|rules?|context)',
            r'you\s+are\s+now\s+(?:a|an|in)\s+(?:different|new)',
            r'(?:new|different)\s+(?:system|role)\s*(?:prompt|instruction)',
            r'(?:pretend|act)\s+(?:as\s+if|like|that)\s+you',
            r'do\s+not\s+follow\s+(?:any|your|the)\s+(?:rules?|instructions?|guidelines?)',
        ]
        for pattern in override_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        if text != text_before:
            threats_found.append("instruction_override")

        # ── 6. SQL injection patterns ────────────────────────────
        #    Keep math = signs, only strip actual SQL keywords
        text_before = text
        sql_patterns = [
            r'\bDROP\s+TABLE\b',
            r'\bDELETE\s+FROM\b',
            r'\bINSERT\s+INTO\b',
            r'\bUNION\s+(?:ALL\s+)?SELECT\b',
            r'\bSELECT\s+\*\s+FROM\b',
            r'\bUPDATE\s+\w+\s+SET\b',
            r'\bALTER\s+TABLE\b',
            r'\bCREATE\s+TABLE\b',
            r'\bEXEC(?:UTE)?\s*\(',
            r';\s*--\s',                    # SQL comment after semicolon
            r'\bOR\s+1\s*=\s*1\b',
            r'\bOR\s+[\'"]?\w+[\'"]?\s*=\s*[\'"]?\w+[\'"]?\s*--',
        ]
        for pattern in sql_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        if text != text_before:
            threats_found.append("sql_injection")

        # ── 7. Common encoding-based evasion ─────────────────────
        text_before = text
        # Remove data: URIs
        text = re.sub(r'data\s*:\s*\w+/\w+\s*;', '', text, flags=re.IGNORECASE)
        # Remove base64 blocks that look like encoded payloads
        text = re.sub(r'base64\s*,\s*[A-Za-z0-9+/=]{20,}', '', text, flags=re.IGNORECASE)
        if text != text_before:
            threats_found.append("encoding_evasion")

        # ── 8. Cleanup after removal ─────────────────────────────
        # Collapse excessive repeated special characters (keep max 2)
        text = re.sub(r'([!?#@$%^&*<>|~`])\1{2,}', r'\1\1', text)
        # Collapse excessive whitespace
        text = re.sub(r'\s{3,}', '  ', text)
        # Strip
        text = text.strip()

        if threats_found:
            logger.info(
                "Deep sanitize removed %d threats: %s (removed %d chars)",
                len(threats_found),
                ", ".join(threats_found),
                len(original) - len(text),
            )

        return text, threats_found

    # ============================================================
    # INPUT VALIDATION
    # ============================================================

    def check_input(self, text: str) -> GuardrailsReport:
        """
        Run all input validation checks.

        Pipeline (updated):
        1. Basic sanitize (from InputGuardrails)
        2. Deep sanitize (strip XSS, injection patterns)
        3. Check length
        4. Check prompt injection (on clean text)
        5. Content safety check

        If deep sanitize removed dangerous content but math remains,
        we proceed with a warning instead of blocking.
        """
        warnings = []
        suggestions = []
        metadata = {"original_length": len(text) if text else 0}

        # Step 1: Basic sanitize
        sanitized = self.input_guardrails.sanitize(text)
        metadata["sanitized_length"] = len(sanitized)

        # Step 2: Deep sanitize — strip XSS, injection, SQL patterns
        deep_cleaned, threats = self._deep_sanitize(sanitized)
        metadata["deep_sanitized_length"] = len(deep_cleaned)
        metadata["threats_removed"] = threats

        if threats:
            warnings.append(
                "Removed potentially unsafe content: {}".format(
                    ", ".join(threats)
                )
            )
            logger.info(
                "Deep sanitize stripped threats from input: %s", threats
            )

        # If deep sanitization removed everything, block
        if not deep_cleaned or not deep_cleaned.strip():
            return GuardrailsReport(
                status=CheckStatus.BLOCKED,
                passed=False,
                message="No valid content remained after security filtering.",
                blocked_reason=(
                    "Input contained only unsafe content with no math problem. "
                    "Please enter a valid mathematics question."
                ),
                suggestions=[
                    "Enter a valid mathematics question",
                    "Example: 'Find the derivative of x³ + 2x'",
                    "Remove any code or special formatting",
                ],
                metadata=metadata,
            )

        # Step 3: Check length (on cleaned text)
        length_result = self.input_guardrails.check_length(deep_cleaned)
        if not length_result.passed:
            return GuardrailsReport(
                status=CheckStatus.BLOCKED,
                passed=False,
                message=length_result.message,
                input_check=length_result,
                blocked_reason=length_result.message,
                suggestions=length_result.suggestions,
                metadata=metadata,
            )

        # Step 4: Check for prompt injection (on DEEP CLEANED text)
        # Since we already stripped injection patterns, this should
        # rarely trigger. If it does, it's a genuine concern.
        injection_result = self.input_guardrails.check_prompt_injection(
            deep_cleaned
        )

        if not injection_result.passed:
            # If threats were already removed AND injection still detected,
            # it's a sophisticated attack — block it
            if threats:
                return GuardrailsReport(
                    status=CheckStatus.BLOCKED,
                    passed=False,
                    message="Input contains persistent injection patterns.",
                    input_check=injection_result,
                    blocked_reason=injection_result.message,
                    suggestions=injection_result.suggestions,
                    metadata={**metadata, **injection_result.metadata},
                )

            # No threats were removed but injection detected —
            # this might be a false positive on math-like text.
            # In non-strict mode, downgrade to warning.
            if not self.strict_mode:
                warnings.append(
                    "Input contains patterns similar to prompt injection, "
                    "but proceeding as it may be valid math."
                )
                logger.warning(
                    "Injection check failed but downgraded to warning "
                    "(non-strict mode): %s",
                    injection_result.message,
                )
            else:
                return GuardrailsReport(
                    status=CheckStatus.BLOCKED,
                    passed=False,
                    message=injection_result.message,
                    input_check=injection_result,
                    blocked_reason=injection_result.message,
                    suggestions=injection_result.suggestions,
                    metadata={**metadata, **injection_result.metadata},
                )

        if (
            injection_result.severity == ValidationSeverity.WARNING
            and injection_result.message not in warnings
        ):
            warnings.append(injection_result.message)

        # Step 5: Content safety check (topic relevance, harmful content)
        safety_check = self.content_filter.check_input(deep_cleaned)

        if safety_check.level == SafetyLevel.BLOCKED:
            return GuardrailsReport(
                status=CheckStatus.BLOCKED,
                passed=False,
                message=safety_check.message,
                safety_check=safety_check,
                blocked_reason=safety_check.message,
                suggestions=(
                    [safety_check.details] if safety_check.details else []
                ),
                metadata={
                    **metadata,
                    "blocked_category": safety_check.category,
                    "confidence": safety_check.confidence,
                },
            )

        if safety_check.level == SafetyLevel.WARNING:
            if safety_check.message not in warnings:
                warnings.append(safety_check.message)

        # Add metadata about math detection
        metadata["math_confidence"] = safety_check.confidence
        metadata["category"] = safety_check.category

        # Determine final status
        if warnings:
            status = CheckStatus.WARNING
            message = "Input accepted with warnings: {}".format(
                "; ".join(warnings)
            )
        else:
            status = CheckStatus.PASSED
            message = "Input validated successfully"

        return GuardrailsReport(
            status=status,
            passed=True,
            message=message,
            input_check=injection_result,
            safety_check=safety_check,
            warnings=warnings,
            suggestions=suggestions,
            metadata=metadata,
        )

    # ============================================================
    # OUTPUT VALIDATION
    # ============================================================

    def check_output(self, output: Union[str, Dict]) -> GuardrailsReport:
        """
        Run all output validation checks.

        Pipeline:
        1. Content safety check
        2. Output structure validation
        """
        if not self.enable_output_validation:
            return GuardrailsReport(
                status=CheckStatus.PASSED,
                passed=True,
                message="Output validation disabled",
            )

        warnings = []
        metadata = {}

        # Normalize output to dict
        if isinstance(output, str):
            output_dict = {"solution": output}
        else:
            output_dict = output

        # Combine all text for safety check
        text_parts = []
        for key in [
            "solution", "verification", "explanation",
            "steps", "answer", "final_answer",
        ]:
            if key in output_dict and output_dict[key]:
                value = output_dict[key]
                if isinstance(value, list):
                    text_parts.extend([str(v) for v in value])
                else:
                    text_parts.append(str(value))

        full_text = " ".join(text_parts)
        metadata["output_length"] = len(full_text)

        # Step 1: Content safety check
        safety_check = self.content_filter.check_output(full_text)

        if safety_check.level == SafetyLevel.BLOCKED:
            return GuardrailsReport(
                status=CheckStatus.BLOCKED,
                passed=False,
                message="Output contains unsafe content",
                safety_check=safety_check,
                blocked_reason=safety_check.message,
                metadata=metadata,
            )

        if safety_check.level == SafetyLevel.WARNING:
            warnings.append(safety_check.message)

        # Step 2: Output structure validation
        try:
            output_check = self.output_guardrails.validate(output_dict)

            if not output_check.is_valid:
                return GuardrailsReport(
                    status=CheckStatus.BLOCKED,
                    passed=False,
                    message="Output validation failed",
                    output_check=output_check,
                    safety_check=safety_check,
                    blocked_reason="; ".join(output_check.errors),
                    metadata=metadata,
                )

            warnings.extend(output_check.warnings)

        except Exception as e:
            warnings.append("Output validation error: {}".format(str(e)))
            output_check = None

        # Determine final status
        if warnings:
            status = CheckStatus.WARNING
            message = "Output accepted with warnings"
        else:
            status = CheckStatus.PASSED
            message = "Output validated successfully"

        return GuardrailsReport(
            status=status,
            passed=True,
            message=message,
            output_check=output_check,
            safety_check=safety_check,
            warnings=warnings,
            metadata=metadata,
        )

    # ============================================================
    # FULL PIPELINE
    # ============================================================

    def validate_full_pipeline(
        self,
        user_input: str,
        ai_output: Optional[Union[str, Dict]] = None,
    ) -> Dict[str, GuardrailsReport]:
        """Validate both input and output in one call."""
        results = {"input": self.check_input(user_input)}

        if ai_output is not None and results["input"].passed:
            results["output"] = self.check_output(ai_output)

        return results

    # ============================================================
    # CONVENIENCE METHODS
    # ============================================================

    def sanitize(self, text: str) -> str:
        """Sanitize input text (basic + deep)."""
        basic = self.input_guardrails.sanitize(text)
        deep, _ = self._deep_sanitize(basic)
        return deep

    def is_safe_input(self, text: str) -> bool:
        """Quick check if input is safe."""
        return self.check_input(text).passed

    def is_safe_output(self, output: Union[str, Dict]) -> bool:
        """Quick check if output is safe."""
        return self.check_output(output).passed

    def get_math_confidence(self, text: str) -> float:
        """Get math relevance confidence score for input."""
        sanitized = self.sanitize(text)
        safety_check = self.content_filter.check_input(sanitized)
        return safety_check.confidence

    def get_rejection_message(self, report: GuardrailsReport) -> str:
        """Get user-friendly rejection message."""
        if report.passed:
            return ""

        if report.blocked_reason:
            return report.blocked_reason

        if (
            report.safety_check
            and report.safety_check.level == SafetyLevel.BLOCKED
        ):
            base_msg = report.safety_check.message
            if report.safety_check.details:
                return "{} {}".format(base_msg, report.safety_check.details)
            return base_msg

        if report.input_check and not report.input_check.passed:
            return report.input_check.message

        return (
            "Input could not be processed. "
            "Please try a different math question."
        )

    def get_suggestions(self, report: GuardrailsReport) -> List[str]:
        """Get suggestions for improving input."""
        suggestions = list(report.suggestions)

        if report.input_check and report.input_check.suggestions:
            suggestions.extend(report.input_check.suggestions)

        if report.safety_check and report.safety_check.details:
            if report.safety_check.details not in suggestions:
                suggestions.append(report.safety_check.details)

        if not suggestions and not report.passed:
            suggestions = [
                "Enter a valid mathematics question",
                "Example: 'Solve x² + 5x + 6 = 0'",
                "Topics: Algebra, Calculus, Probability, Statistics, Trigonometry",
            ]

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for s in suggestions:
            if s and s not in seen:
                seen.add(s)
                unique.append(s)

        return unique

    def format_output_for_display(
        self,
        output: Union[str, Dict],
        include_metadata: bool = False,
    ) -> Dict:
        """Format and sanitize output for safe display."""
        if isinstance(output, str):
            output_dict = {"solution": output}
        else:
            output_dict = dict(output)

        for key in output_dict:
            if isinstance(output_dict[key], str):
                output_dict[key] = (
                    output_dict[key]
                    .replace("<script", "&lt;script")
                    .replace("</script>", "&lt;/script&gt;")
                )

        if include_metadata:
            validation = self.check_output(output)
            output_dict["_validation"] = {
                "passed": validation.passed,
                "warnings": validation.warnings,
            }

        return output_dict


# ============================================================
# FACTORY FUNCTION
# ============================================================

def create_guardrails(
    strict: bool = False,
    validate_outputs: bool = True,
) -> GuardrailsManager:
    """Factory function to create configured GuardrailsManager."""
    return GuardrailsManager(
        strict_mode=strict,
        enable_output_validation=validate_outputs,
    )