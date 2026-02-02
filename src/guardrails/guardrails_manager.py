# src/guardrails/guardrails_manager.py
"""
Unified Guardrails Manager for Math Mentor.
"""

from typing import Dict, Optional, Union
from dataclasses import dataclass, field
from .input_guardrails import InputGuardrails, GuardrailResult
from .output_guardrails import OutputGuardrails, OutputValidation
from .content_filter import ContentFilter, SafetyCheck, SafetyLevel


@dataclass
class GuardrailsReport:
    """Complete guardrails report."""
    passed: bool
    input_check: Optional[GuardrailResult] = None
    safety_check: Optional[SafetyCheck] = None
    output_check: Optional[OutputValidation] = None
    blocked_reason: str = ""
    warnings: list = field(default_factory=list)


class GuardrailsManager:
    """
    Unified manager for all guardrails.
    """
    
    def __init__(self):
        """Initialize all guardrails."""
        self.input_guardrails = InputGuardrails()
        self.output_guardrails = OutputGuardrails()
        self.content_filter = ContentFilter()
    
    def check_input(self, text: str) -> GuardrailsReport:
        """
        Run all input checks.
        """
        warnings = []
        
        # Step 1: Sanitize input
        sanitized = self.input_guardrails.sanitize(text)
        
        # Step 2: Content safety check (includes off-topic detection)
        safety_check = self.content_filter.check_input(sanitized)
        if safety_check.level == SafetyLevel.BLOCKED:
            return GuardrailsReport(
                passed=False,
                safety_check=safety_check,
                blocked_reason=safety_check.message
            )
        elif safety_check.level == SafetyLevel.WARNING:
            warnings.append(safety_check.message)
        
        # Step 3: Additional input validation
        input_check = self.input_guardrails.validate(sanitized)
        if not input_check.passed:
            return GuardrailsReport(
                passed=False,
                input_check=input_check,
                blocked_reason=input_check.message
            )
        
        if input_check.severity == "warning":
            warnings.append(input_check.message)
        
        return GuardrailsReport(
            passed=True,
            input_check=input_check,
            safety_check=safety_check,
            warnings=warnings
        )
    
    def check_output(self, output: Union[str, Dict]) -> GuardrailsReport:
        """
        Run all output checks.
        """
        warnings = []
        
        # Handle both string and dict inputs
        if isinstance(output, str):
            output_dict = {'solution': output}
        else:
            output_dict = output
        
        # Combine output text for safety check
        full_text = " ".join([
            str(output_dict.get('solution', '')),
            str(output_dict.get('verification', '')),
            str(output_dict.get('explanation', ''))
        ])
        
        # Step 1: Content safety check
        safety_check = self.content_filter.check_output(full_text)
        if safety_check.level == SafetyLevel.BLOCKED:
            return GuardrailsReport(
                passed=False,
                safety_check=safety_check,
                blocked_reason=safety_check.message
            )
        
        # Step 2: Output validation
        output_check = self.output_guardrails.validate(output_dict)
        if not output_check.is_valid:
            return GuardrailsReport(
                passed=False,
                output_check=output_check,
                blocked_reason="; ".join(output_check.errors)
            )
        
        warnings.extend(output_check.warnings)
        
        return GuardrailsReport(
            passed=True,
            output_check=output_check,
            safety_check=safety_check,
            warnings=warnings
        )
    
    def get_sanitized_input(self, text: str) -> str:
        """Get sanitized version of input."""
        return self.input_guardrails.sanitize(text)
    
    def get_detected_topic(self, text: str) -> str:
        """Get detected math topic from input."""
        return self.input_guardrails.get_detected_topic(text)
    
    def format_output(self, output: Union[str, Dict]) -> Union[str, Dict]:
        """Format output for safe display."""
        return self.output_guardrails.format_for_display(output)