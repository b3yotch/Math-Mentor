# src/guardrails/output_guardrails.py
"""
Output Guardrails for Math Mentor.
Validates AI-generated responses before showing to users.
"""

import re
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class OutputValidation:
    """Result of output validation."""
    is_valid: bool
    cleaned_output: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    confidence: float = 1.0


class OutputGuardrails:
    """
    Validates and sanitizes AI output.
    """
    
    HALLUCINATION_PATTERNS = [
        r"according to (the )?(textbook|book|source|reference)",
        r"as stated in",
        r"in the (textbook|book|manual|guide)",
        r"page \d+",
        r"chapter \d+",
        r"from (the )?(internet|web|online)",
        r"https?://\S+",
    ]
    
    SOLUTION_COMPONENTS = [
        "step", "answer", "result", "solution",
        "therefore", "thus", "hence", "="
    ]
    
    OFF_TOPIC_PATTERNS = [
        r"(I|i) (cannot|can't|won't|will not) (help|assist) with",
        r"(I|i) (don't|do not) have (access|information)",
        r"as an AI",
        r"I('m| am) (just )?(a|an) (language model|AI|assistant)",
    ]
    
    def __init__(self):
        """Initialize output guardrails."""
        self.hallucination_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.HALLUCINATION_PATTERNS
        ]
        self.off_topic_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.OFF_TOPIC_PATTERNS
        ]
    
    def validate(self, output: Union[str, Dict]) -> OutputValidation:
        """
        Validate AI output.
        
        Args:
            output: String or dictionary with solution, verification, explanation
        
        Returns:
            OutputValidation result
        """
        warnings = []
        errors = []
        confidence = 1.0
        
        # Handle both string and dict inputs
        if isinstance(output, str):
            output_dict = {'solution': output, 'verification': '', 'explanation': ''}
        else:
            output_dict = output
        
        solution = str(output_dict.get('solution', ''))
        verification = str(output_dict.get('verification', ''))
        explanation = str(output_dict.get('explanation', ''))
        
        full_text = f"{solution} {verification} {explanation}"
        
        # Check 1: Empty or too short
        if not solution or len(solution.strip()) < 10:
            errors.append("Solution is empty or too short")
            confidence -= 0.5
        
        # Check 2: Hallucinations
        hallucinations = self._check_hallucinations(full_text)
        if hallucinations:
            warnings.extend(hallucinations)
            confidence -= 0.1 * len(hallucinations)
            solution = self._remove_hallucinations(solution)
            explanation = self._remove_hallucinations(explanation)
        
        # Check 3: Off-topic
        if self._is_off_topic(full_text):
            errors.append("Response appears to be off-topic or a refusal")
            confidence -= 0.3
        
        # Check 4: Solution components
        if not self._has_solution_components(solution):
            warnings.append("Solution may be incomplete")
            confidence -= 0.1
        
        # Check 5: Inappropriate content
        if self._check_inappropriate(full_text):
            errors.append("Response contains inappropriate content")
            confidence = 0.0
        
        # Check 6: Coherence
        if not self._is_coherent(solution):
            warnings.append("Solution may be truncated")
            confidence -= 0.2
        
        confidence = max(0.0, min(1.0, confidence))
        
        cleaned_output = {
            'solution': solution,
            'verification': verification,
            'explanation': explanation
        }
        
        return OutputValidation(
            is_valid=len(errors) == 0,
            cleaned_output=cleaned_output,
            warnings=warnings,
            errors=errors,
            confidence=confidence
        )
    
    def _check_hallucinations(self, text: str) -> List[str]:
        """Check for hallucinated citations."""
        found = []
        for pattern in self.hallucination_patterns:
            if pattern.search(text):
                found.append("Possible hallucinated reference detected")
        return found
    
    def _remove_hallucinations(self, text: str) -> str:
        """Remove hallucinated citations from text."""
        cleaned = text
        for pattern in self.hallucination_patterns:
            cleaned = pattern.sub('', cleaned)
        return cleaned.strip()
    
    def _is_off_topic(self, text: str) -> bool:
        """Check if response is off-topic."""
        for pattern in self.off_topic_patterns:
            if pattern.search(text):
                return True
        return False
    
    def _has_solution_components(self, text: str) -> bool:
        """Check if solution has required components."""
        text_lower = text.lower()
        return any(comp in text_lower for comp in self.SOLUTION_COMPONENTS)
    
    def _check_inappropriate(self, text: str) -> bool:
        """Check for inappropriate content."""
        inappropriate_terms = [
            "kill", "murder", "weapon", "bomb", "drug",
            "porn", "nude", "violence", "hate"
        ]
        text_lower = text.lower()
        return any(term in text_lower for term in inappropriate_terms)
    
    def _is_coherent(self, text: str) -> bool:
        """Check if text appears coherent."""
        if not text:
            return False
        
        truncation_indicators = [
            text.endswith('...') and len(text) > 100,
            text.count('(') != text.count(')'),
            text.count('[') != text.count(']'),
        ]
        
        if any(truncation_indicators):
            return False
        
        return len(text) > 20
    
    def format_for_display(self, output: Union[str, Dict]) -> Union[str, Dict]:
        """
        Format output for safe display.
        
        Args:
            output: String or dictionary output
        
        Returns:
            Formatted output (same type as input)
        """
        # Handle string input
        if isinstance(output, str):
            cleaned = output
            cleaned = re.sub(r'<[^>]+>', '', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            return cleaned
        
        # Handle dictionary input
        formatted = {}
        for key in ['solution', 'verification', 'explanation', 'parsed_problem', 'raw_response']:
            value = output.get(key, '')
            if isinstance(value, str) and value:
                value = re.sub(r'<[^>]+>', '', value)
                value = re.sub(r'\s+', ' ', value).strip()
            formatted[key] = value
        
        return formatted