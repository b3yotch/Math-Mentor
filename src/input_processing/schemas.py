# src/input_processing/schemas.py
from dataclasses import dataclass, field
from typing import Literal, Optional
from datetime import datetime
import uuid

@dataclass
class CanonicalInput:
    input_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_type: Literal["text", "image", "audio"] = "text"
    raw_file_path: Optional[str] = None
    extracted_text: str = ""
    confidence_score: float = 1.0
    was_human_edited: bool = False
    original_extraction: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)
    
    def needs_hitl(self) -> bool:
        """
        Determine if Human-in-the-Loop review is needed.
        
        Rules:
        - Text input: NEVER needs HITL (user typed it directly)
        - Image input: HITL if confidence < 0.70
        - Audio input: HITL if confidence < 0.65
        """
        # Text input never needs HITL - user typed it directly
        if self.input_type == "text":
            return False
        
        # Define thresholds for other input types
        thresholds = {
            "image": 0.70,
            "audio": 0.65
        }
        
        threshold = thresholds.get(self.input_type, 0.70)
        return self.confidence_score < threshold
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "input_id": self.input_id,
            "input_type": self.input_type,
            "extracted_text": self.extracted_text,
            "confidence_score": self.confidence_score,
            "was_human_edited": self.was_human_edited,
            "timestamp": self.timestamp.isoformat()
        }