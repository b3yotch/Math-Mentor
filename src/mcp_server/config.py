"""
MCP Server configuration.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class MCPConfig:
    """Configuration for the MCP server."""
    server_name: str = "math-mentor"
    server_version: str = "1.0.0"
    description: str = "AI-Powered JEE Math Tutor with RAG, HITL, and Guardrails"
    
    # Feature toggles
    enable_rag: bool = True
    enable_evaluation: bool = True
    enable_memory: bool = True
    enable_guardrails: bool = True
    
    # Defaults
    default_top_k: int = 3
    default_rerank: bool = True
    max_history: int = 50
    
    # Supported topics
    supported_topics: List[str] = field(default_factory=lambda: [
        "algebra",
        "calculus", 
        "probability",
        "statistics",
        "linear_algebra",
        "trigonometry",
        "geometry",
    ])