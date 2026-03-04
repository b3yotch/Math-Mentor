# src/evaluation/__init__.py
"""
Evaluation Module for Math Mentor.

Provides comprehensive evaluation of:
- RAG retrieval quality
- Solution correctness
- Agent performance
- HITL effectiveness
- End-to-end pipeline
"""

from .eval_manager import EvaluationManager, get_eval_manager
from .test_datasets import TestDataset, get_test_dataset
from .eval_report import EvalReport

__all__ = [
    "EvaluationManager",
    "get_eval_manager",
    "TestDataset",
    "get_test_dataset",
    "EvalReport",
]