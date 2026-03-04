# src/evaluation/__init__.py
"""
Evaluation Module for Math Mentor.

Two evaluation modes:
1. Batch Benchmark - Test against ground truth dataset (periodic)
2. Evaluator Agent - Per-query quality assessment (continuous)
"""

from .eval_manager import EvaluationManager, get_eval_manager
from .test_datasets import TestDataset, get_test_dataset
from .eval_report import EvalReport
from .evaluator_agent import EvaluatorAgent, QueryEvaluation, get_evaluator_agent

__all__ = [
    # Batch benchmark
    "EvaluationManager",
    "get_eval_manager",
    "TestDataset",
    "get_test_dataset",
    "EvalReport",
    # Per-query evaluator
    "EvaluatorAgent",
    "QueryEvaluation",
    "get_evaluator_agent",
]