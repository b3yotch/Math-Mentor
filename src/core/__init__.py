# src/core/__init__.py
"""
Core utilities for Math Mentor.
"""

from .embedding_manager import (
    EmbeddingManager,
    get_embedding_manager,
    preload_model
)

__all__ = [
    "EmbeddingManager",
    "get_embedding_manager",
    "preload_model"
]