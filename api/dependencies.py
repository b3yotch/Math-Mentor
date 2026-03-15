"""
Shared dependencies for FastAPI routes.
Lazy-loaded singletons for all components.
"""

import os
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


class ComponentManager:
    """Manages lazy-loaded singleton components."""

    def __init__(self):
        self._components = {}

    @property
    def embedding_manager(self):
        if "embedding_manager" not in self._components:
            from src.core.embedding_manager import get_embedding_manager
            self._components["embedding_manager"] = get_embedding_manager()
        return self._components["embedding_manager"]

    @property
    def retriever(self):
        if "retriever" not in self._components:
            from src.rag.retriever import MathRAGRetriever
            self._components["retriever"] = MathRAGRetriever(
                embedding_manager=self.embedding_manager,
                use_reranker=True,
                reranker_type="hybrid",
            )
        return self._components["retriever"]

    @property
    def memory(self):
        if "memory" not in self._components:
            from src.memory.memory_manager import MemoryManager
            self._components["memory"] = MemoryManager()
        return self._components["memory"]

    @property
    def guardrails(self):
        if "guardrails" not in self._components:
            from src.guardrails.guardrails_manager import GuardrailsManager
            self._components["guardrails"] = GuardrailsManager(strict_mode=False)
        return self._components["guardrails"]

    @property
    def normalizer(self):
        if "normalizer" not in self._components:
            from src.input_processing.math_normalizer import MathNormalizer
            self._components["normalizer"] = MathNormalizer()
        return self._components["normalizer"]

    @property
    def groq_client(self):
        if "groq" not in self._components:
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not set")
            self._components["groq"] = Groq(api_key=api_key)
        return self._components["groq"]

    @property
    def evaluator(self):
        if "evaluator" not in self._components:
            from src.evaluation.evaluator_agent import get_evaluator_agent
            self._components["evaluator"] = get_evaluator_agent()
        return self._components["evaluator"]

    def health_check(self) -> dict:
        """Check which components are available."""
        status = {
            "groq_api": bool(os.getenv("GROQ_API_KEY")),
            "rag_available": False,
            "guardrails_active": False,
            "memory_available": False,
        }
        try:
            _ = self.retriever
            status["rag_available"] = True
        except Exception:
            pass
        try:
            _ = self.guardrails
            status["guardrails_active"] = True
        except Exception:
            pass
        try:
            _ = self.memory
            status["memory_available"] = True
        except Exception:
            pass
        return status


# Singleton
_manager = None


def get_components() -> ComponentManager:
    global _manager
    if _manager is None:
        _manager = ComponentManager()
    return _manager