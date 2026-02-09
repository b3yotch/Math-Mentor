# src/hitl/__init__.py
"""
Human-in-the-Loop (HITL) Module for Math Mentor.

This module manages all HITL workflows including:
- OCR/ASR confidence-based triggers
- Parser ambiguity detection
- Verifier confidence checks
- User re-check requests
- Learning from human corrections
"""

from .hitl_manager import (
    HITLManager,
    HITLTrigger,
    HITLStatus,
    HITLRequest,
    HITLResolution,
    HITLConfig,
    get_hitl_manager,
)

__all__ = [
    "HITLManager",
    "HITLTrigger",
    "HITLStatus",
    "HITLRequest",
    "HITLResolution",
    "HITLConfig",
    "get_hitl_manager",
]