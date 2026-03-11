"""
MCP Server for Math Mentor.

Exposes Math Mentor capabilities via Model Context Protocol:
- Tools: solve problems, retrieve context, evaluate, check guardrails
- Resources: knowledge base, problem history, statistics
- Prompts: math solving templates
"""

from .server import create_server

__all__ = ["create_server"]