"""
FastAPI middleware for logging, CORS, and error handling.
"""

import time
import logging
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with timing."""

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        
        response = await call_next(request)
        
        duration = (time.time() - start) * 1000
        logger.info(
            "%s %s - %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration,
        )
        
        response.headers["X-Process-Time-Ms"] = str(round(duration, 1))
        return response


def setup_cors(app):
    """Configure CORS for the app."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def setup_middleware(app):
    """Setup all middleware."""
    setup_cors(app)
    app.add_middleware(RequestLoggingMiddleware)