"""
Math Mentor FastAPI Backend.

Run with: uvicorn api.main:app --reload --port 8000
"""

import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from api.middleware import setup_middleware
from api.routes import health, solve, rag, guardrails, memory, evaluation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Create app
app = FastAPI(
    title="Math Mentor API",
    description="AI-Powered JEE Math Tutor with RAG, Guardrails, HITL, and Evaluation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware
setup_middleware(app)

# Routes
app.include_router(health.router)
app.include_router(solve.router)
app.include_router(rag.router)
app.include_router(guardrails.router)
app.include_router(memory.router)
app.include_router(evaluation.router)


# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logging.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


@app.get("/")
def root():
    return {
        "name": "Math Mentor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }