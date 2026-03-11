"""
MCP Server for Math Mentor.

Exposes all Math Mentor capabilities as MCP tools, resources, and prompts.
Compatible with MCP Inspector, Claude Desktop, VS Code, and any MCP client.
"""

import os
import sys
import json
import logging
import time
import inspect
import re
from typing import Any

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Force UTF-8 on Windows
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, Exception):
        pass

from mcp.server.fastmcp import FastMCP
from .config import MCPConfig

logger = logging.getLogger(__name__)


# ============================================================
# Lazy-loaded components
# ============================================================
_components = {}


def _get_embedding_manager():
    if "embedding_manager" not in _components:
        from src.core.embedding_manager import get_embedding_manager
        _components["embedding_manager"] = get_embedding_manager()
    return _components["embedding_manager"]


def _get_retriever():
    if "retriever" not in _components:
        from src.rag.retriever import MathRAGRetriever
        em = _get_embedding_manager()
        _components["retriever"] = MathRAGRetriever(
            embedding_manager=em,
            use_reranker=True,
            reranker_type="hybrid"
        )
    return _components["retriever"]


def _get_memory():
    if "memory" not in _components:
        from src.memory.memory_manager import MemoryManager
        _components["memory"] = MemoryManager()
    return _components["memory"]


def _get_guardrails():
    if "guardrails" not in _components:
        from src.guardrails.guardrails_manager import GuardrailsManager
        _components["guardrails"] = GuardrailsManager(strict_mode=False)
    return _components["guardrails"]


def _get_normalizer():
    if "normalizer" not in _components:
        from src.input_processing.math_normalizer import MathNormalizer
        _components["normalizer"] = MathNormalizer()
    return _components["normalizer"]


def _get_groq_client():
    if "groq" not in _components:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment")
        _components["groq"] = Groq(api_key=api_key)
    return _components["groq"]


def _get_evaluator():
    if "evaluator" not in _components:
        from src.evaluation.evaluator_agent import get_evaluator_agent
        _components["evaluator"] = get_evaluator_agent()
    return _components["evaluator"]


# ============================================================
# Utility Functions
# ============================================================
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U00002500-\U00002BEF"
    "\U00002705\U0000274C\U000026A0"
    "\U0001F4DD-\U0001F4DF"
    "\U0001F9E0-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U0000FE00-\U0000FE0F"
    "\U0000200D"
    "]+",
    flags=re.UNICODE
)


def _strip_emojis(obj):
    """Recursively remove emoji characters from data structures."""
    if isinstance(obj, str):
        return _EMOJI_PATTERN.sub("", obj).strip()
    elif isinstance(obj, dict):
        return {_strip_emojis(k): _strip_emojis(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_strip_emojis(item) for item in obj]
    else:
        return obj


def _make_serializable(obj):
    """Recursively convert numpy and non-serializable types to native Python types."""
    try:
        import numpy as np
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
    except ImportError:
        pass

    if obj is None:
        return None
    elif isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif hasattr(obj, "to_dict"):
        return _make_serializable(obj.to_dict())
    elif hasattr(obj, "__dict__"):
        return _make_serializable(obj.__dict__)
    else:
        return str(obj)


def _safe_json(obj, indent=2) -> str:
    """
    JSON serialize with full safety.
    Strips emojis, converts numpy, ensures ASCII.
    """
    cleaned = _strip_emojis(_make_serializable(obj))
    return json.dumps(
        cleaned,
        indent=indent,
        ensure_ascii=True,
        default=str,
    )




def _solve_with_llm(question, rag_context, similar_problems=None):
    """Internal: Solve math problem using Groq LLM."""
    client = _get_groq_client()

    context_text = "\n\n".join([
        "{title} ({topic}/{subtopic}):\n{content}".format(
            title=doc.get("type", "Content").title(),
            topic=doc.get("topic", ""),
            subtopic=doc.get("subtopic", ""),
            content=doc.get("content", ""),
        )
        for doc in rag_context[:3]
    ]) if rag_context else "No specific context available."

    similar_text = ""
    if similar_problems:
        similar_text = "\n\nSimilar previously solved problems:\n" + "\n".join([
            "- {q}: {s}".format(
                q=p.get("extracted_text", ""),
                s=p.get("solution", ""),
            )
            for p in similar_problems[:2]
        ])

    system_prompt = (
        "You are an expert JEE Mathematics tutor. "
        "Solve problems step-by-step with clear explanations.\n\n"
        "Your response MUST be in this exact JSON format:\n"
        "{\n"
        '    "parsed_problem": {\n'
        '        "type": "algebra/calculus/probability/linear_algebra",\n'
        '        "what_to_find": "description",\n'
        '        "given": "what information is given"\n'
        "    },\n"
        '    "solution_steps": ["Step 1: ...", "Step 2: ..."],\n'
        '    "final_answer": "The final answer",\n'
        '    "verification": "How to verify this answer",\n'
        '    "explanation": "Student-friendly explanation"\n'
        "}\n\n"
        "Be thorough, show all work. Return valid JSON only."
    )

    user_prompt = (
        "Solve this math problem:\n\n"
        "Problem: {question}\n\n"
        "Relevant Knowledge:\n{context}\n{similar}\n\n"
        "Provide a complete solution. Return valid JSON only."
    ).format(question=question, context=context_text, similar=similar_text)

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=2000,
        )

        content = response.choices[0].message.content

        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            result = json.loads(content)
            solution_text = "\n".join(
                result.get("solution_steps", ["No steps provided"])
            )
            final_answer = result.get("final_answer", "")

            return {
                "parsed_problem": result.get("parsed_problem", {}),
                "solution": "{steps}\n\nFinal Answer: {answer}".format(
                    steps=solution_text, answer=final_answer
                ),
                "final_answer": final_answer,
                "verification": result.get("verification", ""),
                "explanation": result.get("explanation", ""),
            }
        except json.JSONDecodeError:
            return {
                "solution": content,
                "final_answer": "",
                "verification": "Response was not in expected format",
                "explanation": content,
            }

    except Exception as e:
        return {
            "error": str(e),
            "solution": "",
            "final_answer": "",
            "verification": "",
            "explanation": "",
        }


# ============================================================
# Create the MCP Server
# ============================================================
def create_server(config=None):
    """Create and configure the MCP server with all tools, resources, and prompts."""

    if config is None:
        config = MCPConfig()

    # Create FastMCP (compatible with different mcp versions)
    fastmcp_params = inspect.signature(FastMCP.__init__).parameters

    kwargs = {}
    if "description" in fastmcp_params:
        kwargs["description"] = config.description
    if "version" in fastmcp_params:
        kwargs["version"] = config.server_version
    if "instructions" in fastmcp_params and "description" not in fastmcp_params:
        kwargs["instructions"] = config.description

    mcp = FastMCP(config.server_name, **kwargs)

    # ============================================================
    # TOOLS
    # ============================================================

    # -------------------- 1. SOLVE MATH PROBLEM --------------------
    @mcp.tool()
    def solve_math_problem(
        question: str,
        top_k: int = 3,
        include_evaluation: bool = True,
    ) -> str:
        """
        Solve a math problem end-to-end.

        Runs the full pipeline: guardrails, normalization, RAG retrieval,
        LLM solving, output validation, evaluation, and memory save.

        Args:
            question: The math problem to solve (e.g. Solve x^2 - 5x + 6 = 0)
            top_k: Number of RAG documents to retrieve (1-5)
            include_evaluation: Whether to evaluate response quality

        Returns:
            JSON string with solution, explanation, verification, and evaluation
        """
        start_time = time.time()

        result_data = {
            "question": question,
            "status": "success",
            "pipeline_steps": [],
        }

        try:
            # Step 1: Guardrails
            guardrails = _get_guardrails()
            input_report = _safe_call(guardrails.check_input, question)
            result_data["pipeline_steps"].append("guardrails_input")

            if not input_report.passed:
                result_data["status"] = "blocked"
                result_data["blocked_reason"] = (
                    input_report.blocked_reason or "Input blocked by guardrails"
                )
                result_data["suggestions"] = _safe_call(
                    guardrails.get_suggestions, input_report
                )
                return _safe_json(result_data)

            topic = input_report.metadata.get("category", "general")
            result_data["detected_topic"] = topic

            # Step 2: Normalize
            normalizer = _get_normalizer()
            normalized = normalizer.normalize(question)
            result_data["normalized_question"] = normalized
            result_data["pipeline_steps"].append("normalization")

            # Step 3: RAG retrieval
            retriever = _get_retriever()
            top_k = max(1, min(5, top_k))

            rag_results = _safe_call(
                retriever.retrieve,
                normalized,
                top_k=top_k,
                min_score=0.2,
                rerank=True,
            )
            if rag_results is None:
                rag_results = []

            rag_serializable = _make_serializable(rag_results)

            result_data["rag_results_count"] = len(rag_results)
            result_data["rag_sources"] = [
                {
                    "topic": r.get("topic", ""),
                    "subtopic": r.get("subtopic", ""),
                    "score": float(r.get("score", 0)),
                    "content_preview": str(r.get("content", ""))[:200],
                }
                for r in rag_serializable[:top_k]
            ]
            result_data["pipeline_steps"].append("rag_retrieval")

            # Step 4: Memory lookup
            memory = _get_memory()
            similar = _safe_call(
                memory.find_similar_problems, normalized, limit=2
            )
            if similar is None:
                similar = []

            result_data["similar_problems_found"] = len(similar)
            result_data["pipeline_steps"].append("memory_lookup")

            # Step 5: Solve with LLM
            solution = _solve_with_llm(normalized, rag_results, similar)

            if solution.get("error"):
                result_data["status"] = "error"
                result_data["error"] = solution["error"]
                return _safe_json(result_data)

            result_data["solution"] = solution.get("solution", "")
            result_data["final_answer"] = solution.get("final_answer", "")
            result_data["verification"] = solution.get("verification", "")
            result_data["explanation"] = solution.get("explanation", "")
            result_data["parsed_problem"] = solution.get("parsed_problem", {})
            result_data["pipeline_steps"].append("llm_solve")

            # Step 6: Output guardrails
            output_report = _safe_call(guardrails.check_output, solution)

            if output_report is None:
                result_data["output_validation"] = "skipped"
            elif not output_report.passed:
                result_data["status"] = "output_filtered"
                result_data["output_filter_reason"] = output_report.blocked_reason
                return _safe_json(result_data)
            else:
                if output_report.warnings:
                    result_data["output_warnings"] = output_report.warnings

            result_data["pipeline_steps"].append("guardrails_output")

            # Step 7: Save to memory
            problem_id = None
            try:
                problem_id = _safe_call(
                    memory.save_solved_problem,
                    input_type="text",
                    original_input=question,
                    extracted_text=normalized,
                    topic=topic,
                    solution=solution.get("solution", ""),
                    explanation=solution.get("explanation", ""),
                    verification_result=solution.get("verification", ""),
                    rag_context=rag_serializable,
                    confidence_score=1.0,
                    was_human_edited=False,
                )
                result_data["problem_id"] = problem_id
                result_data["pipeline_steps"].append("memory_save")
            except Exception as e:
                result_data["memory_save_error"] = str(e)

            # Step 8: Evaluate
            if include_evaluation:
                try:
                    evaluator = _get_evaluator()
                    latency = (time.time() - start_time) * 1000

                    evaluation = evaluator.evaluate_query(
                        question=normalized,
                        solution=solution.get("solution", ""),
                        explanation=solution.get("explanation", ""),
                        verification=solution.get("verification", ""),
                        rag_context=rag_results,
                        input_type="text",
                        confidence=1.0,
                        was_human_edited=False,
                        latency_ms=latency,
                        topic=topic,
                        query_id=problem_id or "",
                    )

                    result_data["evaluation"] = {
                        "overall_score": round(evaluation.overall_score, 3),
                        "grade": evaluation.grade,
                        "passed": evaluation.passed,
                        "rag_relevance": round(evaluation.rag_relevance_score, 3),
                        "solution_quality": round(
                            evaluation.solution_quality_score, 3
                        ),
                        "explanation_clarity": round(
                            evaluation.explanation_clarity_score, 3
                        ),
                        "issues": evaluation.issues_found,
                        "suggestions": evaluation.suggestions,
                        "solution_assessment": evaluation.solution_assessment,
                        "explanation_assessment": evaluation.explanation_assessment,
                    }
                    result_data["pipeline_steps"].append("evaluation")
                except Exception as e:
                    result_data["evaluation_error"] = str(e)

            result_data["latency_ms"] = round(
                (time.time() - start_time) * 1000, 1
            )

        except Exception as e:
            result_data["status"] = "error"
            result_data["error"] = str(e)
            logger.error("solve_math_problem failed: %s", e, exc_info=True)

        return _safe_json(result_data)

    # -------------------- 2. RETRIEVE MATH CONTEXT --------------------
    @mcp.tool()
    def retrieve_math_context(
        query: str,
        top_k: int = 3,
        topic_filter: str = "",
    ) -> str:
        """
        Retrieve relevant math knowledge from the RAG knowledge base.

        Searches through textbooks, formulas, theorems, and examples.

        Args:
            query: Math question or topic to search for
            top_k: Number of results to return (1-10)
            topic_filter: Optional topic filter (algebra, calculus, probability, etc.)

        Returns:
            JSON string with retrieved documents and scores
        """
        try:
            retriever = _get_retriever()
            top_k = max(1, min(10, top_k))

            results = _safe_call(
                retriever.retrieve,
                query,
                top_k=top_k,
                min_score=0.1,
                rerank=True,
            )
            if results is None:
                results = []

            if topic_filter:
                results = [
                    r
                    for r in results
                    if topic_filter.lower() in str(r.get("topic", "")).lower()
                ]

            formatted = []
            for i, r in enumerate(results):
                formatted.append(
                    {
                        "rank": i + 1,
                        "topic": r.get("topic", ""),
                        "subtopic": r.get("subtopic", ""),
                        "type": r.get("type", ""),
                        "chapter": r.get("chapter", ""),
                        "section": r.get("section", ""),
                        "content": r.get("content", ""),
                        "score": float(r.get("score", 0)),
                        "rerank_score": float(r.get("rerank_score", 0)),
                    }
                )

            return _safe_json(
                {
                    "query": query,
                    "results_count": len(formatted),
                    "topic_filter": topic_filter or "none",
                    "results": formatted,
                }
            )

        except Exception as e:
            return _safe_json({"error": str(e)})

    # -------------------- 3. CHECK GUARDRAILS --------------------
    @mcp.tool()
    def check_input_safety(text: str) -> str:
        """
        Check if a text input passes safety guardrails.

        Validates whether the input is math-related, checks for
        prompt injection attempts, and filters harmful content.

        Args:
            text: The text to validate

        Returns:
            JSON string with pass/fail status, category, and suggestions
        """
        try:
            guardrails = _get_guardrails()
            report = _safe_call(guardrails.check_input, text)

            result = {
                "text": text,
                "passed": report.passed,
                "status": report.status.value,
                "category": report.metadata.get("category", "unknown"),
                "math_confidence": float(
                    report.metadata.get("math_confidence", 0)
                ),
                "blocked_reason": report.blocked_reason,
                "warnings": report.warnings,
            }

            if not report.passed:
                result["suggestions"] = _safe_call(
                    guardrails.get_suggestions, report
                )
                result["rejection_message"] = _safe_call(
                    guardrails.get_rejection_message, report
                )

            return _safe_json(result)

        except Exception as e:
            return _safe_json({"error": str(e)})

    # -------------------- 4. GET PROBLEM HISTORY --------------------
    @mcp.tool()
    def get_problem_history(
        limit: int = 10,
        topic: str = "",
    ) -> str:
        """
        Get previously solved math problems from memory.

        Args:
            limit: Maximum number of problems to return (1-50)
            topic: Optional topic filter

        Returns:
            JSON string with problem history
        """
        try:
            memory = _get_memory()
            limit = max(1, min(50, limit))

            history = _safe_call(memory.get_problem_history, limit=limit)
            if history is None:
                history = []

            if topic:
                history = [
                    p
                    for p in history
                    if topic.lower() in str(p.get("topic", "")).lower()
                ]

            formatted = []
            for p in history:
                formatted.append(
                    {
                        "id": p.get("id", ""),
                        "input_type": p.get("input_type", ""),
                        "question": p.get("extracted_text", ""),
                        "topic": p.get("topic", ""),
                        "solution_preview": str(p.get("solution", ""))[:300],
                        "confidence": float(p.get("confidence_score", 0)),
                        "was_human_edited": p.get("was_human_edited", False),
                        "created_at": str(p.get("created_at", "")),
                    }
                )

            return _safe_json(
                {
                    "total": len(formatted),
                    "topic_filter": topic or "none",
                    "problems": formatted,
                }
            )

        except Exception as e:
            return _safe_json({"error": str(e)})

    # -------------------- 5. FIND SIMILAR PROBLEMS --------------------
    @mcp.tool()
    def find_similar_problems(
        question: str,
        limit: int = 5,
    ) -> str:
        """
        Find previously solved problems similar to the given question.

        Uses semantic similarity to find related problems from memory.

        Args:
            question: The math question to find similar problems for
            limit: Maximum number of similar problems (1-10)

        Returns:
            JSON string with similar problems and their solutions
        """
        try:
            memory = _get_memory()
            limit = max(1, min(10, limit))

            similar = _safe_call(
                memory.find_similar_problems, question, limit=limit
            )
            if similar is None:
                similar = []

            formatted = []
            for p in similar:
                formatted.append(
                    {
                        "question": p.get("extracted_text", ""),
                        "topic": p.get("topic", ""),
                        "solution": p.get("solution", ""),
                        "explanation": p.get("explanation", ""),
                        "input_type": p.get("input_type", ""),
                    }
                )

            return _safe_json(
                {
                    "query": question,
                    "similar_count": len(formatted),
                    "results": formatted,
                }
            )

        except Exception as e:
            return _safe_json({"error": str(e)})

    # -------------------- 6. RECORD FEEDBACK --------------------
    @mcp.tool()
    def record_feedback(
        problem_id: str,
        is_correct: bool,
        comment: str = "",
        corrected_solution: str = "",
    ) -> str:
        """
        Record feedback on a solved problem.

        Helps Math Mentor learn and improve over time.

        Args:
            problem_id: The ID of the problem (from solve_math_problem response)
            is_correct: Whether the solution was correct
            comment: Optional feedback comment
            corrected_solution: Optional corrected solution if answer was wrong

        Returns:
            JSON string confirming feedback was recorded
        """
        try:
            memory = _get_memory()
            _safe_call(
                memory.record_feedback,
                problem_id,
                is_correct=is_correct,
                comment=comment,
                corrected_solution=corrected_solution,
            )

            return _safe_json(
                {
                    "status": "recorded",
                    "problem_id": problem_id,
                    "is_correct": is_correct,
                    "has_comment": bool(comment),
                    "has_correction": bool(corrected_solution),
                }
            )

        except Exception as e:
            return _safe_json({"error": str(e)})

    # -------------------- 7. EVALUATE SOLUTION --------------------
    @mcp.tool()
    def evaluate_solution(
        question: str,
        solution: str,
        explanation: str = "",
    ) -> str:
        """
        Evaluate the quality of a math solution.

        Uses LLM-as-judge to assess mathematical correctness,
        completeness of steps, and explanation clarity.

        Args:
            question: The original math question
            solution: The solution to evaluate
            explanation: Optional explanation text

        Returns:
            JSON string with quality scores and assessment
        """
        try:
            evaluator = _get_evaluator()

            evaluation = evaluator.evaluate_query(
                question=question,
                solution=solution,
                explanation=explanation or "",
                verification="",
                rag_context=[],
                input_type="text",
                confidence=1.0,
                was_human_edited=False,
                latency_ms=0.0,
                topic="",
            )

            return _safe_json(
                {
                    "overall_score": round(evaluation.overall_score, 3),
                    "grade": evaluation.grade,
                    "passed": evaluation.passed,
                    "scores": {
                        "solution_quality": round(
                            evaluation.solution_quality_score, 3
                        ),
                        "explanation_clarity": round(
                            evaluation.explanation_clarity_score, 3
                        ),
                        "rag_relevance": round(
                            evaluation.rag_relevance_score, 3
                        ),
                    },
                    "assessments": {
                        "solution": evaluation.solution_assessment,
                        "explanation": evaluation.explanation_assessment,
                    },
                    "issues": evaluation.issues_found,
                    "suggestions": evaluation.suggestions,
                }
            )

        except Exception as e:
            return _safe_json({"error": str(e)})

    # -------------------- 8. GET SYSTEM STATISTICS --------------------
    @mcp.tool()
    def get_system_stats() -> str:
        """
        Get Math Mentor system statistics.

        Returns memory stats, evaluation history, and system health.

        Returns:
            JSON string with comprehensive system statistics
        """
        try:
            stats = {}

            # Memory stats
            try:
                memory = _get_memory()
                memory_stats = _safe_call(memory.get_statistics)
                stats["memory"] = memory_stats
            except Exception as e:
                stats["memory_error"] = str(e)

            # Evaluation stats
            try:
                evaluator = _get_evaluator()
                eval_stats = evaluator.get_statistics()
                stats["evaluation"] = eval_stats
            except Exception as e:
                stats["evaluation_error"] = str(e)

            # RAG stats
            try:
                retriever = _get_retriever()
                stats["rag"] = {
                    "available": True,
                    "has_reranker": (
                        hasattr(retriever, "reranker")
                        and retriever.reranker is not None
                    ),
                }
            except Exception as e:
                stats["rag"] = {"available": False, "error": str(e)}

            # Guardrails
            stats["guardrails"] = {"active": True}

            # API status
            stats["groq_api"] = {
                "configured": bool(os.getenv("GROQ_API_KEY"))
            }

            return _safe_json(stats)

        except Exception as e:
            return _safe_json({"error": str(e)})

    # -------------------- 9. RUN BATCH EVALUATION --------------------
    @mcp.tool()
    def run_batch_evaluation(
        topic: str = "",
        max_cases: int = 10,
        include_rag: bool = True,
        include_solutions: bool = True,
        include_guardrails: bool = True,
    ) -> str:
        """
        Run batch evaluation against the built-in test dataset.

        Tests system performance across predefined math problems
        with known correct answers.

        Args:
            topic: Filter by topic (algebra, calculus, probability, etc.) or empty for all
            max_cases: Maximum test cases to run (1-26)
            include_rag: Evaluate RAG retrieval quality
            include_solutions: Evaluate solution correctness
            include_guardrails: Evaluate guardrails accuracy

        Returns:
            JSON string with evaluation report and scores
        """
        try:
            from src.evaluation.eval_manager import get_eval_manager

            eval_manager = get_eval_manager()
            max_cases = max(1, min(26, max_cases))

            report = eval_manager.run_full_evaluation(
                include_rag=include_rag,
                include_solution=include_solutions,
                include_guardrails=include_guardrails,
                include_memory=False,
                topic_filter=topic if topic else None,
                max_cases=max_cases,
            )

            return _safe_json(report.to_dict())

        except Exception as e:
            return _safe_json({"error": str(e)})

    # -------------------- 10. NORMALIZE MATH TEXT --------------------
    @mcp.tool()
    def normalize_math(text: str) -> str:
        """
        Normalize mathematical text notation.

        Converts various math notations to a standard format.
        For example: x squared becomes x^2, square root of x becomes sqrt(x).

        Args:
            text: Math text to normalize

        Returns:
            JSON string with original and normalized text
        """
        try:
            normalizer = _get_normalizer()
            normalized = normalizer.normalize(text)

            return _safe_json(
                {
                    "original": text,
                    "normalized": normalized,
                    "changed": text != normalized,
                }
            )

        except Exception as e:
            return _safe_json({"error": str(e)})

    # ============================================================
    # RESOURCES
    # ============================================================

    @mcp.resource("math://topics")
    def get_supported_topics() -> str:
        """List of supported math topics and subtopics."""
        topics_data = {
            "Algebra": [
                "linear_equations",
                "quadratic_equations",
                "systems_of_equations",
                "polynomials",
                "functions",
                "word_problems",
                "inequalities",
            ],
            "Calculus": [
                "limits",
                "derivatives",
                "integrals",
                "optimization",
                "chain_rule",
                "product_rule",
                "applications",
            ],
            "Probability": [
                "basic_probability",
                "conditional_probability",
                "binomial_distribution",
                "permutations",
                "combinations",
                "bayes_theorem",
            ],
            "Linear Algebra": [
                "matrices",
                "determinants",
                "eigenvalues",
                "vector_spaces",
                "matrix_multiplication",
                "inverse_matrix",
            ],
            "Statistics": [
                "mean",
                "median",
                "mode",
                "standard_deviation",
                "variance",
                "distributions",
            ],
            "Trigonometry": [
                "basic_ratios",
                "identities",
                "equations",
                "inverse_functions",
                "applications",
            ],
        }

        return _safe_json(
            {
                "total_topics": len(topics_data),
                "topics": topics_data,
            }
        )

    @mcp.resource("math://stats")
    def get_stats_resource() -> str:
        """Current system statistics and health."""
        try:
            memory = _get_memory()
            stats = _safe_call(memory.get_statistics)
            if stats is None:
                return _safe_json({"error": "Could not retrieve statistics"})
            return _safe_json(stats)
        except Exception as e:
            error_msg = (
                str(e).encode("ascii", errors="replace").decode("ascii")
            )
            return _safe_json({"error": error_msg})

    @mcp.resource("math://test-dataset")
    def get_test_dataset_resource() -> str:
        """Available test cases for batch evaluation."""
        try:
            from src.evaluation.test_datasets import get_test_dataset

            dataset = get_test_dataset()
            summary = dataset.get_summary()

            cases_preview = []
            for tc in dataset.get_all()[:10]:
                cases_preview.append(
                    {
                        "id": tc.id,
                        "question": tc.question,
                        "topic": tc.topic,
                        "subtopic": tc.subtopic,
                        "difficulty": tc.difficulty,
                        "expected_answer": tc.expected_answer,
                    }
                )

            return _safe_json(
                {
                    "summary": summary,
                    "preview_cases": cases_preview,
                    "note": "Use run_batch_evaluation tool to test against these cases",
                }
            )
        except Exception as e:
            return _safe_json({"error": str(e)})

    # ============================================================
    # PROMPTS
    # ============================================================

    @mcp.prompt()
    def solve_step_by_step(question: str) -> str:
        """Create a prompt for step-by-step math solving."""
        return (
            "Please solve this math problem step by step "
            "using the Math Mentor system.\n\n"
            "Question: {q}\n\n"
            "Use the solve_math_problem tool with the question above.\n"
            "After getting the result, present:\n"
            "1. The detected topic\n"
            "2. The solution steps (clearly formatted)\n"
            "3. The final answer\n"
            "4. The verification\n"
            "5. The quality evaluation grade and any issues found\n\n"
            "If the quality grade is below B, suggest how the student "
            "might approach the problem differently."
        ).format(q=question)

    @mcp.prompt()
    def study_topic(topic: str) -> str:
        """Create a prompt for studying a math topic."""
        return (
            "Help me study the topic: {t}\n\n"
            "Steps:\n"
            '1. Use retrieve_math_context to find relevant knowledge about "{t}"\n'
            "2. Summarize the key concepts, formulas, and theorems\n"
            "3. Use solve_math_problem to solve 2-3 example problems on this topic\n"
            "4. Highlight common mistakes students make\n"
            "5. Provide tips for JEE preparation on this topic"
        ).format(t=topic)

    @mcp.prompt()
    def review_performance() -> str:
        """Create a prompt for reviewing learning performance."""
        return (
            "Review Math Mentor performance and learning progress:\n\n"
            "1. Use get_system_stats to see overall statistics\n"
            "2. Use get_problem_history with limit=20 to see recent problems\n"
            "3. Analyze and summarize:\n"
            "   - Topics most frequently asked about\n"
            "   - Average solution quality scores\n"
            "   - Topics where the student struggles\n"
            "   - Topics where performance is strong\n"
            "   - Recommendations for what to study next"
        )

    @mcp.prompt()
    def evaluate_system(topic: str = "") -> str:
        """Create a prompt for running system evaluation."""
        topic_text = ' for topic "{t}"'.format(t=topic) if topic else ""
        topic_arg = ' with topic="{t}"'.format(t=topic) if topic else ""
        return (
            "Run a comprehensive evaluation of Math Mentor{tt}:\n\n"
            "1. Use run_batch_evaluation{ta} with max_cases=10\n"
            "2. Use get_system_stats for current health metrics\n"
            "3. Analyze and report:\n"
            "   - Overall grade and score\n"
            "   - Strongest and weakest components\n"
            "   - RAG retrieval quality\n"
            "   - Solution accuracy broken down by topic\n"
            "   - Specific issues found\n"
            "   - Actionable recommendations for improvement"
        ).format(tt=topic_text, ta=topic_arg)

    @mcp.prompt()
    def compare_approaches(question: str) -> str:
        """Create a prompt for comparing different solution approaches."""
        return (
            "Compare different approaches to solve this problem: {q}\n\n"
            "1. Use solve_math_problem to get the AI solution\n"
            "2. Use retrieve_math_context to find all relevant methods\n"
            "3. Present:\n"
            "   - Method 1: The approach used by the AI\n"
            "   - Method 2: An alternative approach (if applicable)\n"
            "   - Which method is faster for JEE exam conditions\n"
            "   - Which method is more generalizable\n"
            "   - Common pitfalls for each approach"
        ).format(q=question)

    return mcp