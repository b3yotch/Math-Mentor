"""Solve math problems — text, image, and audio input with memory caching."""

import os
import re
import time
import json
import logging
import tempfile
from difflib import SequenceMatcher
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from typing import Optional, Dict, Any
from api.schemas import (
    SolveRequest, SolveResponse, RAGSource, EvaluationResult,
    ParsedProblem, ExtractionResponse,
)
from api.dependencies import get_components, ComponentManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/solve", tags=["Solve"])

# Add near top of solve.py, after existing imports
from src.agents.graph import MathMentorGraph

_math_graph: Optional[MathMentorGraph] = None

def _safe_parse_json(content: str) -> dict:
    """Parse JSON from LLM output, handling LaTeX backslash collisions."""
    if not content:
        return {}

    # Strip code fences
    cleaned = content
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0]
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0]
    cleaned = cleaned.strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fix LaTeX backslashes that collide with JSON escapes
    # \f (formfeed), \b (backspace), \t (tab), \n (newline), \r (return)
    # But in LaTeX: \frac, \beta, \theta, \neq, \right
    fixed = re.sub(
        r'\\(?!["\\/bfnrtu\\])',
        r'\\\\',
        cleaned,
    )
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Extract JSON object
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        extracted = re.sub(r'\\(?!["\\/bfnrtu\\])', r'\\\\', match.group())
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass

    return {}


def _get_graph(components) -> MathMentorGraph:
    """Lazy-initialise the LangGraph pipeline."""
    global _math_graph
    if _math_graph is None:
        _math_graph = MathMentorGraph(components.groq_client)
        logger.info("LangGraph hybrid pipeline initialised")
    return _math_graph


def _to_string(value) -> str:
    """Convert any value to a string safely."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # e.g. {"solution": "x = 3", "method": "substitution"}
        # → "x = 3\nMethod: substitution"
        parts = []
        for k, v in value.items():
            if isinstance(v, (list, tuple)):
                v = ", ".join(str(i) for i in v)
            parts.append(str(v))
        return "\n".join(parts)
    if isinstance(value, (list, tuple)):
        # e.g. ["Step 1: ...", "Step 2: ..."]
        return "\n".join(str(item) for item in value)
    return str(value)


def _to_string_list(value) -> list:
    """Convert any value to a list of strings safely."""
    if value is None:
        return []
    if isinstance(value, str):
        # Split on newlines if it looks like steps
        lines = [line.strip() for line in value.split("\n") if line.strip()]
        return lines if lines else [value]
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            if isinstance(item, dict):
                # e.g. {"step": 1, "description": "..."}
                desc = item.get("description") or item.get("content") or item.get("text") or str(item)
                result.append(str(desc))
            else:
                result.append(str(item))
        return result
    if isinstance(value, dict):
        return ["{}: {}".format(k, v) for k, v in value.items()]
    return [str(value)]


def _normalize_llm_result(result: dict) -> dict:
    """
    Normalize all LLM result fields to expected types.
    
    Handles cases where the LLM returns:
    - dict instead of string for final_answer
    - list instead of string for verification
    - nested dicts in solution_steps
    - missing fields
    """
    if not result or not isinstance(result, dict):
        return result

    # Normalize string fields
    string_fields = [
        "solution", "final_answer", "verification",
        "explanation", "error",
    ]
    for field_name in string_fields:
        if field_name in result:
            result[field_name] = _to_string(result[field_name])

    # Normalize solution_steps to list of strings
    if "solution_steps" in result:
        result["solution_steps"] = _to_string_list(result["solution_steps"])

    # Normalize list fields
    list_fields = ["key_concepts", "common_mistakes"]
    for field_name in list_fields:
        if field_name in result:
            val = result[field_name]
            if isinstance(val, str):
                result[field_name] = [val]
            elif isinstance(val, (list, tuple)):
                result[field_name] = [str(item) for item in val]
            elif val is None:
                result[field_name] = []
            else:
                result[field_name] = [str(val)]

    # Normalize parsed_problem to dict
    if "parsed_problem" in result:
        pp = result["parsed_problem"]
        if isinstance(pp, str):
            result["parsed_problem"] = {"type": pp, "what_to_find": "", "given": ""}
        elif isinstance(pp, dict):
            # Ensure all sub-fields are strings
            for key in ["type", "what_to_find", "given"]:
                if key in pp:
                    pp[key] = _to_string(pp[key])
        elif pp is None:
            result["parsed_problem"] = {}

    return result


def _pre_sanitize(text: str) -> str:
    """
    Strip dangerous patterns from input while preserving math content.
    Runs BEFORE guardrails so that guardrails see clean math input.
    
    This handles:
    - HTML/script tags (XSS)
    - JSON injection / prompt injection patterns
    - SQL injection patterns
    - Control characters and null bytes
    - Excessive special characters
    """
    if not text:
        return text

    original = text

    # 1. Remove null bytes and control characters (except newline/tab)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # 2. Remove HTML tags (XSS) — preserve content inside
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text)

    # 3. Remove JSON-like prompt injection attempts
    # Match {"role": "...", "content": "..."} patterns
    text = re.sub(
        r'\{["\']role["\']:\s*["\'](?:system|assistant|user)["\'],\s*["\']content["\']:\s*["\'][^"\']*["\']\}',
        '',
        text,
        flags=re.IGNORECASE
    )
    # Also catch simpler injection patterns
    text = re.sub(
        r'(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|prompts|rules)',
        '',
        text,
        flags=re.IGNORECASE
    )

    # 4. Remove SQL injection patterns (but keep math = signs)
    sql_patterns = [
        r'\bDROP\s+TABLE\b',
        r'\bDELETE\s+FROM\b',
        r'\bINSERT\s+INTO\b',
        r'\bUNION\s+SELECT\b',
        r'\bSELECT\s+\*\s+FROM\b',
        r';\s*--',
        r'\bOR\s+1\s*=\s*1\b',
    ]
    for pattern in sql_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # 5. Remove javascript: protocol
    text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)

    # 6. Remove excessive repeated special characters (keep 2 max)
    text = re.sub(r'([!?#@$%^&*<>|~`])\1{2,}', r'\1\1', text)

    # 7. Collapse excessive whitespace
    text = re.sub(r'\s{3,}', '  ', text)

    # 8. Strip leading/trailing whitespace
    text = text.strip()

    if text != original:
        logger.info(
            "Pre-sanitized input: removed %d chars of dangerous content",
            len(original) - len(text)
        )

    return text


# ============================================================
# Cache / Similarity Helpers
# ============================================================

_solution_cache: Dict[str, Dict[str, Any]] = {}
_CACHE_MAX_SIZE = 200


def _normalize_for_cache(text: str) -> str:
    """Create a normalized cache key from problem text."""
    if not text:
        return ""
    normalized = " ".join(text.lower().split())
    normalized = normalized.rstrip("?.!,;:")
    return normalized


def _text_similarity(a: str, b: str) -> float:
    """Compute text similarity ratio between two strings."""
    if not a or not b:
        return 0.0
    a_norm = _normalize_for_cache(a)
    b_norm = _normalize_for_cache(b)
    if a_norm == b_norm:
        return 1.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def _check_memory_cache(
    memory, normalized_text: str, similarity_threshold: float = 0.92
) -> tuple:
    """
    Check if we've already solved this exact or very similar problem.
    Returns (cached_problem_dict, similarity_score) or (None, 0.0)
    """
    cache_key = _normalize_for_cache(normalized_text)

    # Strategy 1: Fast in-memory exact match
    if cache_key in _solution_cache:
        cached = _solution_cache[cache_key]
        logger.info("Cache HIT (in-memory exact): %s", cache_key[:50])
        return cached, 1.0

    # Strategy 2: Check DB via find_similar_problems
    try:
        similar = memory.find_similar_problems(normalized_text, limit=5) or []
        for problem in similar:
            stored_text = problem.get("extracted_text") or problem.get("question") or ""
            solution = problem.get("solution") or ""
            if not solution or not solution.strip():
                continue
            similarity = _text_similarity(normalized_text, stored_text)
            if similarity >= similarity_threshold:
                logger.info(
                    "Cache HIT (memory DB, sim=%.3f): '%s' ≈ '%s'",
                    similarity, normalized_text[:40], stored_text[:40],
                )
                _add_to_cache(cache_key, problem)
                return problem, similarity
    except Exception as e:
        logger.warning("Memory cache check failed: %s", e)

    # Strategy 3: Scan recent history
    try:
        history = memory.get_problem_history(limit=50) or []
        for problem in history:
            stored_text = problem.get("extracted_text") or problem.get("question") or ""
            solution = problem.get("solution") or ""
            if not solution or not solution.strip():
                continue
            similarity = _text_similarity(normalized_text, stored_text)
            if similarity >= similarity_threshold:
                logger.info(
                    "Cache HIT (history scan, sim=%.3f): '%s' ≈ '%s'",
                    similarity, normalized_text[:40], stored_text[:40],
                )
                _add_to_cache(cache_key, problem)
                return problem, similarity
    except Exception as e:
        logger.warning("History scan failed: %s", e)

    logger.info("Cache MISS: '%s'", normalized_text[:50])
    return None, 0.0


def _add_to_cache(key: str, problem: dict):
    """Add a solved problem to the in-memory cache."""
    global _solution_cache
    if len(_solution_cache) >= _CACHE_MAX_SIZE:
        evict_count = _CACHE_MAX_SIZE // 5
        keys_to_remove = list(_solution_cache.keys())[:evict_count]
        for k in keys_to_remove:
            del _solution_cache[k]
    _solution_cache[key] = problem


def _build_cached_response(
    cached_problem: dict,
    similarity: float,
    question: str,
    input_type: str,
    confidence: float,
    was_human_edited: bool,
    start_time: float,
    pipeline_steps: list,
) -> SolveResponse:
    """Build a SolveResponse from a cached problem."""
    # Normalize cached fields to strings
    solution = _to_string(cached_problem.get("solution", ""))
    explanation = _to_string(cached_problem.get("explanation", ""))
    verification = _to_string(
        cached_problem.get("verification_result")
        or cached_problem.get("verification", "")
    )
    topic = str(cached_problem.get("topic", "general"))

    solution_steps = []
    if solution:
        lines = solution.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith("Final Answer"):
                solution_steps.append(line)

    final_answer = ""
    if "Final Answer:" in solution:
        final_answer = solution.split("Final Answer:")[-1].strip()

    pipeline_steps.append("memory_cache_hit")

    return SolveResponse(
        status="success",
        question=question,
        input_type=input_type,
        detected_topic=topic,
        normalized_question=question,
        parsed_problem=None,
        solution_steps=solution_steps if solution_steps else None,
        final_answer=final_answer,
        solution=solution,
        verification=verification,
        explanation=explanation,
        rag_results_count=0,
        rag_sources=[],
        similar_problems_found=1,
        problem_id=str(
            cached_problem.get("id")
            or cached_problem.get("_id")
            or cached_problem.get("problem_id")
            or ""
        ),
        evaluation=None,
        pipeline_steps=pipeline_steps,
        latency_ms=round((time.time() - start_time) * 1000, 1),
        confidence=confidence,
        was_human_edited=was_human_edited,
        from_cache=True,
        cache_similarity=round(similarity, 3),
    )


# ============================================================
# Serialization Helper
# ============================================================

def _make_serializable(obj):
    """Convert numpy types to native Python."""
    try:
        import numpy as np
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(i) for i in obj]
    return obj


# ============================================================
# LLM Solver
# ============================================================

def _solve_with_llm(components, question, rag_context, similar_problems=None):
    """Solve using Groq LLM with structured output."""
    client = components.groq_client

    context_text = "\n\n".join([
        "{t} ({topic}/{sub}):\n{c}".format(
            t=doc.get("type", "Content").title(),
            topic=doc.get("topic", ""),
            sub=doc.get("subtopic", ""),
            c=doc.get("content", ""),
        )
        for doc in rag_context[:3]
    ]) if rag_context else "No specific context available."

    similar_text = ""
    if similar_problems:
        similar_text = "\n\nSimilar problems:\n" + "\n".join([
            "- {q}: {s}".format(
                q=p.get("extracted_text", ""),
                s=str(p.get("solution", ""))[:200],
            )
            for p in similar_problems[:2]
        ])

    # Truncate very long questions
    truncated_question = question
    if len(question) > 1500:
        truncated_question = question[:1500] + "\n\n[Problem text truncated for processing]"
        logger.info("Truncated question from %d to 1500 chars for LLM", len(question))

    system_prompt = (
        "You are an expert JEE Mathematics tutor.\n"
        "Solve the problem step by step. Be clear and thorough.\n\n"
        "You MUST return valid JSON in this EXACT format:\n"
        "{\n"
        '  "parsed_problem": {\n'
        '    "type": "algebra/calculus/probability/linear_algebra/statistics/trigonometry",\n'
        '    "what_to_find": "What the question asks us to find",\n'
        '    "given": "What information is given"\n'
        '  },\n'
        '  "solution_steps": [\n'
        '    "Step 1: ...",\n'
        '    "Step 2: ...",\n'
        '    "Step 3: ..."\n'
        '  ],\n'
        '  "final_answer": "The final answer as a single string",\n'
        '  "verification": "How to verify the answer as a single string",\n'
        '  "explanation": "Student-friendly explanation as a single string",\n'
        '  "key_concepts": ["concept1", "concept2"],\n'
        '  "common_mistakes": ["mistake1", "mistake2"]\n'
        "}\n\n"
        "CRITICAL RULES:\n"
        "- final_answer MUST be a simple string, NOT a dict or object\n"
        "- verification MUST be a simple string, NOT a list\n"
        "- solution_steps MUST be a list of strings\n"
        "- Return ONLY valid JSON, no other text"
    )

    user_prompt = (
        "Solve: {q}\n\nContext:\n{ctx}{sim}\n\nReturn valid JSON only."
    ).format(q=truncated_question, ctx=context_text, sim=similar_text)

    try:
        max_tokens = 2500
        if len(question) > 800:
            max_tokens = 3500
        if len(question) > 1200:
            max_tokens = 4000

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content

        try:
            result = _safe_parse_json(content)
            if not result:
                raise json.JSONDecodeError("Empty result", "", 0)

            # ══════════════════════════════════════════════
            # NORMALIZE — fix type mismatches from LLM
            # ══════════════════════════════════════════════
            result = _normalize_llm_result(result)

            steps = result.get("solution_steps", ["No steps provided"])
            answer = result.get("final_answer", "")
            solution_text = "\n".join(
                "{}. {}".format(i + 1, step) if not step.startswith("Step") else step
                for i, step in enumerate(steps)
            )

            return {
                "parsed_problem": result.get("parsed_problem", {}),
                "solution_steps": steps,
                "solution": "{s}\n\nFinal Answer: {a}".format(
                    s=solution_text, a=answer
                ),
                "final_answer": answer,
                "verification": result.get("verification", ""),
                "explanation": result.get("explanation", ""),
                "key_concepts": result.get("key_concepts", []),
                "common_mistakes": result.get("common_mistakes", []),
            }
        except json.JSONDecodeError:
            # LLM returned non-JSON — treat entire response as solution
            return _normalize_llm_result({
                "solution": content,
                "solution_steps": [content],
                "final_answer": "",
                "verification": "",
                "explanation": content,
                "key_concepts": [],
                "common_mistakes": [],
            })

    except Exception as e:
        error_msg = str(e)
        logger.error("LLM solve failed: %s", error_msg)
        return {"error": error_msg}


# ============================================================
# Core Pipeline
# ============================================================

def _run_pipeline(
    components,
    question: str,
    top_k: int,
    include_evaluation: bool,
    input_type: str,
    user_id: str,
    confidence: float = 1.0,
    was_human_edited: bool = False,
):
    """Core solve pipeline with pre-sanitization, caching, and error handling."""
    start_time = time.time()
    pipeline_steps = []

    # Step 0: Pre-sanitize input (strip XSS, injection, etc.)
    sanitized_question = _pre_sanitize(question)
    if sanitized_question != question:
        pipeline_steps.append("pre_sanitize")
        logger.info("Input was pre-sanitized before guardrails")

    # If sanitization removed everything meaningful, error out
    if not sanitized_question or not sanitized_question.strip():
        return SolveResponse(
            status="error",
            question=question,
            input_type=input_type,
            error="After removing unsafe content, no math problem remained. Please enter a valid math question.",
            pipeline_steps=pipeline_steps,
            latency_ms=round((time.time() - start_time) * 1000, 1),
            confidence=confidence,
        )

    # Step 1: Guardrails (on sanitized input)
    guardrails = components.guardrails
    try:
        input_report = guardrails.check_input(sanitized_question)
        pipeline_steps.append("guardrails_input")

        if not input_report.passed:
            return SolveResponse(
                status="blocked",
                question=question,
                input_type=input_type,
                blocked_reason=input_report.blocked_reason or "Input blocked",
                suggestions=guardrails.get_suggestions(input_report),
                pipeline_steps=pipeline_steps,
                latency_ms=round((time.time() - start_time) * 1000, 1),
                confidence=confidence,
            )

        topic = input_report.metadata.get("category", "general")
    except Exception as e:
        logger.warning("Guardrails check failed, proceeding anyway: %s", e)
        pipeline_steps.append("guardrails_bypassed")
        topic = "general"

    # Step 2: Normalize
    try:
        normalizer = components.normalizer
        normalized = normalizer.normalize(sanitized_question)
        pipeline_steps.append("normalization")
    except Exception as e:
        logger.warning("Normalization failed, using raw input: %s", e)
        normalized = sanitized_question
        pipeline_steps.append("normalization_skipped")

    # Step 3: CHECK MEMORY CACHE
    memory = components.memory
    try:
        cached_problem, similarity = _check_memory_cache(memory, normalized)

        if cached_problem and cached_problem.get("solution"):
            logger.info(
                "Returning cached solution (sim=%.3f) for: %s",
                similarity, normalized[:50]
            )
            return _build_cached_response(
                cached_problem=cached_problem,
                similarity=similarity,
                question=question,
                input_type=input_type,
                confidence=confidence,
                was_human_edited=was_human_edited,
                start_time=start_time,
                pipeline_steps=pipeline_steps,
            )
    except Exception as e:
        logger.warning("Cache check failed, proceeding to LLM: %s", e)

    # Step 4: RAG
    try:
        retriever = components.retriever
        top_k = max(1, min(5, top_k))
        rag_results = retriever.retrieve(
            normalized, top_k=top_k, min_score=0.2, rerank=True
        ) or []
        rag_serializable = _make_serializable(rag_results)
        pipeline_steps.append("rag_retrieval")
    except Exception as e:
        logger.warning("RAG retrieval failed: %s", e)
        rag_results = []
        rag_serializable = []
        pipeline_steps.append("rag_failed")

    rag_sources = [
        RAGSource(
            topic=r.get("topic", ""),
            subtopic=r.get("subtopic", ""),
            score=float(r.get("score", 0)),
            content_preview=str(r.get("content", ""))[:200],
        )
        for r in rag_serializable[:top_k]
    ]

    # Step 5: Memory lookup for context
    try:
        similar = memory.find_similar_problems(normalized, limit=2) or []
        pipeline_steps.append("memory_lookup")
    except Exception as e:
        logger.warning("Memory lookup failed: %s", e)
        similar = []

    
    # Step 6: LangGraph Adaptive Solve
    try:
        graph = _get_graph(components)
        solution = graph.solve(
            question=normalized,
            rag_context=rag_results,
            similar_problems=similar,
        )

        # Extract pipeline metadata
        graph_nodes = solution.pop("nodes_executed", [])
        difficulty = solution.get("difficulty", "unknown")
        pipeline_steps.append(f"langgraph({difficulty})")
        pipeline_steps.extend(graph_nodes)

        # Update topic if classifier detected one
        if solution.get("detected_topic") and solution["detected_topic"] != "general":
            topic = solution["detected_topic"]

        logger.info(
            "Solved '%s...' via %s path (%d nodes)",
            normalized[:50], difficulty, len(graph_nodes),
        )
    except Exception as e:
        logger.error("LangGraph failed, falling back to direct LLM: %s", e)
        solution = _solve_with_llm(components, normalized, rag_results, similar)
        pipeline_steps.append("llm_solve_fallback")

    # Step 7: Output guardrails
    try:
        output_report = guardrails.check_output(solution)
        pipeline_steps.append("guardrails_output")

        if not output_report.passed:
            return SolveResponse(
                status="output_filtered",
                question=question,
                input_type=input_type,
                blocked_reason=output_report.blocked_reason,
                pipeline_steps=pipeline_steps,
                latency_ms=round((time.time() - start_time) * 1000, 1),
                confidence=confidence,
            )
    except Exception as e:
        logger.warning("Output guardrails failed, proceeding: %s", e)
        pipeline_steps.append("guardrails_output_skipped")

    # Step 8: Save to memory
    problem_id = None
    try:
        problem_id = memory.save_solved_problem(
            input_type=input_type,
            original_input=question,
            extracted_text=normalized,
            topic=topic,
            solution=solution.get("solution", ""),
            explanation=solution.get("explanation", ""),
            verification_result=solution.get("verification", ""),
            rag_context=rag_serializable,
            confidence_score=confidence,
            was_human_edited=was_human_edited,
            user_id=user_id,
        )
        pipeline_steps.append("memory_save")

        # Add to in-memory cache
        cache_key = _normalize_for_cache(normalized)
        _add_to_cache(cache_key, {
            "id": problem_id,
            "extracted_text": normalized,
            "solution": solution.get("solution", ""),
            "explanation": solution.get("explanation", ""),
            "verification_result": solution.get("verification", ""),
            "topic": topic,
        })

    except Exception:
        try:
            problem_id = memory.save_solved_problem(
                input_type=input_type,
                original_input=question,
                extracted_text=normalized,
                topic=topic,
                solution=solution.get("solution", ""),
                explanation=solution.get("explanation", ""),
                verification_result=solution.get("verification", ""),
                rag_context=rag_serializable,
                confidence_score=confidence,
                was_human_edited=was_human_edited,
            )
            pipeline_steps.append("memory_save")
        except Exception as e2:
            logger.error("Memory save failed: %s", e2)

    # Step 9: Evaluate
    evaluation = None
    if include_evaluation:
        try:
            evaluator = components.evaluator
            latency = (time.time() - start_time) * 1000

            eval_result = evaluator.evaluate_query(
                question=normalized,
                solution=solution.get("solution", ""),
                explanation=solution.get("explanation", ""),
                verification=solution.get("verification", ""),
                rag_context=rag_results,
                input_type=input_type,
                confidence=confidence,
                was_human_edited=was_human_edited,
                latency_ms=latency,
                topic=topic,
                query_id=problem_id or "",
            )

            evaluation = EvaluationResult(
                overall_score=round(float(eval_result.overall_score), 3),
                grade=str(eval_result.grade),
                passed=bool(eval_result.passed),
                rag_relevance=round(float(eval_result.rag_relevance_score), 3),
                solution_quality=round(float(eval_result.solution_quality_score), 3),
                explanation_clarity=round(float(eval_result.explanation_clarity_score), 3),
                issues=list(eval_result.issues_found) if eval_result.issues_found else [],
                suggestions=list(eval_result.suggestions) if eval_result.suggestions else [],
                solution_assessment=str(eval_result.solution_assessment or ""),
                explanation_assessment=str(eval_result.explanation_assessment or ""),
            )
            pipeline_steps.append("evaluation")
        except Exception as e:
            logger.error("Evaluation failed: %s", e)

    # Build parsed problem
    pp = solution.get("parsed_problem", {})
    parsed = ParsedProblem(
        type=pp.get("type", ""),
        what_to_find=pp.get("what_to_find", ""),
        given=pp.get("given", ""),
    ) if pp else None

    return SolveResponse(
        status="success",
        question=question,
        input_type=input_type,
        detected_topic=topic,
        normalized_question=normalized,
        parsed_problem=parsed,
        solution_steps=solution.get("solution_steps"),
        final_answer=solution.get("final_answer", ""),
        solution=solution.get("solution", ""),
        verification=solution.get("verification", ""),
        explanation=solution.get("explanation", ""),
        rag_results_count=len(rag_results),
        rag_sources=rag_sources,
        similar_problems_found=len(similar),
        problem_id=problem_id,
        evaluation=evaluation,
        pipeline_steps=pipeline_steps,
        latency_ms=round((time.time() - start_time) * 1000, 1),
        confidence=confidence,
        was_human_edited=was_human_edited,
        from_cache=False,
        cache_similarity=None,
    )


# ==================== TEXT ENDPOINT ====================
@router.post("", response_model=SolveResponse)
def solve_text(
    request: SolveRequest,
    components: ComponentManager = Depends(get_components),
):
    """Solve a math problem from text input."""
    return _run_pipeline(
        components=components,
        question=request.question,
        top_k=request.top_k,
        include_evaluation=request.include_evaluation,
        input_type=request.input_type,
        user_id=request.user_id or "default",
        confidence=request.confidence,
        was_human_edited=request.was_human_edited,
    )


# ==================== IMAGE ENDPOINT ====================
@router.post("/image", response_model=SolveResponse)
async def solve_image(
    file: UploadFile = File(...),
    top_k: int = Form(default=3),
    include_evaluation: bool = Form(default=True),
    user_id: str = Form(default="default"),
    components: ComponentManager = Depends(get_components),
):
    """Solve from image (OCR)."""
    allowed = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(400, "Invalid file type. Upload JPG, PNG, or WebP.")

    suffix = os.path.splitext(file.filename or "image.png")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from src.input_processing.ocr_processor import OCRProcessor
        ocr = OCRProcessor()
        result = ocr.process(tmp_path)

        extracted_text = result.extracted_text
        conf = result.confidence_score

        if not extracted_text or not extracted_text.strip():
            return SolveResponse(
                status="error",
                question="",
                input_type="image",
                error="Could not extract text from image.",
                pipeline_steps=["ocr_failed"],
                confidence=conf,
            )

        memory = components.memory
        try:
            corrected = memory.apply_corrections(extracted_text, "image")
            was_edited = corrected != extracted_text
            extracted_text = corrected
        except Exception:
            was_edited = False

        return _run_pipeline(
            components=components,
            question=extracted_text,
            top_k=top_k,
            include_evaluation=include_evaluation,
            input_type="image",
            user_id=user_id,
            confidence=conf,
            was_human_edited=was_edited,
        )
    finally:
        os.unlink(tmp_path)


# ==================== AUDIO ENDPOINT ====================
@router.post("/audio", response_model=SolveResponse)
async def solve_audio(
    file: UploadFile = File(...),
    top_k: int = Form(default=3),
    include_evaluation: bool = Form(default=True),
    user_id: str = Form(default="default"),
    components: ComponentManager = Depends(get_components),
):
    """Solve from audio (ASR)."""
    allowed = {
        "audio/wav", "audio/mpeg", "audio/mp3", "audio/ogg",
        "audio/flac", "audio/webm", "audio/mp4", "audio/x-wav",
    }
    if file.content_type and file.content_type not in allowed:
        raise HTTPException(400, "Invalid audio format.")

    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from src.input_processing.asr_processor import ASRProcessor
        asr = ASRProcessor(model_size="base")
        result = asr.process(tmp_path)

        extracted_text = result.extracted_text
        conf = result.confidence_score

        if not extracted_text or not extracted_text.strip():
            return SolveResponse(
                status="error",
                question="",
                input_type="audio",
                error="Could not transcribe audio.",
                pipeline_steps=["asr_failed"],
                confidence=conf,
            )

        memory = components.memory
        try:
            corrected = memory.apply_corrections(extracted_text, "audio")
            was_edited = corrected != extracted_text
            extracted_text = corrected
        except Exception:
            was_edited = False

        return _run_pipeline(
            components=components,
            question=extracted_text,
            top_k=top_k,
            include_evaluation=include_evaluation,
            input_type="audio",
            user_id=user_id,
            confidence=conf,
            was_human_edited=was_edited,
        )
    except ImportError:
        return SolveResponse(
            status="error",
            question="",
            input_type="audio",
            error="Whisper ASR not installed. Run: pip install openai-whisper",
            pipeline_steps=[],
        )
    finally:
        os.unlink(tmp_path)


# ==================== EXTRACT IMAGE (HITL) ====================
@router.post("/extract/image", response_model=ExtractionResponse)
async def extract_image(
    file: UploadFile = File(...),
    components: ComponentManager = Depends(get_components),
):
    """Extract text from image without solving."""
    allowed = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(400, "Invalid file type.")

    suffix = os.path.splitext(file.filename or "image.png")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from src.input_processing.ocr_processor import OCRProcessor
        ocr = OCRProcessor()
        result = ocr.process(tmp_path)

        corrected_text = result.extracted_text
        try:
            memory = components.memory
            corrected_text = memory.apply_corrections(result.extracted_text, "image")
        except Exception:
            pass

        return ExtractionResponse(
            extracted_text=corrected_text,
            confidence=result.confidence_score,
            input_type="image",
            metadata=result.metadata or {},
        )
    finally:
        os.unlink(tmp_path)


# ==================== EXTRACT AUDIO (HITL) ====================
@router.post("/extract/audio", response_model=ExtractionResponse)
async def extract_audio(
    file: UploadFile = File(...),
    components: ComponentManager = Depends(get_components),
):
    """Transcribe audio without solving."""
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from src.input_processing.asr_processor import ASRProcessor
        asr = ASRProcessor(model_size="base")
        result = asr.process(tmp_path)

        corrected_text = result.extracted_text
        try:
            memory = components.memory
            corrected_text = memory.apply_corrections(result.extracted_text, "audio")
        except Exception:
            pass

        return ExtractionResponse(
            extracted_text=corrected_text,
            confidence=result.confidence_score,
            input_type="audio",
            metadata=result.metadata or {},
        )
    except ImportError:
        raise HTTPException(500, "Whisper ASR not installed.")
    finally:
        os.unlink(tmp_path)