"""Solve math problems — text, image, and audio input with memory caching."""

import os
import re
import time
import json
import logging
import tempfile
from difflib import SequenceMatcher
from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException

from api.schemas import (
    SolveRequest, SolveResponse, RAGSource, EvaluationResult,
    ParsedProblem, ExtractionResponse,
)
from api.dependencies import get_components, ComponentManager

try:
    from src.agents.graph import MathMentorGraph
except Exception:
    MathMentorGraph = None

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/solve", tags=["Solve"])

_math_graph = None


# ============================================================
# Graph helper
# ============================================================

def _get_graph(components):
    global _math_graph
    if MathMentorGraph is None:
        return None
    if _math_graph is None:
        _math_graph = MathMentorGraph(components.groq_client)
        logger.info("LangGraph pipeline initialised")
    return _math_graph


# ============================================================
# Input Pre-Sanitization
# ============================================================

def _to_string(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = []
        for _, v in value.items():
            if isinstance(v, (list, tuple)):
                v = ", ".join(str(i) for i in v)
            parts.append(str(v))
        return "\n".join(parts)
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item) for item in value)
    return str(value)


def _to_string_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        lines = [line.strip() for line in value.split("\n") if line.strip()]
        return lines if lines else [value]
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            if isinstance(item, dict):
                desc = (
                    item.get("description")
                    or item.get("content")
                    or item.get("text")
                    or str(item)
                )
                result.append(str(desc))
            else:
                result.append(str(item))
        return result
    if isinstance(value, dict):
        return [f"{k}: {v}" for k, v in value.items()]
    return [str(value)]


def _normalize_llm_result(result: dict) -> dict:
    if not result or not isinstance(result, dict):
        return result

    string_fields = [
        "solution", "final_answer", "verification",
        "explanation", "error",
    ]
    for field_name in string_fields:
        if field_name in result:
            result[field_name] = _to_string(result[field_name])

    if "solution_steps" in result:
        result["solution_steps"] = _to_string_list(result["solution_steps"])

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

    if "parsed_problem" in result:
        pp = result["parsed_problem"]
        if isinstance(pp, str):
            result["parsed_problem"] = {
                "type": pp, "what_to_find": "", "given": ""
            }
        elif isinstance(pp, dict):
            for key in ["type", "what_to_find", "given"]:
                if key in pp:
                    pp[key] = _to_string(pp[key])
        elif pp is None:
            result["parsed_problem"] = {}

    return result


def _safe_parse_json(content: str) -> dict:
    """Parse JSON from LLM output, handling LaTeX backslash collisions."""
    if not content:
        return {}

    cleaned = content.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0]
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', cleaned)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{[\s\S]*\}', cleaned)
    if match:
        extracted = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', match.group())
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass

    return {}


def _postprocess_solution(result: dict) -> dict:
    """
    Fix literal \\n sequences and normalize over-escaped LaTeX backslashes.

    After json.loads, LaTeX in strings should have single backslashes (\frac).
    If the LLM over-escaped (\\\\frac → \\frac after parsing), we fix it here.
    We do NOT attempt regex-based bare-LaTeX wrapping — that belongs to the
    frontend renderer (KaTeX/MathJax).
    """
    if not result or not isinstance(result, dict):
        return result

    string_fields = [
        "solution", "final_answer", "verification",
        "explanation", "error",
    ]

    for field in string_fields:
        if field in result and isinstance(result[field], str):
            value = result[field]

            # Fix literal \n that wasn't converted during JSON parsing
            value = value.replace("\\n", "\n")

            # Collapse excessive blank lines
            while "\n\n\n\n" in value:
                value = value.replace("\n\n\n\n", "\n\n\n")

            # Fix over-escaped LaTeX: \\frac → \frac
            # This handles cases where the LLM used quadruple backslashes
            value = re.sub(r'\\\\([a-zA-Z])', r'\\\1', value)

            result[field] = value

    return result


def _pre_sanitize(text: str) -> str:
    if not text:
        return text

    original = text

    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'<script[^>]*>.*?</script>', '', text,
                  flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text)

    text = re.sub(
        r'\{["\']role["\']:\s*["\'](?:system|assistant|user)["\'],'
        r'\s*["\']content["\']:\s*["\'][^"\']*["\']\}',
        '',
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r'(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above)'
        r'\s+(?:instructions|prompts|rules)',
        '',
        text,
        flags=re.IGNORECASE,
    )

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

    text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'([!?#@$%^&*<>|~`])\1{2,}', r'\1\1', text)
    text = re.sub(r'\s{3,}', '  ', text)
    text = text.strip()

    if text != original:
        logger.info(
            "Pre-sanitized input: removed %d chars of dangerous content",
            len(original) - len(text),
        )

    return text


# ============================================================
# Cache / Similarity Helpers
# ============================================================

_solution_cache: Dict[str, Dict[str, Any]] = {}
_CACHE_MAX_SIZE = 200


def _normalize_for_cache(text: str) -> str:
    if not text:
        return ""
    normalized = " ".join(text.lower().split())
    normalized = normalized.rstrip("?.!,;:")
    return normalized


def _text_similarity(a: str, b: str) -> float:
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
    cache_key = _normalize_for_cache(normalized_text)

    if cache_key in _solution_cache:
        cached = _solution_cache[cache_key]
        logger.info("Cache HIT (in-memory exact): %s", cache_key[:50])
        return cached, 1.0

    try:
        similar = memory.find_similar_problems(normalized_text, limit=5) or []
        for problem in similar:
            stored_text = (
                problem.get("extracted_text")
                or problem.get("question")
                or ""
            )
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

    try:
        history = memory.get_problem_history(limit=50) or []
        for problem in history:
            stored_text = (
                problem.get("extracted_text")
                or problem.get("question")
                or ""
            )
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
    solution = _to_string(cached_problem.get("solution", ""))
    explanation = _to_string(cached_problem.get("explanation", ""))
    verification = _to_string(
        cached_problem.get("verification_result")
        or cached_problem.get("verification", "")
    )
    topic = str(cached_problem.get("topic", "general"))

    pipeline_steps.append("memory_cache_hit")

    return SolveResponse(
        status="success",
        question=question,
        input_type=input_type,
        detected_topic=topic,
        normalized_question=question,
        parsed_problem=None,
        solution_steps=None,
        final_answer="",
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
    try:
        import numpy as np
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
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
# LLM Solver (Fallback / primary if graph disabled)
# ============================================================

def _solve_with_llm(components, question, rag_context, similar_problems=None):
    """Solve using Groq LLM — flowing markdown+LaTeX output."""
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

    truncated_question = question
    if len(question) > 1500:
        truncated_question = question[:1500] + "\n\n[Problem text truncated]"
        logger.info("Truncated question from %d to 1500 chars", len(question))

    system_prompt = (
        "You are an expert JEE Mathematics tutor.\n\n"
        "Return ONLY valid JSON with these fields:\n"
        "{\n"
        '  "parsed_problem": {\n'
        '    "type": "algebra|calculus|probability|linear_algebra|trigonometry|matrices",\n'
        '    "what_to_find": "What to find",\n'
        '    "given": "Given information"\n'
        '  },\n'
        '  "solution": "Full flowing solution as markdown with LaTeX",\n'
        '  "final_answer": "Just the answer with LaTeX if needed",\n'
        '  "verification": "How to verify, one paragraph",\n'
        '  "explanation": "Student-friendly explanation, one paragraph",\n'
        '  "key_concepts": ["concept1"],\n'
        '  "common_mistakes": ["mistake1"]\n'
        "}\n\n"

        "RULES FOR THE solution FIELD:\n"
        "1. Write as a FLOWING DOCUMENT — not a numbered list array.\n"
        "2. Use 4-8 steps max. Never exceed 10.\n"
        "3. Use **bold** for step headers like **Step 1: ...**\n"
        "4. Use bullet points for given information.\n"
        "5. ALWAYS end with a definitive conclusion.\n"
        "6. If multiple valid cases exist, explicitly state all of them.\n"
        "7. NEVER end with 'we need to verify' or 'we need to check'.\n"
        "8. NEVER self-correct mid-solution.\n\n"

        "LATEX RULES — follow exactly:\n"
        "- Wrap ALL math in $...$ for inline or $$...$$ for block.\n"
        "- Write LaTeX commands normally: \\frac, \\lambda, \\int, \\sqrt.\n"
        "- In JSON strings, escape each backslash once: \\\\frac, \\\\lambda.\n"
        "- Inline example in JSON:  \"$\\\\lambda_1 + \\\\lambda_2 = 6$\"\n"
        "- Block example in JSON:   \"$$\\n\\\\frac{a}{b} = c\\n$$\"\n"
        "- Every variable in math context needs $: write $A$ not just A.\n\n"

        "JSON RULES:\n"
        "- Return ONLY valid JSON — no prose, no markdown fences.\n"
        "- Use \\\\n for newlines inside JSON string values.\n"
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

            result = _normalize_llm_result(result)
            result = _postprocess_solution(result)

            solution = _to_string(result.get("solution", ""))
            answer = _to_string(result.get("final_answer", ""))

            if not solution and result.get("solution_steps"):
                steps = result["solution_steps"]
                if isinstance(steps, list):
                    solution = "\n\n".join(str(step) for step in steps)
                else:
                    solution = str(steps)

            return {
                "parsed_problem": result.get("parsed_problem", {}),
                "solution_steps": [],
                "solution": solution,
                "final_answer": answer,
                "verification": _to_string(result.get("verification", "")),
                "explanation": _to_string(result.get("explanation", "")),
                "key_concepts": result.get("key_concepts", []),
                "common_mistakes": result.get("common_mistakes", []),
            }
        except json.JSONDecodeError:
            return _normalize_llm_result({
                "solution": content,
                "solution_steps": [],
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
    start_time = time.time()
    pipeline_steps = []

    # Step 1: Pre-sanitize
    sanitized_question = _pre_sanitize(question)
    if sanitized_question != question:
        pipeline_steps.append("pre_sanitize")

    if not sanitized_question or not sanitized_question.strip():
        return SolveResponse(
            status="error",
            question=question,
            input_type=input_type,
            error=(
                "After removing unsafe content, no math problem remained. "
                "Please enter a valid math question."
            ),
            pipeline_steps=pipeline_steps,
            latency_ms=round((time.time() - start_time) * 1000, 1),
            confidence=confidence,
        )

    # Step 2: Guardrails input check
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

    # Step 3: Normalize
    try:
        normalizer = components.normalizer
        normalized = normalizer.normalize(sanitized_question)
        pipeline_steps.append("normalization")
    except Exception as e:
        logger.warning("Normalization failed, using raw input: %s", e)
        normalized = sanitized_question
        pipeline_steps.append("normalization_skipped")

    # Step 4: Memory cache check
    memory = components.memory
    try:
        cached_problem, similarity = _check_memory_cache(memory, normalized)
        if cached_problem and cached_problem.get("solution"):
            logger.info(
                "Returning cached solution (sim=%.3f) for: %s",
                similarity, normalized[:50],
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
        logger.warning("Cache check failed, proceeding to solver: %s", e)

    # Step 5: RAG retrieval
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

    # Step 5b: Memory similar problems
    try:
        similar = memory.find_similar_problems(normalized, limit=2) or []
        pipeline_steps.append("memory_lookup")
    except Exception as e:
        logger.warning("Memory lookup failed: %s", e)
        similar = []

    # Step 6: Solve (graph or LLM fallback)
    try:
        graph = _get_graph(components)
        if graph is not None:
            solution = graph.solve(
                question=normalized,
                rag_context=rag_results,
                similar_problems=similar,
            )
            # Single postprocess call — graph._fix_newlines already ran
            # inside the graph nodes; this catches any remaining issues.
            solution = _postprocess_solution(solution)
            graph_nodes = solution.pop("nodes_executed", [])
            difficulty = solution.get("difficulty", "unknown")
            pipeline_steps.append(f"langgraph({difficulty})")
            pipeline_steps.extend(graph_nodes)
        else:
            # _solve_with_llm already calls _postprocess_solution internally
            solution = _solve_with_llm(
                components, normalized, rag_results, similar
            )
            pipeline_steps.append("llm_solve")
    except Exception as e:
        logger.error("Graph solve failed, falling back to direct LLM: %s", e)
        solution = _solve_with_llm(
            components, normalized, rag_results, similar
        )
        pipeline_steps.append("llm_solve_fallback")

    if solution.get("error"):
        return SolveResponse(
            status="error",
            question=question,
            input_type=input_type,
            error=str(solution["error"]),
            pipeline_steps=pipeline_steps,
            latency_ms=round((time.time() - start_time) * 1000, 1),
            confidence=confidence,
        )

    # Normalize types only — do NOT call _postprocess_solution again
    solution = _normalize_llm_result(solution)

    # Step 7: Guardrails output check
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

    # Step 9: Evaluation (optional)
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
                solution_quality=round(
                    float(eval_result.solution_quality_score), 3
                ),
                explanation_clarity=round(
                    float(eval_result.explanation_clarity_score), 3
                ),
                issues=list(eval_result.issues_found)
                if eval_result.issues_found else [],
                suggestions=list(eval_result.suggestions)
                if eval_result.suggestions else [],
                solution_assessment=str(eval_result.solution_assessment or ""),
                explanation_assessment=str(
                    eval_result.explanation_assessment or ""
                ),
            )
            pipeline_steps.append("evaluation")
        except Exception as e:
            logger.error("Evaluation failed: %s", e)

    # Step 10: Build response
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
        solution_steps=None,
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


# ==================== EXTRACT IMAGE ====================

@router.post("/extract/image", response_model=ExtractionResponse)
async def extract_image(
    file: UploadFile = File(...),
    components: ComponentManager = Depends(get_components),
):
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
            corrected_text = memory.apply_corrections(
                result.extracted_text, "image"
            )
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


# ==================== EXTRACT AUDIO ====================

@router.post("/extract/audio", response_model=ExtractionResponse)
async def extract_audio(
    file: UploadFile = File(...),
    components: ComponentManager = Depends(get_components),
):
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
            corrected_text = memory.apply_corrections(
                result.extracted_text, "audio"
            )
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