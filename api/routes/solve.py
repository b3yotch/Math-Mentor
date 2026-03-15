"""Solve math problems — text, image, and audio input."""

import os
import time
import json
import logging
import tempfile
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from typing import Optional
from api.schemas import (
    SolveRequest, SolveResponse, RAGSource, EvaluationResult,
    ParsedProblem, ExtractionResponse,
)
from api.dependencies import get_components, ComponentManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/solve", tags=["Solve"])


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
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(i) for i in obj]
    return obj


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
                q=p.get("extracted_text", ""), s=p.get("solution", "")
            )
            for p in similar_problems[:2]
        ])

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
        '    "Step 1: Identify the problem type. This is a quadratic equation...",\n'
        '    "Step 2: Apply the appropriate method...",\n'
        '    "Step 3: Calculate...",\n'
        '    "Step 4: Simplify to get the final result..."\n'
        '  ],\n'
        '  "final_answer": "x = 2 or x = 3",\n'
        '  "verification": "Substitute back: 2^2 - 5(2) + 6 = 4 - 10 + 6 = 0. Correct.",\n'
        '  "explanation": "This is a quadratic equation that can be factored. We look for two numbers that multiply to 6 and add to -5...",\n'
        '  "key_concepts": ["Quadratic equations", "Factoring", "Zero product property"],\n'
        '  "common_mistakes": ["Forgetting to check both roots", "Sign errors when factoring"]\n'
        "}\n\n"
        "IMPORTANT:\n"
        "- Each solution_step should be a complete sentence explaining what you are doing and why\n"
        "- Show all mathematical work within the steps\n"
        "- The explanation should be student-friendly\n"
        "- key_concepts lists the math topics used\n"
        "- common_mistakes warns about typical errors\n"
        "- Return ONLY valid JSON, no other text"
    )

    user_prompt = (
        "Solve: {q}\n\nContext:\n{ctx}{sim}\n\nReturn valid JSON only."
    ).format(q=question, ctx=context_text, sim=similar_text)

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=2500,
        )

        content = response.choices[0].message.content

        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            result = json.loads(content)

            steps = result.get("solution_steps", ["No steps provided"])
            answer = result.get("final_answer", "")
            solution_text = "\n".join(
                "{}. {}".format(i + 1, step) if not step.startswith("Step") else step
                for i, step in enumerate(steps)
            )

            return {
                "parsed_problem": result.get("parsed_problem", {}),
                "solution_steps": steps,
                "solution": "{s}\n\nFinal Answer: {a}".format(s=solution_text, a=answer),
                "final_answer": answer,
                "verification": result.get("verification", ""),
                "explanation": result.get("explanation", ""),
                "key_concepts": result.get("key_concepts", []),
                "common_mistakes": result.get("common_mistakes", []),
            }
        except json.JSONDecodeError:
            return {
                "solution": content,
                "solution_steps": [content],
                "final_answer": "",
                "verification": "",
                "explanation": content,
                "key_concepts": [],
                "common_mistakes": [],
            }

    except Exception as e:
        return {"error": str(e)}


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
    """Core solve pipeline used by text, image, and audio endpoints."""
    start_time = time.time()
    pipeline_steps = []

    # Step 1: Guardrails
    guardrails = components.guardrails
    input_report = guardrails.check_input(question)
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

    # Step 2: Normalize
    normalizer = components.normalizer
    normalized = normalizer.normalize(question)
    pipeline_steps.append("normalization")

    # Step 3: RAG
    retriever = components.retriever
    top_k = max(1, min(5, top_k))
    rag_results = retriever.retrieve(normalized, top_k=top_k, min_score=0.2, rerank=True) or []
    rag_serializable = _make_serializable(rag_results)
    pipeline_steps.append("rag_retrieval")

    rag_sources = [
        RAGSource(
            topic=r.get("topic", ""),
            subtopic=r.get("subtopic", ""),
            score=float(r.get("score", 0)),
            content_preview=str(r.get("content", ""))[:200],
        )
        for r in rag_serializable[:top_k]
    ]

    # Step 4: Memory lookup
    memory = components.memory
    similar = memory.find_similar_problems(normalized, limit=2) or []
    pipeline_steps.append("memory_lookup")

    # Step 5: LLM Solve
    solution = _solve_with_llm(components, normalized, rag_results, similar)
    pipeline_steps.append("llm_solve")

    if solution.get("error"):
        return SolveResponse(
            status="error",
            question=question,
            input_type=input_type,
            error=solution["error"],
            pipeline_steps=pipeline_steps,
            latency_ms=round((time.time() - start_time) * 1000, 1),
            confidence=confidence,
        )

    # Step 6: Output guardrails
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

    # Step 7: Save to memory (with user_id)
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
    except Exception as e:
        # Try without user_id if the memory manager doesn't support it yet
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

    # Step 8: Evaluate
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
                overall_score=round(eval_result.overall_score, 3),
                grade=eval_result.grade,
                passed=eval_result.passed,
                rag_relevance=round(eval_result.rag_relevance_score, 3),
                solution_quality=round(eval_result.solution_quality_score, 3),
                explanation_clarity=round(eval_result.explanation_clarity_score, 3),
                issues=eval_result.issues_found,
                suggestions=eval_result.suggestions,
                solution_assessment=eval_result.solution_assessment,
                explanation_assessment=eval_result.explanation_assessment,
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
        input_type="text",
        user_id=request.user_id or "default",
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
    """Solve a math problem from an uploaded image (OCR)."""
    # Validate file type
    allowed = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(400, "Invalid file type. Upload JPG, PNG, or WebP.")

    # Save temp file
    suffix = os.path.splitext(file.filename or "image.png")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # OCR
        from src.input_processing.ocr_processor import OCRProcessor
        ocr = OCRProcessor()
        result = ocr.process(tmp_path)

        extracted_text = result.extracted_text
        confidence = result.confidence_score

        if not extracted_text or not extracted_text.strip():
            return SolveResponse(
                status="error",
                question="",
                input_type="image",
                error="Could not extract text from image. Please upload a clearer image.",
                pipeline_steps=["ocr_failed"],
                confidence=confidence,
            )

        # Apply learned corrections
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
            confidence=confidence,
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
    """Solve a math problem from an uploaded audio file (ASR)."""
    allowed = {
        "audio/wav", "audio/mpeg", "audio/mp3", "audio/ogg",
        "audio/flac", "audio/webm", "audio/mp4", "audio/x-wav",
    }
    if file.content_type and file.content_type not in allowed:
        raise HTTPException(400, "Invalid audio format. Upload WAV, MP3, OGG, or FLAC.")

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
        confidence = result.confidence_score

        if not extracted_text or not extracted_text.strip():
            return SolveResponse(
                status="error",
                question="",
                input_type="audio",
                error="Could not transcribe audio. Please speak clearly and try again.",
                pipeline_steps=["asr_failed"],
                confidence=confidence,
            )

        # Apply learned corrections
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
            confidence=confidence,
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


# ==================== EXTRACT ONLY (for HITL preview) ====================
@router.post("/extract/image", response_model=ExtractionResponse)
async def extract_image(
    file: UploadFile = File(...),
    components: ComponentManager = Depends(get_components),
):
    """Extract text from image without solving (for HITL review)."""
    suffix = os.path.splitext(file.filename or "image.png")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from src.input_processing.ocr_processor import OCRProcessor
        ocr = OCRProcessor()
        result = ocr.process(tmp_path)

        return ExtractionResponse(
            extracted_text=result.extracted_text,
            confidence=result.confidence_score,
            input_type="image",
            metadata=result.metadata or {},
        )
    finally:
        os.unlink(tmp_path)