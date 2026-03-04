"""
Evaluator Agent - Per-query evaluation after processing.
Runs automatically after each query is solved to assess quality.
"""

import os
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class QueryEvaluation:
    """Evaluation result for a single query."""
    query_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Scores (0.0 to 1.0)
    rag_relevance_score: float = 0.0
    solution_quality_score: float = 0.0
    explanation_clarity_score: float = 0.0
    overall_score: float = 0.0
    
    # Details
    rag_assessment: str = ""
    solution_assessment: str = ""
    explanation_assessment: str = ""
    issues_found: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    # Metadata
    input_type: str = ""
    topic: str = ""
    latency_ms: float = 0.0
    was_human_edited: bool = False
    confidence: float = 0.0
    
    # LLM judge raw response
    raw_judge_response: str = ""
    
    @property
    def grade(self) -> str:
        s = self.overall_score * 100
        if s >= 90: return "A"
        elif s >= 80: return "B"
        elif s >= 70: return "C"
        elif s >= 60: return "D"
        else: return "F"
    
    @property
    def passed(self) -> bool:
        return self.overall_score >= 0.6
    
    def to_dict(self) -> Dict:
        return {
            "query_id": self.query_id,
            "timestamp": self.timestamp.isoformat(),
            "scores": {
                "rag_relevance": round(self.rag_relevance_score, 3),
                "solution_quality": round(self.solution_quality_score, 3),
                "explanation_clarity": round(self.explanation_clarity_score, 3),
                "overall": round(self.overall_score, 3),
            },
            "grade": self.grade,
            "passed": self.passed,
            "assessments": {
                "rag": self.rag_assessment,
                "solution": self.solution_assessment,
                "explanation": self.explanation_assessment,
            },
            "issues": self.issues_found,
            "suggestions": self.suggestions,
            "metadata": {
                "input_type": self.input_type,
                "topic": self.topic,
                "latency_ms": self.latency_ms,
                "was_human_edited": self.was_human_edited,
                "confidence": self.confidence,
            },
        }


class EvaluatorAgent:
    """
    Evaluates each query AFTER it has been processed.
    
    Uses LLM-as-judge to assess:
    1. RAG relevance - Were retrieved docs useful?
    2. Solution quality - Is the math correct and complete?
    3. Explanation clarity - Is it understandable for a student?
    
    No ground truth needed — uses the LLM's own judgment.
    """
    
    def __init__(self):
        self._groq_client = None
        self._eval_history: List[QueryEvaluation] = []
    
    @property
    def groq_client(self):
        if self._groq_client is None:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            except Exception as e:
                logger.error(f"Could not load Groq for evaluator: {e}")
        return self._groq_client
    
    def evaluate_query(
        self,
        question: str,
        solution: str,
        explanation: str,
        verification: str,
        rag_context: List[Dict],
        input_type: str = "text",
        confidence: float = 1.0,
        was_human_edited: bool = False,
        latency_ms: float = 0.0,
        topic: str = "",
        query_id: str = None,
    ) -> QueryEvaluation:
        """
        Evaluate a completed query end-to-end.
        
        Args:
            question: The math question that was asked
            solution: The generated solution
            explanation: The generated explanation
            verification: The verification text
            rag_context: Retrieved documents used
            input_type: text/image/audio
            confidence: Extraction confidence
            was_human_edited: Whether HITL edited the input
            latency_ms: Total processing time
            topic: Detected math topic
            query_id: Problem ID from memory
        
        Returns:
            QueryEvaluation with all scores and assessments
        """
        if query_id is None:
            query_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        evaluation = QueryEvaluation(
            query_id=query_id,
            input_type=input_type,
            topic=topic,
            latency_ms=latency_ms,
            was_human_edited=was_human_edited,
            confidence=confidence,
        )
        
        # ---- 1. Evaluate RAG Relevance (local heuristic) ----
        evaluation.rag_relevance_score, evaluation.rag_assessment = (
            self._evaluate_rag_relevance(question, rag_context)
        )
        
        # ---- 2. Evaluate Solution + Explanation (LLM judge) ----
        llm_scores = self._llm_judge(
            question, solution, explanation, verification, rag_context
        )
        
        evaluation.solution_quality_score = llm_scores.get("solution_quality", 0.0)
        evaluation.explanation_clarity_score = llm_scores.get("explanation_clarity", 0.0)
        evaluation.solution_assessment = llm_scores.get("solution_assessment", "")
        evaluation.explanation_assessment = llm_scores.get("explanation_assessment", "")
        evaluation.issues_found = llm_scores.get("issues", [])
        evaluation.suggestions = llm_scores.get("suggestions", [])
        evaluation.raw_judge_response = llm_scores.get("raw_response", "")
        
        # ---- 3. Calculate Overall Score ----
        # Weighted average: solution matters most
        weights = {
            "rag": 0.2,
            "solution": 0.5,
            "explanation": 0.3,
        }
        evaluation.overall_score = (
            weights["rag"] * evaluation.rag_relevance_score
            + weights["solution"] * evaluation.solution_quality_score
            + weights["explanation"] * evaluation.explanation_clarity_score
        )
        
        # ---- 4. Add contextual issues ----
        if confidence < 0.5:
            evaluation.issues_found.append(
                f"Low input confidence ({confidence:.0%}) - extraction may be unreliable"
            )
        
        if not rag_context:
            evaluation.issues_found.append("No RAG context retrieved")
        
        if was_human_edited:
            evaluation.suggestions.append(
                "Input required human editing - extraction quality may need improvement"
            )
        
        # ---- 5. Store in history ----
        self._eval_history.append(evaluation)
        
        logger.info(
            f"Query {query_id} evaluated: "
            f"overall={evaluation.overall_score:.2f} grade={evaluation.grade}"
        )
        
        return evaluation
    
    # ----------------------------------------------------------------
    # RAG Relevance (heuristic — no LLM call needed)
    # ----------------------------------------------------------------
    def _evaluate_rag_relevance(
        self, question: str, rag_context: List[Dict]
    ) -> tuple:
        """Evaluate RAG retrieval quality using heuristics."""
        
        if not rag_context:
            return 0.0, "No documents retrieved"
        
        question_lower = question.lower()
        question_words = set(question_lower.split())
        
        # Remove stopwords
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "find",
            "solve", "what", "how", "of", "in", "for", "to", "and",
            "or", "if", "it", "this", "that", "with",
        }
        question_keywords = question_words - stopwords
        
        scores = []
        assessments = []
        
        for i, doc in enumerate(rag_context):
            content = str(doc.get("content", "")).lower()
            topic = str(doc.get("topic", "")).lower()
            score = float(doc.get("score", 0))
            
            # Keyword overlap
            if question_keywords:
                overlap = sum(1 for kw in question_keywords if kw in content)
                keyword_score = overlap / len(question_keywords)
            else:
                keyword_score = 0.0
            
            # Retrieval score (already computed by retriever)
            retrieval_score = min(score, 1.0)
            
            # Combined
            doc_score = 0.5 * keyword_score + 0.5 * retrieval_score
            scores.append(doc_score)
            
            relevance = "relevant" if doc_score > 0.4 else "partially relevant" if doc_score > 0.2 else "low relevance"
            assessments.append(f"Doc {i+1} ({topic}): {relevance} ({doc_score:.2f})")
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Bonus if top result is highly relevant
        top_score = scores[0] if scores else 0.0
        final_score = min(0.6 * avg_score + 0.4 * top_score, 1.0)
        
        assessment = f"Retrieved {len(rag_context)} docs. " + "; ".join(assessments[:3])
        
        return final_score, assessment
    
    # ----------------------------------------------------------------
    # LLM Judge (single call evaluates solution + explanation)
    # ----------------------------------------------------------------
    def _llm_judge(
        self,
        question: str,
        solution: str,
        explanation: str,
        verification: str,
        rag_context: List[Dict],
    ) -> Dict:
        """Use LLM to judge solution quality and explanation clarity."""
        
        if self.groq_client is None:
            return self._fallback_judge(solution, explanation)
        
        context_summary = "\n".join(
            f"- {doc.get('topic', '?')}: {doc.get('content', '')[:200]}"
            for doc in rag_context[:3]
        ) if rag_context else "No context retrieved."
        
        judge_prompt = f"""You are a strict math teacher evaluating an AI tutor's response.

QUESTION: {question}

RETRIEVED CONTEXT (used by the AI):
{context_summary}

AI'S SOLUTION:
{solution[:1500]}

AI'S EXPLANATION:
{explanation[:500]}

AI'S VERIFICATION:
{verification[:300]}

Evaluate and respond in EXACTLY this JSON format:
{{
    "solution_quality": <float 0.0-1.0>,
    "solution_assessment": "<one sentence: is the math correct? are steps complete?>",
    "explanation_clarity": <float 0.0-1.0>,
    "explanation_assessment": "<one sentence: is it clear for a JEE student?>",
    "issues": ["<issue 1>", "<issue 2>"],
    "suggestions": ["<suggestion 1>", "<suggestion 2>"]
}}

Scoring guide:
- 1.0 = Perfect, mathematically rigorous, complete
- 0.8 = Correct answer, minor presentation issues
- 0.6 = Mostly correct, some steps unclear or missing
- 0.4 = Partially correct, significant gaps
- 0.2 = Major errors or mostly wrong
- 0.0 = Completely wrong or no real solution

Be strict but fair. JSON only, no other text."""

        try:
            response = self.groq_client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a math evaluation judge. Respond ONLY with valid JSON.",
                    },
                    {"role": "user", "content": judge_prompt},
                ],
                temperature=0.0,
                max_tokens=500,
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            import json
            result = json.loads(content)
            
            return {
                "solution_quality": float(result.get("solution_quality", 0.5)),
                "explanation_clarity": float(result.get("explanation_clarity", 0.5)),
                "solution_assessment": result.get("solution_assessment", ""),
                "explanation_assessment": result.get("explanation_assessment", ""),
                "issues": result.get("issues", []),
                "suggestions": result.get("suggestions", []),
                "raw_response": content,
            }
        
        except Exception as e:
            logger.error(f"LLM judge failed: {e}")
            return self._fallback_judge(solution, explanation)
    
    def _fallback_judge(self, solution: str, explanation: str) -> Dict:
        """Fallback heuristic when LLM judge is unavailable."""
        
        # Basic heuristics
        sol_score = 0.5
        exp_score = 0.5
        issues = []
        
        if not solution or solution.strip() == "":
            sol_score = 0.0
            issues.append("No solution provided")
        else:
            # Check for step-by-step indicators
            step_indicators = ["step", "first", "then", "therefore", "thus", "hence", "so,"]
            has_steps = any(ind in solution.lower() for ind in step_indicators)
            if has_steps:
                sol_score += 0.2
            
            # Check for final answer
            if "final answer" in solution.lower() or "answer:" in solution.lower():
                sol_score += 0.1
            
            # Check length (very short = probably incomplete)
            if len(solution) < 50:
                sol_score -= 0.2
                issues.append("Solution is very short")
            elif len(solution) > 200:
                sol_score += 0.1
        
        if not explanation or explanation.strip() == "":
            exp_score = 0.0
            issues.append("No explanation provided")
        else:
            if len(explanation) > 100:
                exp_score += 0.2
        
        sol_score = max(0.0, min(1.0, sol_score))
        exp_score = max(0.0, min(1.0, exp_score))
        
        return {
            "solution_quality": sol_score,
            "explanation_clarity": exp_score,
            "solution_assessment": "Evaluated by heuristic (LLM unavailable)",
            "explanation_assessment": "Evaluated by heuristic (LLM unavailable)",
            "issues": issues,
            "suggestions": ["LLM judge unavailable - scores are approximate"],
            "raw_response": "",
        }
    
    # ----------------------------------------------------------------
    # History & Statistics
    # ----------------------------------------------------------------
    def get_history(self, limit: int = 50) -> List[QueryEvaluation]:
        """Get recent evaluation history."""
        return self._eval_history[-limit:]
    
    def get_statistics(self) -> Dict:
        """Get aggregate statistics across all evaluations."""
        if not self._eval_history:
            return {
                "total_evaluated": 0,
                "avg_overall_score": 0.0,
                "avg_rag_score": 0.0,
                "avg_solution_score": 0.0,
                "avg_explanation_score": 0.0,
                "pass_rate": 0.0,
                "grade_distribution": {},
                "by_topic": {},
                "by_input_type": {},
            }
        
        evals = self._eval_history
        total = len(evals)
        
        grade_dist = {}
        topic_scores = {}
        input_type_scores = {}
        
        for e in evals:
            # Grade distribution
            grade_dist[e.grade] = grade_dist.get(e.grade, 0) + 1
            
            # By topic
            t = e.topic or "unknown"
            if t not in topic_scores:
                topic_scores[t] = []
            topic_scores[t].append(e.overall_score)
            
            # By input type
            it = e.input_type or "unknown"
            if it not in input_type_scores:
                input_type_scores[it] = []
            input_type_scores[it].append(e.overall_score)
        
        return {
            "total_evaluated": total,
            "avg_overall_score": sum(e.overall_score for e in evals) / total,
            "avg_rag_score": sum(e.rag_relevance_score for e in evals) / total,
            "avg_solution_score": sum(e.solution_quality_score for e in evals) / total,
            "avg_explanation_score": sum(e.explanation_clarity_score for e in evals) / total,
            "pass_rate": sum(1 for e in evals if e.passed) / total,
            "grade_distribution": grade_dist,
            "by_topic": {
                t: sum(s) / len(s) for t, s in topic_scores.items()
            },
            "by_input_type": {
                it: sum(s) / len(s) for it, s in input_type_scores.items()
            },
        }


# Singleton
_evaluator_agent = None


def get_evaluator_agent() -> EvaluatorAgent:
    global _evaluator_agent
    if _evaluator_agent is None:
        _evaluator_agent = EvaluatorAgent()
    return _evaluator_agent