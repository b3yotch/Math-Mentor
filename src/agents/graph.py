# src/agents/graph.py
"""
LangGraph-based HYBRID math-solving pipeline.

Architecture:
  - Classifier: Rule-based for obvious cases, LLM for ambiguous
  - Simple/Medium: Single focused LLM call (1 call, fast)
  - Complex: Full multi-agent pipeline (5 calls, thorough)

Total LLM calls:
  - Simple:  1 (classify=rule) + 1 (fast_solve) = 1 call
  - Medium:  1 (classify=LLM)  + 1 (fast_solve) = 2 calls
  - Complex: 1 (classify=LLM)  + 5 (full pipe)  = 6 calls
"""

import json
import logging
import re
from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


# ================================================================
# State
# ================================================================

class MathState(TypedDict, total=False):
    """State flowing through the pipeline."""
    # Inputs
    question: str
    rag_context: list
    similar_problems: list

    # Classifier
    difficulty: str           # "simple" | "medium" | "complex"
    detected_topic: str
    classification_method: str  # "rule" | "llm"

    # Parser (complex only)
    parsed_problem: dict

    # Strategy (complex only)
    strategy: dict

    # Solver
    solution_steps: list
    final_answer: str
    solution: str

    # Verifier (complex only)
    verification: str
    is_correct: bool
    verification_confidence: float

    # Explainer
    explanation: str
    key_concepts: list
    common_mistakes: list

    # Metadata
    nodes_executed: list
    error: str


# ================================================================
# Rule-Based Pre-Filter (catches ONLY the obvious cases)
# ================================================================

_OBVIOUSLY_SIMPLE = [
    # Pure arithmetic: 2+3, 15*4, 100/5
    r"^\s*[\d\.\s\+\-\*/\(\)\^]+\s*$",
    # "What is X op Y"
    r"^what\s+is\s+\d+\s*[\+\-\*/]\s*\d+",
    # Direct compute: "5 factorial", "10C3"
    r"^\d+\s*!$",
    r"^(?:find\s+)?\d+[CcPp]\d+$",
]

_OBVIOUSLY_COMPLEX = [
    "prove", "proof", "show that", "by induction",
    "differential equation",
    "multiple integrals", "double integral", "triple integral",
    "eigenvalue", "eigenvector", "diagonalize",
    "taylor series", "maclaurin", "fourier",
    "jee advanced", "olympiad",
    "if and only if", "necessary and sufficient",
]


def _try_rule_classify(question: str) -> Optional[str]:
    """
    Only returns a result for OBVIOUS cases.
    Returns None when uncertain → LLM decides.
    """
    q_lower = question.lower().strip()
    word_count = len(q_lower.split())

    # Obviously simple
    for pattern in _OBVIOUSLY_SIMPLE:
        if re.match(pattern, q_lower):
            return "simple"

    # Very short + no complexity markers
    if word_count <= 5:
        return "simple"

    # Obviously complex (2+ indicators or strong single indicator)
    complex_hits = [ind for ind in _OBVIOUSLY_COMPLEX if ind in q_lower]
    if len(complex_hits) >= 2:
        return "complex"
    if complex_hits and word_count > 25:
        return "complex"

    # Uncertain — let LLM decide
    return None


# ================================================================
# Graph
# ================================================================

class MathMentorGraph:
    """
    Hybrid adaptive pipeline with LLM-powered classification.

    Design decisions:
    1. Classifier uses rules first (free), LLM only when uncertain
    2. Simple/Medium use a single comprehensive LLM call (no regression)
    3. Complex problems get full multi-agent decomposition
    4. _solve_with_llm is kept as fallback if graph fails

    LLM calls per difficulty:
      Simple:  0 (classify) + 1 (fast_solve) = 1 total
      Medium:  1 (classify) + 1 (fast_solve) = 2 total
      Complex: 1 (classify) + 5 (pipeline)   = 6 total
    """

    def __init__(
        self,
        groq_client,
        model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    ):
        self.client = groq_client
        self.model = model
        self.compiled_graph = self._build_graph()

    # ──────────────────────────────
    # LLM Helper
    # ──────────────────────────────

    def _llm_call(
        self, system: str, user: str,
        max_tokens: int = 1500, temperature: float = 0.1,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    @staticmethod
    def _parse_json(text: str) -> dict:
        """
        Robustly extract JSON from LLM response.
        
        Handles the critical case where LaTeX backslashes (\frac, \sqrt,
        \pi, \int, \sin, etc.) corrupt JSON parsing because \f, \n, \t, \b
        are valid JSON escape sequences.
        """
        if not text:
            return {}

        # Step 1: Strip markdown code fences
        cleaned = text
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0]
        cleaned = cleaned.strip()

        # Step 2: Try direct parse first (fast path)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Step 3: Fix LaTeX backslashes that break JSON
        # \f (form feed), \b (backspace), \t (tab), \n (newline),
        # \r (carriage return) are valid JSON escapes.
        # But in LaTeX: \frac, \binom, \theta, \neq, \right are NOT
        # meant as JSON escapes. Fix them by double-escaping.
        latex_commands = [
            # \f... commands (form feed collision)
            r'\\frac', r'\\forall',
            # \b... commands (backspace collision)
            r'\\binom', r'\\beta', r'\\begin', r'\\bar', r'\\bf',
            r'\\boxed', r'\\big', r'\\Big',
            # \t... commands (tab collision)
            r'\\theta', r'\\tan', r'\\text', r'\\times', r'\\to',
            r'\\triangle', r'\\therefore',
            # \n... commands (newline collision)
            r'\\neq', r'\\not', r'\\nu', r'\\nabla', r'\\nfrac',
            r'\\newline', r'\\neg',
            # \r... commands (carriage return collision)
            r'\\right', r'\\rangle', r'\\rho', r'\\rightarrow',
            r'\\Rightarrow', r'\\rceil',
            # Other common LaTeX (safe but handle for completeness)
            r'\\sqrt', r'\\sum', r'\\sin', r'\\cos', r'\\log',
            r'\\ln', r'\\lim', r'\\int', r'\\infty', r'\\pi',
            r'\\alpha', r'\\gamma', r'\\delta', r'\\epsilon',
            r'\\lambda', r'\\mu', r'\\sigma', r'\\omega', r'\\phi',
            r'\\partial', r'\\cdot', r'\\ldots', r'\\geq', r'\\leq',
            r'\\pm', r'\\mp', r'\\div', r'\\approx', r'\\equiv',
            r'\\left', r'\\langle', r'\\lceil', r'\\lfloor',
            r'\\rfloor', r'\\mathrm', r'\\mathbf', r'\\overline',
            r'\\underline', r'\\hat', r'\\vec', r'\\dot',
        ]

        fixed = cleaned
        for cmd in latex_commands:
            # cmd is like r'\\frac' — we want to find \frac in the text
            # and ensure it becomes \\frac in JSON (double escaped)
            plain = cmd[1:]  # e.g., r'\frac'
            escaped = cmd     # e.g., r'\\frac'
            # Only fix if it's a single backslash (not already double)
            fixed = fixed.replace(plain, escaped)

        # Also fix any remaining single backslashes before letters
        # that aren't valid JSON escapes (\", \\, \/, \b, \f, \n, \r, \t, \u)
        fixed = re.sub(
            r'\\(?!["\\/bfnrtu\\])',
            r'\\\\',
            fixed,
        )

        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # Step 4: Try to extract JSON object from text
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            extracted = match.group()
            # Apply same LaTeX fix to extracted
            for cmd in latex_commands:
                plain = cmd[1:]
                extracted = extracted.replace(plain, cmd)
            extracted = re.sub(
                r'\\(?!["\\/bfnrtu\\])',
                r'\\\\',
                extracted,
            )
            try:
                return json.loads(extracted)
            except json.JSONDecodeError:
                pass

        # Step 5: Last resort — try to extract key-value pairs manually
        result = {}
        for field in ["explanation", "verification", "final_answer",
                       "verification_summary", "verification_method"]:
            pattern = rf'"{field}"\s*:\s*"((?:[^"\\]|\\.){{0,5000}})"'
            m = re.search(pattern, cleaned, re.DOTALL)
            if m:
                value = m.group(1)
                # Unescape JSON escapes
                value = value.replace('\\"', '"')
                value = value.replace('\\n', '\n')
                value = value.replace('\\\\', '\\')
                result[field] = value

        # Extract arrays
        for field in ["key_concepts", "common_mistakes", "solution_steps",
                       "issues", "edge_cases_checked"]:
            pattern = rf'"{field}"\s*:\s*$$([\s\S]*?)$$'
            m = re.search(pattern, cleaned)
            if m:
                items = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
                result[field] = items

        # Extract booleans
        for field in ["is_correct"]:
            pattern = rf'"{field}"\s*:\s*(true|false)'
            m = re.search(pattern, cleaned, re.IGNORECASE)
            if m:
                result[field] = m.group(1).lower() == 'true'

        # Extract numbers
        for field in ["confidence"]:
            pattern = rf'"{field}"\s*:\s*([\d.]+)'
            m = re.search(pattern, cleaned)
            if m:
                try:
                    result[field] = float(m.group(1))
                except ValueError:
                    pass

        if result:
            logger.info(
                "Extracted %d fields via regex fallback", len(result)
            )

        return result
    # ──────────────────────────────
    # NODE: Classify Difficulty (Hybrid: Rules + LLM)
    # ──────────────────────────────

    def classify_difficulty(self, state: MathState) -> dict:
        """
        Hybrid classifier:
        1. Rule-based check for obvious cases (0 LLM calls)
        2. LLM classification for ambiguous cases (1 LLM call)

        This is itself an "agent" — it makes a reasoning decision
        about how to handle the problem.
        """
        question = state["question"]
        nodes = list(state.get("nodes_executed", []))

        # ── Phase 1: Try rules (free, instant) ──
        rule_result = _try_rule_classify(question)

        if rule_result is not None:
            logger.info(
                "Classify (rule): '%s...' → %s",
                question[:50], rule_result,
            )
            nodes.append(f"classify({rule_result},rule)")
            return {
                "difficulty": rule_result,
                "detected_topic": "general",
                "classification_method": "rule",
                "nodes_executed": nodes,
            }

        # ── Phase 2: LLM classification (for ambiguous cases) ──
        system = (
            "You are a JEE math difficulty classifier.\n\n"
            "Classification criteria:\n"
            "- SIMPLE: Single-step computation, direct formula application, "
            "basic arithmetic, simple substitution. "
            "Examples: 'derivative of x^2', 'solve 2x+3=7', 'find 5C2'\n\n"
            "- MEDIUM: Multi-step problems requiring one technique, "
            "standard JEE Main level. "
            "Examples: 'solve x^2-5x+6=0', 'integrate x*sin(x) dx', "
            "'find probability of getting 3 heads in 5 tosses'\n\n"
            "- COMPLEX: Multi-concept problems, proofs, optimization, "
            "JEE Advanced level, requiring creative approaches. "
            "Examples: 'prove AM-GM inequality', "
            "'maximize area of rectangle inscribed in ellipse', "
            "'solve system of differential equations'\n\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "difficulty": "simple|medium|complex",\n'
            '  "topic": "algebra|calculus|probability|linear_algebra|'
            'trigonometry|statistics|geometry",\n'
            '  "reasoning": "one sentence why this difficulty"\n'
            "}"
        )

        content = self._llm_call(
            system, f"Classify this problem:\n{question}",
            max_tokens=200, temperature=0.0,
        )
        result = self._parse_json(content)

        difficulty = result.get("difficulty", "medium")
        if difficulty not in ("simple", "medium", "complex"):
            difficulty = "medium"

        topic = result.get("topic", "general")
        reasoning = result.get("reasoning", "")

        logger.info(
            "Classify (LLM): '%s...' → %s (%s) — %s",
            question[:50], difficulty, topic, reasoning,
        )

        nodes.append(f"classify({difficulty},llm)")
        return {
            "difficulty": difficulty,
            "detected_topic": topic,
            "classification_method": "llm",
            "nodes_executed": nodes,
        }

    # ──────────────────────────────
    # NODE: Fast Solve (simple + medium — SINGLE LLM call)
    # ──────────────────────────────

    def fast_solve(self, state: MathState) -> dict:
        question = state["question"]
        difficulty = state.get("difficulty", "medium")
        topic = state.get("detected_topic", "general")

        # Build context
        context_parts = []
        if state.get("rag_context"):
            rag_text = "\n".join(
                f"- [{d.get('topic','')}/{d.get('subtopic','')}] "
                f"{d.get('content','')[:200]}"
                for d in state["rag_context"][:3]
            )
            context_parts.append(f"Reference material:\n{rag_text}")
        if state.get("similar_problems"):
            sim_text = "\n".join(
                f"- {p.get('extracted_text','')}: "
                f"{str(p.get('solution',''))[:150]}"
                for p in state["similar_problems"][:2]
            )
            context_parts.append(f"Similar problems:\n{sim_text}")

        context = "\n\n".join(context_parts) or "No additional context."

        # ═══════════════════════════════════════════════════════
        # KEY FIX: Constrain step count and ban self-correction
        # ═══════════════════════════════════════════════════════
        if difficulty == "simple":
            step_constraint = "Use 2-4 steps MAXIMUM."
            max_tokens = 1200
        else:
            step_constraint = "Use 4-8 steps MAXIMUM. Never exceed 10."
            max_tokens = 2500

        system = (
            f"You are an expert JEE Mathematics tutor.\n"
            f"Topic: {topic}. Difficulty: {difficulty}.\n\n"
            "CRITICAL RULES:\n"
            f"1. {step_constraint}\n"
            "2. PLAN your solution BEFORE writing. Do NOT backtrack.\n"
            "3. Each step must move FORWARD. Never write 'wait' or "
            "'actually' or 'let me reconsider'.\n"
            "4. If you find an error, restart cleanly — do NOT show "
            "the wrong work.\n"
            "5. Every step must have exactly ONE clear action.\n"
            "6. Use the RAG context formulas when directly applicable.\n\n"           
            "7. Use LaTeX notation for math: $x^2$, $\\frac{a}{b}$, "
            "$\\sqrt{x}$, $\\int_0^1 f(x)dx$, $\\sum_{i=1}^{n}$\n"
            "8. Wrap ALL mathematical expressions in $ delimiters.\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "parsed_problem": {\n'
            '    "type": "problem type",\n'
            '    "what_to_find": "what to find",\n'
            '    "given": "given information"\n'
            '  },\n'
            '  "solution_steps": [\n'
            '    "Step 1: [Action] — [Result]",\n'
            '    "Step 2: [Action] — [Result]"\n'
            '  ],\n'
            '  "final_answer": "answer as string",\n'
            '  "verification": "substitute answer back to verify",\n'
            '  "explanation": "student-friendly explanation",\n'
            '  "key_concepts": ["concept1"],\n'
            '  "common_mistakes": ["mistake1"],\n'
            '  "rag_formulas_used": ["formula from context if used"]\n'
            "}\n\n"
            "BANNED PHRASES (never use these):\n"
            "- 'Wait, let me reconsider'\n"
            "- 'Actually, I made an error'\n"
            "- 'Let me recalculate'\n"
            "- 'On second thought'\n"
            "- 'I need to correct'\n\n"
            "Return ONLY valid JSON."
        )

        user = f"Solve: {question}\n\nContext:\n{context}"
        content = self._llm_call(system, user, max_tokens=max_tokens)
        result = self._parse_json(content)

        nodes = list(state.get("nodes_executed", []))
        nodes.append("fast_solve(1 call)")

        if not result:
            return {
                "parsed_problem": {},
                "solution_steps": [content],
                "final_answer": "",
                "solution": content,
                "verification": "",
                "explanation": content,
                "key_concepts": [],
                "common_mistakes": [],
                "nodes_executed": nodes,
            }

        steps = result.get("solution_steps", [])
        if isinstance(steps, str):
            steps = [steps]

        # ═══════════════════════════════════════════════════
        # SAFETY: Truncate if LLM ignored step limit
        # ═══════════════════════════════════════════════════
        max_steps = 5 if difficulty == "simple" else 10
        if len(steps) > max_steps:
            logger.warning(
                "LLM returned %d steps, truncating to %d",
                len(steps), max_steps,
            )
            steps = steps[:max_steps]

        answer = str(result.get("final_answer", ""))
        solution_text = "\n".join(
            f"{i+1}. {s}" if not s.startswith("Step") else s
            for i, s in enumerate(steps)
        )

        return {
            "parsed_problem": result.get("parsed_problem", {}),
            "solution_steps": steps,
            "final_answer": answer,
            "solution": f"{solution_text}\n\nFinal Answer: {answer}",
            "verification": str(result.get("verification", "")),
            "explanation": str(result.get("explanation", "")),
            "key_concepts": result.get("key_concepts", []),
            "common_mistakes": result.get("common_mistakes", []),
            "nodes_executed": nodes,
        }
    # ──────────────────────────────
    # NODE: Parse Problem (complex only)
    # ──────────────────────────────

    def parse_problem(self, state: MathState) -> dict:
        """
        Agent 1 (complex path): Decompose the problem.
        Extracts structure, variables, constraints, relationships.
        """
        topic = state.get("detected_topic", "general")

        system = (
            "You are an expert math problem parser specializing in "
            f"{topic} problems.\n"
            "Deeply analyze the problem structure.\n\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "type": "specific problem type",\n'
            '  "what_to_find": "precise goal",\n'
            '  "given": "all given information",\n'
            '  "variables": ["x", "y"],\n'
            '  "constraints": ["x > 0", "y is integer"],\n'
            '  "key_relationships": "core equation or relationship",\n'
            '  "sub_problems": ["sub-task 1 if decomposable"],\n'
            '  "domain_restrictions": "any domain limits"\n'
            "}\n"
            "Be thorough — missing a constraint causes wrong answers."
        )

        content = self._llm_call(
            system, f"Parse this complex problem:\n{state['question']}",
            max_tokens=600,
        )
        result = self._parse_json(content) or {
            "type": topic,
            "what_to_find": state["question"],
            "given": "",
        }

        nodes = list(state.get("nodes_executed", []))
        nodes.append("parse_problem")

        return {
            "parsed_problem": result,
            "nodes_executed": nodes,
        }

    # ──────────────────────────────
    # NODE: Route Strategy (complex only)
    # ──────────────────────────────

    def route_strategy(self, state: MathState) -> dict:
        """
        Agent 2 (complex path): Plan the solution approach.
        Considers multiple methods and picks the best one.
        """
        parsed = json.dumps(state.get("parsed_problem", {}), indent=2)

        # Include RAG context in strategy planning
        rag_hint = ""
        if state.get("rag_context"):
            rag_hint = "\n\nRelevant methods from knowledge base:\n" + "\n".join(
                f"- {d.get('content', '')[:150]}"
                for d in state["rag_context"][:3]
            )

        system = (
            "You are a senior JEE math strategist with 15 years experience.\n"
            "Plan the optimal solution approach.\n\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "primary_approach": "best method and why",\n'
            '  "formulas_needed": ["formula 1", "formula 2"],\n'
            '  "steps_outline": [\n'
            '    "Step 1: what to do and why",\n'
            '    "Step 2: what to do and why"\n'
            '  ],\n'
            '  "alternative_method": "backup approach if primary fails",\n'
            '  "critical_pitfalls": ["mistake to avoid and why"],\n'
            '  "estimated_difficulty_details": "what makes this complex"\n'
            "}"
        )

        user = (
            f"Problem: {state['question']}\n"
            f"Parsed structure:\n{parsed}"
            f"{rag_hint}"
        )
        content = self._llm_call(system, user, max_tokens=700)
        result = self._parse_json(content)

        nodes = list(state.get("nodes_executed", []))
        nodes.append("route_strategy")

        return {
            "strategy": result or {"primary_approach": "direct solving"},
            "nodes_executed": nodes,
        }

    # ──────────────────────────────
    # NODE: Deep Solve (complex only)
    # ──────────────────────────────

    def deep_solve(self, state: MathState) -> dict:
        context_parts = []
        if state.get("parsed_problem"):
            context_parts.append(
                f"Problem structure:\n"
                f"{json.dumps(state['parsed_problem'], indent=2)}"
            )
        if state.get("strategy"):
            context_parts.append(
                f"Planned approach:\n"
                f"{json.dumps(state['strategy'], indent=2)}"
            )
        if state.get("rag_context"):
            rag_text = "\n".join(
                f"- [{d.get('topic','')}/{d.get('subtopic','')}] "
                f"{d.get('content','')[:200]}"
                for d in state["rag_context"][:3]
            )
            context_parts.append(f"Reference material:\n{rag_text}")
        if state.get("similar_problems"):
            sim_text = "\n".join(
                f"- {p.get('extracted_text','')}: "
                f"{str(p.get('solution',''))[:200]}"
                for p in state["similar_problems"][:2]
            )
            context_parts.append(f"Similar solved:\n{sim_text}")

        context = "\n\n".join(context_parts) or "No context."

        # ═══════════════════════════════════════════════════
        # KEY FIX: Structured solving with step limit
        # ═══════════════════════════════════════════════════
        system = (
            "You are an expert JEE math solver on a COMPLEX problem.\n"
            "You have a detailed analysis and strategy. Follow it.\n\n"
            "CRITICAL RULES:\n"
            "1. Use 6-12 steps MAXIMUM. Never exceed 15.\n"
            "2. PLAN completely before writing. NO backtracking.\n"
            "3. Each step: ONE clear action → ONE clear result.\n"
            "4. Reference the strategy's recommended formulas.\n"
            "5. Use RAG context formulas when directly applicable.\n"
            "6. NEVER self-correct mid-solution. If unsure, pick ,\n"
            "7. Use LaTeX notation for math: $x^2$, $\\frac{a}{b}$, "
            "$\\sqrt{x}$, $\\int_0^1 f(x)dx$, $\\sum_{i=1}^{n}$\n"
            "8. Wrap ALL mathematical expressions in $ delimiters.\n"
            "the most likely path and commit.\n\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "solution_steps": [\n'
            '    "Step 1: [Action] — [Result]",\n'
            '    "Step 2: [Action] — [Result]"\n'
            '  ],\n'
            '  "final_answer": "precise answer as string",\n'
            '  "method_used": "method name",\n'
            '  "rag_formulas_used": ["formula if used"]\n'
            "}\n\n"
            "BANNED: 'actually', 'wait', 'let me reconsider', "
            "'I made an error', 'on second thought'\n"
            "Return ONLY valid JSON."
        )

        user = f"Solve:\n{state['question']}\n\n{context}"
        content = self._llm_call(system, user, max_tokens=3500)
        result = self._parse_json(content)

        nodes = list(state.get("nodes_executed", []))
        nodes.append("deep_solve")

        if not result:
            return {
                "solution_steps": [content],
                "final_answer": "",
                "solution": content,
                "nodes_executed": nodes,
            }

        steps = result.get("solution_steps", [])
        if isinstance(steps, str):
            steps = [steps]

        # Safety truncation
        if len(steps) > 15:
            logger.warning(
                "Deep solve: %d steps, truncating to 15", len(steps)
            )
            steps = steps[:15]

        answer = str(result.get("final_answer", ""))
        solution_text = "\n".join(
            f"{i+1}. {s}" if not s.startswith("Step") else s
            for i, s in enumerate(steps)
        )

        return {
            "solution_steps": steps,
            "final_answer": answer,
            "solution": f"{solution_text}\n\nFinal Answer: {answer}",
            "nodes_executed": nodes,
        }
    # ──────────────────────────────
    # NODE: Verify (complex only)
    # ──────────────────────────────

    def verify_solution(self, state: MathState) -> dict:
        """
        Agent 4 (complex path): Independent verification.
        Checks solution using different method when possible.
        """
        system = (
            "You are a rigorous math solution verifier.\n"
            "Verify this solution INDEPENDENTLY — don't just agree.\n\n"
            "Verification checklist:\n"
            "1. Substitute answer back into original equation\n"
            "2. Check each step's arithmetic independently\n"
            "3. Verify edge cases and domain restrictions\n"
            "4. Try solving with alternative method if possible\n"
            "5. Check dimensional consistency\n\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "is_correct": true,\n'
            '  "confidence": 0.95,\n'
            '  "verification_method": "what you did to verify",\n'
            '  "verification_summary": "detailed summary",\n'
            '  "step_check": [\n'
            '    {"step": 1, "correct": true, "note": "checked"},\n'
            '    {"step": 2, "correct": true, "note": "verified"}\n'
            '  ],\n'
            '  "issues": [],\n'
            '  "edge_cases_checked": ["case 1"]\n'
            "}"
        )

        user = (
            f"Original problem: {state['question']}\n\n"
            f"Solution:\n{state.get('solution', '')}\n\n"
            f"Final Answer: {state.get('final_answer', '')}\n\n"
            f"Parsed structure: {json.dumps(state.get('parsed_problem', {}))}"
        )
        content = self._llm_call(system, user, max_tokens=1000)
        result = self._parse_json(content)

        # Build verification summary
        summary_parts = []
        method = result.get("verification_method", "")
        if method:
            summary_parts.append(f"Method: {method}")

        detail = result.get("verification_summary", "")
        if detail:
            summary_parts.append(detail)

        if result.get("issues"):
            summary_parts.append(
                "Issues found: " + ", ".join(result["issues"])
            )

        edges = result.get("edge_cases_checked", [])
        if edges:
            summary_parts.append(
                "Edge cases checked: " + ", ".join(edges)
            )

        summary = "\n".join(summary_parts) or "Solution verified."

        nodes = list(state.get("nodes_executed", []))
        nodes.append("verify_solution")

        return {
            "verification": summary,
            "is_correct": result.get("is_correct", True),
            "verification_confidence": float(
                result.get("confidence", 0.8)
            ),
            "nodes_executed": nodes,
        }

    # ──────────────────────────────
    # NODE: Explain (complex only)
    # ──────────────────────────────

    def explain_solution(self, state: MathState) -> dict:
        """
        Agent 5 (complex path): Student-friendly explanation.
        Has access to verification results for accuracy.
        """
        verification = state.get("verification", "")
        is_correct = state.get("is_correct", True)

        correction_note = ""
        if not is_correct:
            correction_note = (
                "\n\nIMPORTANT: The verifier found issues with this "
                "solution. Acknowledge the issues in your explanation "
                "and clarify the correct approach."
            )

        system = (
            "You are a patient, encouraging JEE math tutor.\n"
            "Explain this complex problem's solution to a student.\n"
            f"{correction_note}\n\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "explanation": "detailed student-friendly explanation",\n'
            '  "key_concepts": ["concept 1 with brief description"],\n'
            '  "common_mistakes": [\n'
            '    "mistake to avoid and why"\n'
            '  ],\n'
            '  "intuition": "the key insight that makes this click",\n'
            '  "exam_tip": "JEE-specific time-saving tip"\n'
            "}\n\n"
            "RULES:\n"
            "- Explain WHY each step works, not just WHAT\n"
            "- Use analogies if helpful\n"
            "- Highlight the key 'aha moment'\n"
            "- explanation must be a single string"
        )

        user = (
            f"Problem: {state['question']}\n\n"
            f"Solution:\n{state.get('solution', '')}\n\n"
            f"Final Answer: {state.get('final_answer', '')}\n\n"
            f"Verification notes: {verification}"
        )
        content = self._llm_call(system, user, max_tokens=1500)
        result = self._parse_json(content)

        explanation = str(result.get("explanation", ""))
        if not explanation:
            explanation = content

        # Append intuition and exam tip if present
        intuition = result.get("intuition", "")
        exam_tip = result.get("exam_tip", "")
        if intuition:
            explanation += f"\n\n💡 Key Insight: {intuition}"
        if exam_tip:
            explanation += f"\n\n📝 JEE Tip: {exam_tip}"

        nodes = list(state.get("nodes_executed", []))
        nodes.append("explain_solution")

        return {
            "explanation": explanation,
            "key_concepts": result.get("key_concepts", []),
            "common_mistakes": result.get("common_mistakes", []),
            "nodes_executed": nodes,
        }

    # ──────────────────────────────
    # Routing
    # ──────────────────────────────
    # Add this method to MathMentorGraph class

    @staticmethod
    def _enhance_with_rag(solution: dict, rag_context: list) -> dict:
        """
        Post-process: If RAG had relevant content but solver didn't
        cite it, append a 'Relevant Formulas' section.
        """
        if not rag_context:
            return solution

        # Check if solver already cited RAG
        rag_used = solution.get("rag_formulas_used", [])
        if rag_used:
            # Solver cited RAG — add to explanation
            citation = "\n\n📚 Formulas Used (from knowledge base):\n"
            citation += "\n".join(f"  • {f}" for f in rag_used)
            solution["explanation"] = (
                solution.get("explanation", "") + citation
            )
            return solution

        # Solver didn't cite RAG — append relevant formulas anyway
        high_score = [
            r for r in rag_context
            if float(r.get("score", 0)) > 0.4
        ]

        if high_score:
            citation = "\n\n📚 Related Formulas (from knowledge base):\n"
            for r in high_score[:3]:
                topic = r.get("subtopic", r.get("topic", ""))
                content = str(r.get("content", ""))[:200]
                score = float(r.get("score", 0))
                citation += f"  • [{topic}] {content} (relevance: {score:.0%})\n"

            solution["explanation"] = (
                solution.get("explanation", "") + citation
            )

        return solution

    @staticmethod
    def _after_classify(state: MathState) -> str:
        """Route based on difficulty classification."""
        if state.get("difficulty") == "complex":
            return "complex"
        return "fast"  # simple AND medium → single LLM call

    # ──────────────────────────────
    # Build Graph
    # ──────────────────────────────

    def _build_graph(self):
        """
        Build the adaptive pipeline.

        ┌──────────────────┐
        │     classify     │ ← Rules first, LLM if uncertain
        │   (difficulty)   │
        └────────┬─────────┘
                 │
          ┌──────┴───────┐
          │              │
       simple/         complex
       medium            │
          │        ┌─────▼──────┐
          │        │   parse    │ Agent 1
          │        └─────┬──────┘
          │        ┌─────▼──────┐
          │        │  strategy  │ Agent 2
          │        └─────┬──────┘
          │        ┌─────▼──────┐
          ▼        │ deep_solve │ Agent 3
     ┌─────────┐  └─────┬──────┘
     │  fast   │  ┌─────▼──────┐
     │  solve  │  │  verify    │ Agent 4
     │(1 call) │  └─────┬──────┘
     └────┬────┘  ┌─────▼──────┐
          │       │  explain   │ Agent 5
          │       └─────┬──────┘
          │             │
          ▼             ▼
         END           END
        """
        graph = StateGraph(MathState)

        # Add nodes
        graph.add_node("classify_difficulty", self.classify_difficulty)
        graph.add_node("fast_solve", self.fast_solve)
        graph.add_node("parse_problem", self.parse_problem)
        graph.add_node("route_strategy", self.route_strategy)
        graph.add_node("deep_solve", self.deep_solve)
        graph.add_node("verify_solution", self.verify_solution)
        graph.add_node("explain_solution", self.explain_solution)

        # Entry
        graph.set_entry_point("classify_difficulty")

        # Conditional: fast path or complex path
        graph.add_conditional_edges(
            "classify_difficulty",
            self._after_classify,
            {
                "fast": "fast_solve",
                "complex": "parse_problem",
            },
        )

        # Fast path → END
        graph.add_edge("fast_solve", END)

        # Complex path: parse → strategy → solve → verify → explain → END
        graph.add_edge("parse_problem", "route_strategy")
        graph.add_edge("route_strategy", "deep_solve")
        graph.add_edge("deep_solve", "verify_solution")
        graph.add_edge("verify_solution", "explain_solution")
        graph.add_edge("explain_solution", END)

        return graph.compile()

    # ──────────────────────────────
    # Public API
    # ──────────────────────────────

    def solve(
        self,
        question: str,
        rag_context: list = None,
        similar_problems: list = None,
    ) -> Dict[str, Any]:
        truncated = question
        if len(question) > 1500:
            truncated = question[:1500] + "\n\n[Truncated]"

        initial_state: MathState = {
            "question": truncated,
            "rag_context": rag_context or [],
            "similar_problems": similar_problems or [],
            "nodes_executed": [],
        }

        try:
            final = self.compiled_graph.invoke(initial_state)

            nodes = final.get("nodes_executed", [])
            difficulty = final.get("difficulty", "unknown")
            classify_method = final.get("classification_method", "unknown")

            logger.info(
                "Pipeline: %s (%s), nodes: %s",
                difficulty, classify_method, " → ".join(nodes),
            )

            result = {
                "parsed_problem": final.get("parsed_problem", {}),
                "solution_steps": final.get("solution_steps", []),
                "solution": final.get("solution", ""),
                "final_answer": final.get("final_answer", ""),
                "verification": final.get("verification", ""),
                "explanation": final.get("explanation", ""),
                "key_concepts": final.get("key_concepts", []),
                "common_mistakes": final.get("common_mistakes", []),
                "difficulty": difficulty,
                "nodes_executed": nodes,
            }

            # ═══════════════════════════════════
            # Enhance with RAG citations
            # ═══════════════════════════════════
            result = self._enhance_with_rag(result, rag_context or [])

            return result

        except Exception as e:
            logger.error("LangGraph failed: %s", e, exc_info=True)
            return {"error": str(e)}

    def get_graph_info(self) -> Dict[str, Any]:
        """Graph info for /health endpoint and debugging."""
        return {
            "type": "LangGraph (Hybrid Adaptive)",
            "model": self.model,
            "classification": "rule-based + LLM fallback",
            "paths": {
                "simple": [
                    "classify(rule, 0 calls)",
                    "fast_solve(1 call)",
                ],
                "medium": [
                    "classify(llm, 1 call)",
                    "fast_solve(1 call)",
                ],
                "complex": [
                    "classify(llm, 1 call)",
                    "parse_problem(1 call)",
                    "route_strategy(1 call)",
                    "deep_solve(1 call)",
                    "verify_solution(1 call)",
                    "explain_solution(1 call)",
                ],
            },
            "total_llm_calls": {
                "simple": 1,
                "medium": 2,
                "complex": 6,
            },
            "agents": {
                "classifier": "Difficulty assessment (hybrid rule+LLM)",
                "parser": "Problem decomposition (complex only)",
                "strategist": "Solution planning (complex only)",
                "solver": "Step-by-step solving",
                "verifier": "Independent verification (complex only)",
                "explainer": "Student-friendly explanation (complex only)",
            },
        }