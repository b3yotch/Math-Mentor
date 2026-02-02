"""
Content Chunker for RAG - Updated with Probability & Statistics
"""

import re
from typing import List, Dict
from dataclasses import dataclass
import hashlib


@dataclass
class Chunk:
    """Represents a content chunk for RAG"""
    chunk_id: str
    content: str
    content_type: str
    topic: str
    subtopic: str
    source: str
    page_number: int
    metadata: Dict
    

class MathContentChunker:
    """
    Chunks mathematical content optimally for RAG retrieval.
    """
    
    def __init__(
        self,
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
        overlap: int = 100
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap
        
        # Topic classification keywords - UPDATED
        self.topic_keywords = {
            "calculus": {
                "limits": ["limit", "lim", "approaches", "continuous", "continuity", "tends to"],
                "derivatives": ["derivative", "differentiate", "d/dx", "f'(x)", "rate of change", "slope", "tangent line", "differentiable"],
                "integrals": ["integral", "integrate", "∫", "antiderivative", "area under", "definite integral", "indefinite integral"],
                "differential_equations": ["differential equation", "ODE", "PDE", "dy/dx", "separable", "linear differential"],
                "series": ["series", "convergence", "divergence", "taylor", "maclaurin", "power series", "infinite series"]
            },
            "algebra": {
                "quadratics": ["quadratic", "x²", "parabola", "discriminant", "roots", "quadratic formula"],
                "polynomials": ["polynomial", "degree", "factor", "remainder theorem", "factor theorem", "zeros"],
                "logarithms": ["logarithm", "log", "ln", "exponential", "e^x", "natural log", "common log"],
                "complex_numbers": ["complex", "imaginary", "i =", "argand", "modulus", "argument", "conjugate"],
                "progressions": ["arithmetic progression", "AP", "geometric progression", "GP", "common difference", "common ratio"],
                "matrices": ["matrix", "matrices", "determinant", "inverse matrix", "transpose", "adjoint", "eigenvalue"],
                "inequalities": ["inequality", "inequation", "greater than", "less than", "absolute value"]
            },
            "trigonometry": {
                "basic_ratios": ["sin", "cos", "tan", "cosec", "sec", "cot", "sine", "cosine", "tangent"],
                "identities": ["identity", "sin²", "cos²", "pythagorean", "trigonometric identity"],
                "equations": ["trigonometric equation", "general solution", "principal solution"],
                "inverse": ["inverse trig", "arcsin", "arccos", "arctan", "sin⁻¹", "cos⁻¹", "tan⁻¹"],
                "properties": ["amplitude", "period", "phase shift", "trigonometric function"]
            },
            "coordinate_geometry": {
                "lines": ["straight line", "slope", "intercept", "parallel", "perpendicular", "distance formula"],
                "circles": ["circle", "radius", "center", "tangent", "chord", "diameter"],
                "conics": ["ellipse", "hyperbola", "parabola", "conic", "focus", "directrix", "eccentricity"]
            },
            "vectors": {
                "basics": ["vector", "magnitude", "direction", "unit vector", "position vector"],
                "operations": ["dot product", "cross product", "scalar product", "vector product"],
                "3d_geometry": ["plane", "line in space", "direction cosines", "direction ratios"]
            },
            
            # ========== NEW: PROBABILITY ==========
            "probability": {
                "basic_probability": [
                    "probability", "sample space", "event", "outcome",
                    "favorable", "equally likely", "P(A)", "P(E)",
                    "trial", "experiment", "sure event", "impossible event",
                    "mutually exclusive", "exhaustive", "complementary"
                ],
                "conditional_probability": [
                    "conditional probability", "P(A|B)", "given that",
                    "bayes", "bayes theorem", "posterior", "prior",
                    "independent events", "dependent events", "multiplication rule"
                ],
                "distributions": [
                    "distribution", "binomial distribution", "poisson distribution", 
                    "normal distribution", "bernoulli", "geometric distribution",
                    "probability distribution", "PMF", "PDF", "CDF",
                    "probability mass function", "probability density function",
                    "cumulative distribution", "bell curve", "gaussian"
                ],
                "random_variables": [
                    "random variable", "discrete random", "continuous random",
                    "expected value", "expectation", "E(X)", "E[X]",
                    "variance", "Var(X)", "standard deviation", "σ",
                    "moment", "moment generating function", "MGF"
                ],
                "combinatorics": [
                    "permutation", "combination", "nCr", "nPr", "factorial",
                    "arrangements", "selections", "choose", "counting principle",
                    "fundamental counting", "circular permutation", "with repetition"
                ]
            },
            
            # ========== NEW: STATISTICS ==========
            "statistics": {
                "descriptive": [
                    "mean", "median", "mode", "average", "central tendency",
                    "arithmetic mean", "geometric mean", "harmonic mean",
                    "weighted mean", "trimmed mean"
                ],
                "dispersion": [
                    "variance", "standard deviation", "dispersion", "spread",
                    "interquartile range", "IQR", "deviation", "range",
                    "coefficient of variation", "mean deviation", "quartile deviation"
                ],
                "data_representation": [
                    "histogram", "frequency", "cumulative frequency", "ogive",
                    "bar graph", "pie chart", "frequency distribution",
                    "grouped data", "ungrouped data", "class interval",
                    "frequency polygon", "stem and leaf", "box plot", "whisker"
                ],
                "percentiles": [
                    "quartile", "percentile", "decile", "Q1", "Q2", "Q3",
                    "first quartile", "third quartile", "median", "five number summary"
                ],
                "correlation_regression": [
                    "correlation", "regression", "scatter plot", "line of best fit",
                    "pearson", "spearman", "coefficient of correlation", "r value",
                    "regression line", "least squares", "linear regression",
                    "positive correlation", "negative correlation"
                ],
                "hypothesis_testing": [
                    "hypothesis", "null hypothesis", "alternative hypothesis", "H0", "H1",
                    "significance level", "p-value", "confidence interval",
                    "t-test", "z-test", "chi-square", "critical value",
                    "type I error", "type II error", "significance"
                ]
            }
        }
    
    def classify_topic(self, content: str) -> tuple:
        """Classify content into topic and subtopic"""
        content_lower = content.lower()
        
        # Score each topic/subtopic
        best_score = 0
        best_topic = "general"
        best_subtopic = "miscellaneous"
        
        for topic, subtopics in self.topic_keywords.items():
            for subtopic, keywords in subtopics.items():
                score = sum(1 for kw in keywords if kw.lower() in content_lower)
                if score > best_score:
                    best_score = score
                    best_topic = topic
                    best_subtopic = subtopic
        
        return best_topic, best_subtopic
    
    def generate_chunk_id(self, content: str, source: str, page: int) -> str:
        """Generate unique chunk ID"""
        hash_input = f"{source}_{page}_{content[:50]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def chunk_definition(self, definition: Dict) -> List[Chunk]:
        """Chunk a definition"""
        content = definition.get("content", "")
        
        if len(content) <= self.max_chunk_size:
            topic, subtopic = self.classify_topic(content)
            return [Chunk(
                chunk_id=self.generate_chunk_id(
                    content,
                    definition.get("metadata", {}).get("source", ""),
                    definition.get("page_number", 0)
                ),
                content=f"Definition: {definition.get('title', '')}\n\n{content}",
                content_type="definition",
                topic=topic,
                subtopic=subtopic,
                source=definition.get("metadata", {}).get("source", ""),
                page_number=definition.get("page_number", 0),
                metadata={
                    "chapter": definition.get("chapter", ""),
                    "section": definition.get("section", ""),
                    "formulas": definition.get("formulas", [])
                }
            )]
        
        return self._split_long_content(definition, "definition")
    
    def chunk_theorem(self, theorem: Dict) -> List[Chunk]:
        """Chunk a theorem"""
        content = theorem.get("content", "")
        
        # Try to separate theorem statement from proof
        proof_markers = ["Proof:", "Proof.", "PROOF"]
        proof_split = None
        
        for marker in proof_markers:
            if marker in content:
                parts = content.split(marker, 1)
                proof_split = (parts[0], marker + parts[1] if len(parts) > 1 else "")
                break
        
        chunks = []
        topic, subtopic = self.classify_topic(content)
        source = theorem.get("metadata", {}).get("source", "")
        page = theorem.get("page_number", 0)
        
        if proof_split:
            statement, proof = proof_split
            
            if statement.strip():
                chunks.append(Chunk(
                    chunk_id=self.generate_chunk_id(statement, source, page),
                    content=f"Theorem: {theorem.get('title', '')}\n\nStatement: {statement.strip()}",
                    content_type="theorem_statement",
                    topic=topic,
                    subtopic=subtopic,
                    source=source,
                    page_number=page,
                    metadata={
                        "chapter": theorem.get("chapter", ""),
                        "section": theorem.get("section", ""),
                        "has_proof": bool(proof.strip())
                    }
                ))
            
            if proof.strip() and len(proof) <= self.max_chunk_size:
                chunks.append(Chunk(
                    chunk_id=self.generate_chunk_id(proof, source, page),
                    content=f"Proof of {theorem.get('title', '')}:\n\n{proof.strip()}",
                    content_type="proof",
                    topic=topic,
                    subtopic=subtopic,
                    source=source,
                    page_number=page,
                    metadata={
                        "chapter": theorem.get("chapter", ""),
                        "section": theorem.get("section", ""),
                        "theorem_title": theorem.get("title", "")
                    }
                ))
        else:
            if len(content) <= self.max_chunk_size:
                chunks.append(Chunk(
                    chunk_id=self.generate_chunk_id(content, source, page),
                    content=f"Theorem: {theorem.get('title', '')}\n\n{content}",
                    content_type="theorem",
                    topic=topic,
                    subtopic=subtopic,
                    source=source,
                    page_number=page,
                    metadata={
                        "chapter": theorem.get("chapter", ""),
                        "section": theorem.get("section", "")
                    }
                ))
            else:
                chunks.extend(self._split_long_content(theorem, "theorem"))
        
        return chunks
    
    def chunk_example(self, example: Dict) -> List[Chunk]:
        """Chunk a solved example"""
        content = example.get("content", "")
        metadata = example.get("metadata", {})
        problem = metadata.get("problem", "")
        solution = metadata.get("solution", "")
        
        chunks = []
        topic, subtopic = self.classify_topic(content)
        source = metadata.get("source", "")
        page = example.get("page_number", 0)
        
        if problem and solution:
            combined = f"Problem:\n{problem}\n\nSolution:\n{solution}"
            
            if len(combined) <= self.max_chunk_size:
                chunks.append(Chunk(
                    chunk_id=self.generate_chunk_id(combined, source, page),
                    content=combined,
                    content_type="solved_example",
                    topic=topic,
                    subtopic=subtopic,
                    source=source,
                    page_number=page,
                    metadata={
                        "chapter": example.get("chapter", ""),
                        "section": example.get("section", ""),
                        "problem_only": problem,
                        "solution_only": solution
                    }
                ))
            else:
                solution_steps = self._split_solution_steps(solution)
                
                for i, step in enumerate(solution_steps):
                    step_content = f"Problem:\n{problem}\n\nSolution Step {i+1}:\n{step}"
                    if len(step_content) <= self.max_chunk_size:
                        chunks.append(Chunk(
                            chunk_id=self.generate_chunk_id(step_content, source, page),
                            content=step_content,
                            content_type="solved_example_step",
                            topic=topic,
                            subtopic=subtopic,
                            source=source,
                            page_number=page,
                            metadata={
                                "chapter": example.get("chapter", ""),
                                "section": example.get("section", ""),
                                "step_number": i + 1,
                                "total_steps": len(solution_steps)
                            }
                        ))
        else:
            if len(content) <= self.max_chunk_size:
                chunks.append(Chunk(
                    chunk_id=self.generate_chunk_id(content, source, page),
                    content=f"Example:\n{content}",
                    content_type="example",
                    topic=topic,
                    subtopic=subtopic,
                    source=source,
                    page_number=page,
                    metadata={
                        "chapter": example.get("chapter", ""),
                        "section": example.get("section", "")
                    }
                ))
            else:
                chunks.extend(self._split_long_content(example, "example"))
        
        return chunks
    
    def chunk_formula(self, formula: Dict, context: Dict) -> Chunk:
        """Create a chunk for a formula with context"""
        content = formula.get("content", "")
        formula_type = formula.get("type", "expression")
        
        contextual_content = f"""Formula: {content}

Context: Found in {context.get('chapter', 'Unknown Chapter')}, {context.get('section', 'Unknown Section')}

Type: {formula_type}"""
        
        topic, subtopic = self.classify_topic(content + " " + context.get('chapter', ''))
        
        return Chunk(
            chunk_id=self.generate_chunk_id(content, context.get("source", ""), context.get("page", 0)),
            content=contextual_content,
            content_type="formula",
            topic=topic,
            subtopic=subtopic,
            source=context.get("source", ""),
            page_number=context.get("page", 0),
            metadata={
                "raw_formula": content,
                "formula_type": formula_type,
                "chapter": context.get("chapter", ""),
                "section": context.get("section", "")
            }
        )
    
    def _split_long_content(self, item: Dict, content_type: str) -> List[Chunk]:
        """Split long content into multiple chunks"""
        content = item.get("content", "")
        chunks = []
        
        text_chunks = self._split_by_sentences(content, self.max_chunk_size)
        topic, subtopic = self.classify_topic(content)
        source = item.get("metadata", {}).get("source", "")
        page = item.get("page_number", 0)
        
        for i, text in enumerate(text_chunks):
            chunks.append(Chunk(
                chunk_id=self.generate_chunk_id(text, source, page),
                content=f"{content_type.title()} (Part {i+1}/{len(text_chunks)}):\n\n{text}",
                content_type=content_type,
                topic=topic,
                subtopic=subtopic,
                source=source,
                page_number=page,
                metadata={
                    "chapter": item.get("chapter", ""),
                    "section": item.get("section", ""),
                    "part": i + 1,
                    "total_parts": len(text_chunks)
                }
            ))
        
        return chunks
    
    def _split_by_sentences(self, text: str, max_size: int) -> List[str]:
        """Split text by sentences"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    def _split_solution_steps(self, solution: str) -> List[str]:
        """Split solution into logical steps"""
        step_patterns = [
            r'Step\s*\d+[:\.]',
            r'$\d+$',
            r'First[,:]',
            r'Second[,:]',
            r'Then[,:]',
            r'Finally[,:]',
            r'Therefore[,:]'
        ]
        
        combined_pattern = '|'.join(step_patterns)
        splits = re.split(f'({combined_pattern})', solution, flags=re.IGNORECASE)
        
        steps = []
        current_step = ""
        
        for part in splits:
            if re.match(combined_pattern, part, re.IGNORECASE):
                if current_step:
                    steps.append(current_step.strip())
                current_step = part
            else:
                current_step += part
        
        if current_step:
            steps.append(current_step.strip())
        
        if len(steps) <= 1:
            steps = [s.strip() for s in solution.split('\n\n') if s.strip()]
        
        return steps if steps else [solution]
    
    def process_book_content(self, book_content: Dict) -> List[Chunk]:
        """Process entire book content and return all chunks"""
        all_chunks = []
        
        for definition in book_content.get("all_definitions", []):
            all_chunks.extend(self.chunk_definition(definition))
        
        for theorem in book_content.get("all_theorems", []):
            all_chunks.extend(self.chunk_theorem(theorem))
        
        for example in book_content.get("all_examples", []):
            all_chunks.extend(self.chunk_example(example))
        
        for chapter in book_content.get("chapters", []):
            for formula in chapter.get("formulas", []):
                context = {
                    "chapter": chapter.get("title", ""),
                    "section": "",
                    "source": book_content.get("metadata", {}).get("source_file", ""),
                    "page": 0
                }
                all_chunks.append(self.chunk_formula(formula, context))
        
        return all_chunks