# src/evaluation/test_datasets.py
"""
Ground truth test datasets for evaluation.
Covers all required math topics: Algebra, Probability, Calculus, Linear Algebra.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class TestCase:
    """A single test case with ground truth."""
    id: str
    question: str
    topic: str
    subtopic: str
    expected_answer: str
    acceptable_answers: List[str] = field(default_factory=list)
    difficulty: int = 1  # 1-5
    input_type: str = "text"  # text, ocr_text, asr_text
    expected_steps: List[str] = field(default_factory=list)
    expected_rag_topics: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


class TestDataset:
    """Collection of test cases for evaluation."""
    
    def __init__(self):
        self.test_cases: List[TestCase] = []
        self._build_dataset()
    
    def _build_dataset(self):
        """Build comprehensive test dataset."""
        
        # ==================== ALGEBRA ====================
        self.test_cases.extend([
            TestCase(
                id="alg_001",
                question="Solve x^2 - 5x + 6 = 0",
                topic="algebra",
                subtopic="quadratic_equations",
                expected_answer="x = 2 or x = 3",
                acceptable_answers=[
                    "x = 2 or x = 3",
                    "x = 3 or x = 2",
                    "x = 2, x = 3",
                    "{2, 3}",
                    "x ∈ {2, 3}",
                ],
                difficulty=1,
                expected_steps=["factor", "solve"],
                expected_rag_topics=["quadratic", "equation", "algebra"],
                tags=["factoring", "quadratic"],
            ),
            TestCase(
                id="alg_002",
                question="Solve 2x + 3 = 11",
                topic="algebra",
                subtopic="linear_equations",
                expected_answer="x = 4",
                acceptable_answers=["x = 4", "4"],
                difficulty=1,
                expected_rag_topics=["linear", "equation"],
                tags=["linear"],
            ),
            TestCase(
                id="alg_003",
                question="Find the roots of x^2 + 4x + 4 = 0",
                topic="algebra",
                subtopic="quadratic_equations",
                expected_answer="x = -2 (double root)",
                acceptable_answers=["x = -2", "-2", "x = -2 (repeated)"],
                difficulty=2,
                tags=["quadratic", "repeated_root"],
            ),
            TestCase(
                id="alg_004",
                question="Simplify (x^2 - 9) / (x - 3)",
                topic="algebra",
                subtopic="simplification",
                expected_answer="x + 3",
                acceptable_answers=["x + 3", "(x+3)", "x + 3, x ≠ 3"],
                difficulty=2,
                tags=["simplification", "factoring"],
            ),
            TestCase(
                id="alg_005",
                question="Solve the system: 2x + y = 7, x - y = 2",
                topic="algebra",
                subtopic="systems_of_equations",
                expected_answer="x = 3, y = 1",
                acceptable_answers=["x = 3, y = 1", "(3, 1)"],
                difficulty=2,
                tags=["simultaneous", "linear_system"],
            ),
            TestCase(
                id="alg_006",
                question="If f(x) = x^2 + 2x + 1, find f(3)",
                topic="algebra",
                subtopic="functions",
                expected_answer="16",
                acceptable_answers=["16", "f(3) = 16"],
                difficulty=1,
                tags=["function_evaluation"],
            ),
            # Word problem
            TestCase(
                id="alg_007",
                question="A train travels at 60 km/h for 2.5 hours. What distance does it cover?",
                topic="algebra",
                subtopic="word_problem",
                expected_answer="150 km",
                acceptable_answers=["150 km", "150", "150 kilometers"],
                difficulty=1,
                tags=["word_problem", "distance"],
            ),
            TestCase(
                id="alg_008",
                question="Ram is twice as old as Shyam. The sum of their ages is 36. Find their ages.",
                topic="algebra",
                subtopic="word_problem",
                expected_answer="Ram = 24, Shyam = 12",
                acceptable_answers=["Ram = 24, Shyam = 12", "24 and 12", "Shyam = 12, Ram = 24"],
                difficulty=2,
                tags=["word_problem", "age_problem"],
            ),
        ])
        
        # ==================== PROBABILITY ====================
        self.test_cases.extend([
            TestCase(
                id="prob_001",
                question="What is the probability of getting heads in a single coin toss?",
                topic="probability",
                subtopic="basic",
                expected_answer="1/2 or 0.5",
                acceptable_answers=["1/2", "0.5", "50%", "0.50"],
                difficulty=1,
                expected_rag_topics=["probability", "coin"],
                tags=["basic_probability"],
            ),
            TestCase(
                id="prob_002",
                question="What is the probability of drawing a king from a standard deck of 52 cards?",
                topic="probability",
                subtopic="basic",
                expected_answer="4/52 = 1/13",
                acceptable_answers=["1/13", "4/52", "0.0769", "7.69%"],
                difficulty=1,
                expected_rag_topics=["probability", "cards", "deck"],
                tags=["cards", "basic_probability"],
            ),
            TestCase(
                id="prob_003",
                question="What is the probability of getting exactly 2 heads in 3 coin tosses?",
                topic="probability",
                subtopic="binomial",
                expected_answer="3/8",
                acceptable_answers=["3/8", "0.375", "37.5%"],
                difficulty=2,
                tags=["binomial", "combinations"],
            ),
            TestCase(
                id="prob_004",
                question="If P(A) = 0.3 and P(B) = 0.4, and A and B are independent, find P(A and B)",
                topic="probability",
                subtopic="independent_events",
                expected_answer="0.12",
                acceptable_answers=["0.12", "12%", "3/25"],
                difficulty=2,
                tags=["independent_events"],
            ),
            TestCase(
                id="prob_005",
                question="How many ways can 5 people be arranged in a line?",
                topic="probability",
                subtopic="permutations",
                expected_answer="120",
                acceptable_answers=["120", "5! = 120"],
                difficulty=2,
                tags=["permutations", "factorial"],
            ),
            TestCase(
                id="prob_006",
                question="In how many ways can you choose 3 items from 10?",
                topic="probability",
                subtopic="combinations",
                expected_answer="120",
                acceptable_answers=["120", "10C3 = 120", "C(10,3) = 120"],
                difficulty=2,
                tags=["combinations"],
            ),
        ])
        
        # ==================== CALCULUS ====================
        self.test_cases.extend([
            TestCase(
                id="calc_001",
                question="Find the derivative of f(x) = x^3",
                topic="calculus",
                subtopic="derivatives",
                expected_answer="3x^2",
                acceptable_answers=["3x^2", "3x²", "f'(x) = 3x^2"],
                difficulty=1,
                expected_rag_topics=["derivative", "power_rule"],
                tags=["derivative", "power_rule"],
            ),
            TestCase(
                id="calc_002",
                question="Find the derivative of f(x) = x^3 + 2x^2 - 5x + 3",
                topic="calculus",
                subtopic="derivatives",
                expected_answer="3x^2 + 4x - 5",
                acceptable_answers=["3x^2 + 4x - 5", "3x² + 4x - 5"],
                difficulty=2,
                tags=["derivative", "polynomial"],
            ),
            TestCase(
                id="calc_003",
                question="Evaluate the limit: lim(x→0) sin(x)/x",
                topic="calculus",
                subtopic="limits",
                expected_answer="1",
                acceptable_answers=["1", "1.0"],
                difficulty=2,
                tags=["limit", "trigonometric"],
            ),
            TestCase(
                id="calc_004",
                question="Find the integral of 2x dx",
                topic="calculus",
                subtopic="integrals",
                expected_answer="x^2 + C",
                acceptable_answers=["x^2 + C", "x² + C", "x^2 + c"],
                difficulty=1,
                tags=["integral", "basic"],
            ),
            TestCase(
                id="calc_005",
                question="Find the derivative of sin(x)",
                topic="calculus",
                subtopic="derivatives",
                expected_answer="cos(x)",
                acceptable_answers=["cos(x)", "cos x", "cosx"],
                difficulty=1,
                tags=["derivative", "trigonometric"],
            ),
            TestCase(
                id="calc_006",
                question="Find the maximum value of f(x) = -x^2 + 4x + 1",
                topic="calculus",
                subtopic="optimization",
                expected_answer="5 at x = 2",
                acceptable_answers=["5", "maximum = 5", "f(2) = 5"],
                difficulty=3,
                tags=["optimization", "maxima"],
            ),
        ])
        
        # ==================== LINEAR ALGEBRA ====================
        self.test_cases.extend([
            TestCase(
                id="la_001",
                question="Find the determinant of the matrix [[1,2],[3,4]]",
                topic="linear_algebra",
                subtopic="determinants",
                expected_answer="-2",
                acceptable_answers=["-2", "det = -2"],
                difficulty=1,
                expected_rag_topics=["determinant", "matrix"],
                tags=["determinant", "2x2"],
            ),
            TestCase(
                id="la_002",
                question="Find the determinant of [[2,0],[0,3]]",
                topic="linear_algebra",
                subtopic="determinants",
                expected_answer="6",
                acceptable_answers=["6", "det = 6"],
                difficulty=1,
                tags=["determinant", "diagonal"],
            ),
            TestCase(
                id="la_003",
                question="Multiply matrices [[1,2],[3,4]] and [[5],[6]]",
                topic="linear_algebra",
                subtopic="matrix_multiplication",
                expected_answer="[[17],[39]]",
                acceptable_answers=["[[17],[39]]", "[17, 39]", "17 and 39"],
                difficulty=2,
                tags=["matrix_multiplication"],
            ),
        ])
        
        # ==================== STATISTICS ====================
        self.test_cases.extend([
            TestCase(
                id="stat_001",
                question="Find the mean of 2, 4, 6, 8, 10",
                topic="statistics",
                subtopic="central_tendency",
                expected_answer="6",
                acceptable_answers=["6", "6.0", "mean = 6"],
                difficulty=1,
                tags=["mean", "average"],
            ),
            TestCase(
                id="stat_002",
                question="Find the median of 3, 1, 4, 1, 5, 9, 2, 6",
                topic="statistics",
                subtopic="central_tendency",
                expected_answer="3.5",
                acceptable_answers=["3.5", "7/2"],
                difficulty=2,
                tags=["median"],
            ),
        ])
    
    def get_all(self) -> List[TestCase]:
        """Get all test cases."""
        return self.test_cases
    
    def get_by_topic(self, topic: str) -> List[TestCase]:
        """Get test cases by topic."""
        return [tc for tc in self.test_cases if tc.topic == topic]
    
    def get_by_difficulty(self, difficulty: int) -> List[TestCase]:
        """Get test cases by difficulty."""
        return [tc for tc in self.test_cases if tc.difficulty == difficulty]
    
    def get_by_tag(self, tag: str) -> List[TestCase]:
        """Get test cases by tag."""
        return [tc for tc in self.test_cases if tag in tc.tags]
    
    def get_by_id(self, test_id: str) -> Optional[TestCase]:
        """Get a specific test case."""
        for tc in self.test_cases:
            if tc.id == test_id:
                return tc
        return None
    
    def get_summary(self) -> Dict:
        """Get dataset summary."""
        topics = {}
        for tc in self.test_cases:
            topics[tc.topic] = topics.get(tc.topic, 0) + 1
        
        return {
            "total": len(self.test_cases),
            "by_topic": topics,
            "difficulties": {
                d: len([tc for tc in self.test_cases if tc.difficulty == d])
                for d in range(1, 6)
            }
        }


# Singleton
_dataset = None

def get_test_dataset() -> TestDataset:
    global _dataset
    if _dataset is None:
        _dataset = TestDataset()
    return _dataset