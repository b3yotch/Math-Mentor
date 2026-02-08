# src/guardrails/math_classifier.py
"""
Embedding-Based Math Classifier for Math Mentor.
Uses SHARED embedding model from EmbeddingManager.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of math classification."""
    is_math: bool
    confidence: float
    topic: str
    subtopic: str
    reasoning: str


class MathClassifier:
    """
    Semantic classifier for math questions using shared embeddings.
    """
    
    # Math anchor examples
    MATH_ANCHORS = [
        # Algebra
        ("Solve x^2 - 5x + 6 = 0", "algebra", "equations"),
        ("Find the roots of the quadratic equation", "algebra", "equations"),
        ("Factor the polynomial x^3 - 8", "algebra", "polynomials"),
        ("Simplify the expression (x+2)(x-3)", "algebra", "expressions"),
        ("If 2x + 5 = 15, find x", "algebra", "equations"),
        ("What is the value of x if 3x = 27", "algebra", "equations"),
        ("x squared plus 2 equals zero", "algebra", "equations"),
        ("x^2 + 2 = 0", "algebra", "equations"),
        
        # Calculus
        ("Find the derivative of f(x) = x^3", "calculus", "derivatives"),
        ("Calculate the integral of sin(x) dx", "calculus", "integrals"),
        ("Find the limit as x approaches 0", "calculus", "limits"),
        ("What is the derivative of e^x", "calculus", "derivatives"),
        ("Evaluate the definite integral from 0 to pi", "calculus", "integrals"),
        ("Find dy/dx for y = x^2 + 3x", "calculus", "derivatives"),
        
        # Probability
        ("What is the probability of getting heads in a coin toss", "probability", "basic"),
        ("Find the probability of drawing a king from a deck of cards", "probability", "basic"),
        ("Calculate the probability of rolling a 6 on a dice", "probability", "basic"),
        ("What are the odds of getting two heads in three tosses", "probability", "combinations"),
        ("If probability of event A is 0.3, find P(not A)", "probability", "basic"),
        ("How many ways can 5 people be arranged in a line", "probability", "permutations"),
        ("Find the number of combinations of 10 choose 3", "probability", "combinations"),
        ("What is the probability of getting a head in a coin toss", "probability", "basic"),
        ("Probability of drawing a red card from a deck", "probability", "basic"),
        
        # Statistics
        ("Find the mean of 2, 4, 6, 8, 10", "statistics", "central_tendency"),
        ("Calculate the standard deviation", "statistics", "dispersion"),
        ("What is the median of the dataset", "statistics", "central_tendency"),
        ("Find the variance of the given numbers", "statistics", "dispersion"),
        
        # Trigonometry
        ("Find the value of sin(30 degrees)", "trigonometry", "ratios"),
        ("Calculate cos(60) + sin(30)", "trigonometry", "ratios"),
        ("Solve the equation sin(x) = 0.5", "trigonometry", "equations"),
        
        # Geometry
        ("Find the area of a circle with radius 5", "geometry", "area"),
        ("Calculate the perimeter of a rectangle", "geometry", "perimeter"),
        ("What is the volume of a sphere with radius 3", "geometry", "volume"),
        
        # Linear Algebra
        ("Find the determinant of the matrix", "linear_algebra", "determinants"),
        ("Calculate the inverse of matrix A", "linear_algebra", "matrices"),
        ("Find the eigenvalues of the matrix", "linear_algebra", "eigenvalues"),
        
        # Word Problems
        ("John has 5 apples and Mary gives him 3 more. How many does he have?", "algebra", "word_problem"),
        ("A train travels at 60 km/h for 2 hours. Find the distance.", "algebra", "word_problem"),
        ("If the price is increased by 20%, what is the new price?", "algebra", "word_problem"),
        ("Ram is twice as old as Shyam. Sum of ages is 36. Find their ages.", "algebra", "word_problem"),
        ("A shopkeeper gives 10% discount. Find the selling price.", "algebra", "word_problem"),
        ("How many ways can 3 books be selected from 10 books?", "probability", "combinations"),
    ]
    
    # Non-math anchor examples
    NON_MATH_ANCHORS = [
        "What is the capital of France",
        "Name the longest river in the world",
        "Which country has the largest population",
        "When did World War 2 end",
        "Who was the first president of USA",
        "What is photosynthesis",
        "Explain the structure of DNA",
        "Who wrote Romeo and Juliet",
        "Recommend a good movie",
        "Tell me a joke",
        "How are you today",
        "What is your name",
        "How to make pasta",
        "What is the recipe for chocolate cake",
        "How to code in Python",
        "What is machine learning",
        "Who is the CEO of Tesla",
        "What is the weather today",
        "Translate hello to Spanish",
        "Who won the last World Cup",
    ]
    
    def __init__(self, embedding_manager=None):
        """
        Initialize classifier with shared embedding manager.
        
        Args:
            embedding_manager: Shared EmbeddingManager instance.
                             If None, will get from singleton.
        """
        self._embedding_manager = embedding_manager
        self._math_embeddings = None
        self._non_math_embeddings = None
        self._math_topics = None
        self._initialized = False
    
    @property
    def embedding_manager(self):
        """Get embedding manager (lazy load from singleton if not provided)."""
        if self._embedding_manager is None:
            from src.core.embedding_manager import get_embedding_manager
            self._embedding_manager = get_embedding_manager()
        return self._embedding_manager
    
    def initialize(self):
        """Pre-compute embeddings for anchor examples."""
        if self._initialized:
            return
        
        logger.info("Initializing math classifier...")
        
        # Get embeddings for math anchors
        math_texts = [anchor[0] for anchor in self.MATH_ANCHORS]
        self._math_embeddings = self.embedding_manager.encode(math_texts, normalize=True)
        self._math_topics = [(anchor[1], anchor[2]) for anchor in self.MATH_ANCHORS]
        
        # Get embeddings for non-math anchors
        self._non_math_embeddings = self.embedding_manager.encode(self.NON_MATH_ANCHORS, normalize=True)
        
        self._initialized = True
        logger.info(f"✓ Classifier initialized ({len(math_texts)} math, {len(self.NON_MATH_ANCHORS)} non-math anchors)")
    
    def classify(self, text: str) -> ClassificationResult:
        """
        Classify whether input is a math question.
        
        Args:
            text: Input text to classify
            
        Returns:
            ClassificationResult
        """
        if not self._initialized:
            self.initialize()
        
        # Encode input
        input_embedding = self.embedding_manager.encode(text, normalize=True)
        
        # Compute similarities
        math_similarities = np.dot(self._math_embeddings, input_embedding)
        non_math_similarities = np.dot(self._non_math_embeddings, input_embedding)
        
        # Get scores
        top_math_sim = np.max(math_similarities)
        top_math_idx = np.argmax(math_similarities)
        top_5_math_avg = np.mean(np.sort(math_similarities)[-5:])
        
        top_non_math_sim = np.max(non_math_similarities)
        top_5_non_math_avg = np.mean(np.sort(non_math_similarities)[-5:])
        
        # Calculate math score
        math_score = (top_math_sim + top_5_math_avg) / 2
        non_math_score = (top_non_math_sim + top_5_non_math_avg) / 2
        
        # Calculate confidence using sigmoid
        score_diff = math_score - non_math_score
        confidence = 1 / (1 + np.exp(-12 * score_diff))  # Steeper sigmoid
        
        # Apply rule-based boost for obvious patterns
        rule_boost = self._get_rule_boost(text)
        confidence = min(1.0, confidence + rule_boost)
        
        # Determine result
        is_math = confidence >= 0.5
        
        if is_math:
            topic, subtopic = self._math_topics[top_math_idx]
        else:
            topic, subtopic = "not_math", "general"
        
        # Reasoning
        if is_math:
            reasoning = f"Similar to math examples (score: {math_score:.2f} vs {non_math_score:.2f})"
        else:
            reasoning = f"More similar to non-math content (math: {math_score:.2f}, non-math: {non_math_score:.2f})"
        
        return ClassificationResult(
            is_math=is_math,
            confidence=float(confidence),
            topic=topic,
            subtopic=subtopic,
            reasoning=reasoning
        )
    
    def _get_rule_boost(self, text: str) -> float:
        """Rule-based boost for obvious math patterns."""
        import re
        
        boost = 0.0
        text_lower = text.lower()
        
        # Strong math indicators
        strong_patterns = [
            r'[xyz]\s*[\^²³]',           # x^2, x², y³
            r'\d+\s*[+\-*/^]\s*\d+',     # 2+3, 5*4
            r'[xyz]\s*[+\-*/=]\s*\d',    # x+2, y=3
            r'\d+\s*[xyz]',              # 2x, 3y
            r'√|∫|∑|∏|∂|∞|π|θ',         # Math symbols
            r'\bsin\b|\bcos\b|\btan\b',  # Trig
            r'\bd/dx\b|\bdy/dx\b',       # Calculus
            r'=\s*0\b',                  # Equals zero
            r'$$\s*\[.*$$\s*\]',         # Matrix notation
        ]
        
        for pattern in strong_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                boost += 0.12
        
        # Word problem indicators
        word_patterns = [
            r'\bhow\s+(many|much)\b',
            r'\bfind\s+the\b',
            r'\bcalculate\b',
            r'\bsolve\b',
            r'\bwhat\s+is\s+the\s+(probability|value|area|sum)\b',
            r'\bprobability\s+of\b',
        ]
        
        for pattern in word_patterns:
            if re.search(pattern, text_lower):
                boost += 0.08
        
        return min(0.35, boost)
    
    def get_topic(self, text: str) -> Tuple[str, str]:
        """Get just topic and subtopic."""
        result = self.classify(text)
        return result.topic, result.subtopic


# ============================================
# SINGLETON
# ============================================
_classifier_instance: Optional[MathClassifier] = None


def get_math_classifier(embedding_manager=None) -> MathClassifier:
    """Get or create classifier singleton."""
    global _classifier_instance
    
    if _classifier_instance is None:
        _classifier_instance = MathClassifier(embedding_manager)
    
    return _classifier_instance


# ============================================
# TEST
# ============================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    
    from src.core.embedding_manager import get_embedding_manager
    
    print("=" * 70)
    print("MATH CLASSIFIER TEST (with shared embedding manager)")
    print("=" * 70)
    
    # Get shared embedding manager
    em = get_embedding_manager()
    print(f"\n📦 Using shared model: {em.model_name}")
    
    # Create classifier with shared manager
    classifier = get_math_classifier(em)
    classifier.initialize()
    
    # Test cases
    test_cases = [
        # High confidence MATH
        ("Solve x^2 - 5x + 6 = 0", True),
        ("x² + 2 = 0", True),
        ("What is 2 + 2?", True),
        ("Find the derivative of x^3", True),
        ("What is the probability of getting heads in a coin toss", True),
        ("Find the probability of drawing a king from 52 cards", True),
        ("Calculate the area of a circle with radius 5", True),
        ("John has 5 apples. Mary gives 3. How many total?", True),
        ("A train goes 60 km/h for 2 hours. Find distance.", True),
        ("Find the mean of 10, 20, 30, 40, 50", True),
        ("What is sin(30)?", True),
        ("Find the determinant of [[1,2],[3,4]]", True),
        
        # NOT MATH
        ("What is the capital of France?", False),
        ("Who wrote Romeo and Juliet?", False),
        ("Tell me a joke", False),
        ("How to cook pasta?", False),
        ("What is photosynthesis?", False),
        ("Recommend a good movie", False),
        ("Who is the president of USA?", False),
        ("Explain machine learning", False),
        ("What is the weather today?", False),
        ("Hello, how are you?", False),
    ]
    
    correct = 0
    total = len(test_cases)
    
    print("\nResults:\n")
    for text, expected_math in test_cases:
        result = classifier.classify(text)
        
        is_correct = result.is_math == expected_math
        if is_correct:
            correct += 1
            status = "✅"
        else:
            status = "❌"
        
        conf = f"{result.confidence:.0%}"
        topic = f"{result.topic}/{result.subtopic}" if result.is_math else "not_math"
        
        print(f"{status} [{conf:>4}] [{topic:<25}] {text[:45]}...")
        
        if not is_correct:
            print(f"         ↳ Expected {'MATH' if expected_math else 'NOT MATH'}, Got {'MATH' if result.is_math else 'NOT MATH'}")
    
    print("\n" + "=" * 70)
    print(f"ACCURACY: {correct}/{total} ({correct/total*100:.1f}%)")
    print("=" * 70)
    
    # Verify same embedding manager
    em2 = classifier.embedding_manager
    print(f"\n✓ Same embedding manager instance: {em is em2}")