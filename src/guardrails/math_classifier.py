# src/guardrails/math_classifier.py
"""
Embedding-Based Math Classifier for Math Mentor.
Uses sentence-transformers (already loaded for RAG) to classify inputs.
Zero extra token cost - runs locally.
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
    confidence: float  # 0.0 to 1.0
    topic: str
    subtopic: str
    reasoning: str


class MathClassifier:
    """
    Semantic classifier for math questions using embeddings.
    
    Instead of keyword matching, this compares the input embedding
    against known "anchor" examples of math and non-math questions.
    
    This is essentially zero-shot classification using your existing
    sentence-transformers model.
    """
    
    # ============================================
    # ANCHOR EXAMPLES - Math Questions
    # ============================================
    MATH_ANCHORS = [
        # Algebra
        ("Solve x^2 - 5x + 6 = 0", "algebra", "equations"),
        ("Find the roots of the quadratic equation", "algebra", "equations"),
        ("Factor the polynomial x^3 - 8", "algebra", "polynomials"),
        ("Simplify the expression (x+2)(x-3)", "algebra", "expressions"),
        ("If 2x + 5 = 15, find x", "algebra", "equations"),
        ("What is the value of x if 3x = 27", "algebra", "equations"),
        
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
        
        # Statistics
        ("Find the mean of 2, 4, 6, 8, 10", "statistics", "central_tendency"),
        ("Calculate the standard deviation", "statistics", "dispersion"),
        ("What is the median of the dataset", "statistics", "central_tendency"),
        ("Find the variance of the given numbers", "statistics", "dispersion"),
        ("Calculate the correlation coefficient", "statistics", "correlation"),
        
        # Trigonometry
        ("Find the value of sin(30 degrees)", "trigonometry", "ratios"),
        ("Calculate cos(60) + sin(30)", "trigonometry", "ratios"),
        ("Solve the equation sin(x) = 0.5", "trigonometry", "equations"),
        ("Find tan(45) without using calculator", "trigonometry", "ratios"),
        ("Prove the identity sin^2(x) + cos^2(x) = 1", "trigonometry", "identities"),
        
        # Geometry
        ("Find the area of a circle with radius 5", "geometry", "area"),
        ("Calculate the perimeter of a rectangle", "geometry", "perimeter"),
        ("What is the volume of a sphere with radius 3", "geometry", "volume"),
        ("Find the hypotenuse of a right triangle", "geometry", "triangles"),
        
        # Linear Algebra
        ("Find the determinant of the matrix", "linear_algebra", "determinants"),
        ("Calculate the inverse of matrix A", "linear_algebra", "matrices"),
        ("Find the eigenvalues of the matrix", "linear_algebra", "eigenvalues"),
        ("Multiply the two matrices", "linear_algebra", "operations"),
        
        # Word Problems
        ("John has 5 apples and Mary gives him 3 more. How many does he have?", "algebra", "word_problem"),
        ("A train travels at 60 km/h for 2 hours. Find the distance.", "algebra", "word_problem"),
        ("If the price is increased by 20%, what is the new price?", "algebra", "word_problem"),
        ("Ram is twice as old as Shyam. Sum of ages is 36. Find their ages.", "algebra", "word_problem"),
        ("A shopkeeper gives 10% discount. Find the selling price.", "algebra", "word_problem"),
        ("How many ways can 3 books be selected from 10 books?", "probability", "combinations"),
    ]
    
    # ============================================
    # ANCHOR EXAMPLES - Non-Math Questions
    # ============================================
    NON_MATH_ANCHORS = [
        # Geography
        "What is the capital of France",
        "Name the longest river in the world",
        "Which country has the largest population",
        "Where is Mount Everest located",
        "What is the currency of Japan",
        
        # History
        "When did World War 2 end",
        "Who was the first president of USA",
        "What caused the French Revolution",
        "When was India independence",
        
        # Science (non-math)
        "What is photosynthesis",
        "Explain the structure of DNA",
        "What is the function of mitochondria",
        "How does the heart pump blood",
        "What is Newton's first law",
        
        # Literature
        "Who wrote Romeo and Juliet",
        "What is the theme of Hamlet",
        "Summarize the novel Pride and Prejudice",
        
        # General Knowledge
        "Who is the CEO of Tesla",
        "What is the latest iPhone model",
        "Tell me about climate change",
        
        # Entertainment
        "Recommend a good movie",
        "Who won the Oscar for best actor",
        "What is the most popular video game",
        
        # Personal/Chat
        "How are you today",
        "Tell me a joke",
        "What is your name",
        "What can you do",
        "Hello how are you",
        
        # Cooking/Lifestyle
        "How to make pasta",
        "What is the recipe for chocolate cake",
        "Best restaurants in New York",
        
        # Technology (non-math)
        "How to code in Python",
        "What is machine learning",
        "Explain blockchain technology",
    ]
    
    def __init__(self, embedding_model=None):
        """
        Initialize the classifier.
        
        Args:
            embedding_model: Optional pre-loaded SentenceTransformer model.
                           If None, will try to load from RAG retriever.
        """
        self.model = embedding_model
        self.math_embeddings = None
        self.non_math_embeddings = None
        self.math_topics = None
        self._initialized = False
    
    def initialize(self, model=None):
        """
        Initialize embeddings for anchor examples.
        Call this once after loading the model.
        """
        if self._initialized:
            return
        
        if model is not None:
            self.model = model
        
        if self.model is None:
            # Try to load model
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
                logger.info("Loaded sentence-transformers model for classifier")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                return
        
        # Compute embeddings for math anchors
        math_texts = [anchor[0] for anchor in self.MATH_ANCHORS]
        self.math_embeddings = self.model.encode(math_texts, convert_to_numpy=True)
        self.math_topics = [(anchor[1], anchor[2]) for anchor in self.MATH_ANCHORS]
        
        # Compute embeddings for non-math anchors
        self.non_math_embeddings = self.model.encode(self.NON_MATH_ANCHORS, convert_to_numpy=True)
        
        self._initialized = True
        logger.info(f"Math classifier initialized with {len(self.MATH_ANCHORS)} math and {len(self.NON_MATH_ANCHORS)} non-math anchors")
    
    def classify(self, text: str) -> ClassificationResult:
        """
        Classify whether input is a math question.
        
        Args:
            text: Input text to classify
            
        Returns:
            ClassificationResult with is_math, confidence, and topic
        """
        if not self._initialized:
            self.initialize()
        
        if self.model is None:
            # Fallback to rule-based if model not available
            return self._rule_based_classify(text)
        
        # Encode the input
        input_embedding = self.model.encode([text], convert_to_numpy=True)[0]
        
        # Compute similarities to math anchors
        math_similarities = self._cosine_similarity(input_embedding, self.math_embeddings)
        
        # Compute similarities to non-math anchors
        non_math_similarities = self._cosine_similarity(input_embedding, self.non_math_embeddings)
        
        # Get top similarities
        top_math_sim = np.max(math_similarities)
        top_math_idx = np.argmax(math_similarities)
        avg_math_sim = np.mean(np.sort(math_similarities)[-5:])  # Top 5 average
        
        top_non_math_sim = np.max(non_math_similarities)
        avg_non_math_sim = np.mean(np.sort(non_math_similarities)[-5:])
        
        # Determine if math
        # Use the gap between math and non-math similarity
        math_score = (avg_math_sim + top_math_sim) / 2
        non_math_score = (avg_non_math_sim + top_non_math_sim) / 2
        
        # Calculate confidence using softmax-like approach
        score_diff = math_score - non_math_score
        confidence = 1 / (1 + np.exp(-10 * score_diff))  # Sigmoid with scaling
        
        # Also apply rule-based boost for obvious math
        rule_boost = self._get_rule_boost(text)
        confidence = min(1.0, confidence + rule_boost)
        
        is_math = confidence >= 0.5
        
        # Get topic from best matching math anchor
        if is_math:
            topic, subtopic = self.math_topics[top_math_idx]
        else:
            topic, subtopic = "not_math", "general"
        
        # Generate reasoning
        if is_math:
            reasoning = f"Semantically similar to math questions (score: {math_score:.2f})"
        else:
            reasoning = f"More similar to non-math content (math: {math_score:.2f}, non-math: {non_math_score:.2f})"
        
        return ClassificationResult(
            is_math=is_math,
            confidence=confidence,
            topic=topic,
            subtopic=subtopic,
            reasoning=reasoning
        )
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between vec1 and each row of vec2."""
        # Normalize
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + 1e-8)
        
        # Dot product
        return np.dot(vec2_norm, vec1_norm)
    
    def _get_rule_boost(self, text: str) -> float:
        """
        Rule-based confidence boost for obvious math patterns.
        This catches things the semantic model might miss.
        """
        boost = 0.0
        text_lower = text.lower()
        
        # Strong math indicators (give significant boost)
        strong_patterns = [
            r'[xyz]\s*[\^²³]\s*[+\-=]',  # x^2 +, x² -, etc
            r'\d+\s*[+\-*/^]\s*\d+',      # 2 + 3, 5 * 4
            r'[xyz]\s*=\s*\d',            # x = 5
            r'\d+\s*[xyz]',               # 2x, 3y
            r'√|∫|∑|∏|∂|∞|π|θ',          # Math symbols
            r'\bsin\b|\bcos\b|\btan\b',   # Trig functions
            r'\bd/dx\b|\bdy/dx\b',        # Calculus notation
            r'\blog\b|\bln\b',            # Logarithms
            r'\blim\b',                   # Limits
            r'\bmatrix\b|\bdet\b',        # Linear algebra
            r'=\s*0\b',                   # Equation equals zero
        ]
        
        for pattern in strong_patterns:
            import re
            if re.search(pattern, text, re.IGNORECASE):
                boost += 0.15
        
        # Word problem indicators (moderate boost)
        word_problem_patterns = [
            r'\bhow\s+many\b',
            r'\bhow\s+much\b',
            r'\bfind\s+the\b',
            r'\bcalculate\b',
            r'\bwhat\s+is\s+the\s+(probability|value|sum|mean|area|volume)\b',
            r'\bsolve\b',
            r'\bif\s+.{5,30}(then|find|what)\b',
            r'\b(probability|chance|odds)\s+of\b',
        ]
        
        for pattern in word_problem_patterns:
            import re
            if re.search(pattern, text_lower):
                boost += 0.1
        
        return min(0.4, boost)  # Cap the boost
    
    def _rule_based_classify(self, text: str) -> ClassificationResult:
        """Fallback rule-based classification if embedding model unavailable."""
        boost = self._get_rule_boost(text)
        
        is_math = boost >= 0.2
        confidence = min(1.0, boost + 0.3) if is_math else max(0.0, 0.3 - boost)
        
        # Simple topic detection
        text_lower = text.lower()
        if any(kw in text_lower for kw in ['probability', 'chance', 'odds', 'coin', 'dice', 'cards']):
            topic, subtopic = "probability", "basic"
        elif any(kw in text_lower for kw in ['derivative', 'integral', 'limit', 'differentiate']):
            topic, subtopic = "calculus", "general"
        elif any(kw in text_lower for kw in ['sin', 'cos', 'tan', 'angle']):
            topic, subtopic = "trigonometry", "general"
        elif any(kw in text_lower for kw in ['mean', 'median', 'mode', 'deviation']):
            topic, subtopic = "statistics", "general"
        elif any(kw in text_lower for kw in ['matrix', 'determinant', 'vector']):
            topic, subtopic = "linear_algebra", "general"
        else:
            topic, subtopic = "algebra", "general"
        
        return ClassificationResult(
            is_math=is_math,
            confidence=confidence,
            topic=topic if is_math else "not_math",
            subtopic=subtopic,
            reasoning="Rule-based classification (embedding model not available)"
        )
    
    def get_topic(self, text: str) -> Tuple[str, str]:
        """Get just the topic and subtopic."""
        result = self.classify(text)
        return result.topic, result.subtopic


# ============================================
# Singleton for reuse
# ============================================
_classifier_instance: Optional[MathClassifier] = None


def get_math_classifier(model=None) -> MathClassifier:
    """Get or create the classifier singleton."""
    global _classifier_instance
    
    if _classifier_instance is None:
        _classifier_instance = MathClassifier(model)
    
    return _classifier_instance


# ============================================
# Test
# ============================================
if __name__ == "__main__":
    print("=" * 70)
    print("MATH CLASSIFIER TEST")
    print("=" * 70)
    
    classifier = MathClassifier()
    classifier.initialize()
    
    test_cases = [
        # Should be MATH (high confidence)
        ("Solve x^2 - 5x + 6 = 0", True),
        ("What is 2 + 2?", True),
        ("Find the derivative of x^3", True),
        ("What is the probability of getting heads in a coin toss", True),
        ("Calculate the area of a circle with radius 5", True),
        ("Find the mean of 10, 20, 30, 40, 50", True),
        ("If x + 5 = 10, find x", True),
        ("John has 5 apples. Mary gives 3. How many total?", True),
        ("A train goes 60 km/h for 2 hours. Find distance.", True),
        ("What is sin(30)?", True),
        ("Find the determinant of matrix [[1,2],[3,4]]", True),
        ("How many ways can 5 people stand in a line?", True),
        ("Find the probability of drawing a king from 52 cards", True),
        ("x²+2=0", True),  # The problematic case!
        
        # Should be NOT MATH
        ("What is the capital of France?", False),
        ("Who wrote Romeo and Juliet?", False),
        ("Tell me a joke", False),
        ("How to cook pasta?", False),
        ("What is photosynthesis?", False),
        ("Recommend a good movie", False),
        ("Who is the president of USA?", False),
        ("Explain machine learning", False),
    ]
    
    correct = 0
    total = len(test_cases)
    
    print("\n")
    for text, expected_math in test_cases:
        result = classifier.classify(text)
        
        is_correct = result.is_math == expected_math
        if is_correct:
            correct += 1
            status = "✅"
        else:
            status = "❌"
        
        # Display
        conf_display = f"{result.confidence:.0%}"
        topic_display = f"{result.topic}/{result.subtopic}" if result.is_math else "not_math"
        
        print(f"{status} [{conf_display:>4}] [{topic_display:<25}] {text[:50]}...")
        
        if not is_correct:
            print(f"         Expected: {'MATH' if expected_math else 'NOT MATH'}, Got: {'MATH' if result.is_math else 'NOT MATH'}")
            print(f"         Reason: {result.reasoning}")
    
    print("\n" + "=" * 70)
    print(f"ACCURACY: {correct}/{total} ({correct/total*100:.1f}%)")
    print("=" * 70)