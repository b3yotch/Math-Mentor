# src/rag/query_expander.py
"""
Query Expander for Math Mentor RAG.
Expands user queries with related terms for better retrieval.
"""

import re
from typing import List, Dict, Set
from dataclasses import dataclass


@dataclass
class ExpandedQuery:
    """Result of query expansion."""
    original: str
    expanded: str
    added_terms: List[str]
    detected_topic: str


class QueryExpander:
    """
    Expands math queries with related terms and synonyms.
    This improves retrieval by matching documents that use
    different terminology for the same concepts.
    """
    
    # Topic-specific expansion terms
    EXPANSIONS = {
        # Probability
        "probability": {
            "keywords": ["probability", "chance", "odds", "likelihood", "P("],
            "expansions": [
                "favorable outcomes", "total outcomes", "sample space",
                "event", "random experiment"
            ]
        },
        "cards": {
            "keywords": ["card", "cards", "deck", "king", "queen", "jack", "ace", 
                        "hearts", "spades", "diamonds", "clubs"],
            "expansions": [
                "52 cards", "standard deck", "playing cards", "suits",
                "face cards", "probability cards"
            ]
        },
        "coins": {
            "keywords": ["coin", "heads", "tails", "toss", "flip"],
            "expansions": [
                "coin toss", "fair coin", "probability heads tails",
                "binomial", "bernoulli trial"
            ]
        },
        "dice": {
            "keywords": ["dice", "die", "roll", "rolling"],
            "expansions": [
                "six-sided", "fair dice", "probability rolling",
                "outcomes dice"
            ]
        },
        
        # Calculus
        "derivative": {
            "keywords": ["derivative", "differentiate", "differentiation", "d/dx", "dy/dx"],
            "expansions": [
                "rate of change", "slope", "tangent", "differential",
                "chain rule", "product rule", "quotient rule"
            ]
        },
        "integral": {
            "keywords": ["integral", "integrate", "integration", "antiderivative"],
            "expansions": [
                "area under curve", "indefinite integral", "definite integral",
                "integration techniques", "substitution", "by parts"
            ]
        },
        "limit": {
            "keywords": ["limit", "lim", "approaches", "tends to"],
            "expansions": [
                "limit definition", "continuity", "L'Hopital",
                "squeeze theorem", "limit laws"
            ]
        },
        
        # Algebra
        "quadratic": {
            "keywords": ["quadratic", "x^2", "x²", "parabola"],
            "expansions": [
                "quadratic formula", "roots", "discriminant",
                "factoring", "completing square", "ax² + bx + c"
            ]
        },
        "equation": {
            "keywords": ["equation", "solve", "solution", "root", "zero"],
            "expansions": [
                "find x", "solving equations", "algebraic equation"
            ]
        },
        "polynomial": {
            "keywords": ["polynomial", "degree", "coefficient"],
            "expansions": [
                "factor theorem", "remainder theorem", "roots polynomial"
            ]
        },
        
        # Trigonometry
        "trigonometry": {
            "keywords": ["sin", "cos", "tan", "sine", "cosine", "tangent",
                        "trig", "trigonometric"],
            "expansions": [
                "trigonometric functions", "trig identities",
                "unit circle", "radians degrees"
            ]
        },
        
        # Statistics
        "mean": {
            "keywords": ["mean", "average", "arithmetic mean"],
            "expansions": [
                "central tendency", "sum divided by n", "expected value"
            ]
        },
        "standard_deviation": {
            "keywords": ["standard deviation", "std", "σ", "variance"],
            "expansions": [
                "dispersion", "spread", "deviation from mean",
                "variance square root"
            ]
        },
        
        # Linear Algebra
        "matrix": {
            "keywords": ["matrix", "matrices", "determinant"],
            "expansions": [
                "matrix operations", "inverse matrix", "transpose",
                "eigenvalue", "eigenvector"
            ]
        },
        
        # Geometry
        "area": {
            "keywords": ["area", "surface area"],
            "expansions": [
                "formula area", "square units", "calculate area"
            ]
        },
        "circle": {
            "keywords": ["circle", "radius", "diameter", "circumference"],
            "expansions": [
                "pi r squared", "πr²", "2πr", "area circle"
            ]
        },
    }
    
    # Math question patterns to preserve
    QUESTION_PATTERNS = [
        (r'\bwhat\s+is\b', 'find'),
        (r'\bcalculate\b', 'find compute'),
        (r'\bdetermine\b', 'find'),
        (r'\bevaluate\b', 'find compute calculate'),
    ]
    
    def __init__(self):
        """Initialize the query expander."""
        # Build keyword to topic mapping
        self._keyword_to_topic: Dict[str, str] = {}
        for topic, data in self.EXPANSIONS.items():
            for keyword in data["keywords"]:
                self._keyword_to_topic[keyword.lower()] = topic
    
    def expand(self, query: str, max_expansions: int = 10) -> ExpandedQuery:
        """
        Expand a query with related terms.
        
        Args:
            query: Original query
            max_expansions: Maximum number of terms to add
            
        Returns:
            ExpandedQuery with original, expanded, and metadata
        """
        query_lower = query.lower()
        detected_topics: Set[str] = set()
        expansion_terms: Set[str] = set()
        
        # Detect topics from keywords
        for keyword, topic in self._keyword_to_topic.items():
            if keyword in query_lower:
                detected_topics.add(topic)
        
        # Gather expansion terms from detected topics
        for topic in detected_topics:
            if topic in self.EXPANSIONS:
                expansion_terms.update(self.EXPANSIONS[topic]["expansions"])
        
        # Also check for numeric patterns
        if re.search(r'\b52\b', query):
            expansion_terms.update(["deck of cards", "standard deck", "playing cards"])
        if re.search(r'\b6\b.*\b(dice|die|sided)\b', query_lower):
            expansion_terms.update(["six-sided die", "fair dice"])
        
        # Remove terms already in query
        query_words = set(query_lower.split())
        expansion_terms = {
            term for term in expansion_terms
            if not any(word in query_words for word in term.lower().split())
        }
        
        # Limit expansions
        expansion_list = list(expansion_terms)[:max_expansions]
        
        # Build expanded query
        expanded = query
        if expansion_list:
            expanded = f"{query} {' '.join(expansion_list)}"
        
        # Detect primary topic
        primary_topic = list(detected_topics)[0] if detected_topics else "general"
        
        return ExpandedQuery(
            original=query,
            expanded=expanded,
            added_terms=expansion_list,
            detected_topic=primary_topic
        )
    
    def get_topic_keywords(self, topic: str) -> List[str]:
        """Get keywords for a specific topic."""
        if topic in self.EXPANSIONS:
            return self.EXPANSIONS[topic]["keywords"]
        return []


# Singleton
_expander_instance = None

def get_query_expander() -> QueryExpander:
    global _expander_instance
    if _expander_instance is None:
        _expander_instance = QueryExpander()
    return _expander_instance


# Test
if __name__ == "__main__":
    expander = QueryExpander()
    
    test_queries = [
        "What is the probability of drawing a king from 52 cards",
        "Find the derivative of x^3",
        "Probability of getting heads in a coin toss",
        "Calculate the mean of 10, 20, 30",
        "Solve x^2 - 5x + 6 = 0",
        "Find the area of a circle with radius 5",
        "What is sin(30)?",
    ]
    
    print("=" * 70)
    print("QUERY EXPANSION TEST")
    print("=" * 70)
    
    for query in test_queries:
        result = expander.expand(query)
        print(f"\n📝 Original: {query}")
        print(f"🔍 Topic: {result.detected_topic}")
        print(f"➕ Added: {result.added_terms[:5]}")
        print(f"📄 Expanded: {result.expanded[:100]}...")