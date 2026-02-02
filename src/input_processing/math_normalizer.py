import re

class MathNormalizer:
    """Convert spoken math phrases to mathematical notation."""
    
    PHRASE_MAP = {
        # Basic operations
        r"\bsquare root of\b": "√",
        r"\bcube root of\b": "∛",
        r"\braised to the power of?\b": "^",
        r"\bto the power of?\b": "^",
        r"\bsquared\b": "^2",
        r"\bcubed\b": "^3",
        r"\bdivided by\b": "/",
        r"\bmultiplied by\b": "*",
        r"\btimes\b": "*",
        r"\bplus\b": "+",
        r"\bminus\b": "-",
        r"\bequals?\b": "=",
        
        # Greek letters
        r"\balpha\b": "α",
        r"\bbeta\b": "β",
        r"\bgamma\b": "γ",
        r"\bdelta\b": "δ",
        r"\btheta\b": "θ",
        r"\blambda\b": "λ",
        r"\bsigma\b": "σ",
        r"\bpi\b": "π",
        
        # Calculus
        r"\bintegral of\b": "∫",
        r"\bderivative of\b": "d/dx",
        r"\blimit as\b": "lim",
        r"\binfinity\b": "∞",
        
        # Common phrases
        r"\bx squared\b": "x^2",
        r"\by squared\b": "y^2",
        r"\bsummation\b": "Σ",
        r"\bfor all\b": "∀",
        r"\bthere exists\b": "∃",
    }
    
    def normalize(self, text: str) -> str:
        """Apply all normalizations to text."""
        result = text.lower()
        
        for pattern, replacement in self.PHRASE_MAP.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result