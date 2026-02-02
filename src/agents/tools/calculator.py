# src/agents/tools/calculator.py
"""
Calculator tool for mathematical computations - Fixed Version
"""

from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import sympy as sp
import re


class CalculatorInput(BaseModel):
    """Input schema for calculator tool."""
    expression: str = Field(description="Mathematical expression to evaluate")


class CalculatorTool(BaseTool):
    """Calculator for math operations."""
    
    name: str = "Calculator"
    description: str = """
    Evaluates mathematical expressions.
    
    Examples:
    - Basic: "2 + 3 * 4" → 14
    - Powers: "2^10" → 1024
    - Sqrt: "sqrt(16)" → 4
    - Derivative: "derivative(x^2, x)" → 2*x
    - Integral: "integrate(x^2, x)" → x**3/3
    - Solve: "solve(x^2 - 4, x)" → [-2, 2]
    - Factor: "factor(x^2 - 4)" → (x-2)*(x+2)
    - Expand: "expand((x+1)^2)" → x**2 + 2*x + 1
    - Simplify: "simplify((x^2-1)/(x-1))" → x + 1
    """
    args_schema: Type[BaseModel] = CalculatorInput
    
    def _run(self, expression: str) -> str:
        """Execute the calculation."""
        try:
            expression = expression.strip()
            expr_lower = expression.lower()
            
            # Define symbols
            x, y, z, a, b, c, n, t = sp.symbols('x y z a b c n t')
            
            # Handle derivative
            if 'derivative' in expr_lower or 'diff' in expr_lower:
                return self._derivative(expression)
            
            # Handle integral
            if 'integrate' in expr_lower or 'integral' in expr_lower:
                return self._integrate(expression)
            
            # Handle solve
            if 'solve' in expr_lower:
                return self._solve(expression)
            
            # Handle factor
            if 'factor' in expr_lower:
                return self._factor(expression)
            
            # Handle expand
            if 'expand' in expr_lower:
                return self._expand(expression)
            
            # Handle simplify
            if 'simplify' in expr_lower:
                return self._simplify(expression)
            
            # Handle limit
            if 'limit' in expr_lower:
                return self._limit(expression)
            
            # Standard numeric evaluation
            return self._evaluate(expression)
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _parse_expr(self, expr_str: str) -> sp.Expr:
        """Parse expression string to SymPy expression."""
        # Replace ^ with **
        expr_str = expr_str.replace('^', '**')
        
        # Define symbols
        x, y, z, a, b, c, n, t = sp.symbols('x y z a b c n t')
        
        # Create namespace for parsing
        namespace = {
            'x': x, 'y': y, 'z': z, 'a': a, 'b': b, 'c': c, 'n': n, 't': t,
            'pi': sp.pi, 'e': sp.E, 'E': sp.E,
            'sqrt': sp.sqrt, 
            'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
            'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
            'log': sp.log, 'ln': sp.ln, 'exp': sp.exp,
            'abs': sp.Abs,
            'factorial': sp.factorial,
            'oo': sp.oo,
        }
        
        return sp.sympify(expr_str, locals=namespace)
    
    def _derivative(self, expression: str) -> str:
        """Calculate derivative."""
        try:
            # Extract: derivative(expr, var)
            match = re.match(r'(?:derivative|diff)\s*$\s*(.+)\s*,\s*(\w)\s*$', expression, re.IGNORECASE)
            
            if match:
                expr_str = match.group(1)
                var_str = match.group(2)
                
                var = sp.Symbol(var_str)
                expr = self._parse_expr(expr_str)
                
                result = sp.diff(expr, var)
                return f"Derivative: {result}"
            
            return "Error: Use format derivative(expression, variable)"
            
        except Exception as e:
            return f"Error in derivative: {str(e)}"
    
    def _integrate(self, expression: str) -> str:
        """Calculate integral."""
        try:
            # Extract: integrate(expr, var)
            match = re.match(r'(?:integrate|integral)\s*$\s*(.+)\s*,\s*(\w)\s*$', expression, re.IGNORECASE)
            
            if match:
                expr_str = match.group(1)
                var_str = match.group(2)
                
                var = sp.Symbol(var_str)
                expr = self._parse_expr(expr_str)
                
                result = sp.integrate(expr, var)
                return f"Integral: {result}"
            
            return "Error: Use format integrate(expression, variable)"
            
        except Exception as e:
            return f"Error in integral: {str(e)}"
    
    def _solve(self, expression: str) -> str:
        """Solve equation."""
        try:
            # Extract: solve(equation, var)
            match = re.match(r'solve\s*$\s*(.+)\s*,\s*(\w)\s*$', expression, re.IGNORECASE)
            
            if match:
                eq_str = match.group(1)
                var_str = match.group(2)
                
                var = sp.Symbol(var_str)
                
                # Handle equations with '='
                if '=' in eq_str:
                    left, right = eq_str.split('=', 1)
                    expr = self._parse_expr(left) - self._parse_expr(right)
                else:
                    expr = self._parse_expr(eq_str)
                
                solutions = sp.solve(expr, var)
                return f"Solutions: {solutions}"
            
            return "Error: Use format solve(equation, variable)"
            
        except Exception as e:
            return f"Error in solve: {str(e)}"
    
    def _factor(self, expression: str) -> str:
        """Factor expression."""
        try:
            match = re.match(r'factor\s*$\s*(.+)\s*$', expression, re.IGNORECASE)
            
            if match:
                expr_str = match.group(1)
                expr = self._parse_expr(expr_str)
                result = sp.factor(expr)
                return f"Factored: {result}"
            
            return "Error: Use format factor(expression)"
            
        except Exception as e:
            return f"Error in factor: {str(e)}"
    
    def _expand(self, expression: str) -> str:
        """Expand expression."""
        try:
            match = re.match(r'expand\s*$\s*(.+)\s*$', expression, re.IGNORECASE)
            
            if match:
                expr_str = match.group(1)
                expr = self._parse_expr(expr_str)
                result = sp.expand(expr)
                return f"Expanded: {result}"
            
            return "Error: Use format expand(expression)"
            
        except Exception as e:
            return f"Error in expand: {str(e)}"
    
    def _simplify(self, expression: str) -> str:
        """Simplify expression."""
        try:
            match = re.match(r'simplify\s*$\s*(.+)\s*$', expression, re.IGNORECASE)
            
            if match:
                expr_str = match.group(1)
                expr = self._parse_expr(expr_str)
                result = sp.simplify(expr)
                return f"Simplified: {result}"
            
            return "Error: Use format simplify(expression)"
            
        except Exception as e:
            return f"Error in simplify: {str(e)}"
    
    def _limit(self, expression: str) -> str:
        """Calculate limit."""
        try:
            # limit(expr, var, point)
            match = re.match(r'limit\s*$\s*(.+)\s*,\s*(\w)\s*,\s*(.+)\s*$', expression, re.IGNORECASE)
            
            if match:
                expr_str = match.group(1)
                var_str = match.group(2)
                point_str = match.group(3)
                
                var = sp.Symbol(var_str)
                expr = self._parse_expr(expr_str)
                
                # Handle infinity
                if 'inf' in point_str.lower() or 'oo' in point_str:
                    point = sp.oo
                else:
                    point = self._parse_expr(point_str)
                
                result = sp.limit(expr, var, point)
                return f"Limit: {result}"
            
            return "Error: Use format limit(expression, variable, point)"
            
        except Exception as e:
            return f"Error in limit: {str(e)}"
    
    def _evaluate(self, expression: str) -> str:
        """Evaluate numeric expression."""
        try:
            expr = self._parse_expr(expression)
            
            # Try to evaluate numerically
            result = expr.evalf()
            
            # If it's a clean integer, return as int
            if result.is_number:
                float_val = float(result)
                if abs(float_val - round(float_val)) < 1e-10:
                    return str(int(round(float_val)))
                # Round to reasonable precision
                if abs(float_val) > 1e-6:
                    return str(round(float_val, 10)).rstrip('0').rstrip('.')
            
            return str(result)
            
        except Exception as e:
            # Fallback to Python eval for simple expressions
            try:
                import math
                expression = expression.replace('^', '**')
                result = eval(expression, {"__builtins__": {}}, {
                    'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos,
                    'tan': math.tan, 'log': math.log, 'exp': math.exp,
                    'pi': math.pi, 'e': math.e, 'abs': abs
                })
                if isinstance(result, float) and result.is_integer():
                    return str(int(result))
                return str(result)
            except:
                return f"Error: {str(e)}"