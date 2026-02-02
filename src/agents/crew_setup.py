# src/agents/crew_setup.py
"""
CrewAI setup using Groq LLM.
"""


import os
import json
from typing import Dict, Any
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from dotenv import load_dotenv
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Load environment variables
load_dotenv()

# Import tools
from .tools.calculator import CalculatorTool

# Import prompts
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.prompts import AGENT_PROMPTS


class MathMentorCrew:
    """
    Multi-agent system for solving math problems using Groq.
    
    Agents:
    1. Parser Agent - Structures the problem
    2. Router Agent - Determines solution strategy  
    3. Solver Agent - Solves the problem
    4. Verifier Agent - Verifies the solution
    5. Explainer Agent - Explains the solution
    """
    
    def __init__(self, model_name: str = "meta-llama/llama-guard-4-12b"):
        """
        Initialize the Math Mentor crew with Groq.
        
        Args:
            model_name: Groq model to use. Options:
                - "llama-3.1-70b-versatile" (recommended)
                - "llama-3.1-8b-instant" (faster)
                - "mixtral-8x7b-32768"
        """
        self.llm = self._setup_llm(model_name)
        self.calculator = CalculatorTool()
        self._setup_agents()
        os.environ.pop("OPENAI_API_KEY", None)
        self.llm = self._setup_llm(model_name)
    def _setup_llm(self, model_name: str):
        """Set up Groq LLM."""
        api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found!\n"
                "1. Get key from: https://console.groq.com/keys\n"
                "2. Create .env file with: GROQ_API_KEY=your_key_here"
            )
        
        return ChatGroq(
            model=f"groq/{model_name}",
            api_key=api_key,
            temperature=0.1
        )

    
    def _setup_agents(self):
        """Create all agents."""
        
        # Parser Agent
        self.parser_agent = Agent(
            role=AGENT_PROMPTS["parser"]["role"],
            goal=AGENT_PROMPTS["parser"]["goal"],
            backstory=AGENT_PROMPTS["parser"]["backstory"],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        # Router Agent
        self.router_agent = Agent(
            role=AGENT_PROMPTS["router"]["role"],
            goal=AGENT_PROMPTS["router"]["goal"],
            backstory=AGENT_PROMPTS["router"]["backstory"],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        # Solver Agent (with calculator)
        self.solver_agent = Agent(
            role=AGENT_PROMPTS["solver"]["role"],
            goal=AGENT_PROMPTS["solver"]["goal"],
            backstory=AGENT_PROMPTS["solver"]["backstory"],
            llm=self.llm,
            tools=[self.calculator],
            verbose=True,
            allow_delegation=False
        )
        
        # Verifier Agent (with calculator)
        self.verifier_agent = Agent(
            role=AGENT_PROMPTS["verifier"]["role"],
            goal=AGENT_PROMPTS["verifier"]["goal"],
            backstory=AGENT_PROMPTS["verifier"]["backstory"],
            llm=self.llm,
            tools=[self.calculator],
            verbose=True,
            allow_delegation=False
        )
        
        # Explainer Agent
        self.explainer_agent = Agent(
            role=AGENT_PROMPTS["explainer"]["role"],
            goal=AGENT_PROMPTS["explainer"]["goal"],
            backstory=AGENT_PROMPTS["explainer"]["backstory"],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
    
    def solve_problem(self, problem_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Solve a math problem using the agent crew.
        
        Args:
            problem_text: The math problem to solve
            context: Optional context from Module 1
        
        Returns:
            Dictionary with solution details
        """
        context = context or {}
        
        # Task 1: Parse
        parse_task = Task(
            description=f"""
            Parse this math problem:
            
            Problem: {problem_text}
            Context: {json.dumps(context)}
            
            Return JSON:
            {{
                "problem_text": "cleaned problem",
                "topic": "algebra/calculus/probability/linear_algebra",
                "variables": ["x", "y"],
                "goal": "what to find",
                "needs_clarification": false
            }}
            """,
            expected_output="JSON with structured problem",
            agent=self.parser_agent
        )
        
        # Task 2: Route
        route_task = Task(
            description="""
            Determine solution strategy.
            
            Return JSON:
            {{
                "approach": "solution method",
                "formulas": ["relevant formulas"],
                "steps_outline": ["step1", "step2"],
                "difficulty": 1-5
            }}
            """,
            expected_output="JSON with solution strategy",
            agent=self.router_agent,
            context=[parse_task]
        )
        
        # Task 3: Solve
        solve_task = Task(
            description="""
            Solve the problem step by step.
            
            Use Calculator tool for computations:
            - "2 + 3 * 4" for arithmetic
            - "derivative(x^2, x)" for derivatives
            - "integrate(x^2, x)" for integrals
            - "solve(x^2 - 4, x)" for equations
            
            Show all steps and state final answer clearly.
            """,
            expected_output="Step-by-step solution with final answer",
            agent=self.solver_agent,
            context=[parse_task, route_task]
        )
        
        # Task 4: Verify
        verify_task = Task(
            description="""
            Verify the solution.
            
            Check calculations using Calculator tool.
            Substitute answer back if possible.
            
            Return JSON:
            {{
                "is_correct": true/false,
                "confidence": 0.0-1.0,
                "verification_steps": ["what checked"],
                "issues": ["any issues"]
            }}
            """,
            expected_output="Verification result",
            agent=self.verifier_agent,
            context=[parse_task, solve_task]
        )
        
        # Task 5: Explain
        explain_task = Task(
            description="""
            Explain the solution for a student.
            
            Include:
            1. Simple overview
            2. Why each step works
            3. Key concepts used
            4. Common mistakes to avoid
            """,
            expected_output="Student-friendly explanation",
            agent=self.explainer_agent,
            context=[parse_task, solve_task, verify_task]
        )
        
        # Create and run crew
        crew = Crew(
            agents=[
                self.parser_agent,
                self.router_agent,
                self.solver_agent,
                self.verifier_agent,
                self.explainer_agent
            ],
            tasks=[parse_task, route_task, solve_task, verify_task, explain_task],
            process=Process.sequential,
            verbose=True
        )
        
        result = crew.kickoff()
        
        return {
            "input": problem_text,
            "parsed_problem": parse_task.output.raw if parse_task.output else None,
            "strategy": route_task.output.raw if route_task.output else None,
            "solution": solve_task.output.raw if solve_task.output else None,
            "verification": verify_task.output.raw if verify_task.output else None,
            "explanation": explain_task.output.raw if explain_task.output else None,
            "final_result": result.raw if result else None
        }
    
    def get_agent_info(self) -> Dict[str, str]:
        """Get info about all agents."""
        return {
            "parser": self.parser_agent.role,
            "router": self.router_agent.role,
            "solver": self.solver_agent.role,
            "verifier": self.verifier_agent.role,
            "explainer": self.explainer_agent.role
        }