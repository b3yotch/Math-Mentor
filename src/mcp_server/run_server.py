"""
Entry point to run the Math Mentor MCP server.

Usage:
    mcp dev src/mcp_server/run_server.py
    
    OR:
    python -m src.mcp_server.run_server
    
    OR for Claude Desktop:
    Add to claude_desktop_config.json
"""

import os
import sys

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from src.mcp_server.server import create_server
from src.mcp_server.config import MCPConfig

# ============================================================
# GLOBAL SERVER OBJECT — required by `mcp dev`
# Must be named `mcp`, `server`, or `app`
# ============================================================
config = MCPConfig()
mcp = create_server(config)


def main():
    """Start the MCP server (used when running directly)."""
    print(f"Starting {config.server_name} v{config.server_version}")
    print(f"Description: {config.description}")
    print("Tools: solve_math_problem, retrieve_math_context, check_input_safety,")
    print("       get_problem_history, find_similar_problems, record_feedback,")
    print("       evaluate_solution, get_system_stats, run_batch_evaluation,")
    print("       normalize_math")
    print("Resources: math://topics, math://stats, math://test-dataset")
    print("Prompts: solve_step_by_step, study_topic, review_performance, evaluate_system")
    
    mcp.run()


if __name__ == "__main__":
    main()