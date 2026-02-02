# src/agents/tools/rag_tool.py
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from src.rag.retriever import RAGRetriever

class RAGInput(BaseModel):
    query: str = Field(description="Math concept or formula to search for")

class RAGTool(BaseTool):
    name: str = "MathKnowledgeBase"
    description: str = """
    Searches the math knowledge base for relevant formulas, theorems, 
    solution templates, and common mistakes. Use this to find:
    - Relevant formulas for a problem type
    - Solution strategies
    - Common pitfalls to avoid
    """
    args_schema: Type[BaseModel] = RAGInput
    retriever: RAGRetriever = None
    
    def __init__(self):
        super().__init__()
        self.retriever = RAGRetriever()
    
    def _run(self, query: str) -> str:
        results = self.retriever.retrieve(query, top_k=3)
        
        if not results:
            return "No relevant information found in knowledge base."
        
        formatted = "Retrieved from Knowledge Base:\n\n"
        for i, doc in enumerate(results, 1):
            formatted += f"[{i}] {doc['content']}\n"
            formatted += f"    Source: {doc['source']}\n\n"
        
        return formatted