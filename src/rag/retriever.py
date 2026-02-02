"""
Updated RAG Retriever for Math Mentor
Integrates with the new knowledge base built from textbooks.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

# Import the knowledge base builder
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_pipeline.knowledge_builder import (
    MathKnowledgeBaseBuilder,
    TopicSpecificKnowledgeBase
)

logger = logging.getLogger(__name__)


class MathRAGRetriever:
    """
    Retriever for mathematical content using FAISS vector store.
    """
    
    def __init__(
        self,
        knowledge_base_path: str = "data/vector_store",
        use_topic_indices: bool = True,
        fallback_to_json: bool = True
    ):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.use_topic_indices = use_topic_indices
        self.fallback_to_json = fallback_to_json
        
        self.kb: Optional[TopicSpecificKnowledgeBase] = None
        self.json_knowledge: Dict = {}
        
        self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """Load the knowledge base"""
        try:
            if self.use_topic_indices and (self.knowledge_base_path / "topics.json").exists():
                logger.info("Loading topic-specific knowledge base...")
                self.kb = TopicSpecificKnowledgeBase()
                self.kb.load(str(self.knowledge_base_path))
                logger.info("✓ Knowledge base loaded successfully")
            elif (self.knowledge_base_path / "global" / "faiss.index").exists():
                logger.info("Loading global knowledge base...")
                self.kb = MathKnowledgeBaseBuilder()
                self.kb.load(str(self.knowledge_base_path / "global"))
                logger.info("✓ Knowledge base loaded successfully")
            else:
                logger.warning("No FAISS index found")
                
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
        
        # Load JSON knowledge base as fallback
        if self.fallback_to_json:
            self._load_json_knowledge()
    
    def _load_json_knowledge(self):
        """Load JSON knowledge base files"""
        json_path = self.knowledge_base_path.parent / "knowledge_base"
        
        if not json_path.exists():
            # Try the old location
            json_path = Path("src/rag/knowledge_base")
        
        if json_path.exists():
            for json_file in json_path.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        topic_name = json_file.stem
                        self.json_knowledge[topic_name] = json.load(f)
                        logger.info(f"Loaded JSON knowledge: {topic_name}")
                except Exception as e:
                    logger.warning(f"Failed to load {json_file}: {e}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        topic_hint: Optional[str] = None,
        content_type_filter: Optional[str] = None,
        min_score: float = 0.3
    ) -> List[Dict]:
        """
        Retrieve relevant content for a query.
        
        Args:
            query: The search query
            top_k: Number of results to return
            topic_hint: Optional topic to focus search (calculus, algebra, etc.)
            content_type_filter: Filter by content type (definition, theorem, example)
            min_score: Minimum similarity score threshold
            
        Returns:
            List of relevant documents with scores
        """
        results = []
        
        # Try FAISS retrieval first
        if self.kb is not None:
            try:
                if hasattr(self.kb, 'search'):
                    raw_results = self.kb.search(query, top_k=top_k * 2, topic=topic_hint)
                else:
                    raw_results = self.kb.global_index.search(query, top_k=top_k * 2)
                
                for result in raw_results:
                    score = result.get("score", 0)
                    
                    if score < min_score:
                        continue
                    
                    chunk = result.get("chunk", {})
                    
                    # Apply content type filter
                    if content_type_filter:
                        if content_type_filter not in chunk.get("content_type", ""):
                            continue
                    
                    results.append({
                        "content": chunk.get("content", ""),
                        "type": chunk.get("content_type", "unknown"),
                        "topic": chunk.get("topic", "general"),
                        "subtopic": chunk.get("subtopic", "general"),
                        "source": chunk.get("source", ""),
                        "page": chunk.get("page_number", 0),
                        "score": score,
                        "metadata": chunk.get("metadata", {})
                    })
                    
                    if len(results) >= top_k:
                        break
                        
            except Exception as e:
                logger.error(f"FAISS retrieval failed: {e}")
        
        # Fallback to JSON if needed
        if len(results) < top_k and self.fallback_to_json:
            json_results = self._search_json_knowledge(
                query, 
                top_k - len(results),
                topic_hint
            )
            results.extend(json_results)
        
        return results[:top_k]
    
    def _search_json_knowledge(
        self,
        query: str,
        top_k: int,
        topic_hint: Optional[str] = None
    ) -> List[Dict]:
        """Search JSON knowledge base (simple keyword matching)"""
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        topics_to_search = [topic_hint] if topic_hint else self.json_knowledge.keys()
        
        for topic in topics_to_search:
            if topic not in self.json_knowledge:
                continue
                
            topic_data = self.json_knowledge[topic]
            
            for subtopic, content in topic_data.get("subtopics", {}).items():
                # Search definitions
                for definition in content.get("definitions", []):
                    score = self._calculate_keyword_score(
                        definition.get("content", ""),
                        query_words
                    )
                    if score > 0:
                        results.append({
                            "content": definition.get("content", ""),
                            "type": "definition",
                            "topic": topic,
                            "subtopic": subtopic,
                            "source": definition.get("source", ""),
                            "score": score * 0.8,  # Discount for keyword search
                            "metadata": {}
                        })
                
                # Search theorems
                for theorem in content.get("theorems", []):
                    score = self._calculate_keyword_score(
                        theorem.get("content", ""),
                        query_words
                    )
                    if score > 0:
                        results.append({
                            "content": theorem.get("content", ""),
                            "type": "theorem",
                            "topic": topic,
                            "subtopic": subtopic,
                            "source": theorem.get("source", ""),
                            "score": score * 0.8,
                            "metadata": {}
                        })
                
                # Search examples
                for example in content.get("examples", []):
                    score = self._calculate_keyword_score(
                        example.get("content", ""),
                        query_words
                    )
                    if score > 0:
                        results.append({
                            "content": example.get("content", ""),
                            "type": "example",
                            "topic": topic,
                            "subtopic": subtopic,
                            "source": example.get("source", ""),
                            "score": score * 0.8,
                            "metadata": example.get("metadata", {})
                        })
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def _calculate_keyword_score(self, text: str, query_words: set) -> float:
        """Calculate simple keyword match score"""
        text_lower = text.lower()
        text_words = set(text_lower.split())
        
        matches = query_words.intersection(text_words)
        
        if not matches:
            return 0.0
        
        return len(matches) / len(query_words)
    
    def get_relevant_formulas(self, topic: str, subtopic: Optional[str] = None) -> List[Dict]:
        """Get formulas for a specific topic"""
        formulas = []
        
        # From FAISS
        if self.kb is not None:
            results = self.kb.search(
                f"{topic} {subtopic or ''} formula",
                top_k=10,
                topic=topic
            )
            
            for result in results:
                chunk = result.get("chunk", {})
                if "formula" in chunk.get("content_type", ""):
                    formulas.append({
                        "formula": chunk.get("metadata", {}).get("raw_formula", ""),
                        "context": chunk.get("content", ""),
                        "source": chunk.get("source", "")
                    })
        
        # From JSON
        if topic in self.json_knowledge:
            topic_data = self.json_knowledge[topic]
            for st, content in topic_data.get("subtopics", {}).items():
                if subtopic and st != subtopic:
                    continue
                for formula in content.get("formulas", []):
                    formulas.append({
                        "formula": formula.get("content", ""),
                        "context": f"From {topic}/{st}",
                        "source": formula.get("source", "")
                    })
        
        return formulas
    
    def get_similar_examples(self, problem: str, top_k: int = 3) -> List[Dict]:
        """Find similar solved examples for a given problem"""
        return self.retrieve(
            query=problem,
            top_k=top_k,
            content_type_filter="example"
        )
    
    def format_context_for_llm(self, results: List[Dict], max_tokens: int = 2000) -> str:
        """Format retrieved results as context for LLM"""
        context_parts = []
        current_length = 0
        
        for result in results:
            content = result.get("content", "")
            content_type = result.get("type", "")
            topic = result.get("topic", "")
            
            formatted = f"[{content_type.upper()} - {topic}]\n{content}\n"
            
            if current_length + len(formatted) > max_tokens * 4:  # Rough char estimate
                break
            
            context_parts.append(formatted)
            current_length += len(formatted)
        
        return "\n---\n".join(context_parts)


# Singleton instance for easy access
_retriever_instance: Optional[MathRAGRetriever] = None


def get_retriever() -> MathRAGRetriever:
    """Get or create the retriever singleton"""
    global _retriever_instance
    
    if _retriever_instance is None:
        _retriever_instance = MathRAGRetriever()
    
    return _retriever_instance