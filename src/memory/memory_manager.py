# src/memory/memory_manager.py
"""
High-level memory manager for Math Mentor.
Provides easy interface for memory operations.
"""

from typing import Dict, List, Optional, Any
from .database import MemoryDatabase


class MemoryManager:
    """
    High-level interface for memory operations.
    
    Handles:
    - Saving and retrieving solved problems
    - Learning from user feedback
    - Applying learned corrections
    - Finding similar problems
    """
    
    def __init__(self, db_path: str = None):
        """Initialize memory manager."""
        self.db = MemoryDatabase(db_path)
    
    def save_solved_problem(
        self,
        input_type: str,
        original_input: str,
        extracted_text: str,
        normalized_text: str = None,
        parsed_problem: Dict = None,
        topic: str = None,
        solution: str = None,
        explanation: str = None,
        verification_result: Dict = None,
        confidence_score: float = 0.0,
        was_human_edited: bool = False,
        rag_context: List[Dict] = None
    ) -> str:
        """
        Save a completely solved problem.
        
        Returns:
            Problem ID
        """
        problem_data = {
            'input_type': input_type,
            'original_input': original_input,
            'extracted_text': extracted_text,
            'normalized_text': normalized_text,
            'parsed_problem': parsed_problem,
            'topic': topic,
            'solution': solution,
            'explanation': explanation,
            'verification_result': verification_result,
            'confidence_score': confidence_score,
            'was_human_edited': was_human_edited,
            'rag_context': rag_context
        }
        
        return self.db.save_problem(problem_data)
    
    def record_feedback(
        self,
        problem_id: str,
        is_correct: bool,
        comment: str = None,
        corrected_solution: str = None
    ):
        """
        Record user feedback on a solution.
        
        Args:
            problem_id: ID of the problem
            is_correct: Whether solution was correct
            comment: User's comment
            corrected_solution: Correct answer if solution was wrong
        """
        self.db.save_feedback(problem_id, is_correct, comment, corrected_solution)
        
        # If incorrect and has correction, we could learn from it
        if not is_correct and corrected_solution:
            # This is where we could add more sophisticated learning
            pass
    
    def learn_extraction_correction(
        self,
        input_type: str,
        original_text: str,
        corrected_text: str
    ):
        """
        Learn from a human correction to OCR/ASR output.
        
        Args:
            input_type: 'image' or 'audio'
            original_text: What was extracted
            corrected_text: What human corrected it to
        """
        self.db.save_extraction_correction(input_type, original_text, corrected_text)
    
    def apply_corrections(self, text: str, input_type: str) -> str:
        """
        Apply learned corrections to extracted text.
        
        Args:
            text: Extracted text from OCR/ASR
            input_type: 'image' or 'audio'
        
        Returns:
            Text with learned corrections applied
        """
        return self.db.apply_learned_corrections(text, input_type)
    
    def find_similar_problems(self, query: str, topic: str = None, limit: int = 3) -> List[Dict]:
        """
        Find similar previously solved problems.
        
        Args:
            query: Problem text to search for
            topic: Optional topic filter
            limit: Maximum results
        
        Returns:
            List of similar problems with solutions
        """
        if topic:
            return self.db.get_problems_by_topic(topic, limit)
        else:
            return self.db.search_problems(query, limit)
    
    def get_successful_examples(self, topic: str, limit: int = 3) -> List[Dict]:
        """
        Get examples of successfully solved problems.
        
        Args:
            topic: Topic to filter by
            limit: Maximum results
        
        Returns:
            List of problems marked as correct
        """
        return self.db.get_successful_problems(topic, limit)
    
    def get_problem_history(self, limit: int = 10) -> List[Dict]:
        """Get recent problem history."""
        return self.db.get_recent_problems(limit)
    
    def get_problem(self, problem_id: str) -> Optional[Dict]:
        """Get a specific problem by ID."""
        return self.db.get_problem(problem_id)
    
    def get_statistics(self) -> Dict:
        """Get memory statistics."""
        return self.db.get_stats()
    
    def get_learning_summary(self) -> Dict:
        """
        Get summary of what the system has learned.
        
        Returns:
            Dictionary with learning statistics
        """
        stats = self.db.get_stats()
        
        # Get correction patterns
        image_corrections = self.db.get_correction_patterns('image')
        audio_corrections = self.db.get_correction_patterns('audio')
        
        return {
            'problems_solved': stats['total_problems'],
            'feedback_received': stats['feedback']['total_feedback'],
            'accuracy_rate': stats['feedback']['accuracy'],
            'topics_covered': list(stats['problems_by_topic'].keys()),
            'image_corrections_learned': len(image_corrections),
            'audio_corrections_learned': len(audio_corrections),
            'top_image_corrections': image_corrections[:5],
            'top_audio_corrections': audio_corrections[:5]
        }