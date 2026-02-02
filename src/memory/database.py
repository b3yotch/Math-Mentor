# src/memory/database.py
"""
SQLite database for Math Mentor memory system.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Any
import uuid


class MemoryDatabase:
    """
    SQLite database for storing:
    - Solved problems and solutions
    - User feedback
    - OCR/ASR correction patterns
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the database.
        
        Args:
            db_path: Path to SQLite database file
        """
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            db_path = os.path.join(base_dir, "data", "math_mentor.db")
        
        self.db_path = db_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        return conn
    
    def _init_db(self):
        """Initialize database tables."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Problems table - stores all solved problems
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS problems (
                id TEXT PRIMARY KEY,
                input_type TEXT NOT NULL,
                original_input TEXT,
                extracted_text TEXT NOT NULL,
                normalized_text TEXT,
                parsed_problem TEXT,
                topic TEXT,
                subtopic TEXT,
                solution TEXT,
                explanation TEXT,
                verification_result TEXT,
                confidence_score REAL,
                was_human_edited INTEGER DEFAULT 0,
                rag_context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Feedback table - stores user feedback on solutions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                problem_id TEXT NOT NULL,
                is_correct INTEGER NOT NULL,
                user_comment TEXT,
                corrected_solution TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (problem_id) REFERENCES problems(id)
            )
        ''')
        
        # Extraction corrections table - learns from OCR/ASR corrections
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS extraction_corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_type TEXT NOT NULL,
                original_text TEXT NOT NULL,
                corrected_text TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(input_type, original_text, corrected_text)
            )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_problems_topic ON problems(topic)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_problems_created ON problems(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_problem ON feedback(problem_id)')
        
        conn.commit()
        conn.close()
        
        print(f"✅ Database initialized at: {self.db_path}")
    
    # ==================== Problem Operations ====================
    
    def save_problem(self, problem_data: Dict[str, Any]) -> str:
        """
        Save a solved problem to the database.
        
        Args:
            problem_data: Dictionary with problem details
        
        Returns:
            Problem ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        problem_id = problem_data.get('id', str(uuid.uuid4()))
        
        cursor.execute('''
            INSERT OR REPLACE INTO problems 
            (id, input_type, original_input, extracted_text, normalized_text,
             parsed_problem, topic, subtopic, solution, explanation,
             verification_result, confidence_score, was_human_edited, rag_context, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            problem_id,
            problem_data.get('input_type', 'text'),
            problem_data.get('original_input'),
            problem_data.get('extracted_text', ''),
            problem_data.get('normalized_text'),
            json.dumps(problem_data.get('parsed_problem')) if problem_data.get('parsed_problem') else None,
            problem_data.get('topic'),
            problem_data.get('subtopic'),
            problem_data.get('solution'),
            problem_data.get('explanation'),
            json.dumps(problem_data.get('verification_result')) if problem_data.get('verification_result') else None,
            problem_data.get('confidence_score', 0.0),
            int(problem_data.get('was_human_edited', False)),
            json.dumps(problem_data.get('rag_context')) if problem_data.get('rag_context') else None,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        return problem_id
    
    def get_problem(self, problem_id: str) -> Optional[Dict]:
        """Get a problem by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM problems WHERE id = ?', (problem_id,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            return self._row_to_dict(row)
        return None
    
    def get_recent_problems(self, limit: int = 10) -> List[Dict]:
        """Get most recent problems."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM problems 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_dict(row) for row in rows]
    
    def get_problems_by_topic(self, topic: str, limit: int = 10) -> List[Dict]:
        """Get problems filtered by topic."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM problems 
            WHERE topic = ?
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (topic, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_dict(row) for row in rows]
    
    def search_problems(self, query: str, limit: int = 5) -> List[Dict]:
        """Search problems by text content."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        search_pattern = f"%{query}%"
        
        cursor.execute('''
            SELECT * FROM problems 
            WHERE extracted_text LIKE ? 
               OR solution LIKE ?
               OR topic LIKE ?
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (search_pattern, search_pattern, search_pattern, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_dict(row) for row in rows]
    
    def get_successful_problems(self, topic: str = None, limit: int = 10) -> List[Dict]:
        """Get problems that were marked as correct."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if topic:
            cursor.execute('''
                SELECT p.* FROM problems p
                INNER JOIN feedback f ON p.id = f.problem_id
                WHERE f.is_correct = 1 AND p.topic = ?
                ORDER BY p.created_at DESC
                LIMIT ?
            ''', (topic, limit))
        else:
            cursor.execute('''
                SELECT p.* FROM problems p
                INNER JOIN feedback f ON p.id = f.problem_id
                WHERE f.is_correct = 1
                ORDER BY p.created_at DESC
                LIMIT ?
            ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_dict(row) for row in rows]
    
    # ==================== Feedback Operations ====================
    
    def save_feedback(
        self, 
        problem_id: str, 
        is_correct: bool, 
        comment: str = None, 
        corrected_solution: str = None
    ) -> int:
        """
        Save user feedback for a problem.
        
        Args:
            problem_id: ID of the problem
            is_correct: Whether the solution was correct
            comment: User's comment
            corrected_solution: Corrected solution if incorrect
        
        Returns:
            Feedback ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback (problem_id, is_correct, user_comment, corrected_solution, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            problem_id,
            int(is_correct),
            comment,
            corrected_solution,
            datetime.now().isoformat()
        ))
        
        feedback_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return feedback_id
    
    def get_feedback(self, problem_id: str) -> List[Dict]:
        """Get all feedback for a problem."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM feedback 
            WHERE problem_id = ?
            ORDER BY created_at DESC
        ''', (problem_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_feedback_stats(self) -> Dict:
        """Get feedback statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) as total FROM feedback')
        total = cursor.fetchone()['total']
        
        cursor.execute('SELECT COUNT(*) as correct FROM feedback WHERE is_correct = 1')
        correct = cursor.fetchone()['correct']
        
        conn.close()
        
        return {
            'total_feedback': total,
            'correct': correct,
            'incorrect': total - correct,
            'accuracy': correct / total if total > 0 else 0.0
        }
    
    # ==================== Extraction Corrections ====================
    
    def save_extraction_correction(
        self, 
        input_type: str, 
        original_text: str, 
        corrected_text: str
    ):
        """
        Save an OCR/ASR correction for learning.
        
        Args:
            input_type: 'image' or 'audio'
            original_text: Original extracted text
            corrected_text: Human-corrected text
        """
        if original_text == corrected_text:
            return  # No correction needed
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Try to update existing, or insert new
        cursor.execute('''
            INSERT INTO extraction_corrections (input_type, original_text, corrected_text, frequency)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(input_type, original_text, corrected_text) 
            DO UPDATE SET frequency = frequency + 1
        ''', (input_type, original_text, corrected_text))
        
        conn.commit()
        conn.close()
    
    def get_correction_patterns(self, input_type: str = None, min_frequency: int = 1) -> List[Dict]:
        """
        Get learned correction patterns.
        
        Args:
            input_type: Filter by input type (optional)
            min_frequency: Minimum frequency threshold
        
        Returns:
            List of correction patterns
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if input_type:
            cursor.execute('''
                SELECT * FROM extraction_corrections
                WHERE input_type = ? AND frequency >= ?
                ORDER BY frequency DESC
            ''', (input_type, min_frequency))
        else:
            cursor.execute('''
                SELECT * FROM extraction_corrections
                WHERE frequency >= ?
                ORDER BY frequency DESC
            ''', (min_frequency,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def apply_learned_corrections(self, text: str, input_type: str) -> str:
        """
        Apply learned corrections to extracted text.
        
        Args:
            text: Original extracted text
            input_type: Type of input ('image' or 'audio')
        
        Returns:
            Corrected text
        """
        patterns = self.get_correction_patterns(input_type, min_frequency=2)
        
        corrected = text
        for pattern in patterns:
            if pattern['original_text'] in corrected:
                corrected = corrected.replace(
                    pattern['original_text'], 
                    pattern['corrected_text']
                )
        
        return corrected
    
    # ==================== Statistics ====================
    
    def get_stats(self) -> Dict:
        """Get overall database statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Total problems
        cursor.execute('SELECT COUNT(*) as count FROM problems')
        total_problems = cursor.fetchone()['count']
        
        # Problems by topic
        cursor.execute('''
            SELECT topic, COUNT(*) as count 
            FROM problems 
            WHERE topic IS NOT NULL
            GROUP BY topic
        ''')
        topics = {row['topic']: row['count'] for row in cursor.fetchall()}
        
        # Problems by input type
        cursor.execute('''
            SELECT input_type, COUNT(*) as count 
            FROM problems 
            GROUP BY input_type
        ''')
        input_types = {row['input_type']: row['count'] for row in cursor.fetchall()}
        
        # Feedback stats
        feedback_stats = self.get_feedback_stats()
        
        # Correction patterns
        cursor.execute('SELECT COUNT(*) as count FROM extraction_corrections')
        correction_patterns = cursor.fetchone()['count']
        
        conn.close()
        
        return {
            'total_problems': total_problems,
            'problems_by_topic': topics,
            'problems_by_input_type': input_types,
            'feedback': feedback_stats,
            'correction_patterns': correction_patterns
        }
    
    # ==================== Utilities ====================
    
    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        """Convert a database row to dictionary with JSON parsing."""
        d = dict(row)
        
        # Parse JSON fields
        for field in ['parsed_problem', 'verification_result', 'rag_context']:
            if d.get(field):
                try:
                    d[field] = json.loads(d[field])
                except json.JSONDecodeError:
                    pass
        
        return d
    
    def clear_all(self):
        """Clear all data (use with caution!)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM feedback')
        cursor.execute('DELETE FROM extraction_corrections')
        cursor.execute('DELETE FROM problems')
        
        conn.commit()
        conn.close()
        
        print("⚠️ All data cleared!")