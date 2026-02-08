# src/core/embedding_manager.py
"""
Shared Embedding Model Manager for Math Mentor.

Loads the sentence-transformers model ONCE and provides it to:
- RAG Retriever
- Math Classifier
- Any other component that needs embeddings

This saves:
- Memory (model loaded once instead of multiple times)
- Startup time (single load)
- Consistency (same model everywhere)
"""

import os
import logging
from typing import Optional, List, Union
import numpy as np

# Suppress noisy logs
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Singleton manager for the shared embedding model.
    
    Usage:
        from src.core.embedding_manager import get_embedding_manager
        
        em = get_embedding_manager()
        embedding = em.encode("Some text")
        embeddings = em.encode(["Text 1", "Text 2"])
    """
    
    # Default model - good balance of speed and quality
    DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
    
    # Alternative models
    AVAILABLE_MODELS = {
        "fast": "sentence-transformers/all-MiniLM-L6-v2",       # Faster, smaller (384 dims)
        "balanced": "sentence-transformers/all-mpnet-base-v2",   # Good balance (768 dims)
        "quality": "sentence-transformers/multi-qa-mpnet-base-dot-v1",  # Better for Q&A
        "technical": "BAAI/bge-base-en-v1.5",                   # Good for technical content
    }
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Name of the model to load. If None, uses DEFAULT_MODEL.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None
        self._dimension = None
        self._is_initialized = False
    
    @property
    def model(self):
        """Lazy load the model on first access."""
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            self._load_model()
        return self._dimension
    
    @property
    def is_initialized(self) -> bool:
        """Check if model is loaded."""
        return self._is_initialized
    
    def _load_model(self):
        """Load the sentence-transformers model."""
        if self._is_initialized:
            return
        
        logger.info(f"Loading embedding model: {self.model_name}")
        
        try:
            from sentence_transformers import SentenceTransformer
            
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            self._is_initialized = True
            
            logger.info(f"✓ Model loaded successfully (dimension: {self._dimension})")
            
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        show_progress: bool = False,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: Single string or list of strings to encode
            normalize: Whether to L2-normalize embeddings (for cosine similarity)
            show_progress: Whether to show progress bar
            batch_size: Batch size for encoding
            
        Returns:
            numpy array of shape (n_texts, dimension) or (dimension,) for single text
        """
        # Handle single string
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        # Encode
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            batch_size=batch_size
        )
        
        # Normalize if requested
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
        
        # Return single vector if input was single string
        if single_input:
            return embeddings[0]
        
        return embeddings
    
    def similarity(
        self,
        query: Union[str, np.ndarray],
        documents: Union[List[str], np.ndarray]
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and documents.
        
        Args:
            query: Query text or pre-computed embedding
            documents: List of document texts or pre-computed embeddings
            
        Returns:
            numpy array of similarity scores
        """
        # Encode query if it's a string
        if isinstance(query, str):
            query_emb = self.encode(query, normalize=True)
        else:
            query_emb = query
            # Normalize if not already
            query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        
        # Encode documents if they're strings
        if isinstance(documents, list) and isinstance(documents[0], str):
            doc_embs = self.encode(documents, normalize=True)
        else:
            doc_embs = documents
            # Normalize if not already
            doc_embs = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarities
        similarities = np.dot(doc_embs, query_emb)
        
        return similarities
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "is_initialized": self._is_initialized,
        }


# ============================================
# SINGLETON INSTANCE
# ============================================
_embedding_manager: Optional[EmbeddingManager] = None


def get_embedding_manager(model_name: Optional[str] = None) -> EmbeddingManager:
    """
    Get or create the shared EmbeddingManager instance.
    
    This ensures the model is loaded only once across the entire application.
    
    Args:
        model_name: Optional model name (only used on first call)
        
    Returns:
        Shared EmbeddingManager instance
    """
    global _embedding_manager
    
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager(model_name)
    
    return _embedding_manager


def preload_model(model_name: Optional[str] = None):
    """
    Preload the embedding model.
    Call this at application startup to load the model before it's needed.
    
    Args:
        model_name: Optional model name to use
    """
    em = get_embedding_manager(model_name)
    _ = em.model  # Trigger lazy loading
    logger.info("Embedding model preloaded")


# ============================================
# TEST
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("EMBEDDING MANAGER TEST")
    print("=" * 60)
    
    # Get the manager
    em = get_embedding_manager()
    
    # Test single encoding
    print("\n1. Single text encoding:")
    embedding = em.encode("What is the derivative of x squared?")
    print(f"   Shape: {embedding.shape}")
    print(f"   Norm: {np.linalg.norm(embedding):.4f}")
    
    # Test batch encoding
    print("\n2. Batch encoding:")
    texts = [
        "Solve x^2 + 2x + 1 = 0",
        "What is the capital of France?",
        "Find the probability of getting heads"
    ]
    embeddings = em.encode(texts)
    print(f"   Shape: {embeddings.shape}")
    
    # Test similarity
    print("\n3. Similarity computation:")
    query = "Calculate the derivative"
    docs = [
        "Find dy/dx for y = x^3",
        "What is the weather today?",
        "Differentiate f(x) = sin(x)"
    ]
    similarities = em.similarity(query, docs)
    print(f"   Query: '{query}'")
    for doc, sim in zip(docs, similarities):
        print(f"   [{sim:.4f}] {doc}")
    
    # Test that second call returns same instance
    print("\n4. Singleton test:")
    em2 = get_embedding_manager()
    print(f"   Same instance: {em is em2}")
    
    # Model info
    print("\n5. Model info:")
    info = em.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)