# src/rag/reranker.py
"""
Reranker Module for Math Mentor RAG System.
Supports multiple reranking strategies:
- Cross-Encoder (sentence-transformers)
- Cohere Rerank API
- LLM-based reranking
- Hybrid scoring
"""

import os
import re
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result from reranking with additional metadata."""
    index: int
    original_score: float
    rerank_score: float
    combined_score: float
    relevance_label: str = ""  # "highly_relevant", "relevant", "partial", "not_relevant"
    

class BaseReranker(ABC):
    """Abstract base class for rerankers."""
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Any],
        top_k: Optional[int] = None
    ) -> List[Any]:
        """Rerank documents based on relevance to query."""
        pass
    
    def _get_content(self, doc: Any) -> str:
        """Extract content from document (handles both dict and dataclass)."""
        if hasattr(doc, 'content'):
            return doc.content
        elif isinstance(doc, dict):
            return doc.get('content', '')
        return str(doc)
    
    def _get_title(self, doc: Any) -> str:
        """Extract title from document."""
        if hasattr(doc, 'title'):
            return doc.title
        elif isinstance(doc, dict):
            return doc.get('title', '')
        return ""


class CrossEncoderReranker(BaseReranker):
    """
    Reranker using Cross-Encoder models.
    Cross-encoders are more accurate than bi-encoders for reranking
    as they process query-document pairs together.
    """
    
    # Available models (accuracy vs speed tradeoff)
    MODELS = {
        "fast": "cross-encoder/ms-marco-MiniLM-L-6-v2",      # Fastest, good accuracy
        "balanced": "cross-encoder/ms-marco-MiniLM-L-12-v2", # Good balance
        "accurate": "cross-encoder/ms-marco-TinyBERT-L-2-v2", # Very fast
        "best": "BAAI/bge-reranker-base",                    # Best accuracy
        "large": "BAAI/bge-reranker-large",                  # Best but slow
    }
    
    def __init__(
        self,
        model_name: str = "fast",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512
    ):
        self.model_name = self.MODELS.get(model_name, model_name)
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = None
        self.device = device
        
        self._load_model()
    
    def _load_model(self):
        """Load the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            import torch
            
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading cross-encoder: {self.model_name} on {self.device}")
            
            self.model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device
            )
            
            logger.info(f"✓ Cross-encoder loaded successfully")
            
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}")
            self.model = None
    
    def rerank(
        self,
        query: str,
        documents: List[Any],
        top_k: Optional[int] = None
    ) -> List[Any]:
        """Rerank documents using cross-encoder."""
        if not documents:
            return []
        
        if self.model is None:
            logger.warning("Cross-encoder not loaded, returning original order")
            return documents[:top_k] if top_k else documents
        
        # Prepare query-document pairs
        pairs = []
        for doc in documents:
            content = self._get_content(doc)
            title = self._get_title(doc)
            
            # Combine title and content for better matching
            doc_text = f"{title}: {content}" if title else content
            
            # Truncate if too long
            if len(doc_text) > 1500:
                doc_text = doc_text[:1500]
            
            pairs.append([query, doc_text])
        
        # Get rerank scores
        try:
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False
            )
            
            # Normalize scores to 0-1 range
            if len(scores) > 1:
                min_score = min(scores)
                max_score = max(scores)
                if max_score > min_score:
                    scores = [(s - min_score) / (max_score - min_score) for s in scores]
                else:
                    scores = [0.5] * len(scores)
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents[:top_k] if top_k else documents
        
        # Combine with original scores
        scored_docs = []
        for i, (doc, rerank_score) in enumerate(zip(documents, scores)):
            # Get original score
            if hasattr(doc, 'combined_score'):
                original_score = doc.combined_score
            elif isinstance(doc, dict):
                original_score = doc.get('score', doc.get('combined_score', 0.5))
            else:
                original_score = 0.5
            
            # Weighted combination: favor rerank score but consider original
            combined = 0.7 * rerank_score + 0.3 * original_score
            
            # Update document score
            if hasattr(doc, 'combined_score'):
                doc.combined_score = combined
                doc.rerank_score = rerank_score
            elif isinstance(doc, dict):
                doc['combined_score'] = combined
                doc['rerank_score'] = rerank_score
            
            scored_docs.append((combined, i, doc))
        
        # Sort by combined score
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Return reranked documents
        reranked = [doc for _, _, doc in scored_docs]
        
        if top_k:
            reranked = reranked[:top_k]
        
        return reranked


class CohereReranker(BaseReranker):
    """
    Reranker using Cohere's Rerank API.
    Requires COHERE_API_KEY environment variable.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "rerank-english-v3.0",
        max_chunks_per_doc: int = 10
    ):
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.model = model
        self.max_chunks_per_doc = max_chunks_per_doc
        self.client = None
        
        self._init_client()
    
    def _init_client(self):
        """Initialize Cohere client."""
        if not self.api_key:
            logger.warning("COHERE_API_KEY not set. Cohere reranker unavailable.")
            return
        
        try:
            import cohere
            self.client = cohere.Client(self.api_key)
            logger.info("✓ Cohere client initialized")
        except ImportError:
            logger.error("cohere not installed. Run: pip install cohere")
    
    def rerank(
        self,
        query: str,
        documents: List[Any],
        top_k: Optional[int] = None
    ) -> List[Any]:
        """Rerank using Cohere API."""
        if not documents:
            return []
        
        if self.client is None:
            logger.warning("Cohere client not available, returning original order")
            return documents[:top_k] if top_k else documents
        
        # Prepare documents for API
        doc_texts = []
        for doc in documents:
            content = self._get_content(doc)
            title = self._get_title(doc)
            doc_text = f"{title}: {content}" if title else content
            doc_texts.append(doc_text[:4000])  # Cohere limit
        
        try:
            response = self.client.rerank(
                query=query,
                documents=doc_texts,
                top_n=top_k or len(documents),
                model=self.model,
                return_documents=False
            )
            
            # Build reranked list
            reranked = []
            for result in response.results:
                idx = result.index
                doc = documents[idx]
                
                # Update score
                if hasattr(doc, 'combined_score'):
                    doc.combined_score = result.relevance_score
                    doc.rerank_score = result.relevance_score
                elif isinstance(doc, dict):
                    doc['combined_score'] = result.relevance_score
                    doc['rerank_score'] = result.relevance_score
                
                reranked.append(doc)
            
            return reranked
            
        except Exception as e:
            logger.error(f"Cohere reranking failed: {e}")
            return documents[:top_k] if top_k else documents


class LLMReranker(BaseReranker):
    """
    Reranker using LLM for scoring relevance.
    Slower but can understand complex relevance signals.
    """
    
    RERANK_PROMPT = """Rate the relevance of this document to the query on a scale of 0-10.

Query: {query}

Document:
{document}

Consider:
1. Does it directly answer the query?
2. Does it contain relevant information?
3. Is the mathematical content appropriate?

Respond with ONLY a number from 0-10."""

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        model: str = "gpt-3.5-turbo",
        batch_size: int = 5
    ):
        self.llm_client = llm_client
        self.model = model
        self.batch_size = batch_size
    
    def rerank(
        self,
        query: str,
        documents: List[Any],
        top_k: Optional[int] = None
    ) -> List[Any]:
        """Rerank using LLM scoring."""
        if not documents:
            return []
        
        if self.llm_client is None:
            logger.warning("LLM client not set, returning original order")
            return documents[:top_k] if top_k else documents
        
        scored_docs = []
        
        for doc in documents:
            content = self._get_content(doc)[:1000]  # Limit for API
            
            prompt = self.RERANK_PROMPT.format(
                query=query,
                document=content
            )
            
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=10
                )
                
                score_text = response.choices[0].message.content.strip()
                score = float(re.search(r'(\d+(?:\.\d+)?)', score_text).group(1))
                score = min(10, max(0, score)) / 10  # Normalize to 0-1
                
            except Exception as e:
                logger.warning(f"LLM scoring failed: {e}")
                score = 0.5
            
            # Update document
            if hasattr(doc, 'combined_score'):
                original = doc.combined_score
                doc.combined_score = 0.6 * score + 0.4 * original
                doc.rerank_score = score
            
            scored_docs.append((score, doc))
        
        # Sort by score
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        reranked = [doc for _, doc in scored_docs]
        
        return reranked[:top_k] if top_k else reranked


class HybridReranker(BaseReranker):
    """
    Hybrid reranker combining multiple signals:
    - Cross-encoder score
    - Keyword/BM25 score
    - Content type relevance
    - Math-specific features
    """
    
    def __init__(
        self,
        use_cross_encoder: bool = True,
        cross_encoder_model: str = "fast",
        cross_encoder_weight: float = 0.5,
        keyword_weight: float = 0.2,
        semantic_weight: float = 0.2,
        feature_weight: float = 0.1,
        device: Optional[str] = None
    ):
        self.cross_encoder_weight = cross_encoder_weight
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
        self.feature_weight = feature_weight
        
        self.cross_encoder = None
        if use_cross_encoder:
            try:
                self.cross_encoder = CrossEncoderReranker(
                    model_name=cross_encoder_model,
                    device=device
                )
            except Exception as e:
                logger.warning(f"Could not load cross-encoder: {e}")
        
        # Math-related keywords for boosting
        self.math_keywords = {
            'probability': ['probability', 'chance', 'likely', 'odds', 'random', 'event', 'sample space'],
            'calculus': ['derivative', 'integral', 'differentiate', 'limit', 'continuous'],
            'algebra': ['equation', 'solve', 'variable', 'polynomial', 'quadratic', 'linear'],
            'statistics': ['mean', 'median', 'mode', 'variance', 'standard deviation', 'distribution'],
            'geometry': ['angle', 'triangle', 'circle', 'area', 'perimeter', 'volume'],
        }
        
        # Content type relevance for different query types
        self.content_type_boost = {
            'how': {'example': 0.3, 'solved_example': 0.4, 'text': 0.1},
            'what': {'definition': 0.4, 'theorem': 0.3, 'text': 0.2},
            'prove': {'proof': 0.5, 'theorem': 0.3, 'theorem_statement': 0.2},
            'solve': {'solved_example': 0.4, 'example': 0.3, 'formula': 0.2},
            'calculate': {'formula': 0.3, 'solved_example': 0.3, 'example': 0.2},
            'find': {'formula': 0.3, 'solved_example': 0.3, 'example': 0.3},
            'define': {'definition': 0.5, 'text': 0.2},
            'explain': {'text': 0.3, 'definition': 0.3, 'example': 0.2},
        }
    
    def rerank(
        self,
        query: str,
        documents: List[Any],
        top_k: Optional[int] = None
    ) -> List[Any]:
        """Rerank using multiple signals."""
        if not documents:
            return []
        
        # Step 1: Get cross-encoder scores (if available)
        if self.cross_encoder and self.cross_encoder.model:
            documents = self.cross_encoder.rerank(query, documents)
        
        # Step 2: Calculate feature-based scores
        query_lower = query.lower()
        query_type = self._detect_query_type(query_lower)
        query_topic = self._detect_topic(query_lower)
        
        scored_docs = []
        for doc in documents:
            # Get existing scores
            if hasattr(doc, 'score'):
                semantic_score = doc.score
            elif isinstance(doc, dict):
                semantic_score = doc.get('semantic_score', doc.get('score', 0.5))
            else:
                semantic_score = 0.5
            
            if hasattr(doc, 'keyword_score'):
                keyword_score = doc.keyword_score
            elif isinstance(doc, dict):
                keyword_score = doc.get('keyword_score', 0)
            else:
                keyword_score = 0
            
            if hasattr(doc, 'rerank_score'):
                cross_encoder_score = doc.rerank_score
            elif isinstance(doc, dict):
                cross_encoder_score = doc.get('rerank_score', semantic_score)
            else:
                cross_encoder_score = semantic_score
            
            # Calculate feature score
            feature_score = self._calculate_feature_score(
                doc, query_type, query_topic, query_lower
            )
            
            # Combine scores
            combined = (
                self.cross_encoder_weight * cross_encoder_score +
                self.semantic_weight * semantic_score +
                self.keyword_weight * keyword_score +
                self.feature_weight * feature_score
            )
            
            # Update document
            if hasattr(doc, 'combined_score'):
                doc.combined_score = combined
            elif isinstance(doc, dict):
                doc['combined_score'] = combined
            
            scored_docs.append((combined, doc))
        
        # Sort by combined score
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        reranked = [doc for _, doc in scored_docs]
        
        return reranked[:top_k] if top_k else reranked
    
    def _detect_query_type(self, query: str) -> str:
        """Detect the type of query."""
        query_words = query.split()
        if not query_words:
            return 'general'
        
        first_word = query_words[0]
        for qtype in self.content_type_boost.keys():
            if first_word == qtype or qtype in query:
                return qtype
        
        return 'general'
    
    def _detect_topic(self, query: str) -> str:
        """Detect the math topic from query."""
        for topic, keywords in self.math_keywords.items():
            if any(kw in query for kw in keywords):
                return topic
        return 'general'
    
    def _calculate_feature_score(
        self,
        doc: Any,
        query_type: str,
        query_topic: str,
        query: str
    ) -> float:
        """Calculate feature-based relevance score."""
        score = 0.5  # Base score
        
        # Get document attributes
        if hasattr(doc, 'content_type'):
            content_type = doc.content_type
        elif isinstance(doc, dict):
            content_type = doc.get('content_type', doc.get('type', 'text'))
        else:
            content_type = 'text'
        
        if hasattr(doc, 'topic'):
            doc_topic = doc.topic
        elif isinstance(doc, dict):
            doc_topic = doc.get('topic', '')
        else:
            doc_topic = ''
        
        # Boost for content type match
        if query_type in self.content_type_boost:
            boost = self.content_type_boost[query_type].get(content_type, 0)
            score += boost
        
        # Boost for topic match
        if query_topic and query_topic.lower() in doc_topic.lower():
            score += 0.2
        
        # Boost for title match
        title = self._get_title(doc).lower()
        query_words = set(query.split())
        title_words = set(title.split())
        overlap = len(query_words & title_words)
        if overlap > 0:
            score += min(0.2, overlap * 0.05)
        
        # Boost for examples when query contains numbers
        if re.search(r'\d', query) and content_type in ['example', 'solved_example']:
            score += 0.15
        
        return min(1.0, score)


class ColBERTReranker(BaseReranker):
    """
    ColBERT-style late interaction reranker.
    Uses token-level similarity for fine-grained matching.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = device
        
        self._load_model()
    
    def _load_model(self):
        """Load model for token embeddings."""
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            
            logger.info(f"✓ ColBERT model loaded: {self.model_name}")
            
        except ImportError:
            logger.error("transformers not installed. Run: pip install transformers")
        except Exception as e:
            logger.error(f"Failed to load ColBERT model: {e}")
    
    def _get_token_embeddings(self, text: str) -> np.ndarray:
        """Get token-level embeddings."""
        import torch
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use last hidden state as token embeddings
            embeddings = outputs.last_hidden_state[0].cpu().numpy()
        
        return embeddings
    
    def _maxsim(self, query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
        """Calculate MaxSim score (ColBERT-style)."""
        # For each query token, find max similarity with any doc token
        # Then sum these max similarities
        
        # Normalize embeddings
        query_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-8)
        doc_norm = doc_emb / (np.linalg.norm(doc_emb, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarity matrix
        sim_matrix = np.dot(query_norm, doc_norm.T)
        
        # MaxSim: for each query token, take max similarity
        max_sims = sim_matrix.max(axis=1)
        
        # Average of max similarities
        score = max_sims.mean()
        
        return float(score)
    
    def rerank(
        self,
        query: str,
        documents: List[Any],
        top_k: Optional[int] = None
    ) -> List[Any]:
        """Rerank using ColBERT-style late interaction."""
        if not documents or self.model is None:
            return documents[:top_k] if top_k else documents
        
        # Get query token embeddings
        query_emb = self._get_token_embeddings(query)
        
        scored_docs = []
        for doc in documents:
            content = self._get_content(doc)[:1000]
            
            # Get document token embeddings
            doc_emb = self._get_token_embeddings(content)
            
            # Calculate MaxSim score
            score = self._maxsim(query_emb, doc_emb)
            
            # Get original score
            if hasattr(doc, 'combined_score'):
                original = doc.combined_score
            elif isinstance(doc, dict):
                original = doc.get('combined_score', 0.5)
            else:
                original = 0.5
            
            # Combine scores
            combined = 0.6 * score + 0.4 * original
            
            # Update document
            if hasattr(doc, 'combined_score'):
                doc.combined_score = combined
                doc.rerank_score = score
            elif isinstance(doc, dict):
                doc['combined_score'] = combined
                doc['rerank_score'] = score
            
            scored_docs.append((combined, doc))
        
        # Sort by score
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        reranked = [doc for _, doc in scored_docs]
        
        return reranked[:top_k] if top_k else reranked


class RerankerFactory:
    """Factory for creating rerankers."""
    
    RERANKERS = {
        "cross_encoder": CrossEncoderReranker,
        "cohere": CohereReranker,
        "llm": LLMReranker,
        "hybrid": HybridReranker,
        "colbert": ColBERTReranker,
    }
    
    @classmethod
    def create(
        cls,
        reranker_type: str = "hybrid",
        **kwargs
    ) -> BaseReranker:
        """Create a reranker instance."""
        if reranker_type not in cls.RERANKERS:
            raise ValueError(f"Unknown reranker type: {reranker_type}. "
                           f"Available: {list(cls.RERANKERS.keys())}")
        
        return cls.RERANKERS[reranker_type](**kwargs)


# Singleton instance
_reranker_instance: Optional[BaseReranker] = None


def get_reranker(
    reranker_type: str = "hybrid",
    **kwargs
) -> BaseReranker:
    """Get or create reranker singleton."""
    global _reranker_instance
    
    if _reranker_instance is None:
        _reranker_instance = RerankerFactory.create(reranker_type, **kwargs)
    
    return _reranker_instance


# Test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    
    print("=" * 70)
    print("RERANKER TEST")
    print("=" * 70)
    
    # Create sample documents
    sample_docs = [
        {
            "content": "The probability of an event A is denoted P(A) and is calculated as the number of favorable outcomes divided by total outcomes.",
            "title": "Definition of Probability",
            "content_type": "definition",
            "topic": "probability",
            "score": 0.7,
            "keyword_score": 0.5,
            "combined_score": 0.65
        },
        {
            "content": "Example: A coin is tossed twice. Find the probability of getting exactly one head. Solution: Sample space = {HH, HT, TH, TT}, Favorable = {HT, TH}, P = 2/4 = 0.5",
            "title": "Coin Toss Example",
            "content_type": "solved_example",
            "topic": "probability",
            "score": 0.65,
            "keyword_score": 0.8,
            "combined_score": 0.7
        },
        {
            "content": "Integration is the reverse process of differentiation. The integral of f(x) gives the area under the curve.",
            "title": "Introduction to Integration",
            "content_type": "text",
            "topic": "calculus",
            "score": 0.5,
            "keyword_score": 0.1,
            "combined_score": 0.35
        },
        {
            "content": "Bayes' Theorem: P(A|B) = P(B|A) * P(A) / P(B). This formula is fundamental in conditional probability.",
            "title": "Bayes' Theorem",
            "content_type": "theorem",
            "topic": "probability",
            "score": 0.6,
            "keyword_score": 0.3,
            "combined_score": 0.5
        },
    ]
    
    query = "What is the probability of getting heads when tossing a coin?"
    
    # Test Hybrid Reranker (without cross-encoder for speed)
    print("\n📊 Testing Hybrid Reranker (no cross-encoder):")
    print("-" * 50)
    
    reranker = HybridReranker(use_cross_encoder=False)
    reranked = reranker.rerank(query, sample_docs.copy(), top_k=3)
    
    for i, doc in enumerate(reranked, 1):
        print(f"{i}. {doc['title']}")
        print(f"   Score: {doc['combined_score']:.3f}")
        print(f"   Content: {doc['content'][:80]}...")
        print()
    
    # Test Cross-Encoder if available
    print("\n📊 Testing Cross-Encoder Reranker:")
    print("-" * 50)
    
    try:
        ce_reranker = CrossEncoderReranker(model_name="fast")
        if ce_reranker.model:
            reranked_ce = ce_reranker.rerank(query, sample_docs.copy(), top_k=3)
            
            for i, doc in enumerate(reranked_ce, 1):
                print(f"{i}. {doc['title']}")
                print(f"   Score: {doc['combined_score']:.3f} (rerank: {doc.get('rerank_score', 0):.3f})")
                print()
        else:
            print("Cross-encoder not available")
    except Exception as e:
        print(f"Cross-encoder test failed: {e}")
    
    print("\n" + "=" * 70)
    print("✅ Reranker test completed!")
    print("=" * 70)