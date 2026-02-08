# src/rag/retriever.py - IMPROVED VERSION WITH FIXES
"""
Improved RAG Retriever for Math Mentor - v2.
Includes content cleaning, length limits, and better filtering.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import logging
# Add after existing imports
from .reranker import get_reranker, HybridReranker, BaseReranker
from typing import List, Dict, Optional, Tuple, Any, Union

from .query_expander import get_query_expander, ExpandedQuery
from .reranker import get_reranker, HybridReranker, BaseReranker

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """A retrieved document with full context."""
    content: str
    title: str
    section: str
    subsection: str
    chapter: str
    topic: str
    subtopic: str
    content_type: str
    score: float
    keyword_score: float = 0.0
    combined_score: float = 0.0
    rerank_score: float = 0.0  # NEW FIELD
    source: str = ""
    page_number: int = 0
    chunk_id: str = ""
    parent_content: str = ""
    sibling_chunks: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "title": self.title,
            "section": self.section,
            "chapter": self.chapter,
            "topic": self.topic,
            "subtopic": self.subtopic,
            "type": self.content_type,
            "score": self.combined_score,
            "semantic_score": self.score,
            "keyword_score": self.keyword_score,
            "rerank_score": self.rerank_score,  # NEW FIELD
            "source": self.source,
            "page": self.page_number,
        }


class ImprovedRAGRetriever:
    """
    Improved RAG Retriever v2 with:
    - Content cleaning
    - Section length limits
    - Better filtering
    - Hybrid search
    - Query expansion
    """
    
    # Configuration
    MAX_SECTION_LENGTH = 3000  # Max chars for a section
    MAX_CHUNK_LENGTH = 1500   # Max chars for a single chunk
    MIN_CONTENT_LENGTH = 50   # Minimum content length
    
    def __init__(
        self,
        knowledge_base_path: str = "data/vector_store",
        embedding_manager=None,
        use_query_expansion: bool = True,
        use_hybrid_search: bool = True,
        return_full_sections: bool = True,
        use_reranker: bool = True,
        reranker_type: str = "hybrid",
        reranker: Optional[BaseReranker] = None
    ):
        self.knowledge_base_path = Path(knowledge_base_path)
        self._embedding_manager = embedding_manager
        self.use_query_expansion = use_query_expansion
        self.use_hybrid_search = use_hybrid_search
        self.return_full_sections = return_full_sections
        self.use_reranker = use_reranker
        
        self.index = None
        self.chunks: List[Dict] = []
        self.sections: Dict[str, Dict] = {}
        self.chunk_to_section: Dict[str, str] = {}
        
        self.query_expander = get_query_expander()
        self.bm25 = None
        self.corpus_tokens = None
        
        # Initialize reranker
        self.reranker = reranker
        if self.use_reranker and self.reranker is None:
            self._init_reranker(reranker_type)
        
        self._initialized = False
        self._load_knowledge_base()

    def _init_reranker(self, reranker_type: str):
        """Initialize the reranker."""
        try:
            from .reranker import CrossEncoderReranker, CohereReranker, HybridReranker
            
            if reranker_type == "cross_encoder":
                self.reranker = CrossEncoderReranker(model_name="fast")
            elif reranker_type == "cohere":
                self.reranker = CohereReranker()
            else:  # hybrid (default)
                self.reranker = HybridReranker(
                    use_cross_encoder=True,
                    cross_encoder_model="fast"
                )
            logger.info(f"✓ Reranker initialized: {reranker_type}")
        except Exception as e:
            logger.warning(f"Could not initialize reranker: {e}. Continuing without reranking.")
            self.reranker = None
            self.use_reranker = False
    
    @property
    def embedding_manager(self):
        if self._embedding_manager is None:
            from src.core.embedding_manager import get_embedding_manager
            self._embedding_manager = get_embedding_manager()
        return self._embedding_manager
    
    def _load_knowledge_base(self):
        """Load all knowledge base components."""
        try:
            import faiss
            
            index_paths = [
                self.knowledge_base_path / "global" / "faiss.index",
                self.knowledge_base_path / "faiss.index",
            ]
            chunks_paths = [
                self.knowledge_base_path / "global" / "chunks.json",
                self.knowledge_base_path / "chunks.json",
            ]
            
            index_path = None
            chunks_path = None
            
            for ip, cp in zip(index_paths, chunks_paths):
                if ip.exists() and cp.exists():
                    index_path = ip
                    chunks_path = cp
                    break
            
            if index_path is None:
                logger.warning("No index found")
                return
            
            self.index = faiss.read_index(str(index_path))
            
            with open(chunks_path, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            
            self._build_section_hierarchy()
            
            if self.use_hybrid_search:
                self._build_bm25_index()
            
            self._initialized = True
            logger.info(f"✓ Loaded {len(self.chunks)} chunks, {len(self.sections)} sections")
            
        except ImportError:
            logger.error("FAISS not installed")
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
    
    def _build_section_hierarchy(self):
        """Build mapping from chunks to their parent sections."""
        for i, chunk in enumerate(self.chunks):
            chunk_id = chunk.get("chunk_id", str(i))
            
            chapter = chunk.get("metadata", {}).get("chapter", "")
            section = chunk.get("metadata", {}).get("section", "")
            
            # Clean chapter name
            chapter = self._clean_chapter_name(chapter)
            
            section_id = f"{chapter}_{section}".replace(" ", "_")
            
            self.chunk_to_section[chunk_id] = section_id
            
            if section_id not in self.sections:
                self.sections[section_id] = {
                    "chapter": chapter,
                    "section": section,
                    "chunks": [],
                    "full_content": "",
                    "topic": chunk.get("topic", "general"),
                    "subtopic": chunk.get("subtopic", "general"),
                }
            
            self.sections[section_id]["chunks"].append({
                "index": i,
                "chunk_id": chunk_id,
                "content": chunk.get("content", ""),
                "type": chunk.get("content_type", "text"),
            })
        
        # Build full section content with length limit
        for section_id, section_data in self.sections.items():
            chunks_content = []
            total_length = 0
            
            for chunk_info in section_data["chunks"]:
                content = chunk_info["content"]
                if total_length + len(content) <= self.MAX_SECTION_LENGTH:
                    chunks_content.append(content)
                    total_length += len(content)
                else:
                    # Add partial content if room
                    remaining = self.MAX_SECTION_LENGTH - total_length
                    if remaining > 200:
                        chunks_content.append(content[:remaining] + "...")
                    break
            
            section_data["full_content"] = "\n\n".join(chunks_content)
    
    def _clean_chapter_name(self, chapter: str) -> str:
        """Clean up chapter names."""
        if not chapter:
            return "General"
        
        # Remove bad chapter names
        bad_names = ["blank page", "untitled", "unknown", ""]
        if chapter.lower().strip() in bad_names:
            return "General"
        
        # Clean up the name
        chapter = chapter.strip()
        
        # Remove leading periods or special chars
        chapter = re.sub(r'^[\.\s]+', '', chapter)
        
        return chapter if chapter else "General"
    
    def _build_bm25_index(self):
        """Build BM25 index for keyword search."""
        try:
            from rank_bm25 import BM25Okapi
            
            self.corpus_tokens = []
            for chunk in self.chunks:
                content = chunk.get("content", "")
                # Better tokenization - preserve math terms
                content = self._clean_for_tokenization(content)
                tokens = re.findall(r'\b\w+\b', content.lower())
                self.corpus_tokens.append(tokens)
            
            self.bm25 = BM25Okapi(self.corpus_tokens)
            logger.info("✓ BM25 index built")
            
        except ImportError:
            logger.warning("rank_bm25 not installed. Run: pip install rank-bm25")
            self.use_hybrid_search = False
    
    def _clean_for_tokenization(self, text: str) -> str:
        """Clean text for BM25 tokenization."""
        # Replace math symbols with words
        replacements = [
            (r'x\^2|x²', 'x squared'),
            (r'x\^3|x³', 'x cubed'),
            (r'\^2', ' squared'),
            (r'\^3', ' cubed'),
            (r'√', 'sqrt square root'),
            (r'∫', 'integral'),
            (r'∑', 'sum summation'),
            (r'π', 'pi'),
            (r'd/dx', 'derivative ddx'),
            (r'dy/dx', 'derivative dydx'),
        ]
        
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.25,
        topic_filter: Optional[str] = None,
        rerank: Optional[bool] = None
    ) -> List[RetrievedDocument]:
        """Retrieve relevant documents with optional reranking."""
        if not self._initialized or self.index is None:
            logger.warning("Index not initialized")
            return []
        
        # Determine if we should rerank
        should_rerank = rerank if rerank is not None else self.use_reranker
        
        # Step 1: Query expansion
        if self.use_query_expansion:
            expanded = self.query_expander.expand(query)
            search_query = expanded.expanded
        else:
            search_query = query
        
        # Step 2: Semantic search - get more candidates if reranking
        candidate_multiplier = 4 if should_rerank else 2
        semantic_results = self._semantic_search(search_query, top_k * candidate_multiplier)
        
        # Step 3: Keyword search
        if self.use_hybrid_search and self.bm25 is not None:
            keyword_results = self._keyword_search(query, top_k * candidate_multiplier)
        else:
            keyword_results = {}
        
        # Step 4: Combine results
        combined_results = self._combine_results(semantic_results, keyword_results, alpha=0.6)
        
        # Step 5: Build candidate documents
        candidate_k = top_k * 3 if should_rerank else top_k
        min_score_for_candidates = min_score * 0.5 if should_rerank else min_score
        
        documents = self._build_candidate_documents(
            combined_results,
            candidate_k,
            min_score_for_candidates,
            topic_filter
        )
        
        # Step 6: RERANK
        if should_rerank and self.reranker and len(documents) > 1:
            logger.debug(f"Reranking {len(documents)} candidates for query: {query[:50]}...")
            try:
                documents = self.reranker.rerank(query, documents, top_k=top_k * 2)
                logger.debug(f"Reranking complete. Top score: {documents[0].combined_score if documents else 'N/A'}")
            except Exception as e:
                logger.warning(f"Reranking failed: {e}. Using original order.")
        
        # Step 7: Final filtering and top_k selection
        final_documents = []
        for doc in documents:
            if len(final_documents) >= top_k:
                break
            
            # Apply final score threshold
            if doc.combined_score < min_score:
                continue
            
            # Skip low quality
            if self._is_low_quality_content(doc.content):
                continue
            
            final_documents.append(doc)
        
        return final_documents
    
    def _build_candidate_documents(
        self,
        combined_results: Dict[int, Dict[str, float]],
        max_candidates: int,
        min_score: float,
        topic_filter: Optional[str]
    ) -> List[RetrievedDocument]:
        """Build candidate documents for reranking."""
        documents = []
        seen_sections = set()
        seen_content_hashes = set()
        
        sorted_results = sorted(
            combined_results.items(),
            key=lambda x: x[1]['combined'],
            reverse=True
        )
        
        for idx, scores in sorted_results:
            if len(documents) >= max_candidates:
                break
            
            if idx >= len(self.chunks):
                continue
            
            chunk = self.chunks[idx]
            
            # Filter by minimum score
            if scores['combined'] < min_score:
                continue
            
            # Topic filter
            if topic_filter:
                chunk_topic = chunk.get("topic", "")
                if topic_filter.lower() not in chunk_topic.lower():
                    continue
            
            # Content quality check
            content = chunk.get("content", "")
            if len(content) < self.MIN_CONTENT_LENGTH:
                continue
            
            # Deduplicate by content hash
            content_hash = hash(content[:200])
            if content_hash in seen_content_hashes:
                continue
            seen_content_hashes.add(content_hash)
            
            # Deduplicate by section
            chunk_id = chunk.get("chunk_id", str(idx))
            section_id = self.chunk_to_section.get(chunk_id, "")
            
            if self.return_full_sections and section_id in seen_sections:
                continue
            seen_sections.add(section_id)
            
            # Build document
            doc = self._build_document(idx, chunk, scores, section_id)
            documents.append(doc)
        
        return documents
    
    def _is_low_quality_content(self, content: str) -> bool:
        """Check if content is low quality."""
        # Too many special characters
        special_ratio = len(re.findall(r'[⎛⎝⎞⎠│├└┘┐┌]', content)) / max(len(content), 1)
        if special_ratio > 0.05:
            return True
        
        # Too short
        if len(content.strip()) < 50:
            return True
        
        # Mostly numbers/symbols
        alpha_ratio = len(re.findall(r'[a-zA-Z]', content)) / max(len(content), 1)
        if alpha_ratio < 0.3:
            return True
        
        return False
    
    def _semantic_search(self, query: str, top_k: int) -> Dict[int, float]:
        """Perform semantic search."""
        query_embedding = self.embedding_manager.encode(query, normalize=True)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results[int(idx)] = float(score)
        
        return results
    
    def _keyword_search(self, query: str, top_k: int) -> Dict[int, float]:
        """Perform keyword search using BM25."""
        if self.bm25 is None:
            return {}
        
        query_clean = self._clean_for_tokenization(query)
        query_tokens = re.findall(r'\b\w+\b', query_clean.lower())
        
        scores = self.bm25.get_scores(query_tokens)
        
        max_score = max(scores) if max(scores) > 0 else 1
        normalized = scores / max_score
        
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = {}
        for idx in top_indices:
            if scores[idx] > 0:
                results[int(idx)] = float(normalized[idx])
        
        return results
    
    def _combine_results(
        self,
        semantic: Dict[int, float],
        keyword: Dict[int, float],
        alpha: float = 0.6
    ) -> Dict[int, Dict[str, float]]:
        """Combine semantic and keyword search results."""
        all_indices = set(semantic.keys()) | set(keyword.keys())
        
        combined = {}
        for idx in all_indices:
            sem_score = semantic.get(idx, 0.0)
            kw_score = keyword.get(idx, 0.0)
            
            # Boost if both match
            if sem_score > 0.3 and kw_score > 0.3:
                boost = 0.1
            else:
                boost = 0.0
            
            combined[idx] = {
                'semantic': sem_score,
                'keyword': kw_score,
                'combined': alpha * sem_score + (1 - alpha) * kw_score + boost
            }
        
        return combined
    
    def _build_document(
        self,
        idx: int,
        chunk: Dict,
        scores: Dict[str, float],
        section_id: str
    ) -> RetrievedDocument:
        """Build a RetrievedDocument with cleaned content."""
        metadata = chunk.get("metadata", {})
        section_data = self.sections.get(section_id, {})
        
        content_type = chunk.get("content_type", "text")
        topic = chunk.get("topic", "general")
        
        # Build title from content type
        title_map = {
            "definition": "Definition",
            "theorem": "Theorem",
            "theorem_statement": "Theorem",
            "proof": "Proof",
            "example": "Example",
            "solved_example": "Solved Example",
            "solved_example_step": "Solved Example Step",
            "formula": "Formula",
            "text": "Content",
        }
        title = title_map.get(content_type, content_type.replace('_', ' ').title())
        
        # Get content - clean it
        chunk_content = self._clean_content(chunk.get("content", ""))
        
        # Get parent content if returning full sections
        if self.return_full_sections and section_data:
            parent_content = self._clean_content(section_data.get("full_content", ""))
            # Use parent if it's better, otherwise use chunk
            if len(parent_content) > len(chunk_content) * 1.5:
                display_content = parent_content
            else:
                display_content = chunk_content
        else:
            parent_content = ""
            display_content = chunk_content
        
        # Get chapter - clean it
        chapter = self._clean_chapter_name(metadata.get("chapter", ""))
        
        return RetrievedDocument(
            content=display_content,
            title=title,
            section=metadata.get("section", ""),
            subsection=chunk.get("subtopic", ""),
            chapter=chapter,
            topic=topic,
            subtopic=chunk.get("subtopic", ""),
            content_type=content_type,
            score=scores['semantic'],
            keyword_score=scores.get('keyword', 0.0),
            combined_score=scores['combined'],
            source=chunk.get("source", ""),
            page_number=chunk.get("page_number", 0),
            chunk_id=chunk.get("chunk_id", str(idx)),
            parent_content=parent_content,
        )
    
    def _clean_content(self, content: str) -> str:
        """Clean content for display."""
        if not content:
            return ""
        
        # Remove excessive Unicode box-drawing characters
        content = re.sub(r'[⎛⎝⎞⎠│├└┘┐┌─═║╔╗╚╝╠╣╦╩╬]+', '', content)
        
        # Clean up fractions notation
        content = re.sub(r'\s+/\s+', '/', content)
        
        # Normalize whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r' {2,}', ' ', content)
        content = re.sub(r'\t+', ' ', content)
        
        # Remove "Part X/Y" prefixes if redundant
        content = re.sub(r'^(Example|Definition|Theorem)\s*\(Part \d+/\d+\):\s*', '', content)
        
        # Limit length
        if len(content) > self.MAX_SECTION_LENGTH:
            cutoff = content[:self.MAX_SECTION_LENGTH].rfind('.')
            if cutoff > self.MAX_SECTION_LENGTH * 0.7:
                content = content[:cutoff + 1]
            else:
                content = content[:self.MAX_SECTION_LENGTH] + "..."
        
        return content.strip()
    
    def format_for_display(self, documents: List[RetrievedDocument]) -> List[Dict]:
        """Format documents for UI display."""
        return [doc.to_dict() for doc in documents]
    
    def format_context_for_llm(
        self,
        documents: List[RetrievedDocument],
        max_tokens: int = 3000
    ) -> str:
        """Format retrieved documents as context for LLM."""
        context_parts = []
        current_length = 0
        char_limit = max_tokens * 4
        
        for i, doc in enumerate(documents, 1):
            section = f"""### Reference {i}: {doc.title} ({doc.topic}/{doc.subtopic})
**From: {doc.chapter}**

{doc.content}

---"""
            
            if current_length + len(section) > char_limit:
                available = char_limit - current_length - 100
                if available > 200:
                    truncated = doc.content[:available] + "..."
                    section = f"""### Reference {i}: {doc.title} ({doc.topic}/{doc.subtopic})

{truncated}

---"""
                    context_parts.append(section)
                break
            
            context_parts.append(section)
            current_length += len(section)
        
        return "\n\n".join(context_parts)


# Backward-compatible wrapper
class MathRAGRetriever(ImprovedRAGRetriever):
    """Backward-compatible alias that returns dictionaries."""
    
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict]:
        """Retrieve and return as list of dictionaries."""
        documents = super().retrieve(query, top_k, **kwargs)
        return self.format_for_display(documents)
    
    def format_for_display(self, documents: List[RetrievedDocument]) -> List[Dict]:
        """Format documents for UI display - ensures JSON serializable."""
        result = []
        for doc in documents:
            if hasattr(doc, 'to_dict'):
                result.append(doc.to_dict())
            elif isinstance(doc, dict):
                result.append(doc)
            else:
                result.append({"content": str(doc), "score": 0})
        return result


_retriever_instance = None

def get_retriever(embedding_manager=None, use_reranker: bool = True) -> MathRAGRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = MathRAGRetriever(
            embedding_manager=embedding_manager,
            use_reranker=use_reranker
        )
    return _retriever_instance


# Test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    
    print("=" * 70)
    print("IMPROVED RAG RETRIEVER TEST v2")
    print("=" * 70)
    
    retriever = ImprovedRAGRetriever(
        use_query_expansion=True,
        use_hybrid_search=True,
        return_full_sections=True
    )
    
    if not retriever._initialized:
        print("❌ Retriever not initialized")
        sys.exit(1)
    
    print(f"\n📊 Stats: {len(retriever.chunks)} chunks, {len(retriever.sections)} sections")
    
    test_queries = [
        "What is the probability of drawing a king from a deck of 52 cards?",
        "Find the derivative of x^3 + 2x",
        "What is the probability of getting heads in a coin toss?",
        "How to calculate the mean of a dataset?",
        "Solve the quadratic equation x^2 - 5x + 6 = 0",
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"🔍 Query: {query}")
        print("="*70)
        
        results = retriever.retrieve(query, top_k=3)
        
        for i, doc in enumerate(results, 1):
            # Score indicator
            score = doc.combined_score
            if score >= 0.6:
                indicator = "🟢"
            elif score >= 0.4:
                indicator = "🟡"
            else:
                indicator = "🔴"
            
            print(f"\n{indicator} [{i}] {doc.title} ({doc.topic}/{doc.subtopic})")
            print(f"   Score: {doc.combined_score:.3f} (sem: {doc.score:.3f}, kw: {doc.keyword_score:.3f})")
            print(f"   Chapter: {doc.chapter}")
            print(f"   Content: {doc.content[:150].replace(chr(10), ' ')}...")
            print(f"   Length: {len(doc.content)} chars")
    
    print("\n" + "=" * 70)
    print("✅ Test completed!")
    print("=" * 70)