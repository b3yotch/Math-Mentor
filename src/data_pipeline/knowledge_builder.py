"""
Knowledge Base Builder
Builds FAISS index and document store from processed content.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import asdict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import logging

from .content_chunker import Chunk, MathContentChunker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MathKnowledgeBaseBuilder:
    """
    Builds a vector knowledge base from processed mathematical content.
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        index_type: str = "flat"  # "flat" or "ivf"
    ):
        logger.info(f"Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        self.index_type = index_type
        
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Dict] = []
        self.chunk_id_to_index: Dict[str, int] = {}
        
    def add_chunks(self, chunks: List[Chunk]):
        """Add chunks to the knowledge base"""
        for chunk in chunks:
            if isinstance(chunk, Chunk):
                chunk_dict = asdict(chunk)
            else:
                chunk_dict = chunk
                
            self.chunk_id_to_index[chunk_dict["chunk_id"]] = len(self.chunks)
            self.chunks.append(chunk_dict)
    
    def build_index(self, use_gpu: bool = False):
        """Build FAISS index from all chunks"""
        if not self.chunks:
            raise ValueError("No chunks added to knowledge base")
        
        logger.info(f"Building index for {len(self.chunks)} chunks...")
        
        # Extract text content for embedding
        texts = [chunk["content"] for chunk in self.chunks]
        
        # Generate embeddings in batches
        logger.info("Generating embeddings...")
        embeddings = self.encoder.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Create FAISS index
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "ivf":
            # IVF index for larger datasets
            nlist = min(100, len(self.chunks) // 10)
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            self.index.train(embeddings)
        
        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except:
                logger.warning("GPU not available, using CPU")
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        logger.info(f"Index built with {self.index.ntotal} vectors")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_topic: Optional[str] = None,
        filter_type: Optional[str] = None
    ) -> List[Dict]:
        """Search the knowledge base"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k * 3)  # Get more for filtering
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
                
            chunk = self.chunks[idx]
            
            # Apply filters
            if filter_topic and chunk.get("topic") != filter_topic:
                continue
            if filter_type and chunk.get("content_type") != filter_type:
                continue
            
            results.append({
                "chunk": chunk,
                "score": float(score)
            })
            
            if len(results) >= top_k:
                break
        
        return results
    
    def save(self, output_dir: str):
        """Save knowledge base to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self.index is not None:
            # Convert GPU index to CPU for saving
            if hasattr(self.index, 'index'):
                cpu_index = faiss.index_gpu_to_cpu(self.index)
            else:
                cpu_index = self.index
            faiss.write_index(cpu_index, str(output_path / "faiss.index"))
        
        # Save chunks
        with open(output_path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
        
        # Save metadata
        metadata = {
            "total_chunks": len(self.chunks),
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "topics": list(set(c.get("topic", "unknown") for c in self.chunks)),
            "content_types": list(set(c.get("content_type", "unknown") for c in self.chunks))
        }
        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Knowledge base saved to {output_dir}")
    
    def load(self, input_dir: str):
        """Load knowledge base from disk"""
        input_path = Path(input_dir)
        
        # Load FAISS index
        self.index = faiss.read_index(str(input_path / "faiss.index"))
        
        # Load chunks
        with open(input_path / "chunks.json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        
        # Rebuild ID mapping
        self.chunk_id_to_index = {
            chunk["chunk_id"]: i for i, chunk in enumerate(self.chunks)
        }
        
        logger.info(f"Loaded knowledge base with {len(self.chunks)} chunks")
    
    def get_stats(self) -> Dict:
        """Get statistics about the knowledge base"""
        topic_counts = {}
        type_counts = {}
        source_counts = {}
        
        for chunk in self.chunks:
            topic = chunk.get("topic", "unknown")
            ctype = chunk.get("content_type", "unknown")
            source = chunk.get("source", "unknown")
            
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
            type_counts[ctype] = type_counts.get(ctype, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1
        
        return {
            "total_chunks": len(self.chunks),
            "by_topic": topic_counts,
            "by_type": type_counts,
            "by_source": source_counts
        }


class TopicSpecificKnowledgeBase:
    """
    Manages multiple topic-specific indices for better retrieval.
    """
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-mpnet-base-v2"):
        self.embedding_model = embedding_model
        self.topic_indices: Dict[str, MathKnowledgeBaseBuilder] = {}
        self.global_index: Optional[MathKnowledgeBaseBuilder] = None
    
    def build_from_chunks(self, chunks: List[Chunk]):
        """Build topic-specific indices from chunks"""
        # Group chunks by topic
        topic_chunks: Dict[str, List[Chunk]] = {}
        
        for chunk in chunks:
            topic = chunk.topic if isinstance(chunk, Chunk) else chunk.get("topic", "general")
            if topic not in topic_chunks:
                topic_chunks[topic] = []
            topic_chunks[topic].append(chunk)
        
        # Build index for each topic
        for topic, t_chunks in topic_chunks.items():
            logger.info(f"Building index for topic: {topic} ({len(t_chunks)} chunks)")
            
            kb = MathKnowledgeBaseBuilder(self.embedding_model)
            kb.add_chunks(t_chunks)
            kb.build_index()
            
            self.topic_indices[topic] = kb
        
        # Build global index
        logger.info("Building global index...")
        self.global_index = MathKnowledgeBaseBuilder(self.embedding_model)
        self.global_index.add_chunks(chunks)
        self.global_index.build_index()
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        topic: Optional[str] = None
    ) -> List[Dict]:
        """Search knowledge base, optionally within specific topic"""
        if topic and topic in self.topic_indices:
            return self.topic_indices[topic].search(query, top_k)
        
        if self.global_index:
            return self.global_index.search(query, top_k)
        
        return []
    
    def save(self, output_dir: str):
        """Save all indices"""
        output_path = Path(output_dir)
        
        # Save topic indices
        for topic, kb in self.topic_indices.items():
            kb.save(str(output_path / "topics" / topic))
        
        # Save global index
        if self.global_index:
            self.global_index.save(str(output_path / "global"))
        
        # Save topic list
        with open(output_path / "topics.json", "w") as f:
            json.dump(list(self.topic_indices.keys()), f)
    
    def load(self, input_dir: str):
        """Load all indices"""
        input_path = Path(input_dir)
        
        # Load topic list
        with open(input_path / "topics.json", "r") as f:
            topics = json.load(f)
        
        # Load topic indices
        for topic in topics:
            kb = MathKnowledgeBaseBuilder(self.embedding_model)
            kb.load(str(input_path / "topics" / topic))
            self.topic_indices[topic] = kb
        
        # Load global index
        self.global_index = MathKnowledgeBaseBuilder(self.embedding_model)
        self.global_index.load(str(input_path / "global"))