#!/usr/bin/env python3
"""
Main script to process textbooks and build knowledge base.

Usage:
    python scripts/process_textbooks.py --pdf-dir data/raw_pdfs --output-dir data
"""

import argparse
import json
from pathlib import Path
from typing import List
import logging
from dataclasses import asdict

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.pdf_processor import MathPDFProcessor, process_pdf
from src.data_pipeline.content_chunker import MathContentChunker
from src.data_pipeline.knowledge_builder import (
    MathKnowledgeBaseBuilder,
    TopicSpecificKnowledgeBase
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_all_pdfs(pdf_dir: str, output_dir: str) -> List[dict]:
    """Process all PDFs in directory"""
    pdf_path = Path(pdf_dir)
    output_path = Path(output_dir) / "processed"
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_content = []
    pdf_files = list(pdf_path.glob("*.pdf"))
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    for pdf_file in pdf_files:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing: {pdf_file.name}")
        logger.info('='*50)
        
        output_file = output_path / f"{pdf_file.stem}.json"
        
        try:
            content = process_pdf(str(pdf_file), str(output_file))
            all_content.append(content)
            logger.info(f"✓ Saved to: {output_file}")
        except Exception as e:
            logger.error(f"✗ Failed to process {pdf_file.name}: {e}")
            continue
    
    return all_content


def chunk_all_content(all_content: List[dict]) -> List:
    """Chunk all processed content"""
    chunker = MathContentChunker(
        max_chunk_size=1000,
        min_chunk_size=100,
        overlap=100
    )
    
    all_chunks = []
    
    for content in all_content:
        logger.info(f"Chunking content from: {content['metadata']['title']}")
        chunks = chunker.process_book_content(content)
        all_chunks.extend(chunks)
        logger.info(f"  → Generated {len(chunks)} chunks")
    
    logger.info(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks


def build_knowledge_base(chunks: List, output_dir: str, use_topic_indices: bool = True):
    """Build FAISS knowledge base"""
    output_path = Path(output_dir) / "vector_store"
    
    if use_topic_indices:
        logger.info("Building topic-specific knowledge base...")
        kb = TopicSpecificKnowledgeBase()
        kb.build_from_chunks(chunks)
        kb.save(str(output_path))
    else:
        logger.info("Building single knowledge base...")
        kb = MathKnowledgeBaseBuilder()
        kb.add_chunks(chunks)
        kb.build_index()
        kb.save(str(output_path / "global"))
    
    # Print stats
    if hasattr(kb, 'global_index') and kb.global_index:
        stats = kb.global_index.get_stats()
    else:
        stats = kb.get_stats()
    
    logger.info("\n" + "="*50)
    logger.info("Knowledge Base Statistics")
    logger.info("="*50)
    logger.info(f"Total chunks: {stats['total_chunks']}")
    logger.info("\nBy Topic:")
    for topic, count in stats['by_topic'].items():
        logger.info(f"  {topic}: {count}")
    logger.info("\nBy Type:")
    for ctype, count in stats['by_type'].items():
        logger.info(f"  {ctype}: {count}")
    
    return kb


def create_topic_json_files(chunks: List, output_dir: str):
    """Create separate JSON files for each topic (for your existing RAG)"""
    output_path = Path(output_dir) / "knowledge_base"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Group by topic
    topic_data = {}
    
    for chunk in chunks:
        if hasattr(chunk, '__dict__'):
            chunk_dict = asdict(chunk)
        else:
            chunk_dict = chunk
            
        topic = chunk_dict.get("topic", "general")
        subtopic = chunk_dict.get("subtopic", "general")
        
        if topic not in topic_data:
            topic_data[topic] = {
                "topic": topic,
                "subtopics": {}
            }
        
        if subtopic not in topic_data[topic]["subtopics"]:
            topic_data[topic]["subtopics"][subtopic] = {
                "definitions": [],
                "theorems": [],
                "examples": [],
                "formulas": []
            }
        
        ctype = chunk_dict.get("content_type", "text")
        
        if "definition" in ctype:
            topic_data[topic]["subtopics"][subtopic]["definitions"].append(chunk_dict)
        elif "theorem" in ctype or "proof" in ctype:
            topic_data[topic]["subtopics"][subtopic]["theorems"].append(chunk_dict)
        elif "example" in ctype:
            topic_data[topic]["subtopics"][subtopic]["examples"].append(chunk_dict)
        elif "formula" in ctype:
            topic_data[topic]["subtopics"][subtopic]["formulas"].append(chunk_dict)
    
    # Save each topic as separate file
    for topic, data in topic_data.items():
        output_file = output_path / f"{topic}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved: {output_file}")


def test_search(kb, queries: List[str]):
    """Test the knowledge base with sample queries"""
    logger.info("\n" + "="*50)
    logger.info("Testing Knowledge Base")
    logger.info("="*50)
    
    for query in queries:
        logger.info(f"\nQuery: {query}")
        logger.info("-" * 40)
        
        results = kb.search(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            chunk = result["chunk"]
            score = result["score"]
            logger.info(f"\n{i}. [Score: {score:.4f}]")
            logger.info(f"   Type: {chunk['content_type']}")
            logger.info(f"   Topic: {chunk['topic']}/{chunk['subtopic']}")
            logger.info(f"   Content: {chunk['content'][:200]}...")


def main():
    parser = argparse.ArgumentParser(description="Process math textbooks for RAG")
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default="data/raw_pdfs",
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="Skip PDF processing, use existing JSON files"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test queries after building"
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Math Mentor - Textbook Processing Pipeline")
    logger.info("="*60)
    
    # Step 1: Process PDFs
    if not args.skip_processing:
        logger.info("\n📚 Step 1: Processing PDFs...")
        all_content = process_all_pdfs(args.pdf_dir, args.output_dir)
    else:
        logger.info("\n📚 Step 1: Loading existing processed files...")
        processed_path = Path(args.output_dir) / "processed"
        all_content = []
        for json_file in processed_path.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                all_content.append(json.load(f))
    
    if not all_content:
        logger.error("No content to process!")
        return
    
    # Step 2: Chunk content
    logger.info("\n✂️  Step 2: Chunking content...")
    all_chunks = chunk_all_content(all_content)
    
    # Step 3: Create topic JSON files (for existing RAG)
    logger.info("\n📁 Step 3: Creating topic-specific JSON files...")
    create_topic_json_files(all_chunks, args.output_dir)
    
    # Step 4: Build FAISS index
    logger.info("\n🔍 Step 4: Building vector knowledge base...")
    kb = build_knowledge_base(all_chunks, args.output_dir)
    
    # Step 5: Test (optional)
    if args.test:
        test_queries = [
            "What is the derivative of sin(x)?",
            "How to find the integral of e^x?",
            "Explain the quadratic formula",
            "What is the chain rule?",
            "How to solve trigonometric equations?",
            "What is L'Hopital's rule?",
            "Explain integration by parts",
            "What is the fundamental theorem of calculus?",
            "How to find limits using substitution?",
            "What are the properties of logarithms?"
        ]
        
        # Use global index for testing
        if hasattr(kb, 'global_index') and kb.global_index:
            test_search(kb.global_index, test_queries)
        else:
            test_search(kb, test_queries)
    
    logger.info("\n" + "="*60)
    logger.info("✅ Processing Complete!")
    logger.info("="*60)
    logger.info(f"\nOutput files:")
    logger.info(f"  📄 Processed JSONs: {args.output_dir}/processed/")
    logger.info(f"  📁 Topic JSONs: {args.output_dir}/knowledge_base/")
    logger.info(f"  🔍 Vector Store: {args.output_dir}/vector_store/")


if __name__ == "__main__":
    main()