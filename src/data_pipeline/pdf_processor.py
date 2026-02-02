"""
PDF Processor for Mathematical Textbooks
Extracts text, formulas, examples, and exercises from PDF textbooks.
"""

import fitz  # PyMuPDF
import pdfplumber
import re
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from enum import Enum
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Types of content that can be extracted"""
    CHAPTER_TITLE = "chapter_title"
    SECTION_TITLE = "section_title"
    DEFINITION = "definition"
    THEOREM = "theorem"
    FORMULA = "formula"
    EXAMPLE = "example"
    EXERCISE = "exercise"
    PROOF = "proof"
    NOTE = "note"
    TEXT = "text"


@dataclass
class ExtractedContent:
    """Represents a piece of extracted content"""
    content_type: str
    title: str
    content: str
    page_number: int
    chapter: str
    section: str
    formulas: List[str]
    metadata: Dict


@dataclass
class ChapterInfo:
    """Chapter information"""
    number: int
    title: str
    start_page: int
    end_page: int
    sections: List[Dict]


class MathPDFProcessor:
    """
    Extracts mathematical content from PDF textbooks.
    Optimized for OpenStax-style textbooks.
    """
    
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self.doc = fitz.open(pdf_path)
        self.pdf_plumber = pdfplumber.open(pdf_path)
        self.total_pages = len(self.doc)
        
        # Regex patterns for mathematical content
        self.patterns = {
            # Chapter patterns (customize based on your PDFs)
            'chapter': re.compile(
                r'^(?:Chapter|CHAPTER)\s*(\d+)\s*[:\-]?\s*(.+?)$',
                re.MULTILINE | re.IGNORECASE
            ),
            'section': re.compile(
                r'^(\d+\.\d+)\s+(.+?)$',
                re.MULTILINE
            ),
            'subsection': re.compile(
                r'^(\d+\.\d+\.\d+)\s+(.+?)$',
                re.MULTILINE
            ),
            
            # Content patterns
            'definition': re.compile(
                r'(?:Definition|DEFINITION)\s*(?:\d+(?:\.\d+)?)?[:\.]?\s*(.*?)(?=(?:Definition|Theorem|Example|Proof|DEFINITION|THEOREM|EXAMPLE|PROOF)|\Z)',
                re.DOTALL | re.IGNORECASE
            ),
            'theorem': re.compile(
                r'(?:Theorem|THEOREM)\s*(?:\d+(?:\.\d+)?)?[:\.]?\s*(.*?)(?=(?:Definition|Theorem|Example|Proof|Corollary|DEFINITION|THEOREM|EXAMPLE|PROOF|COROLLARY)|\Z)',
                re.DOTALL | re.IGNORECASE
            ),
            'example': re.compile(
                r'(?:Example|EXAMPLE)\s*(?:\d+(?:\.\d+)?)?[:\.]?\s*(.*?)(?=(?:Example|Exercise|Try It|EXAMPLE|EXERCISE|TRY IT|Definition|Theorem)|\Z)',
                re.DOTALL | re.IGNORECASE
            ),
            'exercise': re.compile(
                r'(?:Exercise|EXERCISE|Problem|PROBLEM)\s*(?:\d+(?:\.\d+)?)?[:\.]?\s*(.*?)(?=(?:Exercise|Problem|EXERCISE|PROBLEM|\d+\.)|\Z)',
                re.DOTALL | re.IGNORECASE
            ),
            'proof': re.compile(
                r'(?:Proof|PROOF)[:\.]?\s*(.*?)(?:□|QED|∎|(?=(?:Definition|Theorem|Example|Corollary)))',
                re.DOTALL | re.IGNORECASE
            ),
            
            # Formula patterns
            'latex_inline': re.compile(r'\$([^\$]+)\$'),
            'latex_display': re.compile(r'\$\$([^\$]+)\$\$'),
            'math_expression': re.compile(
                r'(?:[a-zA-Z]\s*=\s*[^,\.\n]+)|'
                r'(?:\d+[a-zA-Z]\s*[\+\-]\s*\d+)|'
                r'(?:∫|∑|∏|√|∞|≤|≥|≠|±|×|÷|∂|∇|∈|∉|⊂|⊃|∪|∩)'
            ),
            'equation_number': re.compile(r'$(\d+(?:\.\d+)?)$'),
            
            # Special math notation
            'fraction': re.compile(r'(\d+)\s*/\s*(\d+)'),
            'exponent': re.compile(r'(\w+)\s*\^\s*(\d+|\{[^\}]+\})'),
            'subscript': re.compile(r'(\w+)\s*_\s*(\d+|\{[^\}]+\})'),
        }
        
        # Book-specific configuration
        self.book_config = self._detect_book_type()
        
    def _detect_book_type(self) -> Dict:
        """Detect book type based on content analysis"""
        first_pages_text = ""
        for i in range(min(10, self.total_pages)):
            first_pages_text += self.doc[i].get_text()
        
        # Detect OpenStax
        if "OpenStax" in first_pages_text or "openstax" in first_pages_text.lower():
            return {
                "type": "openstax",
                "chapter_start_pattern": r"^(\d+)\s*\|\s*(.+)$",
                "toc_pages": (3, 15),
                "content_start": 20
            }
        
        # Default configuration
        return {
            "type": "generic",
            "chapter_start_pattern": r"^Chapter\s*(\d+)",
            "toc_pages": (1, 10),
            "content_start": 15
        }
    
    def extract_toc(self) -> List[ChapterInfo]:
        """Extract table of contents"""
        chapters = []
        toc = self.doc.get_toc()
        
        if toc:
            # Use built-in TOC if available
            current_chapter = None
            for level, title, page in toc:
                if level == 1:
                    if current_chapter:
                        chapters.append(current_chapter)
                    
                    # Extract chapter number from title
                    match = re.search(r'(\d+)', title)
                    chapter_num = int(match.group(1)) if match else len(chapters) + 1
                    
                    current_chapter = ChapterInfo(
                        number=chapter_num,
                        title=title,
                        start_page=page,
                        end_page=self.total_pages,
                        sections=[]
                    )
                elif level == 2 and current_chapter:
                    current_chapter.sections.append({
                        "title": title,
                        "page": page
                    })
            
            if current_chapter:
                chapters.append(current_chapter)
                
            # Set end pages
            for i in range(len(chapters) - 1):
                chapters[i].end_page = chapters[i + 1].start_page - 1
        
        else:
            # Manual TOC extraction
            chapters = self._extract_toc_manually()
        
        logger.info(f"Found {len(chapters)} chapters")
        return chapters
    
    def _extract_toc_manually(self) -> List[ChapterInfo]:
        """Manually extract TOC by scanning pages"""
        chapters = []
        
        for page_num in range(self.total_pages):
            text = self.doc[page_num].get_text()
            
            # Look for chapter headings
            chapter_match = self.patterns['chapter'].search(text)
            if chapter_match:
                chapter_num = int(chapter_match.group(1))
                chapter_title = chapter_match.group(2).strip()
                
                chapters.append(ChapterInfo(
                    number=chapter_num,
                    title=chapter_title,
                    start_page=page_num,
                    end_page=self.total_pages,
                    sections=[]
                ))
        
        # Set end pages
        for i in range(len(chapters) - 1):
            chapters[i].end_page = chapters[i + 1].start_page - 1
            
        return chapters
    
    def extract_page_content(self, page_num: int) -> Dict:
        """Extract all content from a single page"""
        page = self.doc[page_num]
        text = page.get_text()
        
        # Also get text with pdfplumber for better table extraction
        plumber_page = self.pdf_plumber.pages[page_num]
        tables = plumber_page.extract_tables()
        
        # Extract text blocks with positioning
        blocks = page.get_text("dict")["blocks"]
        
        content = {
            "page_number": page_num + 1,
            "raw_text": text,
            "text_blocks": [],
            "tables": tables,
            "images": [],
            "formulas": self._extract_formulas(text)
        }
        
        # Process text blocks
        for block in blocks:
            if "lines" in block:
                block_text = ""
                for line in block["lines"]:
                    for span in line["spans"]:
                        block_text += span["text"] + " "
                
                content["text_blocks"].append({
                    "text": block_text.strip(),
                    "bbox": block["bbox"],
                    "font_size": self._get_dominant_font_size(block)
                })
        
        # Extract images
        image_list = page.get_images()
        for img_index, img in enumerate(image_list):
            content["images"].append({
                "index": img_index,
                "xref": img[0],
                "bbox": page.get_image_bbox(img)
            })
        
        return content
    
    def _get_dominant_font_size(self, block: Dict) -> float:
        """Get the most common font size in a block"""
        sizes = []
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    sizes.append(span.get("size", 12))
        return max(set(sizes), key=sizes.count) if sizes else 12
    
    def _extract_formulas(self, text: str) -> List[Dict]:
        """Extract mathematical formulas from text"""
        formulas = []
        
        # LaTeX inline
        for match in self.patterns['latex_inline'].finditer(text):
            formulas.append({
                "type": "latex_inline",
                "content": match.group(1),
                "position": match.span()
            })
        
        # LaTeX display
        for match in self.patterns['latex_display'].finditer(text):
            formulas.append({
                "type": "latex_display",
                "content": match.group(1),
                "position": match.span()
            })
        
        # Mathematical expressions
        for match in self.patterns['math_expression'].finditer(text):
            formulas.append({
                "type": "expression",
                "content": match.group(0),
                "position": match.span()
            })
        
        return formulas
    
    def extract_definitions(self, text: str, page_num: int, chapter: str, section: str) -> List[ExtractedContent]:
        """Extract definitions from text"""
        definitions = []
        
        for match in self.patterns['definition'].finditer(text):
            content = match.group(1).strip()
            if len(content) > 20:  # Filter out too short matches
                definitions.append(ExtractedContent(
                    content_type=ContentType.DEFINITION.value,
                    title=self._extract_definition_title(content),
                    content=content,
                    page_number=page_num,
                    chapter=chapter,
                    section=section,
                    formulas=self._extract_formulas(content),
                    metadata={"source": str(self.pdf_path.name)}
                ))
        
        return definitions
    
    def extract_theorems(self, text: str, page_num: int, chapter: str, section: str) -> List[ExtractedContent]:
        """Extract theorems from text"""
        theorems = []
        
        for match in self.patterns['theorem'].finditer(text):
            content = match.group(1).strip()
            if len(content) > 20:
                theorems.append(ExtractedContent(
                    content_type=ContentType.THEOREM.value,
                    title=self._extract_theorem_title(content),
                    content=content,
                    page_number=page_num,
                    chapter=chapter,
                    section=section,
                    formulas=self._extract_formulas(content),
                    metadata={"source": str(self.pdf_path.name)}
                ))
        
        return theorems
    
    def extract_examples(self, text: str, page_num: int, chapter: str, section: str) -> List[ExtractedContent]:
        """Extract solved examples from text"""
        examples = []
        
        for match in self.patterns['example'].finditer(text):
            content = match.group(1).strip()
            if len(content) > 50:  # Examples should be substantial
                # Try to separate problem and solution
                problem, solution = self._split_example(content)
                
                examples.append(ExtractedContent(
                    content_type=ContentType.EXAMPLE.value,
                    title=f"Example from {section}",
                    content=content,
                    page_number=page_num,
                    chapter=chapter,
                    section=section,
                    formulas=self._extract_formulas(content),
                    metadata={
                        "source": str(self.pdf_path.name),
                        "problem": problem,
                        "solution": solution
                    }
                ))
        
        return examples
    
    def _split_example(self, content: str) -> Tuple[str, str]:
        """Split example into problem and solution"""
        # Common solution indicators
        solution_markers = [
            r'Solution[:\.]',
            r'Answer[:\.]',
            r'Solve[:\.]',
            r'We have',
            r'Let us',
            r'First,',
            r'Step 1'
        ]
        
        for marker in solution_markers:
            match = re.search(marker, content, re.IGNORECASE)
            if match:
                problem = content[:match.start()].strip()
                solution = content[match.start():].strip()
                return problem, solution
        
        # If no marker found, assume first paragraph is problem
        paragraphs = content.split('\n\n')
        if len(paragraphs) > 1:
            return paragraphs[0], '\n\n'.join(paragraphs[1:])
        
        return content, ""
    
    def _extract_definition_title(self, content: str) -> str:
        """Extract title from definition content"""
        # Look for pattern like "A function f is..."
        match = re.search(r'^(?:A|An|The)\s+(\w+(?:\s+\w+)?)', content)
        if match:
            return match.group(1).title()
        
        # First few words
        words = content.split()[:3]
        return ' '.join(words)
    
    def _extract_theorem_title(self, content: str) -> str:
        """Extract title from theorem content"""
        # Named theorems
        named_pattern = re.search(
            r'(?:(\w+(?:\'s)?)\s+(?:Theorem|Rule|Formula|Law))',
            content,
            re.IGNORECASE
        )
        if named_pattern:
            return named_pattern.group(0)
        
        # First sentence
        first_sentence = content.split('.')[0]
        if len(first_sentence) < 100:
            return first_sentence
        
        return "Theorem"
    
    def process_full_book(self) -> Dict:
        """Process the entire PDF and extract all content"""
        logger.info(f"Processing: {self.pdf_path.name}")
        
        chapters = self.extract_toc()
        
        book_content = {
            "metadata": {
                "title": self.pdf_path.stem,
                "total_pages": self.total_pages,
                "chapters_count": len(chapters),
                "source_file": str(self.pdf_path.name)
            },
            "chapters": [],
            "all_definitions": [],
            "all_theorems": [],
            "all_examples": [],
            "all_formulas": []
        }
        
        for chapter in chapters:
            logger.info(f"Processing Chapter {chapter.number}: {chapter.title}")
            
            chapter_data = {
                "number": chapter.number,
                "title": chapter.title,
                "sections": [],
                "definitions": [],
                "theorems": [],
                "examples": [],
                "formulas": []
            }
            
            current_section = "Introduction"
            
            for page_num in range(chapter.start_page, min(chapter.end_page + 1, self.total_pages)):
                try:
                    page_content = self.extract_page_content(page_num)
                    text = page_content["raw_text"]
                    
                    # Check for section change
                    section_match = self.patterns['section'].search(text)
                    if section_match:
                        current_section = f"{section_match.group(1)} {section_match.group(2)}"
                    
                    # Extract content types
                    definitions = self.extract_definitions(
                        text, page_num + 1, chapter.title, current_section
                    )
                    theorems = self.extract_theorems(
                        text, page_num + 1, chapter.title, current_section
                    )
                    examples = self.extract_examples(
                        text, page_num + 1, chapter.title, current_section
                    )
                    
                    # Add to chapter
                    chapter_data["definitions"].extend([asdict(d) for d in definitions])
                    chapter_data["theorems"].extend([asdict(t) for t in theorems])
                    chapter_data["examples"].extend([asdict(e) for e in examples])
                    chapter_data["formulas"].extend(page_content["formulas"])
                    
                    # Add to book totals
                    book_content["all_definitions"].extend([asdict(d) for d in definitions])
                    book_content["all_theorems"].extend([asdict(t) for t in theorems])
                    book_content["all_examples"].extend([asdict(e) for e in examples])
                    book_content["all_formulas"].extend(page_content["formulas"])
                    
                except Exception as e:
                    logger.warning(f"Error processing page {page_num}: {e}")
                    continue
            
            book_content["chapters"].append(chapter_data)
        
        logger.info(f"Extraction complete. Found:")
        logger.info(f"  - {len(book_content['all_definitions'])} definitions")
        logger.info(f"  - {len(book_content['all_theorems'])} theorems")
        logger.info(f"  - {len(book_content['all_examples'])} examples")
        logger.info(f"  - {len(book_content['all_formulas'])} formulas")
        
        return book_content
    
    def save_processed_content(self, output_path: str):
        """Process and save content to JSON"""
        content = self.process_full_book()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved processed content to: {output_file}")
        return content
    
    def close(self):
        """Close PDF files"""
        self.doc.close()
        self.pdf_plumber.close()


# Convenience function
def process_pdf(pdf_path: str, output_path: str) -> Dict:
    """Process a single PDF file"""
    processor = MathPDFProcessor(pdf_path)
    try:
        content = processor.save_processed_content(output_path)
        return content
    finally:
        processor.close()