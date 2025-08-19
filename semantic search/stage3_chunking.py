#!/usr/bin/env python3
"""
Stage 3: Content Chunking Pipeline (Local Version)
Chunks sections from Stage 2.5 into 500-750 token chunks for semantic search

This is the local version for development and testing.
For production use with NAS, use stage3_chunking_nas.py
"""

import json
import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import argparse
from pathlib import Path
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ProtectedZone:
    """Represents a region of content that cannot be split (HTML tags)"""
    start: int
    end: int
    content: str

@dataclass
class BreakPoint:
    """Represents a potential break point in the content"""
    position: int
    type: str  # 'heading', 'paragraph', 'list', 'sentence', 'comma'
    priority: int  # 1 (best) to 5 (worst)

class SimpleTokenizer:
    """Simple token counter that approximates GPT tokenization"""
    
    def __init__(self):
        # Average characters per token based on GPT models
        # GPT models average ~4 characters per token for English text
        self.avg_chars_per_token = 4.0
        
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count using a combination of methods:
        1. Word-based counting with adjustments
        2. Character-based estimation as fallback
        """
        if not text:
            return 0
            
        # Method 1: Word-based with adjustments
        # Split on whitespace and punctuation
        words = re.findall(r'\b\w+\b|[^\w\s]', text)
        
        # Count tokens with adjustments
        token_count = 0
        for word in words:
            if len(word) == 0:
                continue
            elif len(word) <= 3:
                # Short words are usually 1 token
                token_count += 1
            elif len(word) <= 7:
                # Medium words are usually 1-2 tokens
                token_count += 1.3
            else:
                # Longer words need more tokens
                # Estimate based on character count
                token_count += len(word) / 4.5
        
        # Method 2: Character-based validation
        char_estimate = len(text) / self.avg_chars_per_token
        
        # Use weighted average favoring word-based count
        final_estimate = (token_count * 0.7 + char_estimate * 0.3)
        
        return int(final_estimate)
    
    def split_text_at_position(self, text: str, position: int) -> Tuple[str, str]:
        """Split text at a specific character position"""
        return text[:position], text[position:]

class Stage3Chunker:
    """Main chunking processor for Stage 3"""
    
    def __init__(self, min_tokens: int = 500, max_tokens: int = 750, hard_max: int = 800):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.hard_max = hard_max
        self.tokenizer = SimpleTokenizer()
        
    def process_sections(self, sections: List[Dict]) -> List[Dict]:
        """Process all sections and return chunked records"""
        all_chunks = []
        
        # Group sections by document and chapter for proper sequencing
        sections_by_chapter = {}
        for section in sections:
            key = (section['document_id'], section['chapter_number'])
            if key not in sections_by_chapter:
                sections_by_chapter[key] = []
            sections_by_chapter[key].append(section)
        
        # Process each chapter's sections in order
        for (doc_id, chapter_num), chapter_sections in sorted(sections_by_chapter.items()):
            # Sort sections by section_number
            chapter_sections.sort(key=lambda x: x['section_number'])
            
            for section in chapter_sections:
                chunks = self.chunk_section(section)
                all_chunks.extend(chunks)
                
        logger.info(f"Created {len(all_chunks)} chunks from {len(sections)} sections")
        return all_chunks
    
    def chunk_section(self, section: Dict) -> List[Dict]:
        """Chunk a single section into appropriately sized pieces"""
        content = section.get('section_content', '')
        if not content:
            logger.warning(f"Empty content for section {section.get('section_number', 'unknown')}")
            return []
        
        # Check if section is small enough to be a single chunk
        total_tokens = self.tokenizer.count_tokens(content)
        if total_tokens <= self.max_tokens:
            return [self.create_chunk_record(section, content, 1)]
        
        # Find protected zones (HTML tags)
        protected_zones = self.find_protected_zones(content)
        
        # Find all potential break points
        break_points = self.find_break_points(content, protected_zones)
        
        # Create chunks using break points
        chunks = self.create_chunks_with_breaks(section, content, break_points, protected_zones)
        
        return chunks
    
    def find_protected_zones(self, content: str) -> List[ProtectedZone]:
        """Find all HTML comment tags that cannot be split"""
        zones = []
        
        # Pattern for HTML comments (page tags)
        pattern = r'<!--\s*Page(?:Header|Footer)[^>]*?-->'
        
        for match in re.finditer(pattern, content):
            zones.append(ProtectedZone(
                start=match.start(),
                end=match.end(),
                content=match.group()
            ))
        
        return zones
    
    def is_in_protected_zone(self, position: int, zones: List[ProtectedZone]) -> bool:
        """Check if a position is within any protected zone"""
        for zone in zones:
            if zone.start <= position < zone.end:
                return True
        return False
    
    def find_break_points(self, content: str, protected_zones: List[ProtectedZone]) -> List[BreakPoint]:
        """Find all potential break points in the content"""
        break_points = []
        
        # Priority 1: Major headings (##, ###)
        for match in re.finditer(r'\n(#{2,3})\s+[^\n]+', content):
            pos = match.start()
            if not self.is_in_protected_zone(pos, protected_zones):
                break_points.append(BreakPoint(pos, 'heading', 1))
        
        # Priority 2: Paragraph breaks (double newline)
        for match in re.finditer(r'\n\n+', content):
            pos = match.start()
            if not self.is_in_protected_zone(pos, protected_zones):
                # Don't add if too close to a heading
                if not any(abs(bp.position - pos) < 10 for bp in break_points if bp.type == 'heading'):
                    break_points.append(BreakPoint(pos, 'paragraph', 2))
        
        # Priority 3: List boundaries
        # Before bullet points or numbered lists
        for match in re.finditer(r'\n(?=[-*•]\s|\d+\.\s)', content):
            pos = match.start()
            if not self.is_in_protected_zone(pos, protected_zones):
                break_points.append(BreakPoint(pos, 'list', 3))
        
        # Priority 4: Sentence ends
        for match in re.finditer(r'[.!?]\s+(?=[A-Z])', content):
            pos = match.end() - 1  # Position after the space
            if not self.is_in_protected_zone(pos, protected_zones):
                break_points.append(BreakPoint(pos, 'sentence', 4))
        
        # Priority 5: Comma or semicolon (last resort)
        for match in re.finditer(r'[,;]\s+', content):
            pos = match.end() - 1
            if not self.is_in_protected_zone(pos, protected_zones):
                break_points.append(BreakPoint(pos, 'comma', 5))
        
        # Sort by position
        break_points.sort(key=lambda x: x.position)
        
        return break_points
    
    def find_best_break(self, content: str, start_pos: int, target_end: int, 
                       break_points: List[BreakPoint], protected_zones: List[ProtectedZone]) -> int:
        """Find the best break point within a target range"""
        # Define search window
        min_end = start_pos + int(self.min_tokens * self.tokenizer.avg_chars_per_token)
        max_end = min(start_pos + int(self.hard_max * self.tokenizer.avg_chars_per_token), len(content))
        ideal_end = start_pos + int(target_end * self.tokenizer.avg_chars_per_token)
        
        # Find break points in range
        candidates = [bp for bp in break_points 
                     if min_end <= bp.position <= max_end and bp.position > start_pos]
        
        if not candidates:
            # No break points found, try to find next protected zone boundary
            for zone in protected_zones:
                if min_end <= zone.start <= max_end and zone.start > start_pos:
                    return zone.start
            # Last resort: split at max position
            return min(max_end, len(content))
        
        # Sort by priority then distance from ideal
        candidates.sort(key=lambda bp: (bp.priority, abs(bp.position - ideal_end)))
        
        return candidates[0].position
    
    def create_chunks_with_breaks(self, section: Dict, content: str, 
                                 break_points: List[BreakPoint], 
                                 protected_zones: List[ProtectedZone]) -> List[Dict]:
        """Create chunks using the identified break points"""
        chunks = []
        current_pos = 0
        chunk_number = 1
        
        while current_pos < len(content):
            # Estimate where this chunk should end
            remaining_content = content[current_pos:]
            remaining_tokens = self.tokenizer.count_tokens(remaining_content)
            
            if remaining_tokens <= self.max_tokens:
                # Last chunk
                chunk_content = remaining_content
                chunks.append(self.create_chunk_record(section, chunk_content, chunk_number))
                break
            
            # Find best break point
            target_tokens = (self.min_tokens + self.max_tokens) // 2
            break_pos = self.find_best_break(content, current_pos, target_tokens, 
                                            break_points, protected_zones)
            
            # Extract chunk
            chunk_content = content[current_pos:break_pos]
            
            # Validate chunk size
            chunk_tokens = self.tokenizer.count_tokens(chunk_content)
            if chunk_tokens > self.hard_max:
                logger.warning(f"Chunk exceeds hard max: {chunk_tokens} tokens")
            
            chunks.append(self.create_chunk_record(section, chunk_content, chunk_number))
            
            current_pos = break_pos
            chunk_number += 1
        
        return chunks
    
    def create_chunk_record(self, section: Dict, chunk_content: str, chunk_number: int) -> Dict:
        """Create a chunk record with all required fields"""
        
        chunk = {
            # Document fields
            'document_id': section.get('document_id'),
            'filename': section.get('filename'),
            'filepath': section.get('filepath'),
            'source_filename': section.get('source_filename'),
            
            # Chapter fields
            'chapter_number': section.get('chapter_number'),
            'chapter_name': section.get('chapter_name'),
            'chapter_summary': section.get('chapter_summary'),
            'chapter_page_count': section.get('chapter_page_count'),
            
            # Section fields
            'section_number': section.get('section_number'),
            'section_summary': section.get('section_summary'),
            'section_start_page': section.get('section_start_page'),
            'section_end_page': section.get('section_end_page'),
            'section_page_count': section.get('section_page_count'),
            'section_start_reference': section.get('section_start_reference'),
            'section_end_reference': section.get('section_end_reference'),
            
            # Chunk fields
            'chunk_number': chunk_number,
            'chunk_content': chunk_content
        }
        
        return chunk
    
    def extract_page_range(self, content: str) -> Tuple[Optional[int], Optional[int]]:
        """Extract the page range from HTML tags in the content"""
        page_numbers = []
        
        # Find all page numbers in the content
        pattern = r'<!--\s*Page(?:Header|Footer)\s+PageNumber="(\d+)"[^>]*?-->'
        for match in re.finditer(pattern, content):
            page_num = int(match.group(1))
            page_numbers.append(page_num)
        
        if page_numbers:
            return min(page_numbers), max(page_numbers)
        return None, None
    
    def get_page_list(self, start_page: Optional[int], end_page: Optional[int]) -> List[int]:
        """Generate list of pages spanned by chunk"""
        if start_page is None or end_page is None:
            return []
        return list(range(start_page, end_page + 1))

def main():
    """Main entry point for Stage 3 processing"""
    parser = argparse.ArgumentParser(description='Stage 3: Chunk sections into semantic search units')
    parser.add_argument('-i', '--input', 
                       default='semantic search/pipeline_output/stage2_5/stage2_5_corrected_sections.json',
                       help='Input file path (stage 2.5 output)')
    parser.add_argument('-o', '--output',
                       default='semantic search/pipeline_output/stage3/stage3_chunks.json',
                       help='Output file path for chunks')
    parser.add_argument('--min-tokens', type=int, default=500,
                       help='Minimum tokens per chunk')
    parser.add_argument('--max-tokens', type=int, default=750,
                       help='Maximum tokens per chunk (target)')
    parser.add_argument('--hard-max', type=int, default=800,
                       help='Hard maximum tokens (will not exceed)')
    
    args = parser.parse_args()
    
    # Load input data
    logger.info(f"Loading sections from {args.input}")
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            sections = json.load(f)
        logger.info(f"Loaded {len(sections)} sections")
    except Exception as e:
        logger.error(f"Failed to load input file: {e}")
        return 1
    
    # Initialize chunker
    chunker = Stage3Chunker(
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        hard_max=args.hard_max
    )
    
    # Process sections into chunks
    logger.info("Starting chunking process...")
    chunks = chunker.process_sections(sections)
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save chunks
    logger.info(f"Saving {len(chunks)} chunks to {args.output}")
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        logger.info("Chunking complete!")
    except Exception as e:
        logger.error(f"Failed to save output: {e}")
        return 1
    
    # Print statistics
    print("\n=== Chunking Statistics ===")
    print(f"Total sections processed: {len(sections)}")
    print(f"Total chunks created: {len(chunks)}")
    if chunks:
        # Count chunks per section
        chunks_per_section = {}
        for chunk in chunks:
            key = (chunk['chapter_number'], chunk['section_number'])
            chunks_per_section[key] = chunks_per_section.get(key, 0) + 1
        
        avg_chunks = sum(chunks_per_section.values()) / len(chunks_per_section)
        print(f"Average chunks per section: {avg_chunks:.1f}")
        print(f"Max chunks in a section: {max(chunks_per_section.values())}")
        print(f"Sections with 1 chunk: {sum(1 for v in chunks_per_section.values() if v == 1)}")
    
    return 0

if __name__ == "__main__":
    exit(main())