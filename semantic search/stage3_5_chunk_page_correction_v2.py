#!/usr/bin/env python3
"""
Stage 3.5 v2: Chunk Page Boundary Correction (Local Version with Chapter-Level Processing)
Corrects page boundaries for chunks using chapter-wide page tag extraction.

Key Improvements:
- Processes all chunks in a chapter together (like Stage 2.5)
- Enables cross-section inference for sections without tags
- Uses neighboring sections' chunk data for better page inference
- Reduces "sections without tags" to near zero

Input: JSON file from Stage 3 output (stage3_chunks.json)
Output: JSON file with corrected chunk page boundaries (stage3_5_corrected_chunks.json)
"""

import json
import re
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Page Tag Extraction and Position Mapping (Same as original)
# ==============================================================================

def extract_page_tags_with_positions(content: str) -> List[Tuple[int, str, int, str]]:
    """
    Extract all page tags from content with their positions.
    
    Returns:
        List of tuples: (position, tag_type, page_number, page_reference)
        tag_type is either 'header' or 'footer'
    """
    if not content:
        return []
    
    tags = []
    
    # Pattern for page headers and footers
    header_pattern = re.compile(
        r'<!-- PageHeader PageNumber="(\d+)" PageReference="([^"]*)" -->',
        re.IGNORECASE
    )
    footer_pattern = re.compile(
        r'<!-- PageFooter PageNumber="(\d+)" PageReference="([^"]*)" -->',
        re.IGNORECASE
    )
    
    # Find all headers
    for match in header_pattern.finditer(content):
        position = match.start()
        page_num = int(match.group(1))
        page_ref = match.group(2)
        tags.append((position, 'header', page_num, page_ref))
    
    # Find all footers
    for match in footer_pattern.finditer(content):
        position = match.start()
        page_num = int(match.group(1))
        page_ref = match.group(2)
        tags.append((position, 'footer', page_num, page_ref))
    
    # Sort by position
    tags.sort(key=lambda x: x[0])
    
    return tags


def build_page_ranges(page_tags: List[Tuple[int, str, int, str]], content: str) -> Tuple[List[Tuple[int, int, int]], Dict[int, str]]:
    """
    Build page ranges from extracted tags.
    
    Args:
        page_tags: List of extracted page tags with positions
        content: The full content string
    
    Returns:
        Tuple of:
        - List of tuples: (start_position, end_position, page_number)
        - Dict mapping page_number to page_reference
    """
    if not page_tags:
        return [], {}
    
    ranges = []
    page_references = {}  # Map page number to reference
    
    # Process tags in order to build page ranges
    current_page = None
    page_start = 0
    
    for i, (pos, tag_type, page_num, page_ref) in enumerate(page_tags):
        # Store page reference for this page number
        if page_num not in page_references or tag_type == 'header':
            # Prefer header references, but use footer if no header exists
            page_references[page_num] = page_ref
        
        if tag_type == 'header':
            # If we were tracking a previous page without a footer, end it here
            if current_page is not None and current_page != page_num:
                ranges.append((page_start, pos - 1, current_page))
                current_page = page_num
                page_start = pos
            elif current_page != page_num:
                # Start tracking this page (no previous page being tracked)
                current_page = page_num
                page_start = pos
            # else: duplicate header for same page - ignore it
            
        elif tag_type == 'footer':
            # This should be the end of the current page
            if current_page == page_num:
                # Find the actual end of the footer tag
                footer_end_pos = content.find('-->', pos)
                if footer_end_pos != -1:
                    page_end = footer_end_pos + 2
                else:
                    # Fallback
                    page_end = pos + 54  # Typical footer length
                
                ranges.append((page_start, page_end, current_page))
                current_page = None
                
            elif current_page is None:
                # Footer without header
                if ranges:
                    page_start = ranges[-1][1] + 1
                else:
                    page_start = 0
                
                footer_end_pos = content.find('-->', pos)
                if footer_end_pos != -1:
                    page_end = footer_end_pos + 2
                else:
                    page_end = pos + 54
                
                ranges.append((page_start, page_end, page_num))
    
    # If we ended while tracking a page (header without footer), close it
    if current_page is not None:
        ranges.append((page_start, len(content) - 1, current_page))
    
    # Sort ranges by start position
    ranges.sort(key=lambda x: x[0])
    
    return ranges, page_references


def determine_pages_for_position_range(
    start_pos: int, 
    end_pos: int, 
    page_ranges: List[Tuple[int, int, int]]
) -> Set[int]:
    """
    Determine which pages a position range overlaps with.
    
    Args:
        start_pos: Start position in content
        end_pos: End position in content
        page_ranges: List of (start, end, page_number) tuples
    
    Returns:
        Set of page numbers that the range overlaps with
    """
    pages = set()
    
    for range_start, range_end, page_num in page_ranges:
        # Check if there's any overlap
        if start_pos <= range_end and end_pos >= range_start:
            pages.add(page_num)
    
    return pages

# ==============================================================================
# NEW: Chapter-Level Chunk Processing
# ==============================================================================

def build_chapter_wide_position_map(chapter_chunks: List[Dict]) -> Tuple[str, Dict[Tuple[int, int], Tuple[int, int]]]:
    """
    Build a position map for all chunks across the entire chapter.
    
    Returns:
        Tuple of (concatenated_content, chunk_positions)
        chunk_positions maps (section_number, chunk_number) to (start_pos, end_pos)
    """
    # Sort chunks by section and chunk number
    chapter_chunks.sort(key=lambda x: (x.get('section_number', 0), x.get('chunk_number', 0)))
    
    full_content_parts = []
    chunk_positions = {}
    current_pos = 0
    
    for chunk in chapter_chunks:
        section_num = chunk.get('section_number', 0)
        chunk_num = chunk.get('chunk_number', 0)
        content = chunk.get('chunk_content', '')
        
        if not content:
            logger.warning(f"Section {section_num}, Chunk {chunk_num}: Empty content")
            content = ""
        
        start_pos = current_pos
        end_pos = current_pos + len(content) - 1
        
        chunk_positions[(section_num, chunk_num)] = (start_pos, end_pos)
        full_content_parts.append(content)
        
        current_pos = current_pos + len(content)
    
    full_content = ''.join(full_content_parts)
    return full_content, chunk_positions


def process_chapter_chunks(chapter_chunks: List[Dict], debug_mode: bool = False) -> Tuple[List[Dict], Dict]:
    """
    Process all chunks in a chapter to correct page boundaries.
    Uses chapter-wide page tag extraction for better inference.
    
    Returns:
        Tuple of (corrected_chunks, statistics)
    """
    if not chapter_chunks:
        return chapter_chunks, {'total': 0, 'corrected': 0, 'inferred': 0}
    
    stats = {
        'total': len(chapter_chunks),
        'corrected': 0,
        'inferred': 0,
        'tag_found': 0,
        'position_inferred': 0,
        'continuity_fixed': 0,
        'sections_without_tags': 0,
        'sections_with_inference': 0
    }
    
    # Sort chunks by section and chunk number
    chapter_chunks.sort(key=lambda x: (x.get('section_number', 0), x.get('chunk_number', 0)))
    
    # Step 1: Build chapter-wide position map and concatenate all content
    full_content, chunk_positions = build_chapter_wide_position_map(chapter_chunks)
    
    # Step 2: Extract all page tags from entire chapter content
    page_tags = extract_page_tags_with_positions(full_content)
    
    if not page_tags:
        logger.warning(f"Chapter {chapter_chunks[0].get('chapter_number', '?')}: No page tags found in any chunk content")
        # Fall back to section boundaries
        for chunk in chapter_chunks:
            section_start = chunk.get('section_start_page')
            section_end = chunk.get('section_end_page')
            if section_start and section_end:
                chunk['chunk_start_page'] = section_start
                chunk['chunk_end_page'] = section_end
                chunk['chunk_start_reference'] = chunk.get('section_start_reference', '')
                chunk['chunk_end_reference'] = chunk.get('section_end_reference', '')
                stats['inferred'] += 1
        return chapter_chunks, stats
    
    # Step 3: Build page ranges from all tags
    page_ranges, page_references = build_page_ranges(page_tags, full_content)
    
    if not page_ranges:
        logger.warning(f"Chapter {chapter_chunks[0].get('chapter_number', '?')}: Could not build page ranges")
        return chapter_chunks, stats
    
    # Step 4: Map each chunk to pages using position overlaps
    for chunk in chapter_chunks:
        section_num = chunk.get('section_number', 0)
        chunk_num = chunk.get('chunk_number', 0)
        
        # Store original values
        original_start = chunk.get('chunk_start_page')
        original_end = chunk.get('chunk_end_page')
        
        # Get chunk position in concatenated content
        if (section_num, chunk_num) not in chunk_positions:
            logger.warning(f"Section {section_num}, Chunk {chunk_num}: Not found in position map")
            continue
        
        start_pos, end_pos = chunk_positions[(section_num, chunk_num)]
        
        # Determine pages for this position range
        pages = determine_pages_for_position_range(start_pos, end_pos, page_ranges)
        
        if pages:
            # Chunk overlaps with identified pages
            new_start = min(pages)
            new_end = max(pages)
            
            chunk['chunk_start_page'] = new_start
            chunk['chunk_end_page'] = new_end
            
            # Add page references
            chunk['chunk_start_reference'] = page_references.get(new_start, '')
            chunk['chunk_end_reference'] = page_references.get(new_end, '')
            
            # Add debug fields if enabled
            if debug_mode:
                chunk['original_start_page'] = original_start
                chunk['original_end_page'] = original_end
                chunk['page_boundary_method'] = 'tag_found'
                chunk['page_correction_applied'] = (original_start != new_start or original_end != new_end)
            
            stats['tag_found'] += 1
            
            # Check if correction was needed
            if original_start != new_start or original_end != new_end:
                stats['corrected'] += 1
                logger.debug(f"Section {section_num}, Chunk {chunk_num}: Corrected from pages {original_start}-{original_end} to {new_start}-{new_end}")
        else:
            # Mark for inference
            chunk['needs_inference'] = True
            stats['position_inferred'] += 1
    
    # Step 5: Group chunks by section for inference
    chunks_by_section = defaultdict(list)
    for chunk in chapter_chunks:
        section_num = chunk.get('section_number', 0)
        chunks_by_section[section_num].append(chunk)
    
    # Track sections that need inference
    sections_needing_inference = set()
    
    # Step 6: Handle sections where no chunks have pages assigned
    for section_num, section_chunks in chunks_by_section.items():
        # Check if any chunk in this section has pages assigned
        has_pages = any(c.get('chunk_start_page') is not None for c in section_chunks)
        
        if not has_pages:
            sections_needing_inference.add(section_num)
            stats['sections_without_tags'] += 1
    
    # Step 7: Infer pages for sections without tags using neighboring sections
    sorted_sections = sorted(chunks_by_section.keys())
    
    for i, section_num in enumerate(sorted_sections):
        if section_num not in sections_needing_inference:
            continue
        
        section_chunks = chunks_by_section[section_num]
        inferred = False
        
        # Get section boundaries from first chunk
        section_start_page = section_chunks[0].get('section_start_page')
        section_end_page = section_chunks[0].get('section_end_page')
        
        # Try to infer from neighboring sections' chunks
        prev_section_last_page = None
        next_section_first_page = None
        
        # Look at previous section's last chunk
        if i > 0:
            prev_section_num = sorted_sections[i - 1]
            if prev_section_num not in sections_needing_inference:
                prev_chunks = chunks_by_section[prev_section_num]
                if prev_chunks:
                    last_chunk = prev_chunks[-1]
                    if last_chunk.get('chunk_end_page') is not None:
                        prev_section_last_page = last_chunk['chunk_end_page']
        
        # Look at next section's first chunk
        if i < len(sorted_sections) - 1:
            next_section_num = sorted_sections[i + 1]
            if next_section_num not in sections_needing_inference:
                next_chunks = chunks_by_section[next_section_num]
                if next_chunks:
                    first_chunk = next_chunks[0]
                    if first_chunk.get('chunk_start_page') is not None:
                        next_section_first_page = first_chunk['chunk_start_page']
        
        # Determine section's actual pages based on neighbors
        if prev_section_last_page is not None and next_section_first_page is not None:
            # Section is between two known sections
            inferred_start = prev_section_last_page + 1
            inferred_end = next_section_first_page - 1
            
            if inferred_end < inferred_start:
                # Sections are adjacent or overlapping
                inferred_start = prev_section_last_page
                inferred_end = prev_section_last_page
            
            inferred = True
        elif prev_section_last_page is not None:
            # Only previous section known
            inferred_start = prev_section_last_page + 1
            inferred_end = section_end_page if section_end_page else inferred_start
            inferred = True
        elif next_section_first_page is not None:
            # Only next section known
            inferred_end = next_section_first_page - 1
            inferred_start = section_start_page if section_start_page else inferred_end
            inferred = True
        elif section_start_page and section_end_page:
            # Use section boundaries as fallback
            inferred_start = section_start_page
            inferred_end = section_end_page
            inferred = True
        
        if inferred:
            # Distribute pages among section's chunks
            total_chunks = len(section_chunks)
            pages_available = inferred_end - inferred_start + 1
            
            for j, chunk in enumerate(section_chunks):
                if pages_available == 1 or total_chunks == 1:
                    # All chunks on same page
                    chunk['chunk_start_page'] = inferred_start
                    chunk['chunk_end_page'] = inferred_end
                else:
                    # Proportional distribution
                    chunk_start = inferred_start + (j * pages_available // total_chunks)
                    chunk_end = inferred_start + ((j + 1) * pages_available // total_chunks) - 1
                    chunk_end = min(chunk_end, inferred_end)
                    
                    chunk['chunk_start_page'] = chunk_start
                    chunk['chunk_end_page'] = chunk_end
                
                # Add page references if available
                chunk['chunk_start_reference'] = page_references.get(chunk['chunk_start_page'], '')
                chunk['chunk_end_reference'] = page_references.get(chunk['chunk_end_page'], '')
                
                # Remove inference marker
                chunk.pop('needs_inference', None)
                
                # Add debug fields if enabled
                if debug_mode:
                    chunk['original_start_page'] = None
                    chunk['original_end_page'] = None
                    chunk['page_boundary_method'] = 'cross_section_inference'
                    chunk['page_correction_applied'] = True
                
                stats['inferred'] += 1
                stats['corrected'] += 1
            
            stats['sections_with_inference'] += 1
            logger.info(f"Section {section_num}: Inferred pages {inferred_start}-{inferred_end} using neighboring sections")
    
    # Step 8: Handle remaining chunks that need inference (within sections that have some tags)
    for chunk in chapter_chunks:
        if chunk.get('needs_inference') or chunk.get('chunk_start_page') is None:
            section_num = chunk.get('section_number', 0)
            chunk_num = chunk.get('chunk_number', 0)
            section_chunks = chunks_by_section[section_num]
            
            # Find position in section
            chunk_index = next((i for i, c in enumerate(section_chunks) if c['chunk_number'] == chunk_num), None)
            
            if chunk_index is not None:
                # Try to infer from neighboring chunks within the same section
                inferred = False
                
                if chunk_index > 0:
                    prev_chunk = section_chunks[chunk_index - 1]
                    if prev_chunk.get('chunk_end_page') is not None:
                        if chunk_index < len(section_chunks) - 1:
                            next_chunk = section_chunks[chunk_index + 1]
                            if next_chunk.get('chunk_start_page') is not None:
                                # Between two known chunks
                                chunk['chunk_start_page'] = prev_chunk['chunk_end_page']
                                chunk['chunk_end_page'] = next_chunk['chunk_start_page']
                                inferred = True
                        else:
                            # Last chunk in section
                            chunk['chunk_start_page'] = prev_chunk['chunk_end_page']
                            chunk['chunk_end_page'] = chunk.get('section_end_page', prev_chunk['chunk_end_page'])
                            inferred = True
                
                if not inferred and chunk_index == 0:
                    # First chunk in section
                    section_start = chunk.get('section_start_page')
                    if section_start:
                        chunk['chunk_start_page'] = section_start
                        if chunk_index < len(section_chunks) - 1:
                            next_chunk = section_chunks[chunk_index + 1]
                            if next_chunk.get('chunk_start_page'):
                                chunk['chunk_end_page'] = next_chunk['chunk_start_page']
                            else:
                                chunk['chunk_end_page'] = section_start
                        else:
                            chunk['chunk_end_page'] = chunk.get('section_end_page', section_start)
                        inferred = True
                
                if inferred:
                    # Add page references
                    chunk['chunk_start_reference'] = page_references.get(chunk['chunk_start_page'], '')
                    chunk['chunk_end_reference'] = page_references.get(chunk['chunk_end_page'], '')
                    
                    # Remove inference marker
                    chunk.pop('needs_inference', None)
                    
                    if debug_mode:
                        chunk['page_boundary_method'] = 'within_section_inference'
                    
                    stats['inferred'] += 1
                    stats['corrected'] += 1
    
    # Step 9: Validate and fix continuity
    for section_num, section_chunks in chunks_by_section.items():
        for i in range(len(section_chunks) - 1):
            current = section_chunks[i]
            next_chunk = section_chunks[i + 1]
            
            current_end = current.get('chunk_end_page')
            next_start = next_chunk.get('chunk_start_page')
            
            if current_end and next_start:
                if next_start < current_end:
                    # Overlap detected - adjust
                    logger.warning(f"Section {section_num}: Overlap between chunks {current['chunk_number']} and {next_chunk['chunk_number']}")
                    current['chunk_end_page'] = next_start
                    stats['continuity_fixed'] += 1
                    
                    if debug_mode:
                        current['page_boundary_method'] = 'continuity_fixed'
    
    # Clean up temporary markers
    for chunk in chapter_chunks:
        chunk.pop('needs_inference', None)
    
    return chapter_chunks, stats

# ==============================================================================
# Main Processing Function
# ==============================================================================

def main():
    """Main processing function for Stage 3.5 v2 (local version)"""
    
    parser = argparse.ArgumentParser(description='Stage 3.5 v2: Correct chunk page boundaries with chapter-level processing')
    parser.add_argument('-i', '--input', 
                       default='stage3_chunks.json',
                       help='Input file path (stage 3 output)')
    parser.add_argument('-o', '--output',
                       default='stage3_5_corrected_chunks.json',
                       help='Output file path for corrected chunks')
    parser.add_argument('--debug', action='store_true',
                       help='Include debug fields in output')
    parser.add_argument('--sample-limit', type=int,
                       help='Limit processing to first N chapters')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load input data
    print(f"\n📥 Loading chunks from: {args.input}")
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"✅ Loaded {len(chunks)} chunks")
    except Exception as e:
        logger.error(f"Failed to load input file: {e}")
        return 1
    
    # Group chunks by chapter
    chunks_by_chapter = defaultdict(list)
    for chunk in chunks:
        chapter_num = chunk.get('chapter_number')
        if chapter_num is not None:
            chunks_by_chapter[chapter_num].append(chunk)
    
    print(f"📊 Found {len(chunks_by_chapter)} chapters")
    
    # Count sections per chapter
    total_sections = 0
    for chapter_chunks in chunks_by_chapter.values():
        sections_in_chapter = len(set(c.get('section_number') for c in chapter_chunks))
        total_sections += sections_in_chapter
    
    print(f"📊 Total sections across all chapters: {total_sections}")
    
    # Apply sample limit if configured
    if args.sample_limit:
        limited_chapters = sorted(chunks_by_chapter.keys())[:args.sample_limit]
        chunks_by_chapter = {k: chunks_by_chapter[k] for k in limited_chapters}
        print(f"Limited to first {args.sample_limit} chapters")
    
    # Process each chapter's chunks
    all_corrected_chunks = []
    total_stats = defaultdict(int)
    
    chapter_count = 0
    total_chapters = len(chunks_by_chapter)
    
    for chapter_num in sorted(chunks_by_chapter.keys()):
        chapter_chunks = chunks_by_chapter[chapter_num]
        chapter_count += 1
        
        # Count sections in this chapter
        sections_in_chapter = len(set(c.get('section_number') for c in chapter_chunks))
        
        print(f"\n[{chapter_count}/{total_chapters}] Processing Chapter {chapter_num} ({sections_in_chapter} sections, {len(chapter_chunks)} chunks)...")
        
        # Process this chapter's chunks
        corrected_chunks, stats = process_chapter_chunks(chapter_chunks, debug_mode=args.debug)
        all_corrected_chunks.extend(corrected_chunks)
        
        # Aggregate statistics
        for key, value in stats.items():
            total_stats[key] += value
        
        if args.verbose and stats['sections_without_tags'] > 0:
            print(f"  - Sections without tags: {stats['sections_without_tags']}")
            print(f"  - Sections with inference: {stats['sections_with_inference']}")
    
    print(f"\n✅ Processed all {total_chapters} chapters")
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save corrected chunks
    print(f"\n💾 Saving {len(all_corrected_chunks)} corrected chunks to: {args.output}")
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_corrected_chunks, f, indent=2, ensure_ascii=False)
        print(f"✅ Corrected chunks saved successfully")
    except Exception as e:
        logger.error(f"Failed to save output: {e}")
        return 1
    
    # Print statistics
    print("\n" + "=" * 60)
    print("📊 Processing Statistics")
    print("=" * 60)
    print(f"Total chunks processed: {total_stats['total']}")
    print(f"Chunks with page tags found: {total_stats['tag_found']}")
    print(f"Chunks with inferred pages: {total_stats['inferred']}")
    print(f"Chunks corrected: {total_stats['corrected']}")
    print(f"Continuity fixes applied: {total_stats['continuity_fixed']}")
    print(f"\nSections without any tags: {total_stats['sections_without_tags']}")
    print(f"Sections resolved via inference: {total_stats['sections_with_inference']}")
    
    # Calculate correction percentage
    if total_stats['total'] > 0:
        correction_pct = (total_stats['corrected'] / total_stats['total']) * 100
        print(f"\n📈 Correction rate: {correction_pct:.1f}%")
    
    # Calculate inference success rate
    if total_stats['sections_without_tags'] > 0:
        inference_rate = (total_stats['sections_with_inference'] / total_stats['sections_without_tags']) * 100
        print(f"📈 Section inference success rate: {inference_rate:.1f}%")
    
    print("\n" + "=" * 60)
    print("✅ Stage 3.5 v2 Processing Complete!")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit(main())