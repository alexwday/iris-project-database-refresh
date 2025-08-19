#!/usr/bin/env python3
"""
Stage 3.5: Chunk Page Boundary Correction (Local Version)
Corrects page boundaries for chunks after Stage 3 processing using embedded page tags.

This is the local version for development and testing.
For production use with NAS, use stage3_5_chunk_page_correction_nas.py

Key Features:
- Position-based page mapping algorithm (same as Stage 2.5)
- Handles chunks with no tags, mid-page boundaries, missing headers/footers
- Validates and fixes continuity issues within sections
- Adds chunk_start_page, chunk_end_page, chunk_start_reference, chunk_end_reference

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
# Page Tag Extraction and Position Mapping
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
                    page_end = footer_end_pos + 2  # +3 for '-->' -1 for inclusive
                else:
                    # Fallback to regex with larger window
                    footer_match = re.search(r'<!-- PageFooter[^>]*?-->', content[pos:pos+200])
                    if footer_match:
                        page_end = pos + footer_match.end() - 1
                    else:
                        page_end = pos + 54  # Typical footer length
                
                ranges.append((page_start, page_end, current_page))
                current_page = None
                
            elif current_page is None:
                # Footer without header - page starts from beginning or after previous
                if ranges:
                    page_start = ranges[-1][1] + 1
                else:
                    page_start = 0
                
                # Find the actual end of the footer tag
                footer_end_pos = content.find('-->', pos)
                if footer_end_pos != -1:
                    page_end = footer_end_pos + 2
                else:
                    footer_match = re.search(r'<!-- PageFooter[^>]*?-->', content[pos:pos+200])
                    if footer_match:
                        page_end = pos + footer_match.end() - 1
                    else:
                        page_end = pos + 54
                
                ranges.append((page_start, page_end, page_num))
    
    # If we ended while tracking a page (header without footer), close it
    if current_page is not None:
        ranges.append((page_start, len(content) - 1, current_page))
    
    # Sort ranges by start position
    ranges.sort(key=lambda x: x[0])
    
    return ranges, page_references


def build_position_map(chunks: List[Dict]) -> Tuple[str, Dict[int, Tuple[int, int]]]:
    """
    Build a position map for all chunks in concatenated content.
    
    Returns:
        Tuple of (concatenated_content, chunk_positions)
        chunk_positions maps chunk_number to (start_pos, end_pos)
    """
    full_content_parts = []
    chunk_positions = {}
    current_pos = 0
    
    for chunk in chunks:
        chunk_num = chunk.get('chunk_number', 0)
        content = chunk.get('chunk_content', '')
        
        if not content:
            logger.warning(f"Chunk {chunk_num}: Empty content")
            content = ""
        
        start_pos = current_pos
        end_pos = current_pos + len(content) - 1  # -1 for inclusive end position
        
        chunk_positions[chunk_num] = (start_pos, end_pos)
        full_content_parts.append(content)
        
        current_pos = current_pos + len(content)  # Next chunk starts after this content
    
    full_content = ''.join(full_content_parts)
    return full_content, chunk_positions


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
# Chunk Page Boundary Correction
# ==============================================================================

def process_section_chunks(section_chunks: List[Dict], debug_mode: bool = False) -> Tuple[List[Dict], Dict]:
    """
    Process all chunks in a section to correct page boundaries.
    
    Returns:
        Tuple of (corrected_chunks, statistics)
    """
    if not section_chunks:
        return section_chunks, {'total': 0, 'corrected': 0, 'inferred': 0}
    
    stats = {
        'total': len(section_chunks),
        'corrected': 0,
        'inferred': 0,
        'tag_found': 0,
        'position_inferred': 0,
        'continuity_fixed': 0,
        'no_tags_found': 0
    }
    
    # Sort chunks by chunk_number to ensure correct order
    section_chunks.sort(key=lambda x: x.get('chunk_number', 0))
    
    # Get section boundaries for validation
    first_chunk = section_chunks[0]
    section_start_page = first_chunk.get('section_start_page')
    section_end_page = first_chunk.get('section_end_page')
    
    # Step 1: Build position map and concatenate content
    full_content, chunk_positions = build_position_map(section_chunks)
    
    # Step 2: Extract all page tags with positions
    page_tags = extract_page_tags_with_positions(full_content)
    
    if not page_tags:
        logger.warning(f"Section {first_chunk.get('section_number', '?')}: No page tags found in chunk content")
        stats['no_tags_found'] = len(section_chunks)
        
        # Fall back to section boundaries
        if section_start_page and section_end_page:
            # Distribute pages proportionally
            total_chunks = len(section_chunks)
            pages_available = section_end_page - section_start_page + 1
            
            for i, chunk in enumerate(section_chunks):
                if pages_available == 1 or total_chunks == 1:
                    # All chunks on same page
                    chunk['chunk_start_page'] = section_start_page
                    chunk['chunk_end_page'] = section_end_page
                else:
                    # Proportional distribution
                    chunk_start = section_start_page + (i * pages_available // total_chunks)
                    chunk_end = section_start_page + ((i + 1) * pages_available // total_chunks) - 1
                    chunk_end = min(chunk_end, section_end_page)
                    
                    chunk['chunk_start_page'] = chunk_start
                    chunk['chunk_end_page'] = chunk_end
                
                chunk['chunk_start_reference'] = ''
                chunk['chunk_end_reference'] = ''
                stats['inferred'] += 1
        
        return section_chunks, stats
    
    # Step 3: Build page ranges from tags
    page_ranges, page_references = build_page_ranges(page_tags, full_content)
    
    if not page_ranges:
        logger.warning(f"Section {first_chunk.get('section_number', '?')}: Could not build page ranges")
        return section_chunks, stats
    
    # Step 4: Map chunks to pages using position overlaps
    for chunk in section_chunks:
        chunk_num = chunk.get('chunk_number', 0)
        
        # Store original values for comparison (not in output unless debug_mode)
        original_start = chunk.get('chunk_start_page')
        original_end = chunk.get('chunk_end_page')
        
        # Get chunk position in concatenated content
        if chunk_num not in chunk_positions:
            logger.warning(f"Chunk {chunk_num}: Not found in position map")
            continue
        
        start_pos, end_pos = chunk_positions[chunk_num]
        
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
                logger.info(f"Chunk {chunk_num}: Corrected from pages {original_start}-{original_end} to {new_start}-{new_end}")
        else:
            # No direct page overlap - need to infer
            stats['position_inferred'] += 1
            # Will be handled in Step 5
    
    # Special handling for first chunk - should align with section start page
    if section_chunks and section_start_page:
        first = section_chunks[0]
        if first.get('chunk_start_page') != section_start_page:
            logger.warning(f"First chunk doesn't start at section start page {section_start_page}")
            # Could adjust here if needed
    
    # Step 5: Handle chunks without direct page assignments
    for i, chunk in enumerate(section_chunks):
        if chunk.get('chunk_start_page') is None:
            # Try to infer from neighbors
            inferred = False
            
            # Look at previous chunk
            if i > 0:
                prev = section_chunks[i - 1]
                if prev.get('chunk_end_page') is not None:
                    # Check next chunk
                    if i + 1 < len(section_chunks):
                        next_chunk = section_chunks[i + 1]
                        if next_chunk.get('chunk_start_page') is not None:
                            # Chunk is between two known chunks
                            prev_end = prev['chunk_end_page']
                            next_start = next_chunk['chunk_start_page']
                            
                            if prev_end == next_start:
                                # Sandwiched on same page
                                chunk['chunk_start_page'] = prev_end
                                chunk['chunk_end_page'] = prev_end
                            elif next_start > prev_end:
                                # Spans gap between chunks
                                chunk['chunk_start_page'] = prev_end
                                chunk['chunk_end_page'] = next_start
                            else:
                                # Overlap - use previous end
                                chunk['chunk_start_page'] = prev_end
                                chunk['chunk_end_page'] = prev_end
                            
                            inferred = True
                        else:
                            # No next chunk with page info - continue from previous
                            chunk['chunk_start_page'] = prev['chunk_end_page']
                            chunk['chunk_end_page'] = prev['chunk_end_page']
                            inferred = True
            
            # First chunk special case
            if not inferred and i == 0 and section_start_page:
                chunk['chunk_start_page'] = section_start_page
                chunk['chunk_end_page'] = section_start_page
                inferred = True
            
            # Last chunk special case
            if not inferred and i == len(section_chunks) - 1 and section_end_page:
                if i > 0 and section_chunks[i-1].get('chunk_end_page'):
                    chunk['chunk_start_page'] = section_chunks[i-1]['chunk_end_page']
                    chunk['chunk_end_page'] = section_end_page
                else:
                    chunk['chunk_start_page'] = section_end_page
                    chunk['chunk_end_page'] = section_end_page
                inferred = True
            
            if inferred:
                # Add page references (if available)
                chunk['chunk_start_reference'] = page_references.get(chunk['chunk_start_page'], '')
                chunk['chunk_end_reference'] = page_references.get(chunk['chunk_end_page'], '')
                
                # Add debug fields if enabled
                if debug_mode:
                    chunk['original_start_page'] = None
                    chunk['original_end_page'] = None
                    chunk['page_boundary_method'] = 'position_inferred'
                    chunk['page_correction_applied'] = True
                
                stats['inferred'] += 1
                stats['corrected'] += 1
                logger.info(f"Chunk {chunk.get('chunk_number', '?')}: Inferred pages {chunk['chunk_start_page']}-{chunk['chunk_end_page']}")
    
    # Step 6: Validate and fix continuity
    for i in range(len(section_chunks) - 1):
        current = section_chunks[i]
        next_chunk = section_chunks[i + 1]
        
        current_end = current.get('chunk_end_page')
        next_start = next_chunk.get('chunk_start_page')
        
        if current_end and next_start:
            if next_start < current_end:
                # Overlap detected - adjust current chunk's end
                logger.warning(f"Overlap detected: Chunk {current['chunk_number']} ends at {current_end}, "
                              f"but chunk {next_chunk['chunk_number']} starts at {next_start}")
                current['chunk_end_page'] = next_start
                stats['continuity_fixed'] += 1
                
                if debug_mode:
                    current['page_boundary_method'] = 'continuity_fixed'
    
    return section_chunks, stats

# ==============================================================================
# Main Processing Function
# ==============================================================================

def main():
    """Main processing function for Stage 3.5 (local version)"""
    
    parser = argparse.ArgumentParser(description='Stage 3.5: Correct chunk page boundaries')
    parser.add_argument('-i', '--input', 
                       default='stage3_chunks.json',
                       help='Input file path (stage 3 output)')
    parser.add_argument('-o', '--output',
                       default='stage3_5_corrected_chunks.json',
                       help='Output file path for corrected chunks')
    parser.add_argument('--debug', action='store_true',
                       help='Include debug fields in output')
    parser.add_argument('--sample-limit', type=int,
                       help='Limit processing to first N sections')
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
    
    # Group chunks by section
    chunks_by_section = defaultdict(list)
    for chunk in chunks:
        key = (
            chunk['document_id'],
            chunk['chapter_number'],
            chunk['section_number']
        )
        chunks_by_section[key].append(chunk)
    
    print(f"📊 Found {len(chunks_by_section)} unique sections")
    
    # Apply sample limit if configured
    if args.sample_limit:
        limited_keys = list(chunks_by_section.keys())[:args.sample_limit]
        chunks_by_section = {k: chunks_by_section[k] for k in limited_keys}
        print(f"Limited to first {args.sample_limit} sections")
    
    # Process each section's chunks
    all_corrected_chunks = []
    total_stats = defaultdict(int)
    sections_without_tags = []
    
    section_count = 0
    total_sections = len(chunks_by_section)
    
    for (doc_id, chapter_num, section_num), section_chunks in sorted(chunks_by_section.items()):
        section_count += 1
        
        if section_count % 10 == 0 or args.verbose:
            print(f"[{section_count}/{total_sections}] Processing Chapter {chapter_num}, Section {section_num}...")
        
        # Process this section's chunks
        corrected_chunks, stats = process_section_chunks(section_chunks, debug_mode=args.debug)
        all_corrected_chunks.extend(corrected_chunks)
        
        # Aggregate statistics
        for key, value in stats.items():
            total_stats[key] += value
        
        if stats['no_tags_found'] > 0:
            sections_without_tags.append((doc_id, chapter_num, section_num))
    
    print(f"✅ Processed all {total_sections} sections")
    
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
    
    if sections_without_tags:
        print(f"\n⚠️ Sections without any page tags: {len(sections_without_tags)}")
        if args.verbose:
            for doc_id, chapter, section in sections_without_tags[:10]:
                print(f"  - Chapter {chapter}, Section {section}")
    
    # Calculate correction percentage
    if total_stats['total'] > 0:
        correction_pct = (total_stats['corrected'] / total_stats['total']) * 100
        print(f"\n📈 Correction rate: {correction_pct:.1f}%")
    
    print("\n" + "=" * 60)
    print("✅ Stage 3.5 Processing Complete!")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit(main())