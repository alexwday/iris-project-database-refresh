#!/usr/bin/env python3
"""
Stage 2.5: Page Boundary Correction
Corrects page boundaries for sections after Stage 2 processing using embedded page tags.

Key Features:
- Position-based page mapping algorithm
- Handles sections with no tags, mid-page boundaries, missing headers/footers
- Validates and fixes continuity issues
- Preserves all existing fields while correcting page boundaries

Input: JSON file from Stage 2 output (stage2_section_records.json)
Output: JSON file with corrected page boundaries (stage2_5_corrected_sections.json)
"""

import os
import json
import re
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from datetime import datetime


# ==============================================================================
# Configuration
# ==============================================================================

# Default file paths
DEFAULT_INPUT_FILE = "stage2_section_records.json"
DEFAULT_OUTPUT_FILE = "stage2_5_corrected_sections.json"

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Processing parameters
VERBOSE_LOGGING = False
DEBUG_MODE = False  # Set to True to include debug fields in output


# ==============================================================================
# Logging Setup
# ==============================================================================

def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT
    )


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


def build_page_ranges(page_tags: List[Tuple[int, str, int, str]], content: str) -> List[Tuple[int, int, int]]:
    """
    Build page ranges from extracted tags.
    
    Args:
        page_tags: List of extracted page tags with positions
        content: The full content string
    
    Returns:
        List of tuples: (start_position, end_position, page_number)
    """
    if not page_tags:
        return []
    
    ranges = []
    
    # Process tags in order to build page ranges
    current_page = None
    page_start = 0
    
    for i, (pos, tag_type, page_num, page_ref) in enumerate(page_tags):
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
                
                # If there's a gap before the next tag, track it
                if i + 1 < len(page_tags):
                    next_pos = page_tags[i + 1][0]
                    if next_pos > page_end + 1:
                        # There's content between pages - for now skip it
                        pass
            elif current_page is None:
                # Footer without header - page starts from beginning or after previous
                if ranges:
                    page_start = ranges[-1][1] + 1
                else:
                    page_start = 0
                
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
                
                ranges.append((page_start, page_end, page_num))
    
    # If we ended while tracking a page (header without footer), close it
    if current_page is not None:
        ranges.append((page_start, len(content) - 1, current_page))
    
    # Sort ranges by start position
    ranges.sort(key=lambda x: x[0])
    
    return ranges


def build_position_map(sections: List[Dict]) -> Tuple[str, Dict[int, Tuple[int, int]]]:
    """
    Build a position map for all sections in concatenated content.
    
    Returns:
        Tuple of (concatenated_content, section_positions)
        section_positions maps section_number to (start_pos, end_pos)
    """
    full_content_parts = []
    section_positions = {}
    current_pos = 0
    
    for section in sections:
        section_num = section.get('section_number', 0)
        content = section.get('section_content', '')
        
        if not content:
            logging.warning(f"Section {section_num}: Empty content")
            content = ""
        
        start_pos = current_pos
        end_pos = current_pos + len(content) - 1  # -1 for inclusive end position
        
        section_positions[section_num] = (start_pos, end_pos)
        full_content_parts.append(content)
        
        current_pos = current_pos + len(content)  # Next section starts after this content
    
    full_content = ''.join(full_content_parts)
    return full_content, section_positions


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
# Page Boundary Correction
# ==============================================================================

def process_chapter(chapter_sections: List[Dict]) -> Tuple[List[Dict], Dict]:
    """
    Process all sections in a chapter to correct page boundaries.
    
    Returns:
        Tuple of (corrected_sections, statistics)
    """
    if not chapter_sections:
        return chapter_sections, {'total': 0, 'corrected': 0, 'inferred': 0}
    
    stats = {
        'total': len(chapter_sections),
        'corrected': 0,
        'inferred': 0,
        'tag_found': 0,
        'position_inferred': 0,
        'continuity_fixed': 0,
        'no_tags_found': 0
    }
    
    # Sort sections by section_number to ensure correct order
    chapter_sections.sort(key=lambda x: x.get('section_number', 0))
    
    # Step 1: Build position map and concatenate content
    full_content, section_positions = build_position_map(chapter_sections)
    
    # Step 2: Extract all page tags with positions
    page_tags = extract_page_tags_with_positions(full_content)
    
    if not page_tags:
        logging.warning(f"Chapter {chapter_sections[0].get('chapter_number', '?')}: No page tags found in content")
        stats['no_tags_found'] = len(chapter_sections)
        # Fall back to existing boundaries
        return chapter_sections, stats
    
    # Step 3: Build page ranges from tags
    page_ranges = build_page_ranges(page_tags, full_content)
    
    if not page_ranges:
        logging.warning(f"Chapter {chapter_sections[0].get('chapter_number', '?')}: Could not build page ranges")
        return chapter_sections, stats
    
    # Step 4: Map sections to pages using position overlaps
    for section in chapter_sections:
        section_num = section.get('section_number', 0)
        
        # Store original values for comparison (not in output)
        original_start = section.get('section_start_page')
        original_end = section.get('section_end_page')
        
        # Get section position in concatenated content
        if section_num not in section_positions:
            logging.warning(f"Section {section_num}: Not found in position map")
            continue
        
        start_pos, end_pos = section_positions[section_num]
        
        # Determine pages for this position range
        pages = determine_pages_for_position_range(start_pos, end_pos, page_ranges)
        
        if pages:
            # Section overlaps with identified pages
            new_start = min(pages)
            new_end = max(pages)
            
            section['section_start_page'] = new_start
            section['section_end_page'] = new_end
            section['section_page_count'] = new_end - new_start + 1
            
            # Add debug fields if enabled
            if DEBUG_MODE:
                section['original_start_page'] = original_start
                section['original_end_page'] = original_end
                section['page_boundary_method'] = 'tag_found'
                section['page_correction_applied'] = (original_start != new_start or original_end != new_end)
            
            stats['tag_found'] += 1
            
            # Check if correction was needed
            if original_start != new_start or original_end != new_end:
                stats['corrected'] += 1
                logging.info(f"Section {section_num}: Corrected from pages {original_start}-{original_end} to {new_start}-{new_end}")
        else:
            # No direct page overlap - need to infer
            stats['position_inferred'] += 1
            # Will be handled in Step 5
    
    # Special handling for first section - always starts at page 1
    if chapter_sections and chapter_sections[0].get('section_start_page') != 1:
        first = chapter_sections[0]
        if first.get('section_start_page') is None:
            # First section with no page info - set to page 1
            first['section_start_page'] = 1
            first['section_end_page'] = 1  # Will be adjusted if we find next section info
            first['section_page_count'] = 1
            
            # Add debug fields if enabled
            if DEBUG_MODE:
                first['original_start_page'] = None
                first['original_end_page'] = None
                first['page_boundary_method'] = 'first_section_rule'
                first['page_correction_applied'] = True
            
            stats['inferred'] += 1
            stats['corrected'] += 1
            logging.info(f"Section 1: Set to page 1 (first section rule)")
        elif first.get('section_start_page') > 1:
            # First section doesn't start at page 1 - log warning but don't change
            logging.warning(f"Section 1: Starts at page {first['section_start_page']}, not page 1. Consider investigation.")
    
    # Step 5: Handle sections without direct page assignments
    for i, section in enumerate(chapter_sections):
        if section.get('section_start_page') is None:
            # Store original values for debug mode
            original_start = section.get('section_start_page')
            original_end = section.get('section_end_page')
            
            # Try to infer from neighbors
            inferred = False
            
            # Look at previous section
            if i > 0:
                prev = chapter_sections[i - 1]
                if prev.get('section_end_page') is not None:
                    # Check next section
                    if i + 1 < len(chapter_sections):
                        next_section = chapter_sections[i + 1]
                        if next_section.get('section_start_page') is not None:
                            # Section is between two known sections
                            prev_end = prev['section_end_page']
                            next_start = next_section['section_start_page']
                            
                            if prev_end == next_start:
                                # Sandwiched on same page
                                section['section_start_page'] = prev_end
                                section['section_end_page'] = prev_end
                            elif next_start > prev_end:
                                # Spans gap between sections
                                section['section_start_page'] = prev_end + 1
                                section['section_end_page'] = next_start - 1
                            else:
                                # Overlap - use previous end
                                section['section_start_page'] = prev_end
                                section['section_end_page'] = prev_end
                            
                            inferred = True
                        else:
                            # Just use next page after previous section
                            section['section_start_page'] = prev['section_end_page'] + 1
                            section['section_end_page'] = prev['section_end_page'] + 1
                            inferred = True
            
            # Last section special case
            if not inferred and i == len(chapter_sections) - 1 and i > 0:
                prev = chapter_sections[i - 1]
                if prev.get('section_end_page') is not None:
                    # Last section continues from previous
                    section['section_start_page'] = prev['section_end_page'] + 1
                    section['section_end_page'] = prev['section_end_page'] + 1
                    inferred = True
            
            if inferred:
                section['section_page_count'] = section['section_end_page'] - section['section_start_page'] + 1
                
                # Add debug fields if enabled
                if DEBUG_MODE:
                    section['original_start_page'] = original_start  # Note: original_start is from earlier in function
                    section['original_end_page'] = original_end
                    section['page_boundary_method'] = 'position_inferred'
                    section['page_correction_applied'] = True
                
                stats['inferred'] += 1
                stats['corrected'] += 1
                logging.info(f"Section {section.get('section_number', '?')}: Inferred pages {section['section_start_page']}-{section['section_end_page']}")
    
    # Step 6: Validate and fix continuity
    chapter_sections = validate_and_fix_continuity(chapter_sections, stats)
    
    return chapter_sections, stats


def validate_and_fix_continuity(sections: List[Dict], stats: Dict) -> List[Dict]:
    """
    Validate page continuity and fix any gaps or inconsistencies.
    """
    if not sections:
        return sections
    
    # Sort by section number
    sections.sort(key=lambda x: x.get('section_number', 0))
    
    # Check for gaps and overlaps
    for i in range(len(sections) - 1):
        current = sections[i]
        next_section = sections[i + 1]
        
        curr_end = current.get('section_end_page')
        next_start = next_section.get('section_start_page')
        
        if curr_end is None or next_start is None:
            continue
        
        # Check for gap
        if next_start > curr_end + 1:
            logging.warning(f"Gap detected between sections {i+1} and {i+2}: pages {curr_end+1} to {next_start-1}")
            # Could adjust boundaries here if needed
        
        # Check for overlap
        if next_start < curr_end:
            logging.warning(f"Overlap detected between sections {i+1} and {i+2}: section {i+2} starts at {next_start} but {i+1} ends at {curr_end}")
            # Adjust to remove overlap
            if next_start > current.get('section_start_page', 0):
                current['section_end_page'] = next_start - 1
                current['section_page_count'] = current['section_end_page'] - current['section_start_page'] + 1
                stats['continuity_fixed'] += 1
                logging.info(f"Fixed overlap: Section {i+1} now ends at page {current['section_end_page']}")
    
    # Ensure first section starts at page 1
    if sections and sections[0].get('section_start_page', 0) != 1:
        first = sections[0]
        if first.get('section_start_page') is not None:
            logging.warning(f"First section starts at page {first['section_start_page']}, not page 1")
            # Optionally force it to start at page 1
            # first['section_start_page'] = 1
            # first['page_boundary_method'] = 'continuity_fixed'
            # stats['continuity_fixed'] += 1
    
    return sections


# ==============================================================================
# Main Processing Functions
# ==============================================================================

def group_sections_by_chapter(sections: List[Dict]) -> Dict[int, List[Dict]]:
    """Group sections by chapter number."""
    chapters = defaultdict(list)
    
    for section in sections:
        chapter_num = section.get('chapter_number')
        if chapter_num is not None:
            chapters[chapter_num].append(section)
    
    # Sort sections within each chapter
    for chapter_num in chapters:
        chapters[chapter_num].sort(key=lambda x: x.get('section_number', 0))
    
    return dict(chapters)


def process_sections(input_data: List[Dict], chapter_filter: Optional[List[int]] = None) -> Tuple[List[Dict], Dict]:
    """
    Process all sections to correct page boundaries.
    
    Args:
        input_data: List of section records from Stage 2
        chapter_filter: Optional list of chapter numbers to process (None = all)
    
    Returns:
        Tuple of (corrected_sections, overall_statistics)
    """
    # Group sections by chapter
    chapters = group_sections_by_chapter(input_data)
    
    # Filter chapters if requested
    if chapter_filter:
        chapters = {k: v for k, v in chapters.items() if k in chapter_filter}
        logging.info(f"Processing only chapters: {sorted(chapter_filter)}")
    
    logging.info(f"Processing {len(chapters)} chapters with {sum(len(secs) for secs in chapters.values())} total sections")
    
    # Process each chapter
    all_corrected_sections = []
    overall_stats = {
        'total_chapters': len(chapters),
        'total_sections': 0,
        'sections_corrected': 0,
        'sections_inferred': 0,
        'sections_with_tags': 0,
        'sections_without_tags': 0,
        'chapters_without_tags': 0
    }
    
    for chapter_num in sorted(chapters.keys()):
        chapter_sections = chapters[chapter_num]
        logging.info(f"\nProcessing Chapter {chapter_num}: {len(chapter_sections)} sections")
        
        # Process chapter
        corrected_sections, chapter_stats = process_chapter(chapter_sections)
        
        # Update overall statistics
        overall_stats['total_sections'] += chapter_stats['total']
        overall_stats['sections_corrected'] += chapter_stats['corrected']
        overall_stats['sections_inferred'] += chapter_stats['inferred']
        overall_stats['sections_with_tags'] += chapter_stats['tag_found']
        
        if chapter_stats['no_tags_found'] > 0:
            overall_stats['chapters_without_tags'] += 1
            overall_stats['sections_without_tags'] += chapter_stats['no_tags_found']
        
        # Add to results
        all_corrected_sections.extend(corrected_sections)
    
    return all_corrected_sections, overall_stats


def print_statistics(stats: Dict) -> None:
    """Print processing statistics."""
    print("\n" + "=" * 70)
    print("📊 Processing Statistics")
    print("-" * 70)
    print(f"  Total chapters processed: {stats['total_chapters']}")
    print(f"  Total sections processed: {stats['total_sections']}")
    print(f"  Sections with corrections: {stats['sections_corrected']} ({stats['sections_corrected']/max(stats['total_sections'], 1)*100:.1f}%)")
    print(f"  Sections with inferred pages: {stats['sections_inferred']} ({stats['sections_inferred']/max(stats['total_sections'], 1)*100:.1f}%)")
    print(f"  Sections with page tags found: {stats['sections_with_tags']} ({stats['sections_with_tags']/max(stats['total_sections'], 1)*100:.1f}%)")
    
    if stats['sections_without_tags'] > 0:
        print(f"  ⚠️  Sections without any tags: {stats['sections_without_tags']}")
    if stats['chapters_without_tags'] > 0:
        print(f"  ⚠️  Chapters without any tags: {stats['chapters_without_tags']}")
    
    # Summary message
    if stats['sections_corrected'] == 0:
        print("\n  ✅ All sections already had correct page boundaries!")
    elif stats['sections_corrected'] == stats['total_sections']:
        print(f"\n  ✅ Corrected page boundaries for all {stats['total_sections']} sections")
    else:
        print(f"\n  ✅ Corrected {stats['sections_corrected']} out of {stats['total_sections']} sections")
    
    print("=" * 70)


# ==============================================================================
# File I/O
# ==============================================================================

def load_input_file(filepath: str) -> List[Dict]:
    """Load JSON input file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Input file must contain a JSON array")
    
    logging.info(f"Loaded {len(data)} records from {filepath}")
    return data


def save_output_file(filepath: str, data: List[Dict]) -> None:
    """Save JSON output file."""
    output_dir = os.path.dirname(filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Saved {len(data)} records to {filepath}")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    """Main function for Stage 2.5 processing."""
    parser = argparse.ArgumentParser(
        description="Stage 2.5: Page Boundary Correction - Fixes section page boundaries using embedded page tags"
    )
    parser.add_argument(
        '-i', '--input',
        default=DEFAULT_INPUT_FILE,
        help=f'Input JSON file from Stage 2 (default: {DEFAULT_INPUT_FILE})'
    )
    parser.add_argument(
        '-o', '--output',
        default=DEFAULT_OUTPUT_FILE,
        help=f'Output JSON file with corrected boundaries (default: {DEFAULT_OUTPUT_FILE})'
    )
    parser.add_argument(
        '-c', '--chapters',
        type=int,
        nargs='+',
        help='Process only specific chapters (e.g., -c 1 2 3)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Print header
    print("=" * 70)
    print("🔧 Stage 2.5: Page Boundary Correction")
    print("=" * 70)
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    if args.chapters:
        print(f"Chapters to process: {args.chapters}")
    print("-" * 70)
    
    try:
        # Load input data
        input_data = load_input_file(args.input)
        
        # Process sections
        corrected_sections, statistics = process_sections(input_data, args.chapters)
        
        # Save output
        save_output_file(args.output, corrected_sections)
        
        # Print statistics
        print_statistics(statistics)
        
        print("\n✅ Stage 2.5 processing completed successfully!")
        
    except Exception as e:
        logging.error(f"Processing failed: {e}", exc_info=True)
        print(f"\n❌ Processing failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())