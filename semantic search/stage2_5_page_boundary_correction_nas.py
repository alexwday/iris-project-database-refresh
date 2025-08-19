#!/usr/bin/env python3
"""
Stage 2.5: Page Boundary Correction with NAS Integration
Corrects page boundaries for sections after Stage 2 processing using embedded page tags.

Key Features:
- NAS integration for reading/writing files
- Position-based page mapping algorithm
- Handles sections with no tags, mid-page boundaries, missing headers/footers
- Validates and fixes continuity issues
- Preserves all existing fields while correcting page boundaries

Input: JSON file from Stage 2 output on NAS (stage2_section_records.json)
Output: JSON file with corrected page boundaries on NAS (stage2_5_corrected_sections.json)
"""

import os
import json
import re
import logging
import tempfile
import socket
import io
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from datetime import datetime

# --- pysmb imports for NAS access ---
try:
    from smb.SMBConnection import SMBConnection
    from smb import smb_structs
except ImportError:
    SMBConnection = None
    smb_structs = None
    print("ERROR: pysmb library not installed. NAS features unavailable. `pip install pysmb`")

# ==============================================================================
# Configuration (Shared with Stage 2 - update these values)
# ==============================================================================

# --- NAS Configuration ---
NAS_PARAMS = {
    "ip": "your_nas_ip",  # TODO: Replace with actual NAS IP
    "share": "your_share_name",  # TODO: Replace with actual share name
    "user": "your_nas_user",  # TODO: Replace with actual NAS username
    "password": "your_nas_password",  # TODO: Replace with actual NAS password
    "port": 445,  # Default SMB port (can be 139)
}

# --- Directory Paths (Relative to NAS Share) ---
NAS_INPUT_PATH = "semantic_search/pipeline_output/stage2"
INPUT_FILENAME = "stage2_section_records.json"
NAS_OUTPUT_PATH = "semantic_search/pipeline_output/stage2_5"
NAS_LOG_PATH = "semantic_search/pipeline_output/logs"
OUTPUT_FILENAME = "stage2_5_corrected_sections.json"

# --- Sample Processing Limit ---
SAMPLE_CHAPTER_LIMIT = None  # Set to None to process all chapters, or a number to limit

# --- pysmb Configuration ---
if smb_structs is not None:
    smb_structs.SUPPORT_SMB2 = True
    smb_structs.MAX_PAYLOAD_SIZE = 65536
CLIENT_HOSTNAME = socket.gethostname()

# --- Logging Level Control ---
VERBOSE_LOGGING = False  # Set to True for detailed debug output, False for cleaner output

# ==============================================================================
# Configuration Validation
# ==============================================================================

def validate_configuration():
    """Validates that configuration values have been properly set."""
    errors = []

    if NAS_PARAMS["ip"] == "your_nas_ip":
        errors.append("NAS IP address not configured")
    if NAS_PARAMS["share"] == "your_share_name":
        errors.append("NAS share name not configured")
    if NAS_PARAMS["user"] == "your_nas_user":
        errors.append("NAS username not configured")
    if NAS_PARAMS["password"] == "your_nas_password":
        errors.append("NAS password not configured")

    if errors:
        print("❌ Configuration errors detected:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease update the configuration values in the script before running.")
        return False
    return True

# ==============================================================================
# NAS Helper Functions
# ==============================================================================

def create_nas_connection():
    """Creates and returns an authenticated SMBConnection object."""
    try:
        conn = SMBConnection(
            NAS_PARAMS["user"],
            NAS_PARAMS["password"],
            CLIENT_HOSTNAME,
            NAS_PARAMS["ip"],
            use_ntlm_v2=True,
            is_direct_tcp=(NAS_PARAMS["port"] == 445),
        )
        connected = conn.connect(NAS_PARAMS["ip"], NAS_PARAMS["port"], timeout=60)
        if not connected:
            logging.error("Failed to connect to NAS")
            return None
        return conn
    except Exception as e:
        logging.error(f"Exception creating NAS connection: {e}")
        return None


def ensure_nas_dir_exists(conn, share_name, dir_path_relative):
    """Ensures a directory exists on the NAS, creating it if necessary."""
    if not conn:
        return False

    path_parts = dir_path_relative.strip("/").split("/")
    current_path = ""
    try:
        for part in path_parts:
            if not part:
                continue
            current_path = os.path.join(current_path, part).replace("\\", "/")
            try:
                conn.listPath(share_name, current_path)
            except Exception:
                conn.createDirectory(share_name, current_path)
        return True
    except Exception as e:
        logging.error(f"Failed to ensure NAS directory: {e}")
        return False


def write_to_nas(share_name, nas_path_relative, content_bytes):
    """Writes bytes to a file path on the NAS using pysmb."""
    conn = None
    try:
        conn = create_nas_connection()
        if not conn:
            return False

        dir_path = os.path.dirname(nas_path_relative).replace("\\", "/")
        if dir_path and not ensure_nas_dir_exists(conn, share_name, dir_path):
            return False

        file_obj = io.BytesIO(content_bytes)
        bytes_written = conn.storeFile(share_name, nas_path_relative, file_obj)

        if bytes_written == 0 and len(content_bytes) > 0:
            logging.error(f"No bytes written to {nas_path_relative}")
            return False

        return True
    except Exception as e:
        logging.error(f"Error writing to NAS: {e}")
        return False
    finally:
        if conn:
            conn.close()


def read_from_nas(share_name, nas_path_relative):
    """Reads content (as bytes) from a file path on the NAS using pysmb."""
    conn = None
    file_obj = None
    try:
        conn = create_nas_connection()
        if not conn:
            return None

        file_obj = io.BytesIO()
        _, _ = conn.retrieveFile(share_name, nas_path_relative, file_obj)
        file_obj.seek(0)
        content_bytes = file_obj.read()
        return content_bytes
    except Exception as e:
        logging.error(f"Error reading from NAS: {e}")
        return None
    finally:
        if file_obj:
            try:
                file_obj.close()
            except Exception:
                pass
        if conn:
            conn.close()

# ==============================================================================
# Logging Setup
# ==============================================================================

def setup_logging():
    """Setup logging with controlled verbosity."""
    temp_log = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log")
    temp_log_path = temp_log.name
    temp_log.close()

    log_level = logging.DEBUG if VERBOSE_LOGGING else logging.INFO

    # Set the root logger level
    logging.root.setLevel(log_level)

    # Add file handler for complete logging
    root_file_handler = logging.FileHandler(temp_log_path)
    root_file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.root.addHandler(root_file_handler)

    # Set up progress logger (separate from inference logging)
    progress_logger = logging.getLogger("progress")
    progress_logger.setLevel(logging.INFO)
    progress_logger.propagate = False

    progress_console_handler = logging.StreamHandler()
    progress_console_handler.setFormatter(logging.Formatter("%(message)s"))
    progress_logger.addHandler(progress_console_handler)

    progress_file_handler = logging.FileHandler(temp_log_path)
    progress_file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    progress_logger.addHandler(progress_file_handler)

    # Ensure messages show on console
    has_console_handler = any(isinstance(h, logging.StreamHandler) for h in logging.root.handlers)
    if not has_console_handler:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        console_handler.setLevel(logging.DEBUG if VERBOSE_LOGGING else logging.INFO)
        logging.root.addHandler(console_handler)

    return temp_log_path


def log_progress(message, end="\n"):
    """Log a progress message that always shows."""
    progress_logger = logging.getLogger("progress")

    if end == "":
        sys.stdout.write(message)
        sys.stdout.flush()
        for handler in progress_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.stream.write(f"{datetime.now().isoformat()} - {message}\n")
                handler.flush()
    else:
        progress_logger.info(message)

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
    pages_seen = {}  # Track start and end positions for each page
    
    # First pass: collect all positions for each page
    for pos, tag_type, page_num, page_ref in page_tags:
        if page_num not in pages_seen:
            pages_seen[page_num] = {'headers': [], 'footers': []}
        
        if tag_type == 'header':
            pages_seen[page_num]['headers'].append(pos)
        else:  # footer
            pages_seen[page_num]['footers'].append(pos)
    
    # Second pass: build ranges for each page
    for page_num in sorted(pages_seen.keys()):
        page_info = pages_seen[page_num]
        
        # Determine start position
        if page_info['headers']:
            # Page has header(s) - use first header position
            start_pos = min(page_info['headers'])
        elif page_info['footers']:
            # Page has only footer(s) - must start from beginning or after previous page
            # Look for previous page's last position
            prev_page = page_num - 1
            if prev_page in pages_seen and pages_seen[prev_page]['footers']:
                # Start after previous page's last footer
                prev_footer_pos = max(pages_seen[prev_page]['footers'])
                # Find end of footer tag
                footer_match = re.search(r'<!-- PageFooter[^>]*?-->', content[prev_footer_pos:prev_footer_pos+100])
                if footer_match:
                    start_pos = prev_footer_pos + footer_match.end()
                else:
                    start_pos = prev_footer_pos + 50  # Estimate
            else:
                # No previous page or it has no footer - start from beginning
                start_pos = 0
        else:
            # Shouldn't happen but handle gracefully
            continue
        
        # Determine end position
        if page_info['footers']:
            # Page has footer(s) - use last footer position + tag length
            last_footer_pos = max(page_info['footers'])
            footer_match = re.search(r'<!-- PageFooter[^>]*?-->', content[last_footer_pos:last_footer_pos+100])
            if footer_match:
                end_pos = last_footer_pos + footer_match.end()
            else:
                end_pos = last_footer_pos + 50  # Estimate
        elif page_info['headers']:
            # Page has only header(s) - find next page's start or use end of content
            next_page = page_num + 1
            if next_page in pages_seen:
                if pages_seen[next_page]['headers']:
                    # End just before next page's header
                    end_pos = min(pages_seen[next_page]['headers']) - 1
                else:
                    # Next page has no header, use current estimate
                    end_pos = len(content)
            else:
                # No next page - use end of content
                end_pos = len(content)
        else:
            # Shouldn't happen
            continue
        
        ranges.append((start_pos, end_pos, page_num))
    
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
        end_pos = current_pos + len(content)
        
        section_positions[section_num] = (start_pos, end_pos)
        full_content_parts.append(content)
        
        current_pos = end_pos
    
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
        
        # Store original values
        original_start = section.get('section_start_page')
        original_end = section.get('section_end_page')
        section['original_start_page'] = original_start
        section['original_end_page'] = original_end
        
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
            section['page_boundary_method'] = 'tag_found'
            
            stats['tag_found'] += 1
            
            # Check if correction was needed
            if original_start != new_start or original_end != new_end:
                section['page_correction_applied'] = True
                stats['corrected'] += 1
                logging.info(f"Section {section_num}: Corrected from pages {original_start}-{original_end} to {new_start}-{new_end}")
            else:
                section['page_correction_applied'] = False
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
                section['page_boundary_method'] = 'position_inferred'
                section['page_correction_applied'] = True
                stats['inferred'] += 1
                stats['corrected'] += 1
                
                original_start = section.get('original_start_page')
                original_end = section.get('original_end_page')
                logging.info(f"Section {section.get('section_number', '?')}: Inferred pages {section['section_start_page']}-{section['section_end_page']} (was {original_start}-{original_end})")
    
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
                current['page_boundary_method'] = 'continuity_fixed'
                stats['continuity_fixed'] += 1
                logging.info(f"Fixed overlap: Section {i+1} now ends at page {current['section_end_page']}")
    
    # Ensure first section starts at page 1
    if sections and sections[0].get('section_start_page', 0) != 1:
        first = sections[0]
        if first.get('section_start_page') is not None:
            logging.warning(f"First section starts at page {first['section_start_page']}, not page 1")
    
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


def process_sections(input_data: List[Dict]) -> Tuple[List[Dict], Dict]:
    """
    Process all sections to correct page boundaries.
    
    Args:
        input_data: List of section records from Stage 2
    
    Returns:
        Tuple of (corrected_sections, overall_statistics)
    """
    # Group sections by chapter
    chapters = group_sections_by_chapter(input_data)
    
    # Apply sample limit if configured
    if SAMPLE_CHAPTER_LIMIT is not None and SAMPLE_CHAPTER_LIMIT > 0:
        chapter_nums = sorted(chapters.keys())[:SAMPLE_CHAPTER_LIMIT]
        chapters = {k: v for k, v in chapters.items() if k in chapter_nums}
        log_progress(f"⚠️  SAMPLE MODE: Processing only first {SAMPLE_CHAPTER_LIMIT} chapters")
    
    log_progress(f"📊 Processing {len(chapters)} chapters with {sum(len(secs) for secs in chapters.values())} total sections")
    
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
        log_progress(f"\n📚 Processing Chapter {chapter_num}: {len(chapter_sections)} sections")
        
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


def cleanup_logging_handlers():
    """Safely cleanup logging handlers."""
    progress_logger = logging.getLogger("progress")
    handlers_to_remove = list(progress_logger.handlers)
    for handler in handlers_to_remove:
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass
        try:
            progress_logger.removeHandler(handler)
        except Exception:
            pass

    root_handlers_to_remove = list(logging.root.handlers)
    for handler in root_handlers_to_remove:
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass
        try:
            logging.root.removeHandler(handler)
        except Exception:
            pass

    progress_logger.handlers.clear()
    logging.root.handlers.clear()

# ==============================================================================
# Main Entry Point
# ==============================================================================

def run_stage2_5():
    """Main function to execute Stage 2.5 processing."""
    if not validate_configuration():
        return

    temp_log_path = setup_logging()

    log_progress("=" * 70)
    log_progress("🔧 Stage 2.5: Page Boundary Correction")
    log_progress("=" * 70)

    share_name = NAS_PARAMS["share"]
    input_path = os.path.join(NAS_INPUT_PATH, INPUT_FILENAME).replace("\\", "/")
    output_path = os.path.join(NAS_OUTPUT_PATH, OUTPUT_FILENAME).replace("\\", "/")

    # Load input JSON from NAS
    log_progress("📥 Loading Stage 2 output from NAS...")
    input_json_bytes = read_from_nas(share_name, input_path)

    if not input_json_bytes:
        log_progress("❌ Failed to read input JSON from NAS")
        return

    try:
        section_records = json.loads(input_json_bytes.decode("utf-8"))
        if not isinstance(section_records, list):
            log_progress("❌ Input JSON is not a list")
            return
        log_progress(f"✅ Loaded {len(section_records)} section records")
    except json.JSONDecodeError as e:
        log_progress(f"❌ Error decoding JSON: {e}")
        return

    # Process sections
    log_progress("-" * 70)
    corrected_sections, statistics = process_sections(section_records)
    
    # Save output to NAS
    log_progress("-" * 70)
    log_progress(f"💾 Saving {len(corrected_sections)} corrected section records...")

    try:
        output_json = json.dumps(corrected_sections, indent=2, ensure_ascii=False)
        output_bytes = output_json.encode("utf-8")

        if write_to_nas(share_name, output_path, output_bytes):
            log_progress(f"✅ Successfully saved output to {share_name}/{output_path}")
        else:
            log_progress("❌ Failed to write output to NAS")
    except Exception as e:
        log_progress(f"❌ Error saving output: {e}")

    # Upload log file
    try:
        log_file_name = f"stage2_5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path_relative = os.path.join(NAS_LOG_PATH, log_file_name).replace("\\", "/")

        cleanup_logging_handlers()

        with open(temp_log_path, "rb") as f:
            log_content = f.read()

        if write_to_nas(share_name, log_path_relative, log_content):
            print(f"📝 Log file uploaded: {share_name}/{log_path_relative}")
        else:
            print("⚠️ Failed to upload log file")

        os.remove(temp_log_path)
    except Exception as e:
        print(f"⚠️ Error handling log file: {e}")

    # Final summary
    print("=" * 70)
    print("📊 Stage 2.5 Summary")
    print("-" * 70)
    print(f"  Input: {len(section_records)} section records from Stage 2")
    print(f"  Chapters processed: {statistics['total_chapters']}")
    print(f"  Sections corrected: {statistics['sections_corrected']} ({statistics['sections_corrected']/max(statistics['total_sections'], 1)*100:.1f}%)")
    print(f"  Sections with inferred pages: {statistics['sections_inferred']}")
    print(f"  Sections with page tags found: {statistics['sections_with_tags']}")
    
    if statistics['sections_without_tags'] > 0:
        print(f"  ⚠️  Sections without any tags: {statistics['sections_without_tags']}")
    if statistics['chapters_without_tags'] > 0:
        print(f"  ⚠️  Chapters without any tags: {statistics['chapters_without_tags']}")
    
    if SAMPLE_CHAPTER_LIMIT:
        print(f"  ⚠️  SAMPLE MODE: Limited to {SAMPLE_CHAPTER_LIMIT} chapters")
    
    print(f"  Output: {len(corrected_sections)} corrected section records")
    print(f"  Output file: {share_name}/{output_path}")
    print("=" * 70)
    print("✅ Stage 2.5 Completed")


if __name__ == "__main__":
    run_stage2_5()