#!/usr/bin/env python3
"""
Stage 3.5 v2: Chunk Page Boundary Correction with NAS Integration (Chapter-Level Processing)
Corrects page boundaries for chunks using chapter-wide page tag extraction.

Key Improvements:
- Processes all chunks in a chapter together (like Stage 2.5)
- Enables cross-section inference for sections without tags
- Uses neighboring sections' chunk data for better page inference
- Reduces "sections without tags" to near zero
- Full NAS integration for production use

Input: JSON file from Stage 3 output on NAS (stage3_chunks.json)
Output: JSON file with corrected chunk page boundaries on NAS (stage3_5_corrected_chunks.json)
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
# Configuration (Shared with Stage 3 - update these values)
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
NAS_INPUT_PATH = "semantic_search/pipeline_output/stage3"
INPUT_FILENAME = "stage3_chunks.json"
NAS_OUTPUT_PATH = "semantic_search/pipeline_output/stage3_5"
NAS_LOG_PATH = "semantic_search/pipeline_output/logs"
OUTPUT_FILENAME = "stage3_5_corrected_chunks.json"

# --- Sample Processing Limit ---
SAMPLE_CHAPTER_LIMIT = None  # Set to None to process all chapters, or a number to limit

# --- pysmb Configuration ---
if smb_structs is not None:
    smb_structs.SUPPORT_SMB2 = True
    smb_structs.MAX_PAYLOAD_SIZE = 65536
CLIENT_HOSTNAME = socket.gethostname()

# --- Logging Level Control ---
VERBOSE_LOGGING = False  # Set to True for detailed debug output, False for cleaner output
DEBUG_MODE = False  # Set to True to include debug fields in output

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
            if hasattr(handler, "stream") and hasattr(handler.stream, "flush"):
                handler.stream.flush()
    else:
        progress_logger.info(message)

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
    """Main processing function for Stage 3.5 v2 with NAS integration"""
    
    # Validate configuration
    if not validate_configuration():
        return 1
    
    # Setup logging
    temp_log_path = setup_logging()
    log_progress("\n" + "=" * 80)
    log_progress("Stage 3.5 v2: Chunk Page Boundary Correction with NAS Integration")
    log_progress("=" * 80)
    
    # Load input data from NAS
    input_path = f"{NAS_INPUT_PATH}/{INPUT_FILENAME}".replace("\\", "/")
    log_progress(f"\n📥 Loading chunks from NAS: {input_path}")
    
    try:
        input_bytes = read_from_nas(NAS_PARAMS["share"], input_path)
        if input_bytes is None:
            logging.error("Failed to read input file from NAS")
            return 1
        
        chunks = json.loads(input_bytes.decode("utf-8"))
        log_progress(f"✅ Loaded {len(chunks)} chunks")
    except Exception as e:
        logging.error(f"Failed to parse input file: {e}")
        return 1
    
    # Group chunks by chapter
    chunks_by_chapter = defaultdict(list)
    for chunk in chunks:
        chapter_num = chunk.get('chapter_number')
        if chapter_num is not None:
            chunks_by_chapter[chapter_num].append(chunk)
    
    log_progress(f"📊 Found {len(chunks_by_chapter)} chapters")
    
    # Count sections per chapter
    total_sections = 0
    for chapter_chunks in chunks_by_chapter.values():
        sections_in_chapter = len(set(c.get('section_number') for c in chapter_chunks))
        total_sections += sections_in_chapter
    
    log_progress(f"📊 Total sections across all chapters: {total_sections}")
    
    # Apply sample limit if configured
    if SAMPLE_CHAPTER_LIMIT:
        limited_chapters = sorted(chunks_by_chapter.keys())[:SAMPLE_CHAPTER_LIMIT]
        chunks_by_chapter = {k: chunks_by_chapter[k] for k in limited_chapters}
        log_progress(f"Limited to first {SAMPLE_CHAPTER_LIMIT} chapters")
    
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
        
        log_progress(f"\n[{chapter_count}/{total_chapters}] Processing Chapter {chapter_num} ({sections_in_chapter} sections, {len(chapter_chunks)} chunks)...")
        
        # Process this chapter's chunks
        corrected_chunks, stats = process_chapter_chunks(chapter_chunks, debug_mode=DEBUG_MODE)
        all_corrected_chunks.extend(corrected_chunks)
        
        # Aggregate statistics
        for key, value in stats.items():
            total_stats[key] += value
        
        if VERBOSE_LOGGING and stats['sections_without_tags'] > 0:
            log_progress(f"  - Sections without tags: {stats['sections_without_tags']}")
            log_progress(f"  - Sections with inference: {stats['sections_with_inference']}")
    
    log_progress(f"\n✅ Processed all {total_chapters} chapters")
    
    # Save corrected chunks to NAS
    output_path = f"{NAS_OUTPUT_PATH}/{OUTPUT_FILENAME}".replace("\\", "/")
    log_progress(f"\n💾 Saving {len(all_corrected_chunks)} corrected chunks to NAS: {output_path}")
    
    try:
        output_json = json.dumps(all_corrected_chunks, indent=2, ensure_ascii=False)
        success = write_to_nas(NAS_PARAMS["share"], output_path, output_json.encode("utf-8"))
        
        if not success:
            logging.error("Failed to write output file to NAS")
            return 1
        
        log_progress(f"✅ Corrected chunks saved successfully")
    except Exception as e:
        logging.error(f"Failed to save output: {e}")
        return 1
    
    # Print statistics
    log_progress("\n" + "=" * 60)
    log_progress("📊 Processing Statistics")
    log_progress("=" * 60)
    log_progress(f"Total chunks processed: {total_stats['total']}")
    log_progress(f"Chunks with page tags found: {total_stats['tag_found']}")
    log_progress(f"Chunks with inferred pages: {total_stats['inferred']}")
    log_progress(f"Chunks corrected: {total_stats['corrected']}")
    log_progress(f"Continuity fixes applied: {total_stats['continuity_fixed']}")
    log_progress(f"\nSections without any tags: {total_stats['sections_without_tags']}")
    log_progress(f"Sections resolved via inference: {total_stats['sections_with_inference']}")
    
    # Calculate correction percentage
    if total_stats['total'] > 0:
        correction_pct = (total_stats['corrected'] / total_stats['total']) * 100
        log_progress(f"\n📈 Correction rate: {correction_pct:.1f}%")
    
    # Calculate inference success rate
    if total_stats['sections_without_tags'] > 0:
        inference_rate = (total_stats['sections_with_inference'] / total_stats['sections_without_tags']) * 100
        log_progress(f"📈 Section inference success rate: {inference_rate:.1f}%")
    
    # Upload log file to NAS
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"stage3_5_v2_chunk_correction_{log_timestamp}.log"
    log_nas_path = f"{NAS_LOG_PATH}/{log_filename}".replace("\\", "/")
    
    log_progress(f"\n📤 Uploading log to NAS: {log_nas_path}")
    try:
        with open(temp_log_path, "rb") as f:
            log_content = f.read()
        
        if write_to_nas(NAS_PARAMS["share"], log_nas_path, log_content):
            log_progress("✅ Log uploaded successfully")
        else:
            log_progress("⚠️ Failed to upload log file")
    except Exception as e:
        log_progress(f"⚠️ Error uploading log: {e}")
    finally:
        # Clean up temp log file
        try:
            os.unlink(temp_log_path)
        except:
            pass
    
    log_progress("\n" + "=" * 60)
    log_progress("✅ Stage 3.5 v2 Processing Complete!")
    log_progress("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit(main())