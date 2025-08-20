#!/usr/bin/env python3
"""
Stage 3: Content Chunking Pipeline with NAS Integration
Chunks sections from Stage 2.5 into 500-750 token chunks for semantic search

Key Features:
- NAS integration for reading/writing files
- Smart chunking with hierarchical break point detection
- Preserves HTML page tags and never splits within them
- Custom token counting (no tiktoken dependency)
- Outputs streamlined schema for database ingestion

Input: JSON file from Stage 2.5 output on NAS (stage2_5_corrected_sections.json)
Output: JSON file with chunks on NAS (stage3_chunks.json)
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
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

# --- pysmb imports for NAS access ---
try:
    from smb.SMBConnection import SMBConnection
    from smb import smb_structs
except ImportError:
    SMBConnection = None
    smb_structs = None
    print("ERROR: pysmb library not installed. NAS features unavailable. `pip install pysmb`")

# ==============================================================================
# Configuration (Update these values to match your NAS setup)
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
NAS_INPUT_PATH = "semantic_search/pipeline_output/stage2_5"
INPUT_FILENAME = "stage2_5_corrected_sections.json"
NAS_OUTPUT_PATH = "semantic_search/pipeline_output/stage3"
NAS_LOG_PATH = "semantic_search/pipeline_output/logs"
OUTPUT_FILENAME = "stage3_chunks.json"

# --- Chunking Configuration ---
MIN_TOKENS = 500  # Minimum tokens per chunk
MAX_TOKENS = 750  # Target maximum tokens per chunk
HARD_MAX_TOKENS = 800  # Hard limit - will not exceed

# --- Sample Processing Limit ---
SAMPLE_CHAPTER_LIMIT = None  # Set to None to process all chapters, or a number to limit

# --- pysmb Configuration ---
if smb_structs is not None:
    smb_structs.SUPPORT_SMB2 = True
    smb_structs.MAX_PAYLOAD_SIZE = 65536
CLIENT_HOSTNAME = socket.gethostname()

# --- Logging Level Control ---
VERBOSE_LOGGING = False  # Set to True for detailed debug output

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

# ==============================================================================
# Chunking Classes and Functions
# ==============================================================================

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

class Stage3Chunker:
    """Main chunking processor for Stage 3"""
    
    def __init__(self, min_tokens: int = MIN_TOKENS, max_tokens: int = MAX_TOKENS, 
                 hard_max: int = HARD_MAX_TOKENS):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.hard_max = hard_max
        self.tokenizer = SimpleTokenizer()
        
    def process_sections(self, sections: List[Dict]) -> List[Dict]:
        """Process all sections and return chunked records"""
        all_chunks = []
        
        # Apply sample limit if configured
        if SAMPLE_CHAPTER_LIMIT:
            # Group by chapter and limit
            chapters_seen = set()
            limited_sections = []
            for section in sections:
                chapter_num = section.get('chapter_number')
                if chapter_num not in chapters_seen:
                    if len(chapters_seen) >= SAMPLE_CHAPTER_LIMIT:
                        break
                    chapters_seen.add(chapter_num)
                limited_sections.append(section)
            sections = limited_sections
            log_progress(f"Limited to first {SAMPLE_CHAPTER_LIMIT} chapters ({len(sections)} sections)")
        
        # Group sections by document and chapter for proper sequencing
        sections_by_chapter = defaultdict(list)
        for section in sections:
            key = (section['document_id'], section['chapter_number'])
            sections_by_chapter[key].append(section)
        
        total_chapters = len(sections_by_chapter)
        chapter_count = 0
        
        # Process each chapter's sections in order
        for (doc_id, chapter_num), chapter_sections in sorted(sections_by_chapter.items()):
            chapter_count += 1
            # Sort sections by section_number
            chapter_sections.sort(key=lambda x: x['section_number'])
            
            log_progress(f"[{chapter_count}/{total_chapters}] Processing Chapter {chapter_num} ({len(chapter_sections)} sections)...")
            
            for section in chapter_sections:
                chunks = self.chunk_section(section)
                all_chunks.extend(chunks)
                
                if VERBOSE_LOGGING:
                    logging.debug(f"  Section {section['section_number']}: {len(chunks)} chunks")
        
        log_progress(f"✅ Created {len(all_chunks)} chunks from {len(sections)} sections")
        return all_chunks
    
    def chunk_section(self, section: Dict) -> List[Dict]:
        """Chunk a single section into appropriately sized pieces"""
        content = section.get('section_content', '')
        if not content:
            logging.warning(f"Empty content for section {section.get('section_number', 'unknown')}")
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
                logging.warning(f"Chunk exceeds hard max: {chunk_tokens} tokens")
            
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

# ==============================================================================
# Main Processing Function
# ==============================================================================

def main():
    """Main entry point for Stage 3 processing with NAS integration"""
    
    # Validate configuration
    if not validate_configuration():
        return 1
    
    # Setup logging
    temp_log_path = setup_logging()
    log_progress("\n" + "=" * 80)
    log_progress("Stage 3: Content Chunking Pipeline with NAS Integration")
    log_progress("=" * 80)
    
    try:
        # Load input data from NAS
        input_path = f"{NAS_INPUT_PATH}/{INPUT_FILENAME}".replace("\\", "/")
        log_progress(f"\n📥 Loading sections from NAS: {input_path}")
        
        input_bytes = read_from_nas(NAS_PARAMS["share"], input_path)
        if input_bytes is None:
            logging.error("Failed to read input file from NAS")
            return 1
        
        sections = json.loads(input_bytes.decode("utf-8"))
        log_progress(f"✅ Loaded {len(sections)} sections")
        
        # Initialize chunker
        chunker = Stage3Chunker()
        
        # Process sections into chunks
        log_progress("\n🔄 Starting chunking process...")
        chunks = chunker.process_sections(sections)
        
        # Save chunks to NAS
        output_path = f"{NAS_OUTPUT_PATH}/{OUTPUT_FILENAME}".replace("\\", "/")
        log_progress(f"\n💾 Saving {len(chunks)} chunks to NAS: {output_path}")
        
        output_json = json.dumps(chunks, indent=2, ensure_ascii=False)
        success = write_to_nas(NAS_PARAMS["share"], output_path, output_json.encode("utf-8"))
        
        if not success:
            logging.error("Failed to write output file to NAS")
            return 1
        
        log_progress(f"✅ Chunks saved successfully")
        
        # Print statistics
        log_progress("\n" + "=" * 60)
        log_progress("📊 Chunking Statistics")
        log_progress("=" * 60)
        log_progress(f"Total sections processed: {len(sections)}")
        log_progress(f"Total chunks created: {len(chunks)}")
        
        if chunks:
            # Count chunks per section
            chunks_per_section = defaultdict(int)
            for chunk in chunks:
                key = (chunk['chapter_number'], chunk['section_number'])
                chunks_per_section[key] += 1
            
            avg_chunks = sum(chunks_per_section.values()) / len(chunks_per_section)
            log_progress(f"Average chunks per section: {avg_chunks:.1f}")
            log_progress(f"Max chunks in a section: {max(chunks_per_section.values())}")
            log_progress(f"Sections with 1 chunk: {sum(1 for v in chunks_per_section.values() if v == 1)}")
        
        # Upload log file to NAS
        log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"stage3_chunking_{log_timestamp}.log"
        log_nas_path = f"{NAS_LOG_PATH}/{log_filename}".replace("\\", "/")
        
        log_progress(f"\n📤 Uploading log to NAS: {log_nas_path}")
        with open(temp_log_path, "rb") as f:
            log_content = f.read()
        
        if write_to_nas(NAS_PARAMS["share"], log_nas_path, log_content):
            log_progress("✅ Log uploaded successfully")
        else:
            log_progress("⚠️ Failed to upload log file")
        
        log_progress("\n" + "=" * 60)
        log_progress("✅ Stage 3 Processing Complete!")
        log_progress("=" * 60)
        
        return 0
        
    except Exception as e:
        logging.error(f"Fatal error in main: {e}", exc_info=True)
        return 1
    finally:
        # Clean up temp log file
        try:
            os.unlink(temp_log_path)
        except:
            pass

if __name__ == "__main__":
    exit(main())