# -*- coding: utf-8 -*-
"""
Stage 3: Hierarchical Chunking with Page-Aware Storage
Creates searchable chunks from Stage 2 output while maintaining page reconstruction capability

Purpose:
- Takes enriched page records from Stage 2
- Creates 400-500 token chunks within sections
- Tracks which pages each chunk spans
- Stores chunk content split by page boundaries for reconstruction
- Generates embeddings for similarity search
- Outputs flat table suitable for vector database

Input: stage2_enriched_pages.json from Stage 2
Output: stage3_chunks.json with complete chunk records
"""

import os
import json
import re
import time
import logging
import tempfile
import socket
import io
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from datetime import datetime
import hashlib

# --- pysmb imports for NAS access ---
from smb.SMBConnection import SMBConnection
from smb import smb_structs

# --- Dependencies Check ---
try:
    from openai import OpenAI, APIError
except ImportError:
    OpenAI = None
    APIError = None
    print("ERROR: openai library not installed. Embedding features unavailable. `pip install openai`")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x
    print("INFO: tqdm not installed. Progress bars disabled. `pip install tqdm`")

# ==============================================================================
# Configuration
# ==============================================================================

# --- NAS Configuration ---
NAS_PARAMS = {
    "ip": "your_nas_ip",  # TODO: Replace with actual NAS IP
    "share": "your_share_name",  # TODO: Replace with actual share name
    "user": "your_nas_user",  # TODO: Replace with actual NAS username
    "password": "your_nas_password",  # TODO: Replace with actual NAS password
    "port": 445
}

# --- Directory Paths ---
NAS_INPUT_PATH = "semantic_search/pipeline_output/stage2"
INPUT_FILENAME = "stage2_enriched_pages.json"
NAS_OUTPUT_PATH = "semantic_search/pipeline_output/stage3"
NAS_LOG_PATH = "semantic_search/pipeline_output/logs"
OUTPUT_FILENAME = "stage3_chunks.json"

# --- CA Bundle Configuration ---
NAS_SSL_CERT_PATH = "certificates/rbc-ca-bundle.cer"
SSL_LOCAL_PATH = "/tmp/rbc-ca-bundle.cer"

# --- API Configuration ---
BASE_URL = "https://api.example.com/v1"  # TODO: Replace with actual API base URL
EMBEDDING_MODEL = "text-embedding-3-small"  # Embedding model to use
OAUTH_URL = "https://api.example.com/oauth/token"  # TODO: Replace with actual OAuth URL
CLIENT_ID = "your_client_id"  # TODO: Replace with actual client ID
CLIENT_SECRET = "your_client_secret"  # TODO: Replace with actual client secret

# --- Chunking Parameters ---
TARGET_CHUNK_TOKENS = 450  # Target size for chunks
MIN_CHUNK_TOKENS = 400  # Minimum chunk size
MAX_CHUNK_TOKENS = 500  # Maximum chunk size
ULTRA_SMALL_THRESHOLD = 50  # Chunks smaller than this get merged aggressively

# --- Embedding Parameters ---
EMBEDDING_BATCH_SIZE = 100  # Number of texts to embed in one API call
EMBEDDING_DIMENSIONS = 1536  # Dimensions for text-embedding-3-small

# --- API Parameters ---
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY = 5

# --- Token Cost ---
EMBEDDING_TOKEN_COST = 0.00002  # Cost per 1K tokens for embeddings

# --- pysmb Configuration ---
smb_structs.SUPPORT_SMB2 = True
smb_structs.MAX_PAYLOAD_SIZE = 65536
CLIENT_HOSTNAME = socket.gethostname()

# --- Logging Level Control ---
VERBOSE_LOGGING = False

# ==============================================================================
# Configuration Validation
# ==============================================================================

def validate_configuration():
    """Validates that configuration values have been properly set."""
    errors = []
    
    if "your_nas_ip" in NAS_PARAMS["ip"]:
        errors.append("NAS IP address not configured")
    if "your_share_name" in NAS_PARAMS["share"]:
        errors.append("NAS share name not configured")
    if "your_nas_user" in NAS_PARAMS["user"]:
        errors.append("NAS username not configured")
    if "your_nas_password" in NAS_PARAMS["password"]:
        errors.append("NAS password not configured")
    if "api.example.com" in BASE_URL:
        errors.append("API base URL not configured")
    if "api.example.com" in OAUTH_URL:
        errors.append("OAuth URL not configured")
    if "your_client_id" in CLIENT_ID:
        errors.append("Client ID not configured")
    if "your_client_secret" in CLIENT_SECRET:
        errors.append("Client secret not configured")
    
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
            is_direct_tcp=(NAS_PARAMS["port"] == 445)
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
    
    path_parts = dir_path_relative.strip('/').split('/')
    current_path = ''
    try:
        for part in path_parts:
            if not part: continue
            current_path = os.path.join(current_path, part).replace('\\', '/')
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

        dir_path = os.path.dirname(nas_path_relative).replace('\\', '/')
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
        file_attributes, filesize = conn.retrieveFile(share_name, nas_path_relative, file_obj)
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
            except:
                pass
        if conn:
            conn.close()

# ==============================================================================
# Logging Setup
# ==============================================================================

def setup_logging():
    """Setup logging with controlled verbosity."""
    temp_log = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log')
    temp_log_path = temp_log.name
    temp_log.close()
    
    logging.root.handlers = []
    
    log_level = logging.DEBUG if VERBOSE_LOGGING else logging.WARNING
    
    root_file_handler = logging.FileHandler(temp_log_path)
    root_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[root_file_handler]
    )
    
    progress_logger = logging.getLogger('progress')
    progress_logger.setLevel(logging.INFO)
    progress_logger.propagate = False
    
    progress_console_handler = logging.StreamHandler()
    progress_console_handler.setFormatter(logging.Formatter('%(message)s'))
    progress_logger.addHandler(progress_console_handler)
    
    progress_file_handler = logging.FileHandler(temp_log_path)
    progress_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    progress_logger.addHandler(progress_file_handler)
    
    if VERBOSE_LOGGING:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        console_handler.setLevel(logging.WARNING)
        logging.root.addHandler(console_handler)
    
    return temp_log_path

def log_progress(message, end='\n'):
    """Log a progress message that always shows."""
    progress_logger = logging.getLogger('progress')
    
    if end == '':
        sys.stdout.write(message)
        sys.stdout.flush()
        for handler in progress_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.stream.write(f"{datetime.now().isoformat()} - {message}\n")
                handler.flush()
    else:
        progress_logger.info(message)

# ==============================================================================
# Token Counting
# ==============================================================================

def count_tokens(text: str) -> int:
    """Estimates token count using empirically-calibrated formula."""
    if not text:
        return 0
    
    char_count = len(text)
    estimated_tokens = int(char_count / 3.5)
    
    MIN_CHARS_PER_TOKEN = 2
    MAX_CHARS_PER_TOKEN = 10
    
    max_possible_tokens = char_count // MIN_CHARS_PER_TOKEN
    min_possible_tokens = char_count // MAX_CHARS_PER_TOKEN
    
    estimated_tokens = max(min_possible_tokens, min(estimated_tokens, max_possible_tokens))
    
    return estimated_tokens

# ==============================================================================
# API Client Setup
# ==============================================================================

_SSL_CONFIGURED = False
_OPENAI_CLIENT = None

def _setup_ssl_from_nas() -> bool:
    """Downloads SSL cert from NAS and sets environment variables."""
    global _SSL_CONFIGURED
    if _SSL_CONFIGURED:
        return True
    
    try:
        cert_bytes = read_from_nas(NAS_PARAMS["share"], NAS_SSL_CERT_PATH)
        if cert_bytes is None:
            logging.warning("SSL certificate not found on NAS, continuing without it")
            _SSL_CONFIGURED = True
            return True
        
        local_cert = Path(SSL_LOCAL_PATH)
        local_cert.parent.mkdir(parents=True, exist_ok=True)
        with open(local_cert, "wb") as f:
            f.write(cert_bytes)
        
        os.environ["SSL_CERT_FILE"] = str(local_cert)
        os.environ["REQUESTS_CA_BUNDLE"] = str(local_cert)
        _SSL_CONFIGURED = True
        return True
    except Exception as e:
        logging.error(f"Error setting up SSL: {e}")
        return False

def _get_oauth_token(oauth_url=OAUTH_URL, client_id=CLIENT_ID, client_secret=CLIENT_SECRET, ssl_verify_path=SSL_LOCAL_PATH) -> Optional[str]:
    """Retrieves OAuth token."""
    try:
        verify_path = ssl_verify_path if ssl_verify_path and Path(ssl_verify_path).exists() else True
    except (TypeError, OSError):
        verify_path = True

    payload = {'grant_type': 'client_credentials', 'client_id': client_id, 'client_secret': client_secret}
    try:
        import requests
        response = requests.post(oauth_url, data=payload, timeout=30, verify=verify_path)
        response.raise_for_status()
        token_data = response.json()
        oauth_token = token_data.get('access_token')
        if not oauth_token:
            logging.error("No access token in OAuth response")
            return None
        return oauth_token
    except Exception as e:
        logging.error(f"OAuth token request failed: {e}")
        return None

def get_openai_client(base_url=BASE_URL) -> Optional[OpenAI]:
    """Initializes and returns the OpenAI client."""
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT:
        return _OPENAI_CLIENT
    if not OpenAI:
        return None
    if not _setup_ssl_from_nas():
        pass

    api_key = _get_oauth_token()
    if not api_key:
        return None
    try:
        _OPENAI_CLIENT = OpenAI(api_key=api_key, base_url=base_url)
        return _OPENAI_CLIENT
    except Exception as e:
        logging.error(f"Failed to create OpenAI client: {e}")
        return None

# ==============================================================================
# Enhanced Position Mapping for Page-Aware Chunking
# ==============================================================================

def build_enhanced_position_map(pages: List[Dict]) -> Tuple[str, Dict, Dict]:
    """
    Builds enhanced position mapping that tracks exact character positions.
    
    Returns:
        Tuple of (concatenated_content, position_map, page_content_map)
    """
    concatenated_content = ""
    position_map = {}  # page_num -> {start, end}
    page_content_map = {}  # page_num -> original content
    
    for page in pages:
        page_num = page.get('page_number')
        page_content = page.get('content', '')
        
        start_pos = len(concatenated_content)
        concatenated_content += page_content
        
        # Add separator between pages (but not after last page)
        if page != pages[-1]:
            concatenated_content += "\n\n"
        
        end_pos = len(concatenated_content) - (2 if page != pages[-1] else 0)
        
        position_map[page_num] = {
            'start': start_pos,
            'end': end_pos,
            'content_length': len(page_content)
        }
        
        page_content_map[page_num] = page_content
    
    return concatenated_content, position_map, page_content_map

def map_position_to_pages(start_pos: int, end_pos: int, position_map: Dict) -> List[int]:
    """
    Maps character positions to page numbers.
    Returns list of page numbers that the position range spans.
    """
    pages_spanned = []
    
    for page_num, page_info in position_map.items():
        page_start = page_info['start']
        page_end = page_info['end']
        
        # Check if there's any overlap
        if start_pos < page_end and end_pos > page_start:
            pages_spanned.append(page_num)
    
    return sorted(pages_spanned)

def split_content_by_pages(content: str, start_pos: int, end_pos: int, 
                          position_map: Dict, full_content: str) -> Dict[int, str]:
    """
    Splits content by page boundaries.
    Returns dictionary mapping page numbers to content on that page.
    """
    content_by_page = {}
    
    for page_num in map_position_to_pages(start_pos, end_pos, position_map):
        page_info = position_map[page_num]
        page_start = page_info['start']
        page_end = page_info['end']
        
        # Calculate overlap
        overlap_start = max(start_pos, page_start)
        overlap_end = min(end_pos, page_end)
        
        # Extract the overlapping content
        page_content = full_content[overlap_start:overlap_end]
        
        if page_content.strip():  # Only add non-empty content
            content_by_page[page_num] = page_content
    
    return content_by_page

# ==============================================================================
# Smart Chunking Functions
# ==============================================================================

def identify_break_points(content: str) -> List[Dict]:
    """
    Identifies natural break points in content for chunking.
    Returns list of break points with priorities.
    """
    break_points = []
    
    # Headers (highest priority - always break)
    header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    for match in header_pattern.finditer(content):
        break_points.append({
            'position': match.start(),
            'type': 'header',
            'level': len(match.group(1)),
            'priority': 10,
            'force_break': True
        })
    
    # Paragraph boundaries
    paragraph_pattern = re.compile(r'\n\s*\n')
    for match in paragraph_pattern.finditer(content):
        break_points.append({
            'position': match.end(),
            'type': 'paragraph',
            'priority': 7,
            'force_break': False
        })
    
    # Sentence boundaries
    sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    for match in sentence_pattern.finditer(content):
        break_points.append({
            'position': match.start(),
            'type': 'sentence',
            'priority': 5,
            'force_break': False
        })
    
    # Sort by position
    break_points.sort(key=lambda x: x['position'])
    
    return break_points

def create_chunks_from_section(section_content: str, section_id: str, 
                              target_tokens: int = TARGET_CHUNK_TOKENS) -> List[Dict]:
    """
    Creates chunks from section content using smart breaking.
    """
    chunks = []
    break_points = identify_break_points(section_content)
    
    current_start = 0
    chunk_sequence = 1
    
    while current_start < len(section_content):
        # Find optimal chunk end
        chunk_end = find_optimal_chunk_end(
            section_content, current_start, break_points, 
            target_tokens, MAX_CHUNK_TOKENS
        )
        
        chunk_content = section_content[current_start:chunk_end].strip()
        
        if chunk_content:
            chunk_tokens = count_tokens(chunk_content)
            
            # Check if we should merge with previous chunk
            if chunks and chunk_tokens < MIN_CHUNK_TOKENS:
                prev_chunk = chunks[-1]
                combined_tokens = prev_chunk['token_count'] + chunk_tokens
                
                if combined_tokens <= MAX_CHUNK_TOKENS:
                    # Merge with previous
                    prev_chunk['content'] += "\n\n" + chunk_content
                    prev_chunk['end_pos'] = chunk_end
                    prev_chunk['token_count'] = combined_tokens
                    current_start = chunk_end
                    continue
            
            # Create new chunk
            chunks.append({
                'chunk_id': f"{section_id}_c{chunk_sequence}",
                'section_id': section_id,
                'chunk_sequence': chunk_sequence,
                'content': chunk_content,
                'start_pos': current_start,
                'end_pos': chunk_end,
                'token_count': chunk_tokens
            })
            
            chunk_sequence += 1
        
        current_start = chunk_end
    
    return chunks

def find_optimal_chunk_end(content: str, start_pos: int, break_points: List[Dict],
                          target_tokens: int, max_tokens: int) -> int:
    """
    Finds the optimal end position for a chunk.
    """
    # Estimate character positions based on token targets
    chars_per_token = 3.5
    target_chars = int(target_tokens * chars_per_token)
    max_chars = int(max_tokens * chars_per_token)
    
    ideal_end = start_pos + target_chars
    absolute_max = start_pos + max_chars
    
    # Must break at headers
    for bp in break_points:
        if bp['position'] > start_pos and bp['position'] <= ideal_end and bp['force_break']:
            return bp['position']
    
    # Find best break point near ideal position
    best_break = None
    min_distance = float('inf')
    
    for bp in break_points:
        if bp['position'] <= start_pos:
            continue
        if bp['position'] > absolute_max:
            break
            
        distance = abs(bp['position'] - ideal_end)
        if distance < min_distance:
            min_distance = distance
            best_break = bp
    
    if best_break:
        return best_break['position']
    
    # Fallback to character limit
    return min(ideal_end, len(content))

# ==============================================================================
# Chunk Processing Pipeline
# ==============================================================================

def process_chapter_chunks(chapter_num: int, pages: List[Dict]) -> List[Dict]:
    """
    Processes all pages in a chapter to create chunks.
    """
    if not pages:
        return []
    
    # Build enhanced position map
    concatenated_content, position_map, page_content_map = build_enhanced_position_map(pages)
    
    # Group pages by sections
    sections = defaultdict(list)
    for page in pages:
        # Use section summary as a proxy for section identification
        section_key = (page.get('section_summary', ''), page.get('section_page_start'), page.get('section_page_end'))
        if section_key[0]:  # Only if section exists
            sections[section_key].append(page)
    
    all_chunks = []
    
    # Process each section
    for section_key, section_pages in sections.items():
        section_summary, section_start, section_end = section_key
        
        # Create section identifier
        section_id = f"ch{chapter_num}_s{len(all_chunks) + 1}"
        
        # Get section content from concatenated content
        if section_pages:
            first_page_num = min(p['page_number'] for p in section_pages)
            last_page_num = max(p['page_number'] for p in section_pages)
            
            section_start_pos = position_map[first_page_num]['start']
            section_end_pos = position_map[last_page_num]['end']
            
            section_content = concatenated_content[section_start_pos:section_end_pos]
            
            # Create chunks for this section
            section_chunks = create_chunks_from_section(section_content, section_id)
            
            # Add page-aware content splitting
            for chunk in section_chunks:
                # Map chunk to global positions
                global_start = section_start_pos + chunk['start_pos']
                global_end = section_start_pos + chunk['end_pos']
                
                # Split content by pages
                chunk['chunk_content_by_page'] = split_content_by_pages(
                    chunk['content'], global_start, global_end, 
                    position_map, concatenated_content
                )
                
                # Track which pages this chunk spans
                chunk['chunk_pages'] = sorted(chunk['chunk_content_by_page'].keys())
                
                # Add metadata from first page
                first_page = section_pages[0]
                chunk['chapter_number'] = chapter_num
                chunk['chapter_name'] = first_page.get('chapter_name')
                chunk['chapter_summary'] = first_page.get('chapter_summary')
                chunk['section_summary'] = section_summary
                chunk['section_page_start'] = section_start
                chunk['section_page_end'] = section_end
                chunk['section_references'] = first_page.get('section_references', [])
                chunk['document_id'] = first_page.get('document_id')
                chunk['filename'] = first_page.get('filename')
                
                all_chunks.append(chunk)
    
    # Handle pages without sections
    no_section_pages = [p for p in pages if not p.get('section_summary')]
    if no_section_pages:
        # Create chunks for unsectioned content
        for page in no_section_pages:
            page_num = page['page_number']
            page_content = page['content']
            
            if page_content and count_tokens(page_content) > 0:
                chunk_id = f"ch{chapter_num}_unsectioned_p{page_num}"
                
                all_chunks.append({
                    'chunk_id': chunk_id,
                    'section_id': None,
                    'chunk_sequence': page_num,
                    'content': page_content,
                    'chunk_content_by_page': {page_num: page_content},
                    'chunk_pages': [page_num],
                    'token_count': count_tokens(page_content),
                    'chapter_number': chapter_num,
                    'chapter_name': page.get('chapter_name'),
                    'chapter_summary': page.get('chapter_summary'),
                    'section_summary': None,
                    'section_page_start': page_num,
                    'section_page_end': page_num,
                    'section_references': [],
                    'document_id': page.get('document_id'),
                    'filename': page.get('filename')
                })
    
    return all_chunks

# ==============================================================================
# Embedding Generation
# ==============================================================================

def generate_embeddings_batch(texts: List[str], client: Optional[OpenAI]) -> List[List[float]]:
    """
    Generates embeddings for a batch of texts.
    Returns list of embedding vectors.
    """
    if not client:
        # Return empty embeddings if no client
        return [[0.0] * EMBEDDING_DIMENSIONS for _ in texts]
    
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
            dimensions=EMBEDDING_DIMENSIONS
        )
        
        embeddings = [item.embedding for item in response.data]
        return embeddings
        
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        # Return zero vectors on error
        return [[0.0] * EMBEDDING_DIMENSIONS for _ in texts]

def add_embeddings_to_chunks(chunks: List[Dict], client: Optional[OpenAI]) -> List[Dict]:
    """
    Adds embeddings to chunks in batches.
    """
    log_progress(f"  🔤 Generating embeddings for {len(chunks)} chunks...")
    
    # Process in batches
    for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[i:i + EMBEDDING_BATCH_SIZE]
        texts = [chunk['content'] for chunk in batch]
        
        embeddings = generate_embeddings_batch(texts, client)
        
        for chunk, embedding in zip(batch, embeddings):
            chunk['embedding'] = embedding
    
    log_progress(f"  ✅ Embeddings generated")
    return chunks

# ==============================================================================
# Main Processing
# ==============================================================================

def group_pages_by_chapter(page_records: List[Dict]) -> Tuple[Dict[int, List[Dict]], List[Dict]]:
    """Groups pages by chapter number."""
    chapters = defaultdict(list)
    unassigned = []
    
    for record in page_records:
        chapter_num = record.get('chapter_number')
        if chapter_num is not None:
            chapters[chapter_num].append(record)
        else:
            unassigned.append(record)
    
    # Sort pages within each chapter
    for chapter_num in chapters:
        chapters[chapter_num].sort(key=lambda x: x.get('page_number', 0))
    
    return dict(chapters), unassigned

def cleanup_logging_handlers():
    """Safely cleanup logging handlers."""
    progress_logger = logging.getLogger('progress')
    handlers_to_remove = list(progress_logger.handlers)
    for handler in handlers_to_remove:
        try:
            handler.flush()
            handler.close()
        except:
            pass
        try:
            progress_logger.removeHandler(handler)
        except:
            pass
    
    root_handlers_to_remove = list(logging.root.handlers)
    for handler in root_handlers_to_remove:
        try:
            handler.flush()
            handler.close()
        except:
            pass
        try:
            logging.root.removeHandler(handler)
        except:
            pass
    
    progress_logger.handlers.clear()
    logging.root.handlers.clear()

def create_final_chunk_record(chunk: Dict) -> Dict:
    """
    Creates the final chunk record with all required fields for the flat table.
    """
    return {
        # Chunk Identity
        'chunk_id': chunk['chunk_id'],
        'chunk_content': chunk['content'],
        'chunk_embedding': chunk.get('embedding', []),
        'chunk_sequence': chunk['chunk_sequence'],
        'chunk_token_count': chunk['token_count'],
        
        # Section Context
        'section_id': chunk.get('section_id'),
        'section_summary': chunk.get('section_summary'),
        'section_page_start': chunk.get('section_page_start'),
        'section_page_end': chunk.get('section_page_end'),
        
        # Chapter Context
        'chapter_number': chunk.get('chapter_number'),
        'chapter_name': chunk.get('chapter_name'),
        'chapter_summary': chunk.get('chapter_summary'),
        
        # Page Mapping
        'chunk_content_by_page': chunk.get('chunk_content_by_page', {}),
        'chunk_pages': chunk.get('chunk_pages', []),
        
        # Cross-references
        'section_references': chunk.get('section_references', []),
        
        # Document Metadata
        'document_id': chunk.get('document_id'),
        'filename': chunk.get('filename')
    }

def run_stage3():
    """Main function to execute Stage 3 processing."""
    if not validate_configuration():
        return
    
    temp_log_path = setup_logging()
    
    log_progress("=" * 70)
    log_progress("🚀 Starting Stage 3: Hierarchical Chunking with Page-Aware Storage")
    log_progress("=" * 70)
    
    _setup_ssl_from_nas()
    
    share_name = NAS_PARAMS["share"]
    input_path = os.path.join(NAS_INPUT_PATH, INPUT_FILENAME).replace('\\', '/')
    output_path = os.path.join(NAS_OUTPUT_PATH, OUTPUT_FILENAME).replace('\\', '/')
    
    # Load input JSON
    log_progress("📥 Loading Stage 2 output from NAS...")
    input_json_bytes = read_from_nas(share_name, input_path)
    
    if not input_json_bytes:
        log_progress("❌ Failed to read input JSON")
        return
    
    try:
        page_records = json.loads(input_json_bytes.decode('utf-8'))
        if not isinstance(page_records, list):
            log_progress("❌ Input JSON is not a list")
            return
        log_progress(f"✅ Loaded {len(page_records)} page records")
    except json.JSONDecodeError as e:
        log_progress(f"❌ Error decoding JSON: {e}")
        return
    
    # Initialize OpenAI client for embeddings
    client = None
    if OpenAI:
        client = get_openai_client()
        if client:
            log_progress("✅ OpenAI client initialized for embeddings")
        else:
            log_progress("⚠️ Failed to initialize OpenAI client - continuing without embeddings")
    else:
        log_progress("⚠️ OpenAI library not installed - continuing without embeddings")
    
    # Group pages by chapter
    chapters, unassigned_pages = group_pages_by_chapter(page_records)
    log_progress(f"📊 Found {len(chapters)} chapters and {len(unassigned_pages)} unassigned pages")
    log_progress("-" * 70)
    
    # Process each chapter
    all_chunks = []
    
    for chapter_num in sorted(chapters.keys()):
        pages = chapters[chapter_num]
        chapter_name = pages[0].get('chapter_name', f'Chapter {chapter_num}')
        
        log_progress(f"\n📚 Processing Chapter {chapter_num}: {chapter_name}")
        log_progress(f"  📄 {len(pages)} pages to process")
        
        # Create chunks for this chapter
        chapter_chunks = process_chapter_chunks(chapter_num, pages)
        log_progress(f"  ✂️ Created {len(chapter_chunks)} chunks")
        
        # Add embeddings
        if client and chapter_chunks:
            chapter_chunks = add_embeddings_to_chunks(chapter_chunks, client)
        
        all_chunks.extend(chapter_chunks)
    
    # Process unassigned pages
    if unassigned_pages:
        log_progress(f"\n📄 Processing {len(unassigned_pages)} unassigned pages")
        # Create simple chunks for unassigned pages
        for page in unassigned_pages:
            if page.get('content'):
                chunk = {
                    'chunk_id': f"unassigned_p{page['page_number']}",
                    'content': page['content'],
                    'chunk_content_by_page': {page['page_number']: page['content']},
                    'chunk_pages': [page['page_number']],
                    'token_count': count_tokens(page['content']),
                    'chunk_sequence': page['page_number'],
                    'section_id': None,
                    'chapter_number': None,
                    'chapter_name': None,
                    'chapter_summary': None,
                    'section_summary': None,
                    'section_page_start': page['page_number'],
                    'section_page_end': page['page_number'],
                    'section_references': [],
                    'document_id': page.get('document_id'),
                    'filename': page.get('filename')
                }
                
                if client:
                    embeddings = generate_embeddings_batch([chunk['content']], client)
                    chunk['embedding'] = embeddings[0] if embeddings else []
                
                all_chunks.append(chunk)
    
    # Create final records
    log_progress("\n📦 Creating final chunk records...")
    final_chunks = [create_final_chunk_record(chunk) for chunk in all_chunks]
    
    # Calculate statistics
    total_tokens = sum(chunk['chunk_token_count'] for chunk in final_chunks)
    avg_tokens = total_tokens / len(final_chunks) if final_chunks else 0
    
    # Save output
    log_progress("-" * 70)
    log_progress(f"💾 Saving {len(final_chunks)} chunk records...")
    
    try:
        output_json = json.dumps(final_chunks, indent=2, ensure_ascii=False)
        output_bytes = output_json.encode('utf-8')
        
        if write_to_nas(share_name, output_path, output_bytes):
            log_progress(f"✅ Successfully saved output to {share_name}/{output_path}")
        else:
            log_progress("❌ Failed to write output to NAS")
    except Exception as e:
        log_progress(f"❌ Error saving output: {e}")
    
    # Upload log file
    try:
        log_file_name = f"stage3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path_relative = os.path.join(NAS_LOG_PATH, log_file_name).replace('\\', '/')
        
        cleanup_logging_handlers()
        
        with open(temp_log_path, 'rb') as f:
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
    print("📊 Stage 3 Summary")
    print("-" * 70)
    print(f"  Input: {len(page_records)} page records")
    print(f"  Output: {len(final_chunks)} chunks")
    print(f"  Average chunk size: {avg_tokens:.0f} tokens")
    print(f"  Total tokens: {total_tokens:,}")
    if client:
        embedding_cost = (total_tokens / 1000) * EMBEDDING_TOKEN_COST
        print(f"  Estimated embedding cost: ${embedding_cost:.4f}")
    print(f"  Output file: {share_name}/{output_path}")
    print("=" * 70)
    print("✅ Stage 3 Completed")

if __name__ == "__main__":
    run_stage3()