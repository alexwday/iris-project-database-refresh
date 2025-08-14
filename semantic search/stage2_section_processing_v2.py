# -*- coding: utf-8 -*-
"""
Stage 2: Section Processing with Page-Level Output - VERSION 2
Maintains one record per page while enriching with section-level information

Purpose:
Processes page-level data from Stage 1, identifies sections within chapters,
generates section summaries, and maps them back to individual pages.
Each page record is enriched with relevant section summaries while maintaining
the page as the atomic unit for database storage.

Input: JSON file from Stage 1 output (stage1_page_records.json)
Output: JSON file with page-level records enriched with section data (stage2_enriched_pages.json)
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
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from datetime import datetime

# --- pysmb imports for NAS access ---
from smb.SMBConnection import SMBConnection
from smb import smb_structs

# --- Dependencies Check ---
try:
    from openai import OpenAI, APIError
except ImportError:
    OpenAI = None
    APIError = None
    print("ERROR: openai library not installed. GPT features unavailable. `pip install openai`")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x
    print("INFO: tqdm not installed. Progress bars disabled. `pip install tqdm`")

# ==============================================================================
# Configuration (Hardcoded - update these values)
# ==============================================================================

# --- NAS Configuration ---
NAS_PARAMS = {
    "ip": "your_nas_ip",  # TODO: Replace with actual NAS IP
    "share": "your_share_name",  # TODO: Replace with actual share name
    "user": "your_nas_user",  # TODO: Replace with actual NAS username
    "password": "your_nas_password",  # TODO: Replace with actual NAS password
    "port": 445  # Default SMB port (can be 139)
}

# --- Directory Paths (Relative to NAS Share) ---
NAS_INPUT_PATH = "semantic_search/pipeline_output/stage1"
INPUT_FILENAME = "stage1_page_records.json"
NAS_OUTPUT_PATH = "semantic_search/pipeline_output/stage2"
NAS_LOG_PATH = "semantic_search/pipeline_output/logs"
OUTPUT_FILENAME = "stage2_enriched_pages.json"

# --- CA Bundle Configuration ---
NAS_SSL_CERT_PATH = "certificates/rbc-ca-bundle.cer"
SSL_LOCAL_PATH = "/tmp/rbc-ca-bundle.cer"

# --- API Configuration ---
BASE_URL = "https://api.example.com/v1"  # TODO: Replace with actual API base URL
MODEL_NAME_CHAT = "gpt-4-turbo-nonp"  # TODO: Replace with actual model name
OAUTH_URL = "https://api.example.com/oauth/token"  # TODO: Replace with actual OAuth URL
CLIENT_ID = "your_client_id"  # TODO: Replace with actual client ID
CLIENT_SECRET = "your_client_secret"  # TODO: Replace with actual client secret

# --- API Parameters ---
GPT_INPUT_TOKEN_LIMIT = 80000
MAX_COMPLETION_TOKENS = 4000
TEMPERATURE = 0.3
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY = 5
TOKEN_BUFFER = 2000

# Tool response validation retries
TOOL_RESPONSE_RETRIES = 5
TOOL_RESPONSE_RETRY_DELAY = 3

# --- Token Cost ---
PROMPT_TOKEN_COST = 0.01
COMPLETION_TOKEN_COST = 0.03

# --- Section Merging Thresholds ---
MIN_SECTION_TOKENS = 250  # Sections below this trigger merging
MAX_SECTION_TOKENS = 750  # Maximum tokens after merging
ULTRA_SMALL_THRESHOLD = 25  # Very small sections get aggressive merging

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
# NAS Helper Functions (from Stage 1)
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
    import requests
    
    try:
        verify_path = ssl_verify_path if ssl_verify_path and Path(ssl_verify_path).exists() else True
    except (TypeError, OSError):
        verify_path = True

    payload = {'grant_type': 'client_credentials', 'client_id': client_id, 'client_secret': client_secret}
    try:
        response = requests.post(oauth_url, data=payload, timeout=30, verify=verify_path)
        response.raise_for_status()
        token_data = response.json()
        oauth_token = token_data.get('access_token')
        if not oauth_token:
            logging.error("No access token in OAuth response")
            return None
        return oauth_token
    except requests.exceptions.RequestException as e:
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
# API Call with Tool Enforcement
# ==============================================================================

def call_gpt_with_tool_enforcement(client, model, messages, max_tokens, temperature, tool_schema):
    """Makes API call with STRICT tool enforcement."""
    tool_name = tool_schema["function"]["name"]
    
    if not messages:
        logging.error("Messages list is empty")
        return None, None
    
    for attempt in range(TOOL_RESPONSE_RETRIES):
        try:
            if attempt > 0:
                logging.info(f"Tool response retry {attempt + 1}/{TOOL_RESPONSE_RETRIES}")
                enforcement_msg = {
                    "role": "system",
                    "content": f"CRITICAL: You MUST use the '{tool_name}' tool to provide your response. Do not respond with plain text."
                }
                enhanced_messages = messages[:-1] + [enforcement_msg] + messages[-1:]
            else:
                enhanced_messages = messages
            
            response = client.chat.completions.create(
                model=model,
                messages=enhanced_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=[tool_schema],
                tool_choice={"type": "function", "function": {"name": tool_name}},
                stream=False
            )
            
            response_message = response.choices[0].message
            usage_info = response.usage
            
            if not response_message.tool_calls:
                logging.warning(f"Attempt {attempt + 1}: No tool calls in response")
                time.sleep(TOOL_RESPONSE_RETRY_DELAY)
                continue
            
            tool_call = response_message.tool_calls[0]
            
            if tool_call.function.name != tool_name:
                logging.warning(f"Attempt {attempt + 1}: Wrong tool used: {tool_call.function.name}")
                time.sleep(TOOL_RESPONSE_RETRY_DELAY)
                continue
            
            try:
                function_args = json.loads(tool_call.function.arguments)
                
                required_fields = tool_schema["function"]["parameters"].get("required", [])
                for field in required_fields:
                    if field not in function_args:
                        logging.warning(f"Attempt {attempt + 1}: Missing required field '{field}'")
                        time.sleep(TOOL_RESPONSE_RETRY_DELAY)
                        break
                else:
                    return function_args, usage_info
                    
            except json.JSONDecodeError as e:
                logging.warning(f"Attempt {attempt + 1}: Invalid JSON in tool arguments: {e}")
                logging.debug(f"Malformed JSON content: {tool_call.function.arguments[:500]}...")
                time.sleep(TOOL_RESPONSE_RETRY_DELAY)
                continue
                
        except APIError as e:
            logging.warning(f"API Error on attempt {attempt + 1}: {e}")
            time.sleep(TOOL_RESPONSE_RETRY_DELAY * (2 ** min(attempt, 3)))
        except Exception as e:
            logging.warning(f"Unexpected error on attempt {attempt + 1}: {e}")
            time.sleep(TOOL_RESPONSE_RETRY_DELAY)
    
    logging.error(f"Failed to get valid tool response after {TOOL_RESPONSE_RETRIES} attempts")
    return None, None

# ==============================================================================
# Page Position Mapping
# ==============================================================================

def build_page_position_map(pages: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    Builds a position map tracking where each page's content appears in concatenated text.
    Returns: (concatenated_content, position_map)
    """
    position_map = []
    content_parts = []
    current_pos = 0
    
    for i, page in enumerate(pages):
        page_content = page.get('content', '')
        
        # Record position even for empty pages
        position_map.append({
            'page_number': page.get('page_number', i+1),
            'pdf_page_number': page.get('pdf_page_number', page.get('page_number', i+1)),
            'source_page_number': page.get('source_page_number'),
            'start_pos': current_pos,
            'end_pos': current_pos + len(page_content),
            'content_length': len(page_content),
            'is_empty': len(page_content.strip()) == 0,
            'original_index': i  # Preserve original array index
        })
        
        content_parts.append(page_content)
        current_pos += len(page_content)
        
        # Add separator between pages (except after last page)
        if i < len(pages) - 1:
            content_parts.append('\n\n')
            current_pos += 2
    
    concatenated_content = ''.join(content_parts)
    return concatenated_content, position_map

# ==============================================================================
# Section Identification
# ==============================================================================

def identify_sections(concatenated_content: str, chapter_metadata: Dict) -> List[Dict]:
    """
    Identifies sections based on markdown headings in the concatenated content.
    """
    sections = []
    
    # Find all markdown headings
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    headings = []
    
    for match in heading_pattern.finditer(concatenated_content):
        headings.append({
            'level': len(match.group(1)),
            'title': match.group(2).strip(),
            'start_pos': match.start()
        })
    
    # Handle content before first heading (if significant)
    first_heading_pos = headings[0]['start_pos'] if headings else len(concatenated_content)
    
    if first_heading_pos > 50:  # Significant intro content
        intro_content = concatenated_content[:first_heading_pos].strip()
        if intro_content:
            sections.append({
                'section_id': f"ch{chapter_metadata.get('chapter_number', 0)}_intro",
                'level': 0,
                'title': f"{chapter_metadata.get('chapter_name', 'Chapter')} - Introduction",
                'start_pos': 0,
                'end_pos': first_heading_pos,
                'hierarchy': [chapter_metadata.get('chapter_name', 'Chapter')]
            })
    
    # Process each heading to create sections
    for i, heading in enumerate(headings):
        next_pos = headings[i+1]['start_pos'] if i+1 < len(headings) else len(concatenated_content)
        
        # Build hierarchy
        hierarchy = [chapter_metadata.get('chapter_name', 'Chapter')]
        
        # Add parent headings to hierarchy
        for j in range(i-1, -1, -1):
            if headings[j]['level'] < heading['level']:
                hierarchy.insert(1, headings[j]['title'])
                break
        
        hierarchy.append(heading['title'])
        
        sections.append({
            'section_id': f"ch{chapter_metadata.get('chapter_number', 0)}_sec{i+1}",
            'level': heading['level'],
            'title': heading['title'],
            'start_pos': heading['start_pos'],
            'end_pos': next_pos,
            'hierarchy': hierarchy
        })
    
    # If no sections found, treat entire chapter as one section
    if not sections and concatenated_content.strip():
        sections.append({
            'section_id': f"ch{chapter_metadata.get('chapter_number', 0)}_full",
            'level': 1,
            'title': chapter_metadata.get('chapter_name', 'Chapter Content'),
            'start_pos': 0,
            'end_pos': len(concatenated_content),
            'hierarchy': [chapter_metadata.get('chapter_name', 'Chapter')]
        })
    
    return sections

# ==============================================================================
# Section Merging
# ==============================================================================

def merge_small_sections(sections: List[Dict], concatenated_content: str) -> List[Dict]:
    """
    Merges small sections to optimize for GPT processing.
    """
    if not sections:
        return []
    
    # Calculate token counts for each section
    for section in sections:
        section_content = concatenated_content[section['start_pos']:section['end_pos']]
        section['token_count'] = count_tokens(section_content)
    
    merged = []
    i = 0
    
    while i < len(sections):
        current = sections[i]
        
        if current['token_count'] >= MIN_SECTION_TOKENS:
            merged.append(current)
            i += 1
        else:
            # Try to merge with next section if same level
            if i + 1 < len(sections):
                next_sec = sections[i + 1]
                combined_tokens = current['token_count'] + next_sec['token_count']
                
                if current['level'] == next_sec['level'] and combined_tokens <= MAX_SECTION_TOKENS:
                    # Merge sections
                    merged_section = {
                        'section_id': f"{current['section_id']}_merged",
                        'level': current['level'],
                        'title': f"{current['title']} & {next_sec['title']}",
                        'start_pos': current['start_pos'],
                        'end_pos': next_sec['end_pos'],
                        'hierarchy': current['hierarchy'],
                        'token_count': combined_tokens,
                        'is_merged': True
                    }
                    merged.append(merged_section)
                    i += 2
                else:
                    # Can't merge, keep as is
                    merged.append(current)
                    i += 1
            else:
                # Last section, can't merge forward
                merged.append(current)
                i += 1
    
    # Handle ultra-small sections
    final_merged = []
    for section in merged:
        if section['token_count'] < ULTRA_SMALL_THRESHOLD and final_merged:
            # Merge with previous section
            prev = final_merged[-1]
            prev['end_pos'] = section['end_pos']
            prev['token_count'] += section['token_count']
            if not prev.get('is_merged'):
                prev['title'] = f"{prev['title']} & {section['title']}"
                prev['is_merged'] = True
        else:
            final_merged.append(section)
    
    return final_merged

# ==============================================================================
# Section to Page Mapping
# ==============================================================================

def map_sections_to_pages(sections: List[Dict], page_position_map: List[Dict]) -> Dict[int, List[Dict]]:
    """
    Maps sections to pages based on position overlap.
    Returns a dictionary: page_index -> list of sections on that page
    """
    page_sections = {}
    
    for page_info in page_position_map:
        page_idx = page_info['original_index']
        page_start = page_info['start_pos']
        page_end = page_info['end_pos']
        
        if page_info['is_empty']:
            page_sections[page_idx] = []
            continue
        
        sections_on_page = []
        
        for section in sections:
            sec_start = section['start_pos']
            sec_end = section['end_pos']
            
            # Check if section overlaps with page
            if sec_start < page_end and sec_end > page_start:
                # Section appears on this page
                section_info = {
                    'section': section,
                    'starts_on_page': sec_start >= page_start and sec_start < page_end,
                    'ends_on_page': sec_end > page_start and sec_end <= page_end,
                    'spans_entire_page': sec_start <= page_start and sec_end >= page_end
                }
                sections_on_page.append(section_info)
        
        page_sections[page_idx] = sections_on_page
    
    return page_sections

# ==============================================================================
# GPT Section Processing with CO-STAR + XML Format
# ==============================================================================

SECTION_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "extract_section_details",
        "description": "Extracts detailed information about a specific document section based on its content and the overall chapter context.",
        "parameters": {
            "type": "object",
            "properties": {
                "section_summary": {
                    "type": "string",
                    "description": "A concise summary (1-3 sentences) capturing the core topic or purpose of this section, suitable for reranking search results."
                },
                "section_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of meaningful keywords or tags specific to this section's content, scaled appropriately to the section's length and complexity. These tags will be used as metadata for search reranking."
                },
                "section_standard": {
                    "type": "string",
                    "description": "The primary accounting or reporting standard applicable to this section (e.g., 'IFRS', 'US GAAP', 'N/A')."
                },
                "section_standard_codes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of specific standard codes explicitly mentioned or directly relevant in the section (e.g., ['IFRS 16', 'IAS 17']). The number of codes should reflect the section's content. These codes will be used as metadata for search reranking."
                },
                "section_importance_score": {
                    "type": "number",
                    "description": "A score between 0.0 (low importance) and 1.0 (high importance) indicating how crucial this section is to understanding the overall chapter's topic. A score of 0.5 indicates average or unknown importance. This float value will be used for search reranking.",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "section_references": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of explicit references to other sections, chapters, or standard codes found within this section's text (e.g., ['See Section 4.5', 'Refer to Chapter 3', 'IAS 36.12']). Provide an empty list [] if none are found. These references provide context."
                }
            },
            "required": ["section_summary", "section_tags", "section_standard", 
                        "section_standard_codes", "section_importance_score", "section_references"],
            "additionalProperties": False
        }
    }
}

def build_section_prompt(section_text: str, section_metadata: Dict, 
                        chapter_context: Dict, previous_section_summaries: List[str] = None) -> List[Dict]:
    """
    Builds the messages list for section processing using CO-STAR + XML format.
    """
    if previous_section_summaries is None:
        previous_section_summaries = []
    
    # System prompt using CO-STAR + XML format
    system_prompt = """<role>You are an expert financial reporting specialist.</role>
<source_material>You are analyzing a specific section within a chapter from an EY technical accounting guidance manual. You are provided with the overall chapter summary/tags and summaries of recently processed sections from the same chapter.</source_material>
<task>Your primary task is to generate a **concise summary (1-3 sentences)** for the current section, suitable for use in reranking search results. Additionally, extract relevant tags, the primary applicable accounting standard, and specific standard codes mentioned. Use the 'extract_section_details' tool for your response.</task>
<guardrails>Base your analysis strictly on the provided section text and context. Focus on capturing the core topic/purpose concisely for the summary. Ensure tags and standard codes are precise and derived from the section text.</guardrails>"""

    # Build user prompt with CO-STAR + XML elements
    user_prompt_elements = ["<prompt>"]
    user_prompt_elements.append("<style>Concise, factual, keyword-focused for summary; technical and precise for other fields.</style>")
    user_prompt_elements.append("<tone>Professional, objective, expert.</tone>")
    user_prompt_elements.append("<audience>Accounting professionals needing specific guidance on this section.</audience>")
    user_prompt_elements.append('<response_format>Use the "extract_section_details" tool.</response_format>')
    
    # Add chapter context
    user_prompt_elements.append("<overall_chapter_context>")
    user_prompt_elements.append(f"<chapter_summary>{chapter_context.get('chapter_summary', 'N/A')}</chapter_summary>")
    user_prompt_elements.append(f"<chapter_tags>{json.dumps(chapter_context.get('chapter_tags', []))}</chapter_tags>")
    user_prompt_elements.append("</overall_chapter_context>")
    
    # Add recent section context if available
    if previous_section_summaries:
        user_prompt_elements.append("<recent_section_context>")
        for i, summary in enumerate(previous_section_summaries[-5:]):  # Last 5 summaries max
            user_prompt_elements.append(f"<previous_section_{i+1}_summary>{summary}</previous_section_{i+1}_summary>")
        user_prompt_elements.append("</recent_section_context>")
    
    # Add section metadata
    user_prompt_elements.append("<section_metadata>")
    user_prompt_elements.append(f"<section_title>{section_metadata.get('title', 'Unknown')}</section_title>")
    user_prompt_elements.append(f"<section_hierarchy>{' > '.join(section_metadata.get('hierarchy', []))}</section_hierarchy>")
    user_prompt_elements.append(f"<section_level>{section_metadata.get('level', 1)}</section_level>")
    user_prompt_elements.append("</section_metadata>")
    
    # Add current section text
    user_prompt_elements.append(f"<current_section_text>{section_text}</current_section_text>")
    
    # Add detailed instructions
    user_prompt_elements.append("<instructions>")
    user_prompt_elements.append("""
    **Analysis Objective:** Analyze the provided <current_section_text> considering the <overall_chapter_context> and <recent_section_context> (if provided).
    **Action:** Generate the following details for the **current section** using the 'extract_section_details' tool:
    1. **section_summary:** A **very concise summary (1-3 sentences)** capturing the core topic or purpose of this section. This summary will be used to help rerank search results, so it should be distinct and informative at a glance.
    2. **section_tags:** Generate a list of meaningful, granular tags specific to THIS SECTION's content. The number of tags should be dynamic and reflect the section's complexity and key topics. These tags are crucial metadata for search reranking.
    3. **section_standard:** Identify the single, primary accounting standard framework most relevant to THIS SECTION (e.g., 'IFRS', 'US GAAP', 'N/A').
    4. **section_standard_codes:** List specific standard codes (e.g., 'IFRS 16', 'IAS 36.12', 'ASC 842-10-15') explicitly mentioned or directly and significantly relevant within THIS SECTION's text. The number of codes should be dynamic, reflecting the section's content. Provide an empty list [] if none are applicable. These codes are crucial metadata for search reranking.
    5. **section_importance_score:** Assign a score between 0.0 (low importance) and 1.0 (high importance) representing how crucial this section's content is for understanding the overall topic of the chapter provided in the <overall_chapter_context>. Consider the section's scope and depth relative to the chapter summary. A score of 0.5 indicates average or unknown importance. Provide a float value (e.g., 0.7). This score will directly influence search result ranking.
    6. **section_references:** List any explicit textual references made within the <current_section_text> to other sections, chapters, paragraphs, or specific standard codes (e.g., "See Section 4.5", "Refer to Chapter 3", "IAS 36.12"). Provide an empty list [] if no explicit references are found.
    """)
    user_prompt_elements.append("</instructions>")
    user_prompt_elements.append("</prompt>")
    
    user_prompt = "\n".join(user_prompt_elements)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    return messages

def process_section_with_gpt(section_content: str, section_metadata: Dict, 
                             chapter_context: Dict, client: OpenAI,
                             previous_section_summaries: List[str] = None) -> Optional[Dict]:
    """
    Processes a single section with GPT to generate summary and metadata.
    Uses CO-STAR + XML prompting format.
    """
    messages = build_section_prompt(
        section_text=section_content,
        section_metadata=section_metadata,
        chapter_context=chapter_context,
        previous_section_summaries=previous_section_summaries
    )
    
    result, usage_info = call_gpt_with_tool_enforcement(
        client=client,
        model=MODEL_NAME_CHAT,
        messages=messages,
        max_tokens=MAX_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        tool_schema=SECTION_TOOL_SCHEMA
    )
    
    if result:
        if usage_info and VERBOSE_LOGGING:
            prompt_tokens = usage_info.prompt_tokens
            completion_tokens = usage_info.completion_tokens
            total_cost = (prompt_tokens / 1000) * PROMPT_TOKEN_COST + (completion_tokens / 1000) * COMPLETION_TOKEN_COST
            logging.debug(f"Section processing cost: ${total_cost:.4f}")
        
        return result
    
    return None

def process_all_sections(sections: List[Dict], concatenated_content: str, 
                        chapter_context: Dict, client: Optional[OpenAI]) -> Dict[str, Dict]:
    """
    Processes all sections and returns a lookup dictionary by section_id.
    Maintains context by passing previous section summaries.
    """
    processed_sections = {}
    previous_summaries = []  # Track summaries for context
    
    if not client:
        log_progress("  ⚠️ No OpenAI client available, skipping section processing")
        return {}
    
    for section in tqdm(sections, desc=f"  Processing sections"):
        section_id = section['section_id']
        section_content = concatenated_content[section['start_pos']:section['end_pos']]
        
        # Clean Azure tags before processing
        section_content = re.sub(r'<!--\s*Page(Footer|Number|Break|Header)=?(".*?"|\d+)?\s*-->\s*\n?', '', section_content)
        
        # Pass last 5 summaries for context
        result = process_section_with_gpt(
            section_content=section_content,
            section_metadata=section,
            chapter_context=chapter_context,
            client=client,
            previous_section_summaries=previous_summaries[-5:] if previous_summaries else None
        )
        
        if result:
            processed_sections[section_id] = {
                'summary': result.get('section_summary', ''),
                'tags': result.get('section_tags', []),
                'standard': result.get('section_standard', 'N/A'),
                'standard_codes': result.get('section_standard_codes', []),
                'importance_score': result.get('section_importance_score', 0.5),
                'references': result.get('section_references', [])  # Added references field
            }
            # Add to context for next sections
            if result.get('section_summary'):
                previous_summaries.append(result['section_summary'])
        else:
            # Fallback for failed processing
            processed_sections[section_id] = {
                'summary': f"Section covering {section['title']}",
                'tags': [],
                'standard': 'N/A',
                'standard_codes': [],
                'importance_score': 0.5,
                'references': []
            }
    
    return processed_sections

# ==============================================================================
# Page Enrichment
# ==============================================================================

def enrich_pages_with_sections(pages: List[Dict], page_sections_map: Dict[int, List[Dict]], 
                               processed_sections: Dict[str, Dict]) -> List[Dict]:
    """
    Enriches each page record with section information.
    """
    enriched_pages = []
    
    for i, page in enumerate(pages):
        # Copy original page data
        enriched_page = page.copy()
        
        # Get sections on this page
        sections_on_page = page_sections_map.get(i, [])
        
        if sections_on_page:
            # Build section details
            section_summaries = []
            section_titles = []
            section_hierarchies = []
            all_section_tags = set()
            importance_scores = []
            all_standard_codes = set()
            all_references = set()
            
            for section_info in sections_on_page:
                section = section_info['section']
                section_id = section['section_id']
                processed = processed_sections.get(section_id, {})
                
                # Add section summary
                section_summaries.append(processed.get('summary', ''))
                section_titles.append(section['title'])
                section_hierarchies.append(' > '.join(section['hierarchy']))
                
                # Collect tags, codes, and references
                all_section_tags.update(processed.get('tags', []))
                all_standard_codes.update(processed.get('standard_codes', []))
                all_references.update(processed.get('references', []))
                importance_scores.append(processed.get('importance_score', 0.5))
            
            # Add section enrichments to page
            enriched_page['sections_on_page'] = len(sections_on_page)
            enriched_page['section_summaries'] = section_summaries
            enriched_page['section_titles'] = section_titles
            enriched_page['section_hierarchies'] = section_hierarchies
            enriched_page['section_tags'] = list(all_section_tags)
            enriched_page['section_standard_codes'] = list(all_standard_codes)
            enriched_page['section_references'] = list(all_references)
            
            # Calculate average importance score
            if importance_scores:
                enriched_page['page_importance_score'] = sum(importance_scores) / len(importance_scores)
            else:
                enriched_page['page_importance_score'] = 0.5
                
        else:
            # No sections on this page
            enriched_page['sections_on_page'] = 0
            enriched_page['section_summaries'] = []
            enriched_page['section_titles'] = []
            enriched_page['section_hierarchies'] = []
            enriched_page['section_tags'] = []
            enriched_page['section_standard_codes'] = []
            enriched_page['section_references'] = []
            enriched_page['page_importance_score'] = 0.3
        
        enriched_pages.append(enriched_page)
    
    return enriched_pages

# ==============================================================================
# Chapter Processing
# ==============================================================================

def process_chapter(chapter_num: int, pages: List[Dict], client: Optional[OpenAI]) -> List[Dict]:
    """
    Processes all pages in a chapter, identifying and mapping sections.
    """
    if not pages:
        return []
    
    # Get chapter metadata from first page
    first_page = pages[0]
    chapter_metadata = {
        'chapter_number': chapter_num,
        'chapter_name': first_page.get('chapter_name', f'Chapter {chapter_num}'),
        'chapter_summary': first_page.get('chapter_summary'),
        'chapter_tags': first_page.get('chapter_tags', [])
    }
    
    log_progress("")
    log_progress(f"📚 Processing Chapter {chapter_num}: {chapter_metadata['chapter_name']}")
    log_progress(f"  📄 {len(pages)} pages to process")
    
    # Step 1: Build page position map
    concatenated_content, position_map = build_page_position_map(pages)
    log_progress(f"  📏 Concatenated content: {len(concatenated_content):,} characters")
    
    # Step 2: Identify sections
    sections = identify_sections(concatenated_content, chapter_metadata)
    log_progress(f"  📑 Found {len(sections)} initial sections")
    
    # Step 3: Merge small sections
    merged_sections = merge_small_sections(sections, concatenated_content)
    log_progress(f"  🔗 {len(merged_sections)} sections after merging")
    
    # Step 4: Map sections to pages
    page_sections_map = map_sections_to_pages(merged_sections, position_map)
    
    # Count pages with sections
    pages_with_sections = sum(1 for sections in page_sections_map.values() if sections)
    log_progress(f"  📍 Sections mapped to {pages_with_sections} pages")
    
    # Step 5: Process sections with GPT
    processed_sections = process_all_sections(
        merged_sections, concatenated_content, chapter_metadata, client
    )
    
    # Step 6: Enrich pages with section data
    enriched_pages = enrich_pages_with_sections(pages, page_sections_map, processed_sections)
    
    # Calculate statistics for this chapter
    multi_section_pages = sum(1 for p in enriched_pages if p.get('sections_on_page', 0) > 1)
    if multi_section_pages > 0:
        log_progress(f"  📊 {multi_section_pages} pages have multiple sections in this chapter")
    
    log_progress(f"  ✅ Chapter {chapter_num} processing complete")
    
    return enriched_pages

# ==============================================================================
# Main Processing Functions
# ==============================================================================

def group_pages_by_chapter(page_records: List[Dict]) -> Dict[int, List[Dict]]:
    """Groups page records by chapter_number."""
    chapters = defaultdict(list)
    
    for record in page_records:
        chapter_num = record.get('chapter_number')
        if chapter_num is not None:
            chapters[chapter_num].append(record)
    
    # Sort pages within each chapter
    for chapter_num in chapters:
        chapters[chapter_num].sort(key=lambda x: (
            x.get('pdf_page_number', x.get('page_number', 0))
        ))
    
    return dict(chapters)

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

def run_stage2():
    """Main function to execute Stage 2 processing."""
    if not validate_configuration():
        return
    
    temp_log_path = setup_logging()
    
    log_progress("=" * 70)
    log_progress("🚀 Starting Stage 2: Section Processing with Page-Level Output (V2)")
    log_progress("   Using CO-STAR + XML prompting format")
    log_progress("=" * 70)
    
    _setup_ssl_from_nas()
    
    share_name = NAS_PARAMS["share"]
    output_path_relative = os.path.join(NAS_OUTPUT_PATH, OUTPUT_FILENAME).replace('\\', '/')
    
    # Load Stage 1 output
    stage1_path_relative = os.path.join(NAS_INPUT_PATH, INPUT_FILENAME).replace('\\', '/')
    log_progress("📥 Loading Stage 1 output from NAS...")
    
    stage1_json_bytes = read_from_nas(share_name, stage1_path_relative)
    if not stage1_json_bytes:
        log_progress(f"❌ Failed to read Stage 1 output")
        return
    
    try:
        page_records = json.loads(stage1_json_bytes.decode('utf-8'))
        if not isinstance(page_records, list):
            log_progress("❌ Stage 1 output is not a list")
            return
        log_progress(f"✅ Loaded {len(page_records)} page records")
    except json.JSONDecodeError as e:
        log_progress(f"❌ Error decoding Stage 1 JSON: {e}")
        return
    
    # Initialize OpenAI client
    client = None
    if OpenAI:
        client = get_openai_client()
        if client:
            log_progress("✅ OpenAI client initialized")
        else:
            log_progress("⚠️ Failed to initialize OpenAI client - will skip section summaries")
    else:
        log_progress("⚠️ OpenAI library not installed - will skip section summaries")
    
    # Group pages by chapter
    chapters = group_pages_by_chapter(page_records)
    unassigned_pages = [r for r in page_records if r.get('chapter_number') is None]
    
    log_progress(f"📊 Found {len(chapters)} chapters and {len(unassigned_pages)} unassigned pages")
    log_progress("-" * 70)
    
    # Process each chapter
    all_enriched_pages = []
    
    for chapter_num in sorted(chapters.keys()):
        pages = chapters[chapter_num]
        enriched_pages = process_chapter(chapter_num, pages, client)
        all_enriched_pages.extend(enriched_pages)
    
    # Add unassigned pages without section processing
    for page in unassigned_pages:
        enriched_page = page.copy()
        enriched_page['sections_on_page'] = 0
        enriched_page['section_summaries'] = []
        enriched_page['section_titles'] = []
        enriched_page['section_hierarchies'] = []
        enriched_page['section_tags'] = []
        enriched_page['section_standard_codes'] = []
        enriched_page['page_importance_score'] = 0.3
        all_enriched_pages.append(enriched_page)
    
    # Sort by original order (source_page_number)
    all_enriched_pages.sort(key=lambda x: (
        x.get('source_page_number', x.get('page_number', 0))
    ))
    
    # Save output
    log_progress("-" * 70)
    log_progress(f"💾 Saving {len(all_enriched_pages)} enriched page records...")
    
    try:
        output_json = json.dumps(all_enriched_pages, indent=2, ensure_ascii=False)
        output_bytes = output_json.encode('utf-8')
        
        if write_to_nas(share_name, output_path_relative, output_bytes):
            log_progress(f"✅ Successfully saved output")
        else:
            log_progress("❌ Failed to write output")
    except Exception as e:
        log_progress(f"❌ Error saving output: {e}")
    
    # Upload log file
    try:
        log_file_name = f"stage2_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path_relative = os.path.join(NAS_LOG_PATH, log_file_name).replace('\\', '/')
        
        cleanup_logging_handlers()
        
        with open(temp_log_path, 'rb') as f:
            log_content = f.read()
        
        if write_to_nas(share_name, log_path_relative, log_content):
            print(f"📝 Log file uploaded: {share_name}/{log_path_relative}")
        else:
            print(f"⚠️ Failed to upload log file")
        
        os.remove(temp_log_path)
    except Exception as e:
        print(f"⚠️ Error handling log file: {e}")
    
    # Calculate statistics for pages with multiple sections
    pages_with_multiple_sections = sum(1 for p in all_enriched_pages if p.get('sections_on_page', 0) > 1)
    pages_with_one_section = sum(1 for p in all_enriched_pages if p.get('sections_on_page', 0) == 1)
    pages_with_no_sections = sum(1 for p in all_enriched_pages if p.get('sections_on_page', 0) == 0)
    
    # Calculate distribution of sections per page
    section_distribution = {}
    max_sections_on_page = 0
    for p in all_enriched_pages:
        num_sections = p.get('sections_on_page', 0)
        section_distribution[num_sections] = section_distribution.get(num_sections, 0) + 1
        if num_sections > max_sections_on_page:
            max_sections_on_page = num_sections
    
    # Final summary
    print("=" * 70)
    print("📊 Stage 2 Summary")
    print("-" * 70)
    print(f"  Input: {len(page_records)} page records")
    print(f"  Chapters processed: {len(chapters)}")
    print(f"  Output: {len(all_enriched_pages)} enriched page records")
    print("-" * 70)
    print("📄 Section Distribution per Page:")
    print(f"  Pages with 0 sections: {pages_with_no_sections}")
    print(f"  Pages with 1 section: {pages_with_one_section}")
    print(f"  Pages with 2+ sections: {pages_with_multiple_sections}")
    
    # Show detailed distribution if there are pages with multiple sections
    if pages_with_multiple_sections > 0:
        print("\n  Detailed distribution:")
        for num_sections in sorted(section_distribution.keys()):
            if num_sections > 1:
                count = section_distribution[num_sections]
                print(f"    {num_sections} sections: {count} pages")
        print(f"\n  💡 Consider: {pages_with_multiple_sections} pages would benefit from combined section summaries")
        print(f"  Maximum sections on a single page: {max_sections_on_page}")
    
    if unassigned_pages:
        print(f"\n  Unassigned pages (no chapter): {len(unassigned_pages)}")
    print("-" * 70)
    print(f"  Output file: {share_name}/{output_path_relative}")
    print("=" * 70)
    print("✅ Stage 2 Completed - Page-level records enriched with section data")

if __name__ == "__main__":
    run_stage2()