# -*- coding: utf-8 -*-
"""
Stage 2: Section Processing with Streamlined Output - FINAL VERSION
Maintains one record per page while enriching with section-level information

Purpose:
Processes page-level data from Stage 1, identifies sections within chapters,
generates section summaries with embedded metadata, and creates cross-references
between related sections for enhanced RAG retrieval.

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
from typing import List, Dict, Tuple, Optional, Set
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

# --- Section Identification Parameters ---
MAX_HEADING_LEVEL = 3  # Only consider headings up to level 3
MAX_CROSS_REFERENCES = 2  # Maximum number of cross-references per section

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
    
    # Clear any existing handlers to prevent duplication
    logging.root.handlers = []
    
    log_level = logging.DEBUG if VERBOSE_LOGGING else logging.WARNING
    
    # Only add file handler to root logger (no console handler)
    root_file_handler = logging.FileHandler(temp_log_path)
    root_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[root_file_handler]
    )
    
    # Progress logger handles all console output
    progress_logger = logging.getLogger('progress')
    progress_logger.setLevel(logging.INFO)
    progress_logger.propagate = False
    
    # Console handler for progress messages
    progress_console_handler = logging.StreamHandler()
    progress_console_handler.setFormatter(logging.Formatter('%(message)s'))
    progress_logger.addHandler(progress_console_handler)
    
    # File handler for progress messages
    progress_file_handler = logging.FileHandler(temp_log_path)
    progress_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    progress_logger.addHandler(progress_file_handler)
    
    # If verbose logging is enabled, also show warnings/errors on console
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
# Tool Enforcement for API Calls
# ==============================================================================

def call_gpt_with_tool_enforcement(client, model, messages, max_tokens, temperature, tool_schema):
    """
    Makes API call with STRICT tool enforcement.
    Will retry if response doesn't use the specified tool.
    """
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
                return function_args, usage_info
                
            except json.JSONDecodeError as e:
                logging.warning(f"Attempt {attempt + 1}: Invalid JSON in tool arguments: {e}")
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
# Section Identification and Processing
# ==============================================================================

def identify_sections(content: str, chapter_metadata: Dict) -> List[Dict]:
    """
    Identifies sections based on markdown headings (levels 1 to MAX_HEADING_LEVEL).
    Returns list of sections with their positions and content.
    """
    sections = []
    section_pattern = re.compile(f'^(#{{1,{MAX_HEADING_LEVEL}}})\\s+(.+)$', re.MULTILINE)
    
    matches = list(section_pattern.finditer(content))
    
    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        start_pos = match.start()
        
        if i < len(matches) - 1:
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(content)
        
        section_content = content[start_pos:end_pos].strip()
        
        sections.append({
            'id': f"ch{chapter_metadata['chapter_number']}_s{i+1}",
            'title': title,
            'level': level,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'content': section_content,
            'token_count': count_tokens(section_content)
        })
    
    return sections

def merge_small_sections(sections: List[Dict]) -> List[Dict]:
    """
    Merges small sections with adjacent sections to avoid fragmentation.
    """
    if not sections:
        return sections
    
    merged = []
    i = 0
    
    while i < len(sections):
        current = sections[i]
        
        # Check if current section is too small
        if current['token_count'] < MIN_SECTION_TOKENS:
            # Try to merge with previous or next section
            if merged and merged[-1]['token_count'] + current['token_count'] <= MAX_SECTION_TOKENS:
                # Merge with previous
                prev = merged[-1]
                prev['content'] += "\n\n" + current['content']
                prev['title'] += f" & {current['title']}"
                prev['end_pos'] = current['end_pos']
                prev['token_count'] += current['token_count']
            elif i + 1 < len(sections) and current['token_count'] + sections[i + 1]['token_count'] <= MAX_SECTION_TOKENS:
                # Merge with next
                next_section = sections[i + 1]
                current['content'] += "\n\n" + next_section['content']
                current['title'] += f" & {next_section['title']}"
                current['end_pos'] = next_section['end_pos']
                current['token_count'] += next_section['token_count']
                merged.append(current)
                i += 1  # Skip next section
            else:
                # Can't merge, keep as is
                merged.append(current)
        else:
            merged.append(current)
        
        i += 1
    
    # Re-ID sections after merging
    for idx, section in enumerate(merged):
        original_chapter = section['id'].split('_')[0]
        section['id'] = f"{original_chapter}_s{idx+1}"
    
    return merged

# ==============================================================================
# Section Summary Generation with Embedded Metadata
# ==============================================================================

SECTION_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "provide_section_analysis",
        "description": "Provides structured analysis of section content",
        "parameters": {
            "type": "object",
            "properties": {
                "section_summary": {
                    "type": "string",
                    "description": "A comprehensive summary (2-4 sentences) that includes the core topic, relevant accounting standards (e.g., IFRS 16, ASC 842), and key concepts or tags naturally embedded in the text."
                }
            },
            "required": ["section_summary"],
            "additionalProperties": False
        }
    }
}

def build_section_analysis_prompt(section: Dict, chapter_summary: str, previous_summaries: List[str] = None):
    """
    Builds prompt for section analysis with embedded metadata.
    """
    system_prompt = """You are an expert financial reporting specialist analyzing EY technical accounting guidance.
Your task is to create concise, information-rich summaries that embed all relevant metadata naturally within the text."""

    user_prompt_parts = []
    
    # Add chapter context
    user_prompt_parts.append(f"<chapter_context>\n{chapter_summary}\n</chapter_context>")
    
    # Add previous section context if available
    if previous_summaries and len(previous_summaries) > 0:
        recent_context = "\n".join(previous_summaries[-3:])  # Last 3 sections
        user_prompt_parts.append(f"<previous_sections>\n{recent_context}\n</previous_sections>")
    
    # Add current section
    user_prompt_parts.append(f"<current_section>\n{section['content']}\n</current_section>")
    
    # Add instructions
    user_prompt_parts.append("""<instructions>
Create a comprehensive summary (2-4 sentences) that:
1. Captures the core topic and purpose of this section
2. Naturally embeds any relevant accounting standards mentioned (e.g., "This section explains IFRS 16 lease classification...")
3. Includes key concepts and technical terms as part of the narrative
4. Maintains clarity and readability while being information-dense

The summary should read naturally while containing all searchable metadata within the text itself.
Do NOT create separate lists or categories - embed everything in flowing sentences.
</instructions>""")
    
    user_prompt_parts.append("YOU MUST use the 'provide_section_analysis' tool to provide your response.")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(user_prompt_parts)}
    ]
    
    return messages

def process_sections(sections: List[Dict], chapter_metadata: Dict, client: Optional[OpenAI]) -> Tuple[Dict, Dict]:
    """
    Process sections to generate summaries with embedded metadata.
    Returns processed sections and section page ranges.
    """
    processed_sections = {}
    section_page_ranges = {}
    
    if not client:
        # Fallback without GPT
        for section in sections:
            processed_sections[section['id']] = {
                'title': section['title'],
                'summary': f"Section covering {section['title']}"
            }
        return processed_sections, section_page_ranges
    
    previous_summaries = []
    chapter_summary = chapter_metadata.get('chapter_summary', '')
    
    for section in sections:
        log_progress(f"    Processing section: {section['id']} - {section['title'][:50]}...", end='')
        
        messages = build_section_analysis_prompt(section, chapter_summary, previous_summaries)
        
        result, usage_info = call_gpt_with_tool_enforcement(
            client=client,
            model=MODEL_NAME_CHAT,
            messages=messages,
            max_tokens=MAX_COMPLETION_TOKENS,
            temperature=TEMPERATURE,
            tool_schema=SECTION_TOOL_SCHEMA
        )
        
        if result:
            summary = result.get('section_summary', '')
            processed_sections[section['id']] = {
                'title': section['title'],
                'summary': summary
            }
            if summary:
                previous_summaries.append(f"{section['title']}: {summary}")
            log_progress(" ✅")
        else:
            processed_sections[section['id']] = {
                'title': section['title'],
                'summary': f"Section covering {section['title']}"
            }
            log_progress(" ❌")
    
    return processed_sections, section_page_ranges

# ==============================================================================
# Cross-Reference Generation
# ==============================================================================

CROSS_REFERENCE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "identify_cross_references",
        "description": "Identifies contextually relevant sections within the chapter",
        "parameters": {
            "type": "object",
            "properties": {
                "cross_references": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "section_id": {
                                "type": "string",
                                "description": "The section ID (e.g., ch1_s2)"
                            },
                            "related_sections": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of up to 2 section IDs that are crucial for understanding this section"
                            }
                        },
                        "required": ["section_id", "related_sections"]
                    },
                    "description": "Cross-references between sections"
                }
            },
            "required": ["cross_references"],
            "additionalProperties": False
        }
    }
}

def generate_cross_references(processed_sections: Dict, chapter_metadata: Dict, client: Optional[OpenAI]) -> Dict[str, List[str]]:
    """
    Generates cross-references between sections in a chapter.
    Returns a dictionary mapping section_id to list of related section_ids.
    """
    if not client or len(processed_sections) < 2:
        return {sid: [] for sid in processed_sections.keys()}
    
    # Prepare section summaries for analysis
    section_list = []
    for section_id, section_data in processed_sections.items():
        section_list.append(f"{section_id}: {section_data['title']} - {section_data['summary']}")
    
    system_prompt = """You are analyzing sections within a technical accounting chapter to identify crucial cross-references.
Your task is to identify which sections are most important for understanding each other section."""
    
    user_prompt = f"""<chapter_context>
Chapter {chapter_metadata['chapter_number']}: {chapter_metadata['chapter_name']}
</chapter_context>

<sections>
{chr(10).join(section_list)}
</sections>

<instructions>
For each section, identify up to {MAX_CROSS_REFERENCES} other sections within this chapter that are CRUCIAL for understanding it.
Only create cross-references when sections are strongly related (e.g., a section on "recognition" and "measurement" of the same topic).
Cross-references should be bidirectional - if section A references B, then B should reference A.
Return empty list for sections that don't have crucial relationships.
</instructions>

YOU MUST use the 'identify_cross_references' tool to provide your response."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    log_progress(f"  🔗 Generating cross-references for {len(processed_sections)} sections...", end='')
    
    result, usage_info = call_gpt_with_tool_enforcement(
        client=client,
        model=MODEL_NAME_CHAT,
        messages=messages,
        max_tokens=MAX_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        tool_schema=CROSS_REFERENCE_TOOL_SCHEMA
    )
    
    cross_references = {sid: [] for sid in processed_sections.keys()}
    
    if result and 'cross_references' in result:
        # Process the results
        for ref in result['cross_references']:
            section_id = ref.get('section_id')
            related = ref.get('related_sections', [])
            if section_id in cross_references:
                cross_references[section_id] = related[:MAX_CROSS_REFERENCES]
        
        # Ensure bidirectionality
        for section_id, related_ids in list(cross_references.items()):
            for related_id in related_ids:
                if related_id in cross_references and section_id not in cross_references[related_id]:
                    if len(cross_references[related_id]) < MAX_CROSS_REFERENCES:
                        cross_references[related_id].append(section_id)
        
        log_progress(" ✅")
    else:
        log_progress(" ❌")
    
    return cross_references

# ==============================================================================
# Page Mapping and Enrichment
# ==============================================================================

def build_page_position_map(pages: List[Dict]) -> Tuple[str, Dict]:
    """
    Concatenates page content and builds a position map.
    Returns concatenated content and position map.
    """
    concatenated_content = ""
    position_map = {}
    
    for page in pages:
        page_num = page.get('page_number')
        start_pos = len(concatenated_content)
        page_content = page.get('content', '')
        concatenated_content += page_content + "\n\n"
        end_pos = len(concatenated_content)
        
        position_map[page_num] = {
            'start': start_pos,
            'end': end_pos
        }
    
    return concatenated_content, position_map

def map_sections_to_pages(sections: List[Dict], position_map: Dict, pages: List[Dict]) -> Dict:
    """
    Maps sections to pages based on character positions.
    Returns a dictionary mapping page numbers to list of section IDs.
    """
    page_sections_map = defaultdict(list)
    
    for section in sections:
        section_start = section['start_pos']
        section_end = section['end_pos']
        
        for page in pages:
            page_num = page.get('page_number')
            if page_num not in position_map:
                continue
                
            page_start = position_map[page_num]['start']
            page_end = position_map[page_num]['end']
            
            # Check if section overlaps with page
            if section_start < page_end and section_end > page_start:
                page_sections_map[page_num].append(section['id'])
    
    return dict(page_sections_map)

def find_section_page_bounds(section_id: str, page_sections_map: Dict, pages: List[Dict]) -> Tuple[int, int]:
    """
    Finds the first and last page numbers where a section appears.
    Returns (start_page, end_page) tuple.
    """
    pages_with_section = []
    
    for page_num, section_ids in page_sections_map.items():
        if section_id in section_ids:
            pages_with_section.append(page_num)
    
    if pages_with_section:
        return min(pages_with_section), max(pages_with_section)
    return None, None

def enrich_pages_with_sections(pages: List[Dict], page_sections_map: Dict, 
                              processed_sections: Dict, cross_references: Dict) -> List[Dict]:
    """
    Enriches page records with section information using the new schema.
    """
    enriched_pages = []
    
    for page in pages:
        enriched_page = page.copy()
        page_num = page.get('page_number')
        
        sections_on_page = page_sections_map.get(page_num, [])
        
        if sections_on_page:
            # Find section page bounds
            first_section_start = None
            last_section_end = None
            
            for i, section_id in enumerate(sections_on_page):
                start_page, end_page = find_section_page_bounds(section_id, page_sections_map, pages)
                
                if i == 0 and start_page is not None:
                    first_section_start = start_page
                if i == len(sections_on_page) - 1 and end_page is not None:
                    last_section_end = end_page
            
            # Build combined section summary
            section_summaries = []
            all_references = set()
            
            for section_id in sections_on_page:
                if section_id in processed_sections:
                    section_data = processed_sections[section_id]
                    # Combine title and summary
                    combined = f"{section_data['title']}: {section_data['summary']}"
                    section_summaries.append(combined)
                    
                    # Collect cross-references
                    if section_id in cross_references:
                        all_references.update(cross_references[section_id])
            
            # Join summaries with double line breaks
            enriched_page['section_summary'] = "\n\n".join(section_summaries) if section_summaries else ""
            enriched_page['section_page_start'] = first_section_start
            enriched_page['section_page_end'] = last_section_end
            enriched_page['section_references'] = list(all_references)
            
        else:
            # No sections on this page
            enriched_page['section_summary'] = ""
            enriched_page['section_page_start'] = None
            enriched_page['section_page_end'] = None
            enriched_page['section_references'] = []
        
        # Keep only required fields
        final_page = {
            'document_id': enriched_page.get('document_id'),
            'filename': enriched_page.get('filename'),
            'filepath': enriched_page.get('filepath'),
            'source_filename': enriched_page.get('source_filename'),
            'chapter_number': enriched_page.get('chapter_number'),
            'chapter_name': enriched_page.get('chapter_name'),
            'chapter_summary': enriched_page.get('chapter_summary'),
            'chapter_page_count': enriched_page.get('chapter_page_count'),
            'page_number': enriched_page.get('page_number'),
            'page_reference': enriched_page.get('page_reference'),
            'source_page_number': enriched_page.get('source_page_number'),
            'content': enriched_page.get('content'),
            'section_page_start': enriched_page.get('section_page_start'),
            'section_page_end': enriched_page.get('section_page_end'),
            'section_summary': enriched_page.get('section_summary'),
            'section_references': enriched_page.get('section_references')
        }
        
        enriched_pages.append(final_page)
    
    return enriched_pages

# ==============================================================================
# Chapter Processing
# ==============================================================================

def process_chapter(chapter_num: int, pages: List[Dict], client: Optional[OpenAI]) -> List[Dict]:
    """
    Processes all pages in a chapter, identifying sections and generating cross-references.
    """
    if not pages:
        return []
    
    # Get chapter metadata from first page
    first_page = pages[0]
    chapter_metadata = {
        'chapter_number': chapter_num,
        'chapter_name': first_page.get('chapter_name', f'Chapter {chapter_num}'),
        'chapter_summary': first_page.get('chapter_summary')
    }
    
    log_progress("")
    log_progress(f"📚 Processing Chapter {chapter_num}: {chapter_metadata['chapter_name']}")
    log_progress(f"  📄 {len(pages)} pages to process")
    
    # Step 1: Build page position map
    concatenated_content, position_map = build_page_position_map(pages)
    log_progress(f"  📏 Concatenated content: {len(concatenated_content):,} characters")
    
    # Step 2: Identify sections
    sections = identify_sections(concatenated_content, chapter_metadata)
    log_progress(f"  📑 Found {len(sections)} sections")
    
    # Step 3: Merge small sections
    sections = merge_small_sections(sections)
    log_progress(f"  🔀 After merging: {len(sections)} sections")
    
    # Step 4: Map sections to pages
    page_sections_map = map_sections_to_pages(sections, position_map, pages)
    
    # Step 5: Process sections to generate summaries
    processed_sections, section_page_ranges = process_sections(sections, chapter_metadata, client)
    
    # Step 6: Generate cross-references
    cross_references = generate_cross_references(processed_sections, chapter_metadata, client)
    
    # Step 7: Enrich pages with section data
    enriched_pages = enrich_pages_with_sections(pages, page_sections_map, processed_sections, cross_references)
    
    log_progress(f"  ✅ Chapter {chapter_num} processing complete")
    
    return enriched_pages

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

def run_stage2():
    """Main function to execute Stage 2 processing."""
    if not validate_configuration():
        return
    
    temp_log_path = setup_logging()
    
    log_progress("=" * 70)
    log_progress("🚀 Starting Stage 2: Section Processing (Final Version)")
    log_progress("=" * 70)
    
    _setup_ssl_from_nas()
    
    share_name = NAS_PARAMS["share"]
    input_path = os.path.join(NAS_INPUT_PATH, INPUT_FILENAME).replace('\\', '/')
    output_path = os.path.join(NAS_OUTPUT_PATH, OUTPUT_FILENAME).replace('\\', '/')
    
    # Load input JSON
    log_progress("📥 Loading Stage 1 output from NAS...")
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
    
    # Initialize OpenAI client
    client = None
    if OpenAI:
        client = get_openai_client()
        if client:
            log_progress("✅ OpenAI client initialized")
        else:
            log_progress("⚠️ Failed to initialize OpenAI client - continuing without GPT features")
    else:
        log_progress("⚠️ OpenAI library not installed - continuing without GPT features")
    
    # Group pages by chapter
    chapters, unassigned_pages = group_pages_by_chapter(page_records)
    log_progress(f"📊 Found {len(chapters)} chapters and {len(unassigned_pages)} unassigned pages")
    log_progress("-" * 70)
    
    # Process each chapter
    all_enriched_pages = []
    
    for chapter_num in sorted(chapters.keys()):
        pages = chapters[chapter_num]
        enriched_pages = process_chapter(chapter_num, pages, client)
        all_enriched_pages.extend(enriched_pages)
    
    # Add unassigned pages with empty section fields
    for page in unassigned_pages:
        enriched_page = {
            'document_id': page.get('document_id'),
            'filename': page.get('filename'),
            'filepath': page.get('filepath'),
            'source_filename': page.get('source_filename'),
            'chapter_number': page.get('chapter_number'),
            'chapter_name': page.get('chapter_name'),
            'chapter_summary': page.get('chapter_summary'),
            'chapter_page_count': page.get('chapter_page_count'),
            'page_number': page.get('page_number'),
            'page_reference': page.get('page_reference'),
            'source_page_number': page.get('source_page_number'),
            'content': page.get('content'),
            'section_page_start': None,
            'section_page_end': None,
            'section_summary': "",
            'section_references': []
        }
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
        
        if write_to_nas(share_name, output_path, output_bytes):
            log_progress(f"✅ Successfully saved output to {share_name}/{output_path}")
        else:
            log_progress("❌ Failed to write output to NAS")
    except Exception as e:
        log_progress(f"❌ Error saving output: {e}")
    
    # Upload log file
    try:
        log_file_name = f"stage2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    print("📊 Stage 2 Summary")
    print("-" * 70)
    print(f"  Input: {len(page_records)} page records")
    print(f"  Chapters processed: {len(chapters)}")
    print(f"  Output: {len(all_enriched_pages)} enriched page records")
    print(f"  Output file: {share_name}/{output_path}")
    print("=" * 70)
    print("✅ Stage 2 Completed")

if __name__ == "__main__":
    run_stage2()