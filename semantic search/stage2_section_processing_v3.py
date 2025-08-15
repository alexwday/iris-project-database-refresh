# -*- coding: utf-8 -*-
"""
Stage 2: Section Processing with Embedded Page Tags - Version 3
Creates section-based records with HTML page tags embedded in content

Purpose:
Processes page-level data from Stage 1, identifies sections within chapters,
embeds HTML page tags in section content, generates enriched section summaries,
and creates cross-references between related sections.

Input: JSON file from Stage 1 output (stage1_page_records.json)
Output: JSON file with section-level records (stage2_section_records.json)
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
import html
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
OUTPUT_FILENAME = "stage2_section_records.json"

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

# --- Section Merging Thresholds (from old script) ---
MIN_SECTION_TOKENS = 250  # Sections below this trigger merging
MAX_SECTION_TOKENS = 750  # Maximum tokens after merging
ULTRA_SMALL_THRESHOLD = 25  # Very small sections get aggressive merging

# --- Section Identification Parameters ---
MAX_HEADING_LEVEL = 6  # Consider all heading levels (H1-H6) like old script
MAX_CROSS_REFERENCES = 3  # Maximum number of cross-references per section

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
# Page Tag Embedding Functions
# ==============================================================================

def embed_page_tags(pages: List[Dict]) -> str:
    """
    Combines page content with embedded HTML page tags.
    Each page gets header and footer tags with page number and reference.
    """
    if not pages:
        return ""
    
    tagged_content_parts = []
    
    for page in pages:
        page_num = page.get('page_number')
        page_ref = page.get('page_reference', '')
        content = page.get('content', '')
        
        # Escape special characters in page reference for HTML attributes
        page_ref_escaped = html.escape(page_ref, quote=True)
        
        # Build tagged content for this page
        tagged_page = []
        
        # Add header tag
        tagged_page.append(f'<!-- PageHeader PageNumber="{page_num}" PageReference="{page_ref_escaped}" -->')
        
        # Add actual content
        tagged_page.append(content)
        
        # Add footer tag
        tagged_page.append(f'<!-- PageFooter PageNumber="{page_num}" PageReference="{page_ref_escaped}" -->')
        
        # Join this page's parts and add to overall content
        tagged_content_parts.append('\n'.join(tagged_page))
    
    # Join all pages with newlines
    return '\n'.join(tagged_content_parts)

def extract_page_range_from_content(content: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Extracts the first and last page numbers from content with embedded tags.
    """
    # Pattern to find all page number tags (in both header and footer)
    page_pattern = re.compile(r'<!-- Page(?:Header|Footer) PageNumber="(\d+)"')
    matches = page_pattern.findall(content)
    
    if not matches:
        return None, None
    
    page_numbers = [int(m) for m in matches]
    return min(page_numbers), max(page_numbers)

# ==============================================================================
# Section Identification and Processing
# ==============================================================================

def identify_sections_from_pages(pages: List[Dict], chapter_metadata: Dict) -> List[Dict]:
    """
    Identifies sections based on markdown headings across all pages.
    Returns list of sections with their page ranges and content.
    """
    # First, create the full tagged content
    full_content = embed_page_tags(pages)
    
    sections = []
    section_pattern = re.compile(f'^(#{{1,{MAX_HEADING_LEVEL}}})\\s+(.+)$', re.MULTILINE)
    
    matches = list(section_pattern.finditer(full_content))
    
    # Handle content before first heading (if any)
    first_heading_pos = matches[0].start() if matches else len(full_content)
    if first_heading_pos > 0:
        intro_content = full_content[:first_heading_pos].strip()
        if intro_content:
            # Extract page range from intro content
            start_page, end_page = extract_page_range_from_content(intro_content)
            sections.append({
                'section_number': 1,
                'title': chapter_metadata.get('chapter_name', 'Introduction'),
                'level': 1,
                'content': intro_content,
                'token_count': count_tokens(intro_content),
                'start_page': start_page,
                'end_page': end_page
            })
    
    # Process each heading-defined section
    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        start_pos = match.start()
        
        if i < len(matches) - 1:
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(full_content)
        
        section_content = full_content[start_pos:end_pos].strip()
        
        # Extract page range from section content
        start_page, end_page = extract_page_range_from_content(section_content)
        
        sections.append({
            'section_number': len(sections) + 1,
            'title': title,
            'level': level,
            'content': section_content,
            'token_count': count_tokens(section_content),
            'start_page': start_page,
            'end_page': end_page
        })
    
    return sections

def generate_hierarchy_string(section: Dict, all_sections: List[Dict], current_idx: int) -> str:
    """
    Generates a breadcrumb-style hierarchy string for a section.
    Looks back through previous sections to build the hierarchy.
    """
    parts = []
    current_level = section.get('level', 1)
    
    # Track the most recent section at each level
    level_titles = {}
    
    # Look back through all previous sections to build hierarchy
    for i in range(current_idx):
        prev_section = all_sections[i]
        prev_level = prev_section.get('level', 1)
        # Store the most recent title at each level
        level_titles[prev_level] = prev_section.get('title', '')
        # Clear any deeper levels when we hit a higher level
        for deeper_level in list(level_titles.keys()):
            if deeper_level > prev_level:
                del level_titles[deeper_level]
    
    # Build the hierarchy from level 1 up to current level - 1
    for level in range(1, current_level):
        if level in level_titles:
            parts.append(level_titles[level])
    
    # Add current section
    parts.append(section.get('title', ''))
    
    # Filter out empty parts and join
    parts = [p for p in parts if p]
    return ' > '.join(parts)

def merge_small_sections(sections: List[Dict]) -> List[Dict]:
    """
    Merges small sections with adjacent sections to avoid fragmentation.
    Preserves page tags during merging.
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
                prev['content'] += "\n" + current['content']
                # Keep the original section's title (don't concatenate)
                # prev['title'] remains unchanged
                prev['token_count'] += current['token_count']
                # Update page range properly - take min of starts and max of ends
                if current.get('start_page') is not None:
                    if prev.get('start_page') is not None:
                        prev['start_page'] = min(prev['start_page'], current['start_page'])
                    else:
                        prev['start_page'] = current['start_page']
                if current.get('end_page') is not None:
                    if prev.get('end_page') is not None:
                        prev['end_page'] = max(prev['end_page'], current['end_page'])
                    else:
                        prev['end_page'] = current['end_page']
            elif i + 1 < len(sections) and current['token_count'] + sections[i + 1]['token_count'] <= MAX_SECTION_TOKENS:
                # Merge with next
                next_section = sections[i + 1]
                current['content'] += "\n" + next_section['content']
                # Keep the current section's title (don't concatenate)
                # current['title'] remains unchanged
                current['token_count'] += next_section['token_count']
                # Update page range properly - take min of starts and max of ends
                if next_section.get('start_page') is not None:
                    if current.get('start_page') is not None:
                        current['start_page'] = min(current['start_page'], next_section['start_page'])
                    else:
                        current['start_page'] = next_section['start_page']
                if next_section.get('end_page') is not None:
                    if current.get('end_page') is not None:
                        current['end_page'] = max(current['end_page'], next_section['end_page'])
                    else:
                        current['end_page'] = next_section['end_page']
                merged.append(current)
                i += 1  # Skip next section
            else:
                # Can't merge, keep as is
                merged.append(current)
        else:
            merged.append(current)
        
        i += 1
    
    # Re-number sections after merging
    for idx, section in enumerate(merged):
        section['section_number'] = idx + 1
    
    return merged

# ==============================================================================
# Section Summary Generation
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
                    "description": "A comprehensive summary (3-5 sentences) that explains the section's purpose and content. Must naturally embed relevant accounting standards (e.g., IFRS 16, ASC 842) and key concepts/tags within the narrative."
                }
            },
            "required": ["section_summary"],
            "additionalProperties": False
        }
    }
}

def build_section_analysis_prompt(section: Dict, chapter_summary: str, hierarchy: str):
    """
    Builds prompt for section analysis with requirement to embed metadata.
    Uses structured XML tags for cleaner prompting.
    """
    system_prompt = """<role>You are an expert financial reporting specialist analyzing EY technical accounting guidance.</role>
<task>Create comprehensive summaries that naturally embed all relevant metadata within the text.</task>"""

    user_prompt_parts = ["<prompt>"]
    
    # Add chapter context
    user_prompt_parts.append("<chapter_context>")
    user_prompt_parts.append(f"<chapter_summary>{chapter_summary}</chapter_summary>")
    user_prompt_parts.append("</chapter_context>")
    
    # Add section hierarchy
    user_prompt_parts.append(f"<section_hierarchy>{hierarchy}</section_hierarchy>")
    
    # Add current section
    user_prompt_parts.append("<current_section>")
    user_prompt_parts.append(section['content'])
    user_prompt_parts.append("</current_section>")
    
    # Add instructions
    user_prompt_parts.append("<instructions>")
    user_prompt_parts.append("""Create a comprehensive summary (3-5 sentences) that:
1. Explains the core purpose and content of this section
2. MUST embed any relevant accounting standards directly in the text (e.g., "This section explains IFRS 16 lease classification...")
3. MUST include key technical terms and concepts as part of the narrative
4. MUST mention any specific standard codes referenced (e.g., "Following ASC 842-10-15 requirements...")
5. Should flow naturally while being information-dense

The summary should read as a cohesive explanation while containing all searchable metadata.
Do NOT create bullet points or lists - write flowing sentences that embed all information.""")
    user_prompt_parts.append("</instructions>")
    
    user_prompt_parts.append("<response_format>YOU MUST use the 'provide_section_analysis' tool to provide your response.</response_format>")
    user_prompt_parts.append("</prompt>")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(user_prompt_parts)}
    ]
    
    return messages

def process_section_summary(section: Dict, chapter_metadata: Dict, hierarchy: str, client: Optional[OpenAI]) -> str:
    """
    Generates an enriched summary for a section with retry logic.
    """
    if not client:
        return f"Section covering {section['title']}"
    
    # Validate section size doesn't exceed GPT limits
    content_tokens = section.get('token_count', 0)
    if content_tokens > (GPT_INPUT_TOKEN_LIMIT - TOKEN_BUFFER - 2000):  # Extra buffer for prompt
        logging.warning(f"Section {section.get('section_number', '?')} may exceed token limit: {content_tokens} tokens")
        # Truncate content if too large
        max_chars = int((GPT_INPUT_TOKEN_LIMIT - TOKEN_BUFFER - 2000) * 3.5)
        section_copy = section.copy()
        section_copy['content'] = section['content'][:max_chars]
        messages = build_section_analysis_prompt(section_copy, chapter_metadata.get('chapter_summary', ''), hierarchy)
    else:
        messages = build_section_analysis_prompt(section, chapter_metadata.get('chapter_summary', ''), hierarchy)
    
    # Try to get summary with retries
    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            result, usage_info = call_gpt_with_tool_enforcement(
                client=client,
                model=MODEL_NAME_CHAT,
                messages=messages,
                max_tokens=MAX_COMPLETION_TOKENS,
                temperature=TEMPERATURE,
                tool_schema=SECTION_TOOL_SCHEMA
            )
            
            if result and 'section_summary' in result:
                # Log token usage if verbose
                if usage_info and VERBOSE_LOGGING:
                    prompt_tokens = usage_info.prompt_tokens
                    completion_tokens = usage_info.completion_tokens
                    total_cost = (prompt_tokens / 1000) * PROMPT_TOKEN_COST + (completion_tokens / 1000) * COMPLETION_TOKEN_COST
                    logging.debug(f"Section summary tokens - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Cost: ${total_cost:.4f}")
                
                return result['section_summary']
                
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1}/{API_RETRY_ATTEMPTS} failed for section summary: {e}")
            if attempt < API_RETRY_ATTEMPTS - 1:
                time.sleep(API_RETRY_DELAY * (attempt + 1))  # Exponential backoff
    
    # Fallback if all attempts fail
    logging.error(f"Failed to generate summary for section after {API_RETRY_ATTEMPTS} attempts")
    return f"Section covering {section['title']}"

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
                            "section_number": {
                                "type": "integer",
                                "description": "The section number"
                            },
                            "related_sections": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "List of up to 3 related section numbers that are crucial for understanding this section"
                            }
                        },
                        "required": ["section_number", "related_sections"]
                    },
                    "description": "Cross-references between sections"
                }
            },
            "required": ["cross_references"],
            "additionalProperties": False
        }
    }
}

def generate_cross_references(sections: List[Dict], chapter_metadata: Dict, client: Optional[OpenAI]) -> Dict[int, List[str]]:
    """
    Generates cross-references between sections using GPT to identify semantic relationships.
    Uses complete summaries (hierarchy + GPT summary) for better context.
    Returns a dictionary mapping section_number to list of related section IDs.
    """
    cross_refs = {s['section_number']: [] for s in sections}
    
    # If no client or too few sections, return empty references
    if not client or len(sections) < 2:
        return cross_refs
    
    # Build section index with complete summaries for GPT analysis
    section_summaries = []
    for section in sections:
        section_num = section['section_number']
        complete_summary = section.get('complete_summary', f"Section {section_num}: {section.get('title', 'Unknown')}")
        # Format: Section ID + complete summary
        section_summaries.append(f"[Section {section_num}]\n{complete_summary}")
    
    system_prompt = """<role>You are analyzing sections within a technical accounting chapter to identify crucial cross-references.</role>
<task>Identify which sections are most important for understanding each other section.</task>"""
    
    user_prompt_parts = ["<prompt>"]
    
    user_prompt_parts.append("<chapter_context>")
    user_prompt_parts.append(f"<chapter_name>{chapter_metadata.get('chapter_name', 'Unknown Chapter')}</chapter_name>")
    user_prompt_parts.append(f"<chapter_summary>{chapter_metadata.get('chapter_summary', '')}</chapter_summary>")
    user_prompt_parts.append("</chapter_context>")
    
    user_prompt_parts.append("<sections>")
    user_prompt_parts.append("\n\n".join(section_summaries))
    user_prompt_parts.append("</sections>")
    
    user_prompt_parts.append("<instructions>")
    user_prompt_parts.append(f"""Analyze the complete section summaries above (which include hierarchy paths and detailed content descriptions).
For each section, identify up to {MAX_CROSS_REFERENCES} other sections within this chapter that are CRUCIAL for understanding it.
Only create cross-references when sections are strongly related based on their actual content (e.g., a section on "recognition" and "measurement" of the same topic).
Cross-references should be bidirectional - if section A references B, then B should reference A.
Return empty list for sections that don't have crucial relationships.
Use section numbers (integers) not section IDs.""")
    user_prompt_parts.append("</instructions>")
    
    user_prompt_parts.append("<response_format>YOU MUST use the 'identify_cross_references' tool to provide your response.</response_format>")
    user_prompt_parts.append("</prompt>")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(user_prompt_parts)}
    ]
    
    log_progress(f"  🔗 Generating cross-references for {len(sections)} sections...", end='')
    
    result, usage_info = call_gpt_with_tool_enforcement(
        client=client,
        model=MODEL_NAME_CHAT,
        messages=messages,
        max_tokens=MAX_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        tool_schema=CROSS_REFERENCE_TOOL_SCHEMA
    )
    
    if result and 'cross_references' in result:
        # Process the results
        for ref in result['cross_references']:
            section_num = ref.get('section_number')
            related = ref.get('related_sections', [])
            if section_num in cross_refs:
                # Convert section numbers to IDs and limit to MAX_CROSS_REFERENCES
                cross_refs[section_num] = [f"s{r}" for r in related[:MAX_CROSS_REFERENCES]]
        
        # Ensure bidirectionality
        for section_num, related_nums in list(cross_refs.items()):
            for related_str in related_nums:
                # Extract number from "s2" format
                related_num = int(related_str[1:]) if related_str.startswith('s') else int(related_str)
                if related_num in cross_refs:
                    ref_id = f"s{section_num}"
                    if ref_id not in cross_refs[related_num] and len(cross_refs[related_num]) < MAX_CROSS_REFERENCES:
                        cross_refs[related_num].append(ref_id)
        
        log_progress(" ✅")
    else:
        log_progress(" ❌")
    
    return cross_refs

# ==============================================================================
# Chapter Processing
# ==============================================================================

def process_chapter(chapter_num: int, pages: List[Dict], client: Optional[OpenAI]) -> List[Dict]:
    """
    Processes all pages in a chapter to create section records.
    """
    if not pages:
        return []
    
    # Get chapter metadata from first page
    first_page = pages[0]
    chapter_metadata = {
        'document_id': first_page.get('document_id'),
        'filename': first_page.get('filename'),
        'filepath': first_page.get('filepath'),
        'source_filename': first_page.get('source_filename'),
        'chapter_number': chapter_num,
        'chapter_name': first_page.get('chapter_name', f'Chapter {chapter_num}'),
        'chapter_summary': first_page.get('chapter_summary'),
        'chapter_page_count': first_page.get('chapter_page_count')
    }
    
    log_progress("")
    log_progress(f"📚 Processing Chapter {chapter_num}: {chapter_metadata['chapter_name']}")
    log_progress(f"  📄 {len(pages)} pages to process")
    
    # Step 1: Identify sections from pages
    sections = identify_sections_from_pages(pages, chapter_metadata)
    log_progress(f"  📑 Found {len(sections)} initial sections")
    
    # Step 2: Merge small sections
    sections = merge_small_sections(sections)
    log_progress(f"  🔀 After merging: {len(sections)} sections")
    
    # Step 3: Generate hierarchies for all sections
    for idx, section in enumerate(sections):
        hierarchy = generate_hierarchy_string(section, sections, idx)
        section['hierarchy'] = hierarchy
    
    # Step 4: Generate section summaries for all sections
    log_progress(f"  📝 Generating section summaries...")
    for idx, section in enumerate(tqdm(sections, desc=f"Chapter {chapter_num} Summaries")):
        hierarchy = section['hierarchy']
        
        # Generate enriched summary
        gpt_summary = process_section_summary(section, chapter_metadata, hierarchy, client)
        
        # Store both the GPT summary and complete summary for cross-referencing
        section['gpt_summary'] = gpt_summary
        # Complete summary includes hierarchy for context in cross-referencing
        section['complete_summary'] = f"{hierarchy}\n\n{gpt_summary}"
    
    # Step 5: Generate cross-references using complete summaries
    cross_refs = generate_cross_references(sections, chapter_metadata, client)
    
    # Step 6: Build final section records
    section_records = []
    
    for section in sections:
        # Calculate page count for this section with validation
        start_page = section.get('start_page')
        end_page = section.get('end_page')
        if start_page is not None and end_page is not None:
            section_page_count = max(0, end_page - start_page + 1)
        else:
            section_page_count = 0
            logging.warning(f"Section {section['section_number']} has invalid page range: {start_page}-{end_page}")
        
        # Build section record
        section_record = {
            'document_id': chapter_metadata['document_id'],
            'filename': chapter_metadata['filename'],
            'filepath': chapter_metadata['filepath'],
            'source_filename': chapter_metadata['source_filename'],
            
            'chapter_number': chapter_metadata['chapter_number'],
            'chapter_name': chapter_metadata['chapter_name'],
            'chapter_summary': chapter_metadata['chapter_summary'],
            'chapter_page_count': chapter_metadata['chapter_page_count'],
            
            'section_number': section['section_number'],
            'section_summary': f"{section['hierarchy']}\n\n{section['gpt_summary']}",  # Hierarchy + GPT summary as designed
            'section_page_count': section_page_count,
            'section_references': [f"ch{chapter_num}_{ref}" for ref in cross_refs.get(section['section_number'], [])],
            
            # Include content with embedded page tags
            'section_content': section['content']
        }
        
        section_records.append(section_record)
    
    log_progress(f"  ✅ Chapter {chapter_num} processing complete: {len(section_records)} sections")
    
    return section_records

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
    log_progress("🚀 Starting Stage 2: Section Processing (Version 3)")
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
    all_section_records = []
    
    for chapter_num in sorted(chapters.keys()):
        pages = chapters[chapter_num]
        section_records = process_chapter(chapter_num, pages, client)
        all_section_records.extend(section_records)
    
    # Note: We skip unassigned pages as they don't belong to chapters
    if unassigned_pages:
        log_progress(f"⚠️ Skipping {len(unassigned_pages)} unassigned pages (no chapter)")
    
    # Save output
    log_progress("-" * 70)
    log_progress(f"💾 Saving {len(all_section_records)} section records...")
    
    try:
        output_json = json.dumps(all_section_records, indent=2, ensure_ascii=False)
        output_bytes = output_json.encode('utf-8')
        
        if write_to_nas(share_name, output_path, output_bytes):
            log_progress(f"✅ Successfully saved output to {share_name}/{output_path}")
        else:
            log_progress("❌ Failed to write output to NAS")
    except Exception as e:
        log_progress(f"❌ Error saving output: {e}")
    
    # Upload log file
    try:
        log_file_name = f"stage2_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    print(f"  Output: {len(all_section_records)} section records")
    print(f"  Output file: {share_name}/{output_path}")
    print("=" * 70)
    print("✅ Stage 2 Completed")

if __name__ == "__main__":
    run_stage2()