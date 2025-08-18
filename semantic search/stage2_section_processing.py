#!/usr/bin/env python3
"""
Stage 2: Section Processing with Natural Page Tags and GPT Summaries - Version 5
Creates section-based records with natural page boundaries and enriched summaries

Key Features:
- Natural page tag preservation (from v4)
- GPT-generated section summaries (from v3)
- Correct input/output schema alignment with Stage 1
- Section hierarchy in summary field

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

    def tqdm(x, **kwargs):
        return x

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
    "port": 445,  # Default SMB port (can be 139)
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

# --- Section Merging Thresholds ---
MIN_SECTION_TOKENS = 250  # Sections below this trigger merging
MAX_SECTION_TOKENS = 750  # Maximum tokens after merging
ULTRA_SMALL_THRESHOLD = 25  # Very small sections get aggressive merging

# --- Section Identification Parameters ---
MAX_HEADING_LEVEL = 6  # Consider all heading levels (H1-H6)

# --- Sample Processing Limit ---
SAMPLE_CHAPTER_LIMIT = None  # Set to None to process all chapters, or a number to limit

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

    if NAS_PARAMS["ip"] == "your_nas_ip":
        errors.append("NAS IP address not configured")
    if NAS_PARAMS["share"] == "your_share_name":
        errors.append("NAS share name not configured")
    if NAS_PARAMS["user"] == "your_nas_user":
        errors.append("NAS username not configured")
    if NAS_PARAMS["password"] == "your_nas_password":
        errors.append("NAS password not configured")
    if BASE_URL == "https://api.example.com/v1":
        errors.append("API base URL not configured")
    if OAUTH_URL == "https://api.example.com/oauth/token":
        errors.append("OAuth URL not configured")
    if CLIENT_ID == "your_client_id":
        errors.append("Client ID not configured")
    if CLIENT_SECRET == "your_client_secret":
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

    logging.root.handlers = []

    log_level = logging.DEBUG if VERBOSE_LOGGING else logging.WARNING

    root_file_handler = logging.FileHandler(temp_log_path)
    root_file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[root_file_handler]
    )

    progress_logger = logging.getLogger("progress")
    progress_logger.setLevel(logging.INFO)
    progress_logger.propagate = False

    progress_console_handler = logging.StreamHandler()
    progress_console_handler.setFormatter(logging.Formatter("%(message)s"))
    progress_logger.addHandler(progress_console_handler)

    progress_file_handler = logging.FileHandler(temp_log_path)
    progress_file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    progress_logger.addHandler(progress_file_handler)

    if VERBOSE_LOGGING:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        console_handler.setLevel(logging.WARNING)
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
# Token Counting
# ==============================================================================


def count_tokens(text: str) -> int:
    """Estimates token count using empirically-calibrated formula."""
    if not text:
        return 0

    char_count = len(text)
    if char_count == 0:
        return 0

    CHARS_PER_TOKEN_ESTIMATE = 3.5
    MIN_CHARS_PER_TOKEN = 2
    MAX_CHARS_PER_TOKEN = 10

    estimated_tokens = int(char_count / CHARS_PER_TOKEN_ESTIMATE)

    max_possible_tokens = char_count // MIN_CHARS_PER_TOKEN
    min_possible_tokens = max(1, char_count // MAX_CHARS_PER_TOKEN)

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


def _get_oauth_token(
    oauth_url=OAUTH_URL, client_id=CLIENT_ID, client_secret=CLIENT_SECRET, ssl_verify_path=SSL_LOCAL_PATH
) -> Optional[str]:
    """Retrieves OAuth token."""
    try:
        verify_path = ssl_verify_path if ssl_verify_path and Path(ssl_verify_path).exists() else True
    except (TypeError, OSError):
        verify_path = True

    payload = {"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret}
    try:
        import requests

        response = requests.post(oauth_url, data=payload, timeout=30, verify=verify_path)
        response.raise_for_status()
        token_data = response.json()
        oauth_token = token_data.get("access_token")
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
                    "content": f"CRITICAL: You MUST use the '{tool_name}' tool to provide your response. Do not respond with plain text.",
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
                stream=False,
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
# Page Tag Functions
# ==============================================================================


def clean_existing_page_tags(content: str) -> str:
    """Removes any existing Azure PageHeader/PageFooter tags from content."""
    if not content:
        return ""

    content = re.sub(r"<!--\s*PageHeader[^>]*?-->", "", content, flags=re.IGNORECASE)
    content = re.sub(r"<!--\s*PageFooter[^>]*?-->", "", content, flags=re.IGNORECASE)
    content = re.sub(r"<!--\s*Page[Nn]umber[^>]*?-->", "", content)
    content = re.sub(r"<!--\s*PageBreak[^>]*?-->", "", content)
    content = re.sub(r"\n\n\n+", "\n\n", content)

    return content.strip()


def embed_page_tags(pages: List[Dict]) -> str:
    """
    Combines page content with embedded HTML page tags.
    Each page gets header and footer tags with page number and reference.
    """
    if not pages:
        return ""

    content_parts = []

    for page in pages:
        # Use page_number (page within chapter PDF) for tags
        page_num = page.get("page_number")
        if page_num is None:
            logging.warning("Page missing page_number, skipping")
            continue

        page_ref = page.get("page_reference") or ""
        content = page.get("content") or ""

        # Clean any existing page tags from the content
        clean_content = clean_existing_page_tags(content)

        # Escape special characters in page reference for HTML attributes
        page_ref_escaped = html.escape(page_ref, quote=True)

        # Add header tag
        header = f'<!-- PageHeader PageNumber="{page_num}" PageReference="{page_ref_escaped}" -->\n'
        content_parts.append(header)

        # Add cleaned content
        content_parts.append(clean_content)
        if clean_content and not clean_content.endswith("\n"):
            content_parts.append("\n")

        # Add footer tag
        footer = f'<!-- PageFooter PageNumber="{page_num}" PageReference="{page_ref_escaped}" -->\n'
        content_parts.append(footer)

    return "".join(content_parts)


def extract_page_metadata(content: str) -> Dict:
    """
    Extracts page range and boundary information from content.
    Does NOT modify the content, only analyzes it.
    """
    page_pattern = re.compile(r'<!-- Page(?:Header|Footer) PageNumber="(\d+)" PageReference="([^"]*)" -->')
    matches = list(page_pattern.finditer(content))

    if not matches:
        return {"section_start_page": None, "section_end_page": None, "section_page_count": 0}

    # Extract page numbers
    page_numbers = []
    for match in matches:
        page_num = int(match.group(1))
        page_numbers.append(page_num)

    # Get unique pages in order
    unique_pages = sorted(set(page_numbers))

    return {
        "section_start_page": min(unique_pages) if unique_pages else None,
        "section_end_page": max(unique_pages) if unique_pages else None,
        "section_page_count": len(unique_pages),
    }


def infer_page_boundaries(sections: List[Dict], full_content: str) -> List[Dict]:
    """
    Infers missing page boundaries for sections that fall within a single page.
    Uses position tracking and surrounding sections to determine the correct page.
    """
    # First, track the page context throughout the content
    page_pattern = re.compile(r'<!-- Page(?:Header|Footer) PageNumber="(\d+)" PageReference="([^"]*)" -->')
    page_positions = [(match.start(), int(match.group(1))) for match in page_pattern.finditer(full_content)]
    
    if not page_positions:
        return sections
    
    # Sort by position to ensure correct order
    page_positions.sort(key=lambda x: x[0])
    
    for section in sections:
        # If section already has page metadata, skip
        if section.get("section_start_page") is not None:
            continue
            
        # Find section position in full content
        section_start = full_content.find(section["content"])
        if section_start == -1:
            continue
            
        # Find the last page marker before this section
        current_page = None
        for pos, page_num in page_positions:
            if pos < section_start:
                current_page = page_num
            else:
                break
        
        # If we found a page context, use it
        if current_page is not None:
            section["section_start_page"] = current_page
            section["section_end_page"] = current_page
            section["section_page_count"] = 1
    
    # Second pass: use neighboring sections to fill any remaining gaps
    for i, section in enumerate(sections):
        if section.get("section_start_page") is not None:
            continue
            
        # Look at previous section with valid page info
        prev_page = None
        for j in range(i - 1, -1, -1):
            if sections[j].get("section_end_page") is not None:
                prev_page = sections[j]["section_end_page"]
                break
        
        # Look at next section with valid page info
        next_page = None
        for j in range(i + 1, len(sections)):
            if sections[j].get("section_start_page") is not None:
                next_page = sections[j]["section_start_page"]
                break
        
        # Infer based on neighbors
        if prev_page is not None and next_page is not None:
            # Section is between two known pages
            if prev_page == next_page:
                # All on the same page
                section["section_start_page"] = prev_page
                section["section_end_page"] = prev_page
                section["section_page_count"] = 1
            else:
                # Likely on the previous page's end or next page's start
                # Use the previous page as a conservative estimate
                section["section_start_page"] = prev_page
                section["section_end_page"] = prev_page
                section["section_page_count"] = 1
        elif prev_page is not None:
            # Only have previous context
            section["section_start_page"] = prev_page
            section["section_end_page"] = prev_page
            section["section_page_count"] = 1
        elif next_page is not None:
            # Only have next context
            section["section_start_page"] = next_page
            section["section_end_page"] = next_page
            section["section_page_count"] = 1
    
    return sections


# ==============================================================================
# Section Identification and Processing
# ==============================================================================


def split_by_heading_level(content: str, level: int, parent_title: str = "") -> List[Dict]:
    """
    Splits content by a specific heading level.
    Returns list of sections at that level.
    Adjusts boundaries to include page tags that appear right before headings.
    """
    pattern = re.compile(f"^(#{{{level}}})\\s+(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(content))
    
    if not matches:
        # No headings at this level, return content as single section
        page_metadata = extract_page_metadata(content)
        return [{
            "title": parent_title or "Content",
            "level": level,
            "content": content,
            "token_count": count_tokens(content),
            "parent_title": parent_title,
            **page_metadata,
        }]
    
    sections = []
    
    # Page tag pattern to check for headers/footers before headings
    page_tag_pattern = re.compile(r'<!-- Page(?:Header|Footer) PageNumber="\d+" PageReference="[^"]*" -->')
    
    # Adjust match positions to include page tags that appear right before headings
    adjusted_positions = []
    for i, match in enumerate(matches):
        start_pos = match.start()
        
        # Look backwards from the heading to find any page tags
        # Check the content immediately before the heading
        search_start = max(0, start_pos - 200)  # Look back up to 200 chars
        preceding_content = content[search_start:start_pos]
        
        # Find the last page tag before this heading
        page_tags_before = list(page_tag_pattern.finditer(preceding_content))
        if page_tags_before:
            last_tag = page_tags_before[-1]
            # Check if there's only whitespace between the tag and the heading
            between_content = preceding_content[last_tag.end():].strip()
            if not between_content:  # Only whitespace between tag and heading
                # Adjust start position to include this page tag
                actual_start = search_start + last_tag.start()
                adjusted_positions.append((actual_start, match))
            else:
                adjusted_positions.append((start_pos, match))
        else:
            adjusted_positions.append((start_pos, match))
    
    # Handle content before first heading
    first_heading_pos = adjusted_positions[0][0] if adjusted_positions else len(content)
    if first_heading_pos > 0:
        intro_content = content[:first_heading_pos].strip()
        if intro_content:
            page_metadata = extract_page_metadata(intro_content)
            sections.append({
                "title": parent_title or "Introduction",
                "level": level,
                "content": intro_content,
                "token_count": count_tokens(intro_content),
                "parent_title": parent_title,
                **page_metadata,
            })
    
    # Process each heading-defined section with adjusted boundaries
    for i, (start_pos, match) in enumerate(adjusted_positions):
        title = match.group(2).strip()
        
        if i < len(adjusted_positions) - 1:
            end_pos = adjusted_positions[i + 1][0]
        else:
            end_pos = len(content)
        
        section_content = content[start_pos:end_pos].strip()
        page_metadata = extract_page_metadata(section_content)
        
        sections.append({
            "title": title,
            "level": level,
            "content": section_content,
            "token_count": count_tokens(section_content),
            "parent_title": parent_title,
            **page_metadata,
        })
    
    return sections


def recursive_split_section(section: Dict, current_level: int, max_level: int = 6, page_threshold: int = 3) -> List[Dict]:
    """
    Recursively splits a section if it spans more than page_threshold pages.
    Stops at max_level (default 6 for H6).
    """
    # Calculate page span
    start_page = section.get("section_start_page")
    end_page = section.get("section_end_page")
    
    # If we can't determine pages, keep as is
    if start_page is None or end_page is None:
        section["splitting_level"] = current_level
        return [section]
    
    page_span = end_page - start_page + 1
    
    # If within threshold or at max level, return as is
    if page_span <= page_threshold or current_level >= max_level:
        section["splitting_level"] = current_level
        if page_span > page_threshold and current_level >= max_level:
            logging.info(f"Section '{section.get('title', 'Untitled')[:50]}' spans {page_span} pages but reached max level H{max_level}")
        return [section]
    
    # Log the split attempt
    logging.debug(f"Splitting section '{section.get('title', 'Untitled')[:50]}' (spans {page_span} pages) at H{current_level + 1}")
    
    # Try splitting at next level
    next_level = current_level + 1
    subsections = split_by_heading_level(
        section["content"], 
        next_level, 
        section["title"]
    )
    
    # If no meaningful split occurred (only 1 section)
    if len(subsections) <= 1:
        # No headings at this level, but section is still too large
        # Mark it as split at current level and return (don't skip to H6)
        section["splitting_level"] = current_level
        if next_level < max_level:
            logging.debug(f"No H{next_level} headings found in section '{section.get('title', 'Untitled')[:50]}', keeping as H{current_level}")
        return [section]
    
    # Process each subsection recursively
    result = []
    for subsection in subsections:
        # Inherit parent's page info if subsection has none
        if subsection.get("section_start_page") is None:
            subsection["section_start_page"] = section.get("section_start_page")
            subsection["section_end_page"] = section.get("section_end_page")
            subsection["section_page_count"] = section.get("section_page_count", 0)
        
        split_results = recursive_split_section(subsection, next_level, max_level, page_threshold)
        result.extend(split_results)
    
    return result


def hierarchical_split_sections(pages: List[Dict], chapter_metadata: Dict = None) -> List[Dict]:
    """
    Performs hierarchical splitting of chapter content.
    Starts with H1, recursively splits sections that span > 3 pages.
    """
    # Embed page tags
    full_content = embed_page_tags(pages)
    
    # Start with H1 level split
    initial_sections = split_by_heading_level(full_content, level=1, parent_title=chapter_metadata.get("chapter_name", "") if chapter_metadata else "")
    
    # Recursively split large sections
    final_sections = []
    for section in initial_sections:
        split_sections = recursive_split_section(section, current_level=1, max_level=6, page_threshold=3)
        final_sections.extend(split_sections)
    
    # Infer missing page boundaries
    final_sections = infer_page_boundaries(final_sections, full_content)
    
    # Renumber sections
    for idx, section in enumerate(final_sections):
        section["section_number"] = idx + 1
    
    return final_sections


def identify_sections_from_pages(pages: List[Dict], chapter_metadata: Dict = None) -> List[Dict]:
    """
    Identifies sections using hierarchical splitting approach.
    Adaptively splits based on page span threshold.
    """
    return hierarchical_split_sections(pages, chapter_metadata)


def generate_hierarchy_string(section: Dict, all_sections: List[Dict], current_idx: int) -> str:
    """
    Generates a breadcrumb-style hierarchy string for a section.
    """
    parts = []
    current_level = section.get("level", 1)

    level_titles = {}

    for i in range(current_idx):
        prev_section = all_sections[i]
        prev_level = prev_section.get("level", 1)
        level_titles[prev_level] = prev_section.get("title", "")
        for deeper_level in list(level_titles.keys()):
            if deeper_level > prev_level:
                del level_titles[deeper_level]

    for level in range(1, current_level):
        if level in level_titles:
            parts.append(level_titles[level])

    parts.append(section.get("title", ""))

    parts = [p for p in parts if p]
    return " > ".join(parts)


def merge_small_sections(sections: List[Dict]) -> List[Dict]:
    """
    Merges small sections with adjacent sections to avoid fragmentation.
    Respects heading level boundaries - only merges sections at the same level
    or child sections with their parent.
    Updates page metadata after merging.
    """
    if not sections:
        return sections

    merged = []
    consumed = set()  # Track which sections have been consumed by merging
    i = 0

    while i < len(sections):
        if i in consumed:
            i += 1
            continue

        current = sections[i]
        current_level = current.get("level", 1)

        if current["token_count"] < MIN_SECTION_TOKENS:
            # Try to merge with previous section first
            if merged:
                prev = merged[-1]
                prev_level = prev.get("level", 1)
                
                # Only merge if:
                # 1. Same level (sibling sections)
                # 2. Current is deeper level (child merging with parent)
                can_merge_prev = (current_level >= prev_level and 
                                 prev["token_count"] + current["token_count"] <= MAX_SECTION_TOKENS)
                
                if can_merge_prev:
                    prev["content"] += "\n" + current["content"]
                    prev["token_count"] += current["token_count"]

                    # Update page metadata after merging
                    combined_metadata = extract_page_metadata(prev["content"])
                    prev.update(combined_metadata)

                    consumed.add(i)  # Mark current as consumed
                    i += 1
                    continue

            # Try to merge with next section
            if i + 1 < len(sections) and i + 1 not in consumed:
                next_section = sections[i + 1]
                next_level = next_section.get("level", 1)
                
                # Only merge if:
                # 1. Same level (sibling sections)
                # 2. Next is deeper level (parent merging with child)
                can_merge_next = (next_level >= current_level and 
                                current["token_count"] + next_section["token_count"] <= MAX_SECTION_TOKENS)
                
                if can_merge_next:
                    current["content"] += "\n" + next_section["content"]
                    current["token_count"] += next_section["token_count"]

                    # Update page metadata after merging
                    combined_metadata = extract_page_metadata(current["content"])
                    current.update(combined_metadata)

                    merged.append(current)
                    consumed.add(i + 1)  # Mark next as consumed
                else:
                    # Can't merge, keep as is
                    merged.append(current)
            else:
                # Can't merge, keep as is
                merged.append(current)
        else:
            # Large enough, no merge needed
            merged.append(current)

        i += 1

    # Renumber sections after merging
    for idx, section in enumerate(merged):
        section["section_number"] = idx + 1

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
                    "description": "A condensed summary (2-3 sentences) that explains the section's purpose and content. Must naturally embed relevant accounting standards (e.g., IFRS 16, ASC 842) and key concepts within the narrative.",
                }
            },
            "required": ["section_summary"],
            "additionalProperties": False,
        },
    },
}


def build_section_analysis_prompt(
    section: Dict, chapter_summary: str, hierarchy: str, previous_summaries: List[str] = None
):
    """
    Builds prompt for section analysis using CO-STAR framework with XML structure.
    Uses structured XML tags for cleaner prompting.
    """
    # System prompt with role definition
    system_prompt = """<role>You are an expert financial reporting specialist analyzing EY technical accounting guidance.</role>
<expertise>Deep knowledge of IFRS, US GAAP, accounting standards, and technical implementation guidance.</expertise>"""

    # User prompt using CO-STAR framework
    user_prompt_parts = ["<prompt>"]

    # CONTEXT - Background information
    user_prompt_parts.append("<context>")
    user_prompt_parts.append("<document_type>EY Technical Accounting Guidance</document_type>")
    user_prompt_parts.append(f"<chapter_summary>{chapter_summary}</chapter_summary>")
    user_prompt_parts.append(f"<section_hierarchy>{hierarchy}</section_hierarchy>")

    # Add previous section summaries for continuity
    if previous_summaries and len(previous_summaries) > 0:
        recent_context = "\n\n".join(previous_summaries[-5:])  # Last 5 sections for context
        user_prompt_parts.append("<previous_sections>")
        user_prompt_parts.append(recent_context)
        user_prompt_parts.append("</previous_sections>")

    user_prompt_parts.append("</context>")

    # OBJECTIVE - What we want to achieve
    user_prompt_parts.append("<objective>")
    user_prompt_parts.append(
        """Create a condensed summary that:
1. Captures the essential purpose and content of this section
2. Naturally embeds relevant accounting standards and technical references
3. Provides sufficient detail for semantic search and retrieval
4. Maintains continuity with previous sections in the chapter"""
    )
    user_prompt_parts.append("</objective>")

    # STYLE - How to write
    user_prompt_parts.append("<style>")
    user_prompt_parts.append(
        """Technical and precise, using domain-specific terminology.
Embed standards naturally: "Explains IFRS 16 lease classification criteria including..."
Include specific references: "per ASC 842-10-15" when mentioned in content.
Write in present tense, third person."""
    )
    user_prompt_parts.append("</style>")

    # TONE - The voice to use
    user_prompt_parts.append("<tone>")
    user_prompt_parts.append("Professional, authoritative, and concise. Neutral and factual.")
    user_prompt_parts.append("</tone>")

    # AUDIENCE - Who will read this
    user_prompt_parts.append("<audience>")
    user_prompt_parts.append(
        "Professional accountants, auditors, and financial reporting specialists searching for specific technical guidance."
    )
    user_prompt_parts.append("</audience>")

    # The actual content to summarize
    user_prompt_parts.append("<current_section>")
    user_prompt_parts.append(section["content"])
    user_prompt_parts.append("</current_section>")

    # RESPONSE - Format requirements
    user_prompt_parts.append("<response_requirements>")
    user_prompt_parts.append(
        """EXACTLY 2-3 complete sentences.
Must be self-contained and understandable without reading the full section.
Naturally embed all relevant metadata, standards, and technical terms.
Focus on WHAT the section covers and WHY it matters."""
    )
    user_prompt_parts.append("</response_requirements>")

    user_prompt_parts.append(
        "<response_format>YOU MUST use the 'provide_section_analysis' tool to provide your response.</response_format>"
    )
    user_prompt_parts.append("</prompt>")

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": "\n".join(user_prompt_parts)}]

    return messages


def process_section_summary(
    section: Dict, chapter_metadata: Dict, hierarchy: str, previous_summaries: List[str], client: Optional[OpenAI]
) -> str:
    """
    Generates an enriched summary for a section with retry logic.
    Uses segmentation approach for large sections (similar to Stage 1).
    """
    if not client:
        return f"Section covering {section['title']}"

    # Validate section content exists and is not empty
    section_content = section.get("content", "")
    if not section_content or not section_content.strip():
        logging.warning(f"Section {section.get('section_number', '?')} has empty content. Using default summary.")
        return f"Section covering {section.get('title', 'untitled section')}"

    content_tokens = section.get("token_count", 0)

    # Additional validation for edge cases
    if content_tokens == 0:
        logging.warning(f"Section {section.get('section_number', '?')} has 0 tokens. Using default summary.")
        return f"Section covering {section.get('title', 'untitled section')}"

    available_tokens = GPT_INPUT_TOKEN_LIMIT - TOKEN_BUFFER - 2000

    # Ensure available_tokens is positive
    if available_tokens <= 0:
        logging.error(f"No available token space for processing: {available_tokens}")
        return f"Section covering {section.get('title', 'untitled section')}"

    # Check if segmentation is needed
    if content_tokens > available_tokens:
        logging.info(f"Section {section.get('section_number', '?')} requires segmentation: {content_tokens} tokens")

        # Calculate number of segments needed with protection against division by zero
        num_segments = max(1, (content_tokens + available_tokens - 1) // available_tokens)
        content = section_content  # Use the validated content from above
        total_chars = len(content)
        chars_per_token = total_chars / content_tokens if content_tokens > 0 else 3.5
        segment_len_chars = int(available_tokens * chars_per_token)

        # Create segments
        segments = []
        start = 0
        for i in range(num_segments):
            if start >= len(content):
                break

            if i == num_segments - 1:
                # Last segment gets remaining content
                segment = content[start:]
            else:
                # Find a good break point (sentence boundary)
                end = min(start + segment_len_chars, len(content))
                # Try to break at a sentence boundary
                last_period = content.rfind(". ", start, end)
                last_newline = content.rfind("\n", start, end)
                break_point = max(last_period, last_newline)
                if break_point > start:
                    end = break_point + 1
                segment = content[start:end]
                start = end

            if segment and segment.strip():
                segments.append(segment)

        # Process segments incrementally (like Stage 1)
        current_summary = None
        for seg_idx, segment_text in enumerate(segments):
            # is_final = seg_idx == len(segments) - 1  # Unused variable

            # Create a temporary section dict for this segment
            segment_section = section.copy()
            segment_section["content"] = segment_text

            # Build prompt with previous segment summary as context
            if current_summary:
                # Add the current accumulated summary to previous summaries
                segment_previous = previous_summaries + [f"Previous segment summary: {current_summary}"]
            else:
                segment_previous = previous_summaries

            # Build messages for this segment
            messages = build_section_analysis_prompt(
                segment_section, chapter_metadata.get("chapter_summary", ""), hierarchy, segment_previous
            )

            # Process segment
            for attempt in range(API_RETRY_ATTEMPTS):
                try:
                    result, usage_info = call_gpt_with_tool_enforcement(
                        client=client,
                        model=MODEL_NAME_CHAT,
                        messages=messages,
                        max_tokens=MAX_COMPLETION_TOKENS,
                        temperature=TEMPERATURE,
                        tool_schema=SECTION_TOOL_SCHEMA,
                    )

                    if result and "section_summary" in result:
                        current_summary = result["section_summary"]
                        if usage_info and VERBOSE_LOGGING:
                            prompt_tokens = usage_info.prompt_tokens
                            completion_tokens = usage_info.completion_tokens
                            total_cost = (prompt_tokens / 1000) * PROMPT_TOKEN_COST + (
                                completion_tokens / 1000
                            ) * COMPLETION_TOKEN_COST
                            logging.debug(
                                f"Segment {seg_idx+1}/{len(segments)} - Tokens: {prompt_tokens}+{completion_tokens}, Cost: ${total_cost:.4f}"
                            )
                        break

                except Exception as e:
                    logging.warning(f"Segment {seg_idx+1} attempt {attempt + 1}/{API_RETRY_ATTEMPTS} failed: {e}")
                    if attempt < API_RETRY_ATTEMPTS - 1:
                        time.sleep(API_RETRY_DELAY * (attempt + 1))

            if not current_summary:
                # Fallback if segment processing fails
                current_summary = f"Section segment {seg_idx+1} of {section.get('title', 'section')}"

        return current_summary if current_summary else f"Section covering {section.get('title', 'section')}"

    else:
        # Normal processing for sections within token limits
        messages = build_section_analysis_prompt(
            section, chapter_metadata.get("chapter_summary", ""), hierarchy, previous_summaries
        )

        for attempt in range(API_RETRY_ATTEMPTS):
            try:
                result, usage_info = call_gpt_with_tool_enforcement(
                    client=client,
                    model=MODEL_NAME_CHAT,
                    messages=messages,
                    max_tokens=MAX_COMPLETION_TOKENS,
                    temperature=TEMPERATURE,
                    tool_schema=SECTION_TOOL_SCHEMA,
                )

                if result and "section_summary" in result:
                    if usage_info and VERBOSE_LOGGING:
                        prompt_tokens = usage_info.prompt_tokens
                        completion_tokens = usage_info.completion_tokens
                        total_cost = (prompt_tokens / 1000) * PROMPT_TOKEN_COST + (
                            completion_tokens / 1000
                        ) * COMPLETION_TOKEN_COST
                        logging.debug(
                            f"Section summary tokens - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Cost: ${total_cost:.4f}"
                        )

                    return result["section_summary"]

            except Exception as e:
                logging.warning(f"Attempt {attempt + 1}/{API_RETRY_ATTEMPTS} failed for section summary: {e}")
                if attempt < API_RETRY_ATTEMPTS - 1:
                    time.sleep(API_RETRY_DELAY * (attempt + 1))

        logging.error(f"Failed to generate summary for section after {API_RETRY_ATTEMPTS} attempts")
        return f"Section covering {section.get('title', 'section')}"


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
        "document_id": first_page.get("document_id"),
        "filename": first_page.get("filename"),
        "filepath": first_page.get("filepath"),
        "source_filename": first_page.get("source_filename"),
        "chapter_number": chapter_num,
        "chapter_name": first_page.get("chapter_name", f"Chapter {chapter_num}"),
        "chapter_summary": first_page.get("chapter_summary"),
        "chapter_page_count": first_page.get("chapter_page_count"),
    }

    log_progress("")
    log_progress(f"📚 Processing Chapter {chapter_num}: {chapter_metadata['chapter_name']}")
    log_progress(f"  📄 {len(pages)} pages to process")

    # Step 1: Identify sections using hierarchical splitting
    sections = identify_sections_from_pages(pages, chapter_metadata)
    log_progress(f"  📑 After hierarchical splitting: {len(sections)} sections")
    
    # Log splitting details
    splitting_levels = {}
    for section in sections:
        level = section.get("splitting_level", section.get("level", 1))
        splitting_levels[level] = splitting_levels.get(level, 0) + 1
    
    if len(splitting_levels) > 1:
        level_summary = ", ".join([f"H{k}: {v}" for k, v in sorted(splitting_levels.items())])
        log_progress(f"  🎯 Splitting breakdown: {level_summary}")

    # Step 2: Merge small sections
    sections = merge_small_sections(sections)
    log_progress(f"  🔀 After merging small sections: {len(sections)} sections")

    # Step 3: Generate hierarchies for all sections
    for idx, section in enumerate(sections):
        hierarchy = generate_hierarchy_string(section, sections, idx)
        section["hierarchy"] = hierarchy

    # Step 4: Generate section summaries if client available
    if client:
        log_progress("  📝 Generating section summaries...")
        previous_summaries = []

        for idx, section in enumerate(tqdm(sections, desc=f"Chapter {chapter_num} Summaries")):
            hierarchy = section["hierarchy"]

            # Generate enriched summary with previous context
            gpt_summary = process_section_summary(section, chapter_metadata, hierarchy, previous_summaries, client)

            # Build section summary with hierarchy and GPT summary
            section["section_summary"] = f"{hierarchy}\n\n{gpt_summary}"

            # Add to previous summaries for next section's context
            previous_summaries.append(f"[Section {section['section_number']}] {section['section_summary']}")
    else:
        # No client, just use hierarchy as summary
        for section in sections:
            section["section_summary"] = section["hierarchy"]

    # Step 5: Build final section records with CORRECT schema (no single-page fields)
    section_records = []

    for section in sections:
        # Build section record - sections span multiple pages, so no single page fields!
        section_record = {
            # Document metadata
            "document_id": chapter_metadata["document_id"],
            "filename": chapter_metadata["filename"],
            "filepath": chapter_metadata["filepath"],
            "source_filename": chapter_metadata["source_filename"],
            # Chapter metadata
            "chapter_number": chapter_metadata["chapter_number"],
            "chapter_name": chapter_metadata["chapter_name"],
            "chapter_summary": chapter_metadata["chapter_summary"],
            "chapter_page_count": chapter_metadata["chapter_page_count"],
            # Section metadata - NO SINGLE PAGE FIELDS (they don't make sense for multi-page sections)
            "section_number": section["section_number"],
            "section_summary": section["section_summary"],  # Hierarchy + GPT summary
            # Section page RANGE information
            "section_start_page": section.get("section_start_page"),
            "section_end_page": section.get("section_end_page"),
            "section_page_count": section.get("section_page_count", 0),
            # Content with natural page tags showing actual page transitions
            "section_content": section["content"],
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
        chapter_num = record.get("chapter_number")
        if chapter_num is not None:
            chapters[chapter_num].append(record)
        else:
            unassigned.append(record)

    # Sort pages within each chapter by source_page_number
    for chapter_num in chapters:
        chapters[chapter_num].sort(key=lambda x: x.get("source_page_number", 0))

    return dict(chapters), unassigned


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


def run_stage2():
    """Main function to execute Stage 2 processing."""
    if not validate_configuration():
        return

    temp_log_path = setup_logging()

    log_progress("=" * 70)
    log_progress("🚀 Starting Stage 2: Section Processing (Version 5)")
    log_progress("=" * 70)

    _setup_ssl_from_nas()

    share_name = NAS_PARAMS["share"]
    input_path = os.path.join(NAS_INPUT_PATH, INPUT_FILENAME).replace("\\", "/")
    output_path = os.path.join(NAS_OUTPUT_PATH, OUTPUT_FILENAME).replace("\\", "/")

    # Load input JSON
    log_progress("📥 Loading Stage 1 output from NAS...")
    input_json_bytes = read_from_nas(share_name, input_path)

    if not input_json_bytes:
        log_progress("❌ Failed to read input JSON")
        return

    try:
        page_records = json.loads(input_json_bytes.decode("utf-8"))
        if not isinstance(page_records, list):
            log_progress("❌ Input JSON is not a list")
            return
        log_progress(f"✅ Loaded {len(page_records)} page records")
    except json.JSONDecodeError as e:
        log_progress(f"❌ Error decoding JSON: {e}")
        return

    # Validate input records have required fields
    required_fields = ["chapter_number", "content", "page_number"]
    valid_records = []
    invalid_count = 0

    for record in page_records:
        missing_fields = [field for field in required_fields if field not in record or record[field] is None]
        if missing_fields:
            invalid_count += 1
            logging.warning(
                f"Record missing required fields {missing_fields}: page {record.get('page_number', 'unknown')}"
            )
        else:
            valid_records.append(record)

    if invalid_count > 0:
        log_progress(f"⚠️ Skipped {invalid_count} records with missing required fields")

    if not valid_records:
        log_progress("❌ No valid records to process after validation")
        return

    page_records = valid_records

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

    # Apply sample limit if configured
    chapters_to_process = sorted(chapters.keys())
    if SAMPLE_CHAPTER_LIMIT is not None and SAMPLE_CHAPTER_LIMIT > 0:
        chapters_to_process = chapters_to_process[:SAMPLE_CHAPTER_LIMIT]
        log_progress(f"⚠️  SAMPLE MODE: Processing only first {SAMPLE_CHAPTER_LIMIT} chapters")

    log_progress("-" * 70)

    # Process each chapter
    all_section_records = []

    for chapter_num in chapters_to_process:
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
        output_bytes = output_json.encode("utf-8")

        if write_to_nas(share_name, output_path, output_bytes):
            log_progress(f"✅ Successfully saved output to {share_name}/{output_path}")
        else:
            log_progress("❌ Failed to write output to NAS")
    except Exception as e:
        log_progress(f"❌ Error saving output: {e}")

    # Upload log file
    try:
        log_file_name = f"stage2_v5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    print("📊 Stage 2 Summary")
    print("-" * 70)
    print(f"  Input: {len(page_records)} page records")
    print(f"  Chapters available: {len(chapters)}")
    print(f"  Chapters processed: {len(chapters_to_process)}")
    if SAMPLE_CHAPTER_LIMIT:
        print(f"  ⚠️  SAMPLE MODE: Limited to {SAMPLE_CHAPTER_LIMIT} chapters")
    print(f"  Output: {len(all_section_records)} section records")
    print(f"  Output file: {share_name}/{output_path}")
    print("=" * 70)
    print("✅ Stage 2 Completed")


if __name__ == "__main__":
    run_stage2()
