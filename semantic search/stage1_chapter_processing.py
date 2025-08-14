# -*- coding: utf-8 -*-
"""
Stage 1: Chapter Processing - ROBUST VERSION with Improved Tool Calling

Key improvements:
- Enforces tool calling for consistent JSON responses
- Robust retry mechanism specifically for missing tags
- Better handling of segmented chapters
- Validates tool responses before accepting them
"""

import os
import json
import traceback
import re
import time
import logging
import requests
import tempfile
import socket
import io
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
from datetime import datetime
from collections import defaultdict

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
NAS_INPUT_JSON_PATH = "semantic_search/prep_output/ey/ey_prep_output_with_chapters.json"
NAS_OUTPUT_PATH = "semantic_search/pipeline_output/stage1"
NAS_LOG_PATH = "semantic_search/pipeline_output/logs"
OUTPUT_FILENAME = "stage1_page_records.json"

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
GPT_INPUT_TOKEN_LIMIT = 80000  # Maximum tokens for input/prompt
MAX_COMPLETION_TOKENS = 4000   # Maximum tokens for output/completion
TEMPERATURE = 0.3
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY = 5
TOKEN_BUFFER = 2000  # Safety buffer for prompt overhead

# Retry parameters specifically for tool response validation
TOOL_RESPONSE_RETRIES = 5  # More retries for getting proper tool responses
TOOL_RESPONSE_RETRY_DELAY = 3

# --- Token Cost ---
PROMPT_TOKEN_COST = 0.01
COMPLETION_TOKEN_COST = 0.03

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
# NAS Helper Functions (Simplified for brevity)
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
    """Setup logging with controlled verbosity - fixed duplication."""
    temp_log = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log')
    temp_log_path = temp_log.name
    temp_log.close()
    
    # Clear any existing handlers to prevent duplication
    logging.root.handlers = []
    
    log_level = logging.DEBUG if VERBOSE_LOGGING else logging.WARNING
    
    # Only add file handler to root logger (no console handler)
    # This prevents duplicate console output
    root_file_handler = logging.FileHandler(temp_log_path)
    root_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[root_file_handler]  # Only file handler, no StreamHandler
    )
    
    # Progress logger handles all console output
    progress_logger = logging.getLogger('progress')
    progress_logger.setLevel(logging.INFO)
    progress_logger.propagate = False  # Don't propagate to root
    
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
        console_handler.setLevel(logging.WARNING)  # Only warnings and errors
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
    # Handle None or empty ssl_verify_path safely
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
# IMPROVED API Call with Strict Tool Enforcement
# ==============================================================================

def call_gpt_with_tool_enforcement(client, model, messages, max_tokens, temperature, tool_schema):
    """
    Makes API call with STRICT tool enforcement.
    Will retry if response doesn't use the specified tool.
    """
    tool_name = tool_schema["function"]["name"]
    
    # Validate messages list is not empty
    if not messages:
        logging.error("Messages list is empty")
        return None, None
    
    for attempt in range(TOOL_RESPONSE_RETRIES):
        try:
            if attempt > 0:
                logging.info(f"Tool response retry {attempt + 1}/{TOOL_RESPONSE_RETRIES}")
                # Add enforcement message to prompt
                enforcement_msg = {
                    "role": "system",
                    "content": f"CRITICAL: You MUST use the '{tool_name}' tool to provide your response. Do not respond with plain text."
                }
                # Insert enforcement message before the last user message
                # Safe because we checked messages is not empty above
                # messages[-1:] returns a list, so this is correct
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
            
            # Validate we got a tool call
            if not response_message.tool_calls:
                logging.warning(f"Attempt {attempt + 1}: No tool calls in response")
                time.sleep(TOOL_RESPONSE_RETRY_DELAY)
                continue
            
            tool_call = response_message.tool_calls[0]
            
            # Validate it's the correct tool
            if tool_call.function.name != tool_name:
                logging.warning(f"Attempt {attempt + 1}: Wrong tool used: {tool_call.function.name}")
                time.sleep(TOOL_RESPONSE_RETRY_DELAY)
                continue
            
            # Parse and validate the tool arguments
            try:
                function_args = json.loads(tool_call.function.arguments)
                
                # Validate required fields are present and non-empty
                if 'summary' not in function_args or not function_args['summary']:
                    logging.warning(f"Attempt {attempt + 1}: Missing or empty summary")
                    time.sleep(TOOL_RESPONSE_RETRY_DELAY)
                    continue
                
                if 'tags' not in function_args or not isinstance(function_args['tags'], list):
                    logging.warning(f"Attempt {attempt + 1}: Missing or invalid tags")
                    time.sleep(TOOL_RESPONSE_RETRY_DELAY)
                    continue
                
                # Validate we have at least some tags
                if len(function_args['tags']) < 3:
                    logging.warning(f"Attempt {attempt + 1}: Too few tags ({len(function_args['tags'])})")
                    time.sleep(TOOL_RESPONSE_RETRY_DELAY)
                    continue
                
                # Success! Return the validated response
                return function_args, usage_info
                
            except json.JSONDecodeError as e:
                # Log the actual malformed JSON for debugging
                logging.warning(f"Attempt {attempt + 1}: Invalid JSON in tool arguments: {e}")
                logging.debug(f"Malformed JSON content: {tool_call.function.arguments[:500]}...")  # First 500 chars
                time.sleep(TOOL_RESPONSE_RETRY_DELAY)
                continue
                
        except APIError as e:
            logging.warning(f"API Error on attempt {attempt + 1}: {e}")
            time.sleep(TOOL_RESPONSE_RETRY_DELAY * (2 ** min(attempt, 3)))
        except Exception as e:
            logging.warning(f"Unexpected error on attempt {attempt + 1}: {e}")
            time.sleep(TOOL_RESPONSE_RETRY_DELAY)
    
    # All retries exhausted
    logging.error(f"Failed to get valid tool response after {TOOL_RESPONSE_RETRIES} attempts")
    return None, None

# ==============================================================================
# IMPROVED GPT Prompting with Better Structure
# ==============================================================================

CHAPTER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "provide_chapter_analysis",
        "description": "Provides structured analysis of chapter content including summary and tags",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Comprehensive summary with sections: Purpose, Key Topics/Standards, Context/Applicability, Key Outcomes/Decisions"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 5,
                    "maxItems": 15,
                    "description": "Specific topic tags for retrieval (standards, concepts, procedures, terms)"
                }
            },
            "required": ["summary", "tags"],
            "additionalProperties": False
        }
    }
}

def build_chapter_analysis_prompt(segment_text, prev_summary=None, prev_tags=None, is_final_segment=False):
    """
    Builds prompt using CO-STAR + XML format with explicit tool requirement.
    """
    system_prompt = """<role>
You are an expert financial reporting specialist analyzing EY technical accounting guidance.
</role>

<context>
You are processing content from comprehensive accounting guidance manuals covering IFRS and US GAAP.
The content will be used to build a searchable knowledge base for accounting professionals.
</context>

<objective>
Extract and structure key information from the provided text segment to create:
1. A detailed, structured summary following specific guidelines
2. Granular topic tags for efficient retrieval
</objective>

<style>
- Technical and precise
- Structured with clear sections
- Comprehensive yet concise
- Professional tone
</style>

<tone>
Expert, analytical, objective
</tone>

<audience>
Accounting professionals requiring specific technical guidance
</audience>

<response_format>
YOU MUST use the 'provide_chapter_analysis' tool to structure your response.
DO NOT provide a plain text response.
</response_format>"""

    # Build user prompt
    user_prompt_parts = []
    
    # Add previous context if available
    if prev_summary or prev_tags:
        user_prompt_parts.append("<previous_context>")
        if prev_summary:
            user_prompt_parts.append(f"<previous_summary>\n{prev_summary}\n</previous_summary>")
        if prev_tags:
            user_prompt_parts.append(f"<previous_tags>\n{json.dumps(prev_tags)}\n</previous_tags>")
        user_prompt_parts.append("</previous_context>")
    
    # Add current segment
    user_prompt_parts.append(f"<current_segment>\n{segment_text}\n</current_segment>")
    
    # Add specific instructions
    user_prompt_parts.append("<instructions>")
    user_prompt_parts.append("<summary_requirements>")
    user_prompt_parts.append("""Create a comprehensive summary with these REQUIRED sections:

**Purpose:** State the primary objective of this content (1-2 sentences)

**Key Topics/Standards:** List and explain:
- Specific accounting standards referenced (e.g., IFRS 16, ASC 842)
- Core concepts and principles discussed
- Important procedures or methodologies

**Context/Applicability:** Describe:
- Which entities or transactions this applies to
- Industry-specific considerations
- Exceptions or special cases

**Key Outcomes/Decisions:** Identify:
- Critical judgments required
- Decision points for practitioners
- Important implications""")
    user_prompt_parts.append("</summary_requirements>")
    
    user_prompt_parts.append("<tag_requirements>")
    user_prompt_parts.append("""Generate 5-15 specific tags including:
- Standard references (e.g., 'IFRS 15', 'ASC 606')
- Technical concepts (e.g., 'revenue recognition', 'lease classification')
- Procedures/methods (e.g., 'five-step model', 'expected credit loss')
- Key defined terms (e.g., 'performance obligation', 'right-of-use asset')
- Application contexts (e.g., 'software industry', 'transition provisions')

Tags must be:
- Specific and granular (not generic)
- Directly mentioned or clearly implied in the text
- Useful for search and retrieval""")
    user_prompt_parts.append("</tag_requirements>")
    
    # Add segment-specific guidance
    if is_final_segment and (prev_summary or prev_tags):
        user_prompt_parts.append("""<task>
This is the FINAL segment. Synthesize ALL information from previous and current segments.
Ensure the summary and tags comprehensively cover the ENTIRE chapter content.
</task>""")
    elif prev_summary or prev_tags:
        user_prompt_parts.append("""<task>
Integrate this segment with previous context. 
Update and expand the summary and tags to include new information.
Maintain continuity with previous analysis.
</task>""")
    else:
        user_prompt_parts.append("""<task>
Analyze this initial segment and create the foundation summary and tags.
Focus only on the content provided in the current segment.
</task>""")
    
    user_prompt_parts.append("</instructions>")
    
    user_prompt_parts.append("""<critical_requirement>
YOU MUST USE THE 'provide_chapter_analysis' TOOL TO PROVIDE YOUR RESPONSE.
The tool must include both 'summary' and 'tags' fields with appropriate content.
</critical_requirement>""")
    
    user_prompt = "\n".join(user_prompt_parts)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    return messages

def process_chapter_segment_robust(segment_text, client, model_name, prev_summary=None, prev_tags=None, is_final_segment=False):
    """
    Process a chapter segment with robust tool calling and validation.
    """
    messages = build_chapter_analysis_prompt(segment_text, prev_summary, prev_tags, is_final_segment)
    
    # Call API with strict tool enforcement
    result, usage_info = call_gpt_with_tool_enforcement(
        client=client,
        model=model_name,
        messages=messages,
        max_tokens=MAX_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        tool_schema=CHAPTER_TOOL_SCHEMA
    )
    
    if result:
        # Log success
        if usage_info and VERBOSE_LOGGING:
            prompt_tokens = usage_info.prompt_tokens
            completion_tokens = usage_info.completion_tokens
            total_cost = (prompt_tokens / 1000) * PROMPT_TOKEN_COST + (completion_tokens / 1000) * COMPLETION_TOKEN_COST
            logging.debug(f"API Usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Cost: ${total_cost:.4f}")
        
        # Validate and clean the result
        summary = result.get('summary', '').strip()
        tags = result.get('tags', [])
        
        # Filter out empty or duplicate tags
        tags = list(set(tag.strip() for tag in tags if tag and tag.strip()))
        
        return {'summary': summary, 'tags': tags}
    
    return None

def get_chapter_summary_and_tags_robust(chapter_text: str, client: OpenAI, model_name: str = MODEL_NAME_CHAT) -> Tuple[Optional[str], Optional[List[str]]]:
    """
    Generate summary and tags with improved segmentation handling.
    """
    total_tokens = count_tokens(chapter_text)
    total_chars = len(chapter_text)
    
    # Ensure we have positive available tokens
    available_for_content = max(1000, GPT_INPUT_TOKEN_LIMIT - TOKEN_BUFFER)  # Minimum 1000 tokens
    needs_segmentation = total_tokens > available_for_content
    
    if needs_segmentation:
        # Calculate number of segments needed using ceiling division
        # Ensure available_for_content is positive to avoid division by zero
        if available_for_content <= 0:
            logging.error(f"Invalid available_for_content: {available_for_content}")
            return None, None
        
        num_segments = max(1, (total_tokens + available_for_content - 1) // available_for_content)
        
        # Calculate target tokens per segment for better distribution
        # num_segments is guaranteed to be >= 1 from above
        target_tokens_per_segment = max(1, total_tokens // num_segments)
        
        # Estimate characters per token for this specific text
        chars_per_token = total_chars / total_tokens if total_tokens > 0 else 3.5
        
        # Calculate segment length in characters
        segment_len_chars = int(target_tokens_per_segment * chars_per_token)
        
        # Create segments with correct indexing
        segments = []
        start = 0
        
        for i in range(num_segments):
            # Check if we've already consumed all text
            if start >= len(chapter_text):
                break
                
            if i == num_segments - 1:
                # Last segment gets all remaining text from start to end
                segment = chapter_text[start:]
            else:
                # Calculate end position for this segment
                end = min(start + segment_len_chars, len(chapter_text))
                segment = chapter_text[start:end]
                start = end  # Update start for next iteration
            
            # Only add non-empty segments
            if segment and segment.strip():
                segments.append(segment)
        
        # Log segment distribution for debugging
        if VERBOSE_LOGGING:
            logging.debug(f"Segmentation: {total_tokens:,} tokens into {num_segments} segments")
            logging.debug(f"Target per segment: {target_tokens_per_segment:,} tokens")
            for idx, seg in enumerate(segments):
                seg_tokens = count_tokens(seg)
                logging.debug(f"Segment {idx+1}: {len(seg):,} chars, {seg_tokens:,} tokens")
        
        log_progress(f"  📄 Chapter size: {total_chars:,} chars ({total_tokens:,} tokens)")
        log_progress(f"  ✂️  Split into {len(segments)} segments for processing")
        
        # Show segment breakdown
        total_segment_tokens = 0
        for idx, seg in enumerate(segments):
            seg_tokens = count_tokens(seg)
            total_segment_tokens += seg_tokens
            if seg_tokens == 0:
                log_progress(f"    ⚠️ Segment {idx+1}: EMPTY (0 tokens) - will be skipped")
            else:
                log_progress(f"    📑 Segment {idx+1}: {len(seg):,} chars (~{seg_tokens:,} tokens)")
        
        # Use proportional threshold for token count validation (1% tolerance)
        token_difference = abs(total_segment_tokens - total_tokens)
        token_tolerance = max(100, int(total_tokens * 0.01))  # 1% or minimum 100 tokens
        if token_difference > token_tolerance:
            log_progress(f"    ⚠️ Token count mismatch: original {total_tokens:,} vs sum {total_segment_tokens:,} (diff: {token_difference:,})")
        
        current_summary = None
        current_tags = []
        all_tags_collected = set()  # Track all unique tags across segments
        
        # Filter out empty segments before processing
        non_empty_segments = [(i, seg) for i, seg in enumerate(segments) if seg.strip()]
        
        if len(non_empty_segments) == 0:
            log_progress("  ❌ All segments are empty after splitting!")
            return None, None
        
        if len(non_empty_segments) < len(segments):
            log_progress(f"  ⚠️ Filtered out {len(segments) - len(non_empty_segments)} empty segment(s)")
        
        for seg_idx, (original_idx, segment_text) in enumerate(non_empty_segments):
            segment_tokens = count_tokens(segment_text)
            
            # Determine if this is the final segment (clearer logic)
            # seg_idx is the index in the non_empty_segments list
            is_final = (seg_idx == len(non_empty_segments) - 1)
            
            log_progress(f"  🔄 Processing segment {seg_idx+1}/{len(non_empty_segments)} ({segment_tokens:,} tokens)...", end='')
            
            # Process segment with retries
            segment_result = None
            for retry in range(3):
                try:
                    segment_result = process_chapter_segment_robust(
                        segment_text, client, model_name,
                        prev_summary=current_summary, 
                        prev_tags=list(all_tags_collected) if all_tags_collected else None,
                        is_final_segment=is_final
                    )
                    if segment_result:
                        break
                    elif retry < 2:
                        logging.info(f"Segment {seg_idx+1} processing failed, retrying...")
                        time.sleep(5 * (2 ** retry))
                except Exception as e:
                    logging.warning(f"Exception processing segment {seg_idx+1}: {e}")
                    if retry < 2:
                        time.sleep(5 * (2 ** retry))
            
            if segment_result:
                current_summary = segment_result.get('summary')
                segment_tags = segment_result.get('tags', [])
                
                # Collect all unique tags
                all_tags_collected.update(segment_tags)
                current_tags = list(all_tags_collected)
                
                log_progress(f" ✅ (added {len(segment_tags)} tags)")
            else:
                log_progress(" ❌ Failed")
                # Continue with what we have
        
        # Return the final accumulated results
        if current_summary:
            # Limit tags to 15 most relevant (if we have too many)
            if len(current_tags) > 15:
                # In production, you might want to use a more sophisticated selection
                current_tags = current_tags[:15]
            return current_summary, current_tags
            
    else:
        # Single call processing
        log_progress(f"  📄 Chapter size: {total_chars:,} chars ({total_tokens:,} tokens)")
        log_progress(f"  🔄 Processing in single call...", end='')
        
        result = None
        for retry in range(3):
            try:
                result = process_chapter_segment_robust(
                    chapter_text, client, model_name, is_final_segment=True
                )
                if result:
                    break
                elif retry < 2:
                    logging.info("Single call processing failed, retrying...")
                    time.sleep(5 * (2 ** retry))
            except Exception as e:
                logging.warning(f"Exception in single call: {e}")
                if retry < 2:
                    time.sleep(5 * (2 ** retry))
        
        if result:
            log_progress(" ✅")
            return result.get('summary'), result.get('tags')
        else:
            log_progress(" ❌ Failed")
    
    return None, None

# ==============================================================================
# Main Processing Functions
# ==============================================================================

def group_pages_by_chapter(json_data: List[Dict]) -> Dict[int, List[Dict]]:
    """Groups page records by chapter_number."""
    chapters = defaultdict(list)
    
    for record in json_data:
        chapter_num = record.get('chapter_number')
        if chapter_num is not None:
            chapters[chapter_num].append(record)
    
    for chapter_num in chapters:
        chapters[chapter_num].sort(key=lambda x: x.get('page_number', 0))
    
    return dict(chapters)

def process_chapter_pages(chapter_num: int, pages: List[Dict], client: Optional[OpenAI]) -> List[Dict]:
    """Process pages in a chapter with robust tag generation."""
    # Check if pages list is empty
    if not pages:
        log_progress(f"⚠️ Chapter {chapter_num} has no pages")
        return []
    
    first_page = pages[0]
    chapter_name = first_page.get('chapter_name', f'Chapter {chapter_num}')
    chapter_filename = first_page.get('filename', 'unknown.pdf')
    
    pdf_page_numbers = [p.get('page_number', 0) for p in pages]
    
    log_progress("")
    log_progress(f"📚 Chapter {chapter_num}: {chapter_name}")
    log_progress(f"  📁 File: {chapter_filename}")
    log_progress(f"  📄 Pages: {min(pdf_page_numbers)}-{max(pdf_page_numbers)} ({len(pages)} pages)")
    
    # Concatenate all page content
    content_parts = []
    for page in pages:
        page_content = page.get('content', '')
        if page_content:
            content_parts.append(page_content)
    
    concatenated_content = "\n\n".join(content_parts)
    del content_parts  # Free memory
    
    # Generate summary and tags with robust method
    chapter_summary, chapter_tags = None, None
    if client:
        try:
            chapter_summary, chapter_tags = get_chapter_summary_and_tags_robust(
                concatenated_content, client, model_name=MODEL_NAME_CHAT
            )
            if chapter_summary and chapter_tags:
                log_progress(f"  ✅ Generated summary and {len(chapter_tags)} tags")
                if VERBOSE_LOGGING:
                    logging.debug(f"  Tags generated: {chapter_tags}")
            else:
                log_progress(f"  ⚠️ Failed to generate summary/tags")
        except Exception as e:
            log_progress(f"  ❌ Error: {str(e)[:100]}")
            logging.error(f"Full error: {e}")
    else:
        log_progress("  ⚠️ OpenAI client not available")
    
    # Apply to all pages
    enriched_pages = []
    chapter_token_count = count_tokens(concatenated_content)
    
    for page in pages:
        enriched_page = page.copy()
        enriched_page['chapter_summary'] = chapter_summary
        enriched_page['chapter_tags'] = chapter_tags
        enriched_page['chapter_token_count'] = chapter_token_count
        
        page_content = page.get('content', '')
        enriched_page['page_token_count'] = count_tokens(page_content)
        enriched_page['pdf_filename'] = page.get('filename')
        enriched_page['pdf_page_number'] = page.get('page_number')
        
        enriched_pages.append(enriched_page)
    
    return enriched_pages

def process_unassigned_pages(pages: List[Dict]) -> List[Dict]:
    """Process pages without chapter assignment."""
    if not pages:
        return []
    
    log_progress(f"📄 Processing {len(pages)} unassigned pages")
    
    enriched_pages = []
    for page in pages:
        enriched_page = page.copy()
        enriched_page['chapter_summary'] = None
        enriched_page['chapter_tags'] = None
        enriched_page['chapter_token_count'] = None
        
        page_content = page.get('content', '')
        enriched_page['page_token_count'] = count_tokens(page_content)
        enriched_page['pdf_filename'] = page.get('filename')
        enriched_page['pdf_page_number'] = page.get('page_number')
        
        enriched_pages.append(enriched_page)
    
    return enriched_pages

def cleanup_logging_handlers():
    """Safely cleanup logging handlers."""
    # Clean up progress logger handlers
    progress_logger = logging.getLogger('progress')
    handlers_to_remove = list(progress_logger.handlers)  # Create a snapshot
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
    
    # Clean up root logger handlers
    root_handlers_to_remove = list(logging.root.handlers)  # Create a snapshot
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
    
    # Clear the handlers list to be sure
    progress_logger.handlers.clear()
    logging.root.handlers.clear()

def run_stage1():
    """Main function to execute Stage 1 processing."""
    if not validate_configuration():
        return
    
    temp_log_path = setup_logging()
    
    log_progress("=" * 70)
    log_progress("🚀 Starting Stage 1: Chapter Processing (Robust Version)")
    log_progress("=" * 70)
    
    _setup_ssl_from_nas()
    
    share_name = NAS_PARAMS["share"]
    output_path_relative = os.path.join(NAS_OUTPUT_PATH, OUTPUT_FILENAME).replace('\\', '/')
    
    # Load input JSON
    log_progress("📥 Loading input JSON from NAS...")
    input_json_bytes = read_from_nas(share_name, NAS_INPUT_JSON_PATH)
    
    if not input_json_bytes:
        log_progress(f"❌ Failed to read input JSON")
        return
    
    try:
        input_data = json.loads(input_json_bytes.decode('utf-8'))
        if not isinstance(input_data, list):
            log_progress("❌ Input JSON is not a list")
            return
        log_progress(f"✅ Loaded {len(input_data)} page records")
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
            log_progress("⚠️ Failed to initialize OpenAI client")
    else:
        log_progress("⚠️ OpenAI library not installed")
    
    # Group and process pages
    chapters = group_pages_by_chapter(input_data)
    unassigned_pages = [r for r in input_data if r.get('chapter_number') is None]
    
    log_progress(f"📊 Found {len(chapters)} chapters and {len(unassigned_pages)} unassigned pages")
    log_progress("-" * 70)
    
    all_enriched_pages = []
    
    for chapter_num in sorted(chapters.keys()):
        pages = chapters[chapter_num]
        enriched_pages = process_chapter_pages(chapter_num, pages, client)
        all_enriched_pages.extend(enriched_pages)
    
    if unassigned_pages:
        enriched_unassigned = process_unassigned_pages(unassigned_pages)
        all_enriched_pages.extend(enriched_unassigned)
    
    # Sort by source_page_number to maintain original document order
    all_enriched_pages.sort(key=lambda x: x.get('source_page_number', x.get('page_number', 0)))
    
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
        log_file_name = f"stage1_robust_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    
    # Final summary
    print("=" * 70)
    print("📊 Stage 1 Summary")
    print("-" * 70)
    print(f"  Input: {len(input_data)} page records")
    print(f"  Output: {len(all_enriched_pages)} enriched page records")
    print(f"  Chapters processed: {len(chapters)}")
    if unassigned_pages:
        print(f"  Unassigned pages: {len(unassigned_pages)}")
    print(f"  Output file: {share_name}/{output_path_relative}")
    print("=" * 70)
    print("✅ Stage 1 Completed")

if __name__ == "__main__":
    run_stage1()