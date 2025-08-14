# -*- coding: utf-8 -*-
"""
Stage 1: Chapter Processing (Page-Level Records Version) - FIXED

Purpose:
Processes JSON output from Chapter Assignment Tool containing page-level records.
Generates chapter summaries and tags once per chapter using an LLM, then applies
them to all pages in that chapter. Keeps page-level records for perfect citation tracking.

Input: JSON file from Chapter Assignment Tool with split PDF references
Output: JSON file with enriched page-level records (one record per page)
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
NAS_INPUT_JSON_PATH = "semantic_search/prep_output/ey/ey_prep_output_with_chapters.json"  # TODO: Adjust path
NAS_OUTPUT_PATH = "semantic_search/pipeline_output/stage1"
NAS_LOG_PATH = "semantic_search/pipeline_output/logs"
OUTPUT_FILENAME = "stage1_page_records.json"

# --- CA Bundle Configuration ---
NAS_SSL_CERT_PATH = "certificates/rbc-ca-bundle.cer"  # TODO: Adjust to match your NAS location
SSL_LOCAL_PATH = "/tmp/rbc-ca-bundle.cer"  # Temp path for cert

# --- API Configuration (Hardcoded) ---
BASE_URL = "https://api.example.com/v1"  # TODO: Replace with actual API base URL
MODEL_NAME_CHAT = "gpt-4-turbo-nonp"  # TODO: Replace with actual model name
OAUTH_URL = "https://api.example.com/oauth/token"  # TODO: Replace with actual OAuth URL
CLIENT_ID = "your_client_id"  # TODO: Replace with actual client ID
CLIENT_SECRET = "your_client_secret"  # TODO: Replace with actual client secret

# --- API Parameters ---
# FIXED: Separate input and output token limits
GPT_INPUT_TOKEN_LIMIT = 80000  # Maximum tokens for input/prompt
MAX_COMPLETION_TOKENS = 4000   # Maximum tokens for output/completion
TEMPERATURE = 0.3
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY = 5
TOKEN_BUFFER = 2000  # Safety buffer for prompt overhead

# --- Token Cost (Optional) ---
PROMPT_TOKEN_COST = 0.01
COMPLETION_TOKEN_COST = 0.03

# --- pysmb Configuration ---
smb_structs.SUPPORT_SMB2 = True
smb_structs.MAX_PAYLOAD_SIZE = 65536
CLIENT_HOSTNAME = socket.gethostname()

# --- Logging Level Control ---
# Set to True to see detailed debug logs, False for minimal output
VERBOSE_LOGGING = False

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
            return None
        return conn
    except Exception:
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
    except Exception:
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
        return True
    except Exception:
        return False
    finally:
        if conn:
            conn.close()

def read_from_nas(share_name, nas_path_relative):
    """Reads content (as bytes) from a file path on the NAS using pysmb."""
    conn = None
    try:
        conn = create_nas_connection()
        if not conn:
            return None

        file_obj = io.BytesIO()
        file_attributes, filesize = conn.retrieveFile(share_name, nas_path_relative, file_obj)
        file_obj.seek(0)
        content_bytes = file_obj.read()
        return content_bytes
    except Exception:
        return None
    finally:
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
    
    # Set logging level based on verbosity setting
    log_level = logging.DEBUG if VERBOSE_LOGGING else logging.WARNING
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(temp_log_path),
            logging.StreamHandler()
        ]
    )
    
    # Create a separate logger for progress messages that always shows
    progress_logger = logging.getLogger('progress')
    progress_logger.setLevel(logging.INFO)
    progress_logger.propagate = False
    
    # Add handlers to progress logger
    progress_handler = logging.StreamHandler()
    progress_handler.setFormatter(logging.Formatter('%(message)s'))
    progress_logger.addHandler(progress_handler)
    
    file_handler = logging.FileHandler(temp_log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    progress_logger.addHandler(file_handler)
    
    return temp_log_path

def log_progress(message):
    """Log a progress message that always shows."""
    progress_logger = logging.getLogger('progress')
    progress_logger.info(message)

# ==============================================================================
# Utility Functions
# ==============================================================================

# --- Token Counting ---
def count_tokens(text: str) -> int:
    """
    Estimates token count using empirically-calibrated formula.
    
    Based on GPT-4 tokenization patterns:
    - Average ratio: ~1 token per 3.5-4 characters for English text
    - More conservative estimate to avoid underestimation
    """
    if not text:
        return 0
    
    char_count = len(text)
    
    # Use chars/3.5 for a slightly conservative estimate
    estimated_tokens = int(char_count / 3.5)
    
    # Apply reasonable bounds
    MIN_CHARS_PER_TOKEN = 2     # For code/numbers
    MAX_CHARS_PER_TOKEN = 10    # For sparse text
    
    min_tokens = char_count // MAX_CHARS_PER_TOKEN
    max_tokens = char_count // MIN_CHARS_PER_TOKEN
    
    estimated_tokens = max(min_tokens, min(estimated_tokens, max_tokens))
    
    return estimated_tokens

# --- API Client ---
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
    except Exception:
        return False

def _get_oauth_token(oauth_url=OAUTH_URL, client_id=CLIENT_ID, client_secret=CLIENT_SECRET, ssl_verify_path=SSL_LOCAL_PATH) -> Optional[str]:
    """Retrieves OAuth token."""
    verify_path = ssl_verify_path if Path(ssl_verify_path).exists() else True

    payload = {'grant_type': 'client_credentials', 'client_id': client_id, 'client_secret': client_secret}
    try:
        response = requests.post(oauth_url, data=payload, timeout=30, verify=verify_path)
        response.raise_for_status()
        token_data = response.json()
        oauth_token = token_data.get('access_token')
        if not oauth_token:
            return None
        return oauth_token
    except requests.exceptions.RequestException:
        return None

def get_openai_client(base_url=BASE_URL) -> Optional[OpenAI]:
    """Initializes and returns the OpenAI client."""
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT:
        return _OPENAI_CLIENT
    if not OpenAI:
        return None
    if not _setup_ssl_from_nas():
        pass  # Continue anyway

    api_key = _get_oauth_token()
    if not api_key:
        return None
    try:
        _OPENAI_CLIENT = OpenAI(api_key=api_key, base_url=base_url)
        return _OPENAI_CLIENT
    except Exception:
        return None

# --- API Call ---
def call_gpt_chat_completion(client, model, messages, max_tokens, temperature, tools=None, tool_choice=None):
    """Makes the API call with retry logic, supporting tool calls."""
    last_exception = None
    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            completion_kwargs = {
                "model": model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "stream": False,
            }
            if tools and tool_choice:
                completion_kwargs["tools"] = tools
                completion_kwargs["tool_choice"] = tool_choice
            else:
                completion_kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**completion_kwargs)
            response_message = response.choices[0].message
            usage_info = response.usage

            if response_message.tool_calls:
                tool_call = response_message.tool_calls[0]
                if tool_choice and isinstance(tool_choice, dict):
                    expected_tool_name = tool_choice.get("function", {}).get("name")
                    if expected_tool_name and tool_call.function.name != expected_tool_name:
                        raise ValueError(f"Expected tool '{expected_tool_name}' but received '{tool_call.function.name}'")
                function_args_json = tool_call.function.arguments
                return function_args_json, usage_info
            elif response_message.content:
                return response_message.content, usage_info
            else:
                raise ValueError("API response missing both tool calls and content.")

        except APIError as e:
            last_exception = e
            time.sleep(API_RETRY_DELAY * (attempt + 1))
        except Exception as e:
            last_exception = e
            time.sleep(API_RETRY_DELAY)

    if last_exception:
        raise last_exception
    else:
        raise Exception("API call failed for unknown reasons.")

def parse_gpt_json_response(response_content_str: str, expected_keys: List[str] = None) -> Optional[Dict]:
    """
    Parses JSON response string from GPT.
    FIXED: Made expected_keys optional and added better error handling.
    """
    try:
        # Strip code block markers if present
        if response_content_str.strip().startswith("```json"):
            response_content_str = response_content_str.strip()[7:-3].strip()
        elif response_content_str.strip().startswith("```"):
            response_content_str = response_content_str.strip()[3:-3].strip()

        data = json.loads(response_content_str)
        if not isinstance(data, dict):
            logging.error("Response is not a JSON object")
            return None

        # Only validate keys if expected_keys provided
        if expected_keys:
            missing_keys = [key for key in expected_keys if key not in data]
            if missing_keys:
                logging.error(f"Missing expected keys in response: {', '.join(missing_keys)}")
                logging.error(f"Available keys: {', '.join(data.keys())}")
                # Don't fail completely - return partial data
                # This allows handling cases where 'tags' might be missing
                return data

        return data
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding GPT JSON response: {e}")
        logging.error(f"Raw response (first 500 chars): {response_content_str[:500]}...")
        return None
    except Exception as e:
        logging.error(f"Unexpected error parsing GPT response: {e}")
        return None

# ==============================================================================
# GPT Prompting for Chapter Details
# ==============================================================================

CHAPTER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "extract_chapter_details",
        "description": "Extracts the summary and topic tags from a chapter segment based on provided guidance.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "A detailed summary of the chapter segment, following the structure outlined in the prompt."
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of 5-15 specific, granular topic tags relevant for retrieval by accounting professionals."
                }
            },
            "required": ["summary", "tags"]
        }
    }
}

def _build_chapter_prompt(segment_text, prev_summary=None, prev_tags=None, is_final_segment=False):
    """Builds the messages list for the chapter/segment processing call using CO-STAR and XML."""
    system_prompt = """<role>You are an expert financial reporting specialist with deep knowledge of IFRS and US GAAP.</role>
<source_material>You are analyzing segments of a chapter from a comprehensive EY technical accounting guidance manual.</source_material>
<task>Your primary task is to extract key information and generate a highly detailed, structured summary and a set of specific, granular topic tags for the provided text segment. This output will be used to build a knowledge base for accurate retrieval by accounting professionals. You will provide the output using the available 'extract_chapter_details' tool.</task>
<guardrails>Base your analysis strictly on the provided text segment and any previous context given. Do not infer information not explicitly present or heavily implied. Focus on factual extraction and objective summarization. Ensure tags are precise and directly relevant to accounting standards, concepts, or procedures mentioned.</guardrails>"""

    user_prompt_elements = ["<prompt>"]
    user_prompt_elements.append("<context>You are processing a text segment from a chapter within an EY technical accounting guidance manual (likely IFRS or US GAAP focused). The ultimate goal is to populate a knowledge base for efficient and accurate information retrieval by accounting professionals performing research.</context>")
    if prev_summary: user_prompt_elements.append(f"<previous_summary>{prev_summary}</previous_summary>")
    if prev_tags: user_prompt_elements.append(f"<previous_tags>{json.dumps(prev_tags)}</previous_tags>")
    user_prompt_elements.append("<style>Highly detailed, structured, technical, analytical, precise, and informative. Use clear headings within the summary string as specified.</style>")
    user_prompt_elements.append("<tone>Professional, objective, expert.</tone>")
    user_prompt_elements.append("<audience>Accounting professionals needing specific guidance; requires accuracy, completeness (within the scope of the text), and easy identification of key concepts.</audience>")
    user_prompt_elements.append('<response_format>Use the "extract_chapter_details" tool to provide the summary and tags.</response_format>')
    user_prompt_elements.append(f"<current_segment>{segment_text}</current_segment>")
    user_prompt_elements.append("<instructions>")
    summary_structure_guidance = """
    **Summary Structure Guidance:** Structure the 'summary' string using the following headings. Provide detailed information under each:
    *   **Purpose:** Concisely state the primary objective of this chapter/segment.
    *   **Key Topics/Standards:** List primary standards (e.g., IFRS 16, ASC 842) and detail significant topics, concepts, principles, or procedures discussed. Be specific.
    *   **Context/Applicability:** Describe the scope precisely (entities, transactions, industries, exceptions).
    *   **Key Outcomes/Decisions:** Identify main outcomes, critical judgments, or key decisions needed.
    """
    tag_guidance = """
    **Tag Generation Guidance:** Generate specific, granular tags (5-15) for retrieval. Include:
    *   Relevant standard names/paragraphs (e.g., 'IFRS 15', 'IAS 36.12').
    *   Core accounting concepts (e.g., 'revenue recognition', 'lease modification').
    *   Specific procedures/models (e.g., 'five-step revenue model', 'ECL model').
    *   Key terms defined (e.g., 'performance obligation', 'lease term').
    *   Applicability context (e.g., 'SME considerations', 'interim reporting').
    """
    if prev_summary or prev_tags:
        user_prompt_elements.append(summary_structure_guidance)
        user_prompt_elements.append(tag_guidance)
        if is_final_segment:
            user_prompt_elements.append("**Objective:** Consolidate all context with the <current_segment> for the FINAL chapter summary/tags.")
            user_prompt_elements.append("**Action:** Synthesize all info, adhering to guidance. Ensure final output reflects the entire chapter processed.")
        else:
            user_prompt_elements.append("**Objective:** Refine cumulative understanding with <current_segment>.")
            user_prompt_elements.append("**Action:** Integrate <current_segment> with previous context. Provide UPDATED summary/tags, adhering to guidance.")
    else:
        user_prompt_elements.append(summary_structure_guidance)
        user_prompt_elements.append(tag_guidance)
        user_prompt_elements.append("**Objective:** Analyze the initial segment.")
        user_prompt_elements.append("**Action:** Generate summary/tags based ONLY on <current_segment>, adhering to guidance.")
    user_prompt_elements.append("</instructions>")
    user_prompt_elements.append("</prompt>")
    user_prompt = "\n".join(user_prompt_elements)

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    return messages

def process_chapter_segment_for_details(segment_text, client, model_name, max_completion_tokens, temperature, prev_summary=None, prev_tags=None, is_final_segment=False):
    """
    Processes a single chapter segment using GPT to get summary and tags.
    FIXED: Better error handling for missing keys in response.
    """
    messages = _build_chapter_prompt(segment_text, prev_summary, prev_tags, is_final_segment)

    try:
        response_content_json_str, usage_info = call_gpt_chat_completion(
            client=client,
            messages=messages,
            model=model_name,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            tools=[CHAPTER_TOOL_SCHEMA],
            tool_choice={"type": "function", "function": {"name": "extract_chapter_details"}}
        )

        if not response_content_json_str:
            raise ValueError("API call returned empty response.")

        # Parse response with better error handling
        parsed_data = parse_gpt_json_response(response_content_json_str)
        
        if not parsed_data:
            logging.error("Failed to parse GPT response")
            return None
        
        # Ensure both summary and tags exist, provide defaults if missing
        if 'summary' not in parsed_data:
            logging.error("Response missing 'summary' field")
            parsed_data['summary'] = "Summary generation failed"
        
        if 'tags' not in parsed_data:
            logging.error("Response missing 'tags' field")
            parsed_data['tags'] = []

        if usage_info and VERBOSE_LOGGING:
            prompt_tokens = usage_info.prompt_tokens
            completion_tokens = usage_info.completion_tokens
            total_tokens = usage_info.total_tokens
            prompt_cost = (prompt_tokens / 1000) * PROMPT_TOKEN_COST
            completion_cost = (completion_tokens / 1000) * COMPLETION_TOKEN_COST
            total_cost = prompt_cost + completion_cost
            logging.debug(f"API Usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}, Cost: ${total_cost:.4f}")

        return parsed_data

    except Exception as e:
        logging.error(f"Error processing chapter segment: {e}")
        return None

def get_chapter_summary_and_tags(chapter_text: str, client: OpenAI, model_name: str = MODEL_NAME_CHAT) -> Tuple[Optional[str], Optional[List[str]]]:
    """
    Generates summary and tags for the chapter text, handling segmentation.
    FIXED: Corrected token limit calculation and improved progress display.
    """
    total_tokens = count_tokens(chapter_text)
    total_chars = len(chapter_text)
    
    # FIXED: Calculate available tokens for input correctly
    # We have GPT_INPUT_TOKEN_LIMIT for input and need to reserve TOKEN_BUFFER for prompt overhead
    available_for_content = GPT_INPUT_TOKEN_LIMIT - TOKEN_BUFFER
    
    # Check if content fits in single call
    needs_segmentation = total_tokens > available_for_content
    
    if needs_segmentation:
        num_segments = (total_tokens // available_for_content) + 1
        segment_len_approx = len(chapter_text) // num_segments
        segments = [chapter_text[i:i + segment_len_approx] for i in range(0, len(chapter_text), segment_len_approx)]
        
        # Progress display for segmented processing
        log_progress(f"  📄 Chapter size: {total_chars:,} chars ({total_tokens:,} tokens)")
        log_progress(f"  ✂️  Split into {len(segments)} segments for processing")
        
        current_summary = None
        current_tags = None
        final_result = None

        for i, segment_text in enumerate(segments):
            segment_tokens = count_tokens(segment_text)
            log_progress(f"  🔄 Processing segment {i+1}/{len(segments)} ({segment_tokens:,} tokens)...", end='')
            
            is_final = (i == len(segments) - 1)
            
            # Retry logic for segment processing
            segment_result = None
            for retry in range(3):
                try:
                    segment_result = process_chapter_segment_for_details(
                        segment_text, client, model_name, MAX_COMPLETION_TOKENS, TEMPERATURE,
                        prev_summary=current_summary, prev_tags=current_tags, is_final_segment=is_final
                    )
                    if segment_result:
                        break
                    elif retry < 2:
                        time.sleep(5 * (retry + 1))
                except Exception as e:
                    if retry < 2:
                        time.sleep(5 * (retry + 1))

            if segment_result:
                current_summary = segment_result.get('summary')
                current_tags = segment_result.get('tags')
                if is_final:
                    final_result = segment_result
                log_progress(" ✅")
            else:
                log_progress(" ❌ Failed")
                if i == len(segments) - 1 and current_summary:
                    final_result = {'summary': current_summary, 'tags': current_tags}

        if final_result:
            return final_result.get('summary'), final_result.get('tags')
    else:
        # Single call processing
        log_progress(f"  📄 Chapter size: {total_chars:,} chars ({total_tokens:,} tokens)")
        log_progress(f"  🔄 Processing in single call...", end='')
        
        result = None
        for retry in range(3):
            try:
                result = process_chapter_segment_for_details(
                    chapter_text, client, model_name, MAX_COMPLETION_TOKENS, TEMPERATURE, is_final_segment=True
                )
                if result:
                    break
                elif retry < 2:
                    time.sleep(5 * (retry + 1))
            except Exception as e:
                if retry < 2:
                    time.sleep(5 * (retry + 1))
        
        if result:
            log_progress(" ✅")
            return result.get('summary'), result.get('tags')
        else:
            log_progress(" ❌ Failed")
    
    return None, None

# ==============================================================================
# Main Processing Logic
# ==============================================================================

def group_pages_by_chapter(json_data: List[Dict]) -> Dict[int, List[Dict]]:
    """
    Groups page records by chapter_number and sorts them by page_number.
    Returns a dictionary mapping chapter_number to list of page records.
    """
    chapters = defaultdict(list)
    
    for record in json_data:
        chapter_num = record.get('chapter_number')
        if chapter_num is not None:  # Include chapter 0
            chapters[chapter_num].append(record)
    
    # Sort pages within each chapter by page_number
    for chapter_num in chapters:
        chapters[chapter_num].sort(key=lambda x: x.get('page_number', 0))
    
    return dict(chapters)

def process_chapter_pages(chapter_num: int, pages: List[Dict], client: Optional[OpenAI]) -> List[Dict]:
    """
    Processes pages in a chapter: generates ONE summary/tags for the chapter, 
    then applies to ALL pages. Returns enriched page-level records.
    """
    # Get chapter metadata from first page
    first_page = pages[0]
    chapter_name = first_page.get('chapter_name', f'Chapter {chapter_num}')
    chapter_filename = first_page.get('filename', 'unknown.pdf')
    
    # Get page ranges
    pdf_page_numbers = [p.get('page_number', 0) for p in pages]
    original_page_numbers = [p.get('original_page_number', p.get('page_number', 0)) for p in pages]
    
    # Progress display
    log_progress("")
    log_progress(f"📚 Chapter {chapter_num}: {chapter_name}")
    log_progress(f"  📁 File: {chapter_filename}")
    log_progress(f"  📄 Pages: {min(pdf_page_numbers)}-{max(pdf_page_numbers)} ({len(pages)} pages)")
    
    # Concatenate all page content for GPT processing
    content_parts = []
    for page in pages:
        page_content = page.get('content', '')
        if page_content:
            content_parts.append(page_content)
    
    concatenated_content = "\n\n".join(content_parts)
    
    # Generate summary and tags via GPT (ONCE for the whole chapter)
    chapter_summary, chapter_tags = None, None
    if client:
        try:
            chapter_summary, chapter_tags = get_chapter_summary_and_tags(
                concatenated_content, client, model_name=MODEL_NAME_CHAT
            )
            if chapter_summary and chapter_tags:
                log_progress(f"  ✅ Generated summary and {len(chapter_tags)} tags")
            else:
                log_progress(f"  ⚠️ Failed to generate summary/tags")
        except Exception as e:
            log_progress(f"  ❌ Error: {str(e)[:100]}")
    else:
        log_progress("  ⚠️ OpenAI client not available - skipping summary/tags")
    
    # Apply chapter summary/tags to each page record
    enriched_pages = []
    for page in pages:
        enriched_page = page.copy()  # Keep all original fields
        
        # Add chapter-level enrichments
        enriched_page['chapter_summary'] = chapter_summary
        enriched_page['chapter_tags'] = chapter_tags
        enriched_page['chapter_token_count'] = count_tokens(concatenated_content)
        
        # Calculate page-specific token count
        page_content = page.get('content', '')
        enriched_page['page_token_count'] = count_tokens(page_content)
        
        # Ensure citation fields are clear
        enriched_page['pdf_filename'] = page.get('filename')  # Split PDF name
        enriched_page['pdf_page_number'] = page.get('page_number')  # Page in split PDF
        
        enriched_pages.append(enriched_page)
    
    return enriched_pages

def process_unassigned_pages(pages: List[Dict]) -> List[Dict]:
    """
    Processes pages that don't have a chapter assignment.
    Returns enriched page records without chapter summaries.
    """
    if not pages:
        return []
    
    log_progress(f"📄 Processing {len(pages)} unassigned pages")
    
    enriched_pages = []
    for page in pages:
        enriched_page = page.copy()  # Keep all original fields
        
        # No chapter summary/tags for unassigned pages
        enriched_page['chapter_summary'] = None
        enriched_page['chapter_tags'] = None
        enriched_page['chapter_token_count'] = None
        
        # Calculate page-specific token count
        page_content = page.get('content', '')
        enriched_page['page_token_count'] = count_tokens(page_content)
        
        # Ensure citation fields
        enriched_page['pdf_filename'] = page.get('filename')
        enriched_page['pdf_page_number'] = page.get('page_number')
        
        enriched_pages.append(enriched_page)
    
    return enriched_pages

def run_stage1():
    """Main function to execute Stage 1 processing with page-level records."""
    # Setup logging
    temp_log_path = setup_logging()
    
    log_progress("=" * 70)
    log_progress("🚀 Starting Stage 1: Chapter Processing (Page-Level Records)")
    log_progress("=" * 70)
    
    # Setup SSL
    _setup_ssl_from_nas()
    
    share_name = NAS_PARAMS["share"]
    output_path_relative = os.path.join(NAS_OUTPUT_PATH, OUTPUT_FILENAME).replace('\\', '/')
    
    # --- Load Input JSON from NAS ---
    log_progress("📥 Loading input JSON from NAS...")
    input_json_bytes = read_from_nas(share_name, NAS_INPUT_JSON_PATH)
    
    if not input_json_bytes:
        log_progress(f"❌ Failed to read input JSON from {share_name}/{NAS_INPUT_JSON_PATH}")
        return
    
    try:
        input_data = json.loads(input_json_bytes.decode('utf-8'))
        if not isinstance(input_data, list):
            log_progress("❌ Input JSON is not a list. Expected array of page records.")
            return
        log_progress(f"✅ Loaded {len(input_data)} page records")
        
    except json.JSONDecodeError as e:
        log_progress(f"❌ Error decoding input JSON: {e}")
        return
    
    # --- Initialize OpenAI Client ---
    client = None
    if OpenAI:
        client = get_openai_client()
        if client:
            log_progress("✅ OpenAI client initialized")
        else:
            log_progress("⚠️ Failed to initialize OpenAI client - will proceed without summaries/tags")
    else:
        log_progress("⚠️ OpenAI library not installed - summaries and tags will not be generated")
    
    # --- Group Pages by Chapter ---
    chapters = group_pages_by_chapter(input_data)
    unassigned_pages = [r for r in input_data if r.get('chapter_number') is None]
    
    log_progress(f"📊 Found {len(chapters)} chapters and {len(unassigned_pages)} unassigned pages")
    log_progress("-" * 70)
    
    # --- Process Each Chapter ---
    all_enriched_pages = []
    
    # Process chapters in order
    for chapter_num in sorted(chapters.keys()):
        pages = chapters[chapter_num]
        enriched_pages = process_chapter_pages(chapter_num, pages, client)
        all_enriched_pages.extend(enriched_pages)
    
    # Process unassigned pages if any
    if unassigned_pages:
        enriched_unassigned = process_unassigned_pages(unassigned_pages)
        all_enriched_pages.extend(enriched_unassigned)
    
    # Sort final output by original page number to maintain document order
    all_enriched_pages.sort(key=lambda x: x.get('original_page_number', x.get('page_number', 0)))
    
    # --- Save Output to NAS ---
    log_progress("-" * 70)
    log_progress(f"💾 Saving {len(all_enriched_pages)} enriched page records to NAS...")
    
    try:
        output_json = json.dumps(all_enriched_pages, indent=2, ensure_ascii=False)
        output_bytes = output_json.encode('utf-8')
        
        if write_to_nas(share_name, output_path_relative, output_bytes):
            log_progress(f"✅ Successfully saved output to: {share_name}/{output_path_relative}")
        else:
            log_progress("❌ Failed to write output to NAS")
    except Exception as e:
        log_progress(f"❌ Error saving output: {e}")
    
    # --- Upload Log File to NAS ---
    try:
        log_file_name = f"stage1_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path_relative = os.path.join(NAS_LOG_PATH, log_file_name).replace('\\', '/')
        
        # Close logging handlers to flush content
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)
        
        # Also close progress logger handlers
        progress_logger = logging.getLogger('progress')
        for handler in progress_logger.handlers[:]:
            handler.close()
            progress_logger.removeHandler(handler)
        
        # Read log content and upload to NAS
        with open(temp_log_path, 'rb') as f:
            log_content = f.read()
        
        if write_to_nas(share_name, log_path_relative, log_content):
            print(f"📝 Log file uploaded to NAS: {share_name}/{log_path_relative}")
        else:
            print(f"⚠️ Failed to upload log file to NAS")
        
        # Clean up temp log file
        os.remove(temp_log_path)
    except Exception as e:
        print(f"⚠️ Error handling log file: {e}")
    
    # --- Final Summary ---
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

# ==============================================================================
# Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    run_stage1()