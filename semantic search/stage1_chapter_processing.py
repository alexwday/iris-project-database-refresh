# -*- coding: utf-8 -*-
"""
Stage 1: Chapter Processing (Page-Level Records Version)

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
# Note: tiktoken removed due to SSL/network issues. Using reliable local estimation instead.

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
# Path on NAS where input JSON is stored (output from EY prep with chapter assignments)
NAS_INPUT_JSON_PATH = "semantic_search/prep_output/ey/ey_prep_output_with_chapters.json"  # TODO: Adjust path
# Path on NAS where output will be saved (relative to share root)
NAS_OUTPUT_PATH = "semantic_search/pipeline_output/stage1"
# Path on NAS where logs will be saved
NAS_LOG_PATH = "semantic_search/pipeline_output/logs"
OUTPUT_FILENAME = "stage1_page_records.json"

# --- CA Bundle Configuration ---
# Path on NAS where the SSL certificate is stored (relative to share root)
NAS_SSL_CERT_PATH = "certificates/rbc-ca-bundle.cer"  # TODO: Adjust to match your NAS location
SSL_LOCAL_PATH = "/tmp/rbc-ca-bundle.cer"  # Temp path for cert

# --- API Configuration (Hardcoded) ---
BASE_URL = "https://api.example.com/v1"  # TODO: Replace with actual API base URL
MODEL_NAME_CHAT = "gpt-4-turbo-nonp"  # TODO: Replace with actual model name
OAUTH_URL = "https://api.example.com/oauth/token"  # TODO: Replace with actual OAuth URL
CLIENT_ID = "your_client_id"  # TODO: Replace with actual client ID
CLIENT_SECRET = "your_client_secret"  # TODO: Replace with actual client secret

# --- API Parameters ---
MAX_COMPLETION_TOKENS_CHAPTER = 4000
TEMPERATURE = 0.3
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY = 5
GPT_INPUT_TOKEN_LIMIT = 80000
TOKEN_BUFFER = 2000

# --- Token Cost (Optional) ---
PROMPT_TOKEN_COST = 0.01
COMPLETION_TOKEN_COST = 0.03

# --- pysmb Configuration ---
smb_structs.SUPPORT_SMB2 = True
smb_structs.MAX_PAYLOAD_SIZE = 65536
CLIENT_HOSTNAME = socket.gethostname()

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
            logging.error("Failed to connect to NAS.")
            return None
        logging.debug(f"Successfully connected to NAS: {NAS_PARAMS['ip']}:{NAS_PARAMS['port']}")
        return conn
    except Exception as e:
        logging.error(f"Exception creating NAS connection: {e}")
        return None

def ensure_nas_dir_exists(conn, share_name, dir_path_relative):
    """Ensures a directory exists on the NAS, creating it if necessary."""
    if not conn:
        logging.error("Cannot ensure NAS directory: No connection.")
        return False
    
    path_parts = dir_path_relative.strip('/').split('/')
    current_path = ''
    try:
        for part in path_parts:
            if not part: continue
            current_path = os.path.join(current_path, part).replace('\\', '/')
            try:
                conn.listPath(share_name, current_path)
                logging.debug(f"Directory exists: {current_path}")
            except Exception:
                logging.info(f"Creating directory on NAS: {share_name}/{current_path}")
                conn.createDirectory(share_name, current_path)
        return True
    except Exception as e:
        logging.error(f"Failed to ensure/create NAS directory '{share_name}/{dir_path_relative}': {e}")
        return False

def write_to_nas(share_name, nas_path_relative, content_bytes):
    """Writes bytes to a file path on the NAS using pysmb."""
    conn = None
    logging.info(f"Attempting to write to NAS path: {share_name}/{nas_path_relative}")
    try:
        conn = create_nas_connection()
        if not conn:
            return False

        dir_path = os.path.dirname(nas_path_relative).replace('\\', '/')
        if dir_path and not ensure_nas_dir_exists(conn, share_name, dir_path):
            logging.error(f"Failed to ensure output directory exists: {dir_path}")
            return False

        file_obj = io.BytesIO(content_bytes)
        bytes_written = conn.storeFile(share_name, nas_path_relative, file_obj)
        logging.info(f"Successfully wrote {bytes_written} bytes to: {share_name}/{nas_path_relative}")
        return True
    except Exception as e:
        logging.error(f"Unexpected error writing to NAS '{share_name}/{nas_path_relative}': {e}")
        return False
    finally:
        if conn:
            conn.close()

def read_from_nas(share_name, nas_path_relative):
    """Reads content (as bytes) from a file path on the NAS using pysmb."""
    conn = None
    logging.debug(f"Attempting to read from NAS path: {share_name}/{nas_path_relative}")
    try:
        conn = create_nas_connection()
        if not conn:
            return None

        file_obj = io.BytesIO()
        file_attributes, filesize = conn.retrieveFile(share_name, nas_path_relative, file_obj)
        file_obj.seek(0)
        content_bytes = file_obj.read()
        logging.debug(f"Successfully read {filesize} bytes from: {share_name}/{nas_path_relative}")
        return content_bytes
    except Exception as e:
        logging.error(f"Unexpected error reading from NAS '{share_name}/{nas_path_relative}': {e}")
        return None
    finally:
        if conn:
            conn.close()

# ==============================================================================
# Logging Setup
# ==============================================================================

def setup_logging():
    """Setup logging to write to NAS."""
    temp_log = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log')
    temp_log_path = temp_log.name
    temp_log.close()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler(temp_log_path),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Temporary log file: {temp_log_path}")
    return temp_log_path

# ==============================================================================
# Utility Functions
# ==============================================================================

# --- Token Counting ---
def count_tokens(text: str) -> int:
    """
    Estimates token count using a reliable formula based on GPT-4 patterns.
    Includes sanity checks to prevent unreasonable values.
    
    Formula based on empirical analysis:
    - Average English word: ~1.3 tokens
    - Average character per word: ~5 characters (including spaces)
    - Therefore: tokens ≈ chars / 3.8
    - We use chars / 3.5 to be slightly conservative
    """
    if not text:
        return 0
    
    char_count = len(text)
    
    # Basic estimation: characters divided by 3.5
    # This is more accurate than /4 for typical English text
    estimated_tokens = int(char_count / 3.5)
    
    # Sanity checks
    MIN_CHARS_PER_TOKEN = 2     # Absolute minimum (handles code/numbers)
    MAX_CHARS_PER_TOKEN = 10    # Absolute maximum (handles sparse text)
    
    # Apply bounds
    min_tokens = char_count // MAX_CHARS_PER_TOKEN
    max_tokens = char_count // MIN_CHARS_PER_TOKEN
    
    # Ensure estimate is within reasonable bounds
    estimated_tokens = max(min_tokens, min(estimated_tokens, max_tokens))
    
    # Warning for unusually high token counts
    if estimated_tokens > 500000:
        logging.warning(f"Unusually high token count detected: {estimated_tokens:,} tokens from {char_count:,} characters")
        logging.warning("This might indicate concatenated content issues or data corruption")
    
    # Log token estimation details for debugging (only for large texts)
    if char_count > 100000:
        words_estimate = char_count // 5  # Rough word count
        logging.debug(f"Token estimation: {char_count:,} chars → ~{words_estimate:,} words → {estimated_tokens:,} tokens")
    
    return estimated_tokens

# --- API Client ---
_SSL_CONFIGURED = False
_OPENAI_CLIENT = None

def _setup_ssl_from_nas() -> bool:
    """Downloads SSL cert from NAS and sets environment variables."""
    global _SSL_CONFIGURED
    if _SSL_CONFIGURED:
        return True
    
    logging.info("Setting up SSL certificate from NAS...")
    try:
        cert_bytes = read_from_nas(NAS_PARAMS["share"], NAS_SSL_CERT_PATH)
        if cert_bytes is None:
            logging.warning(f"SSL certificate not found on NAS at {NAS_SSL_CERT_PATH}. API calls may fail.")
            _SSL_CONFIGURED = True
            return True
        
        local_cert = Path(SSL_LOCAL_PATH)
        local_cert.parent.mkdir(parents=True, exist_ok=True)
        with open(local_cert, "wb") as f:
            f.write(cert_bytes)
        
        os.environ["SSL_CERT_FILE"] = str(local_cert)
        os.environ["REQUESTS_CA_BUNDLE"] = str(local_cert)
        logging.info(f"SSL certificate configured successfully at: {local_cert}")
        _SSL_CONFIGURED = True
        return True
    except Exception as e:
        logging.error(f"Error setting up SSL certificate from NAS: {e}", exc_info=True)
        return False

def _get_oauth_token(oauth_url=OAUTH_URL, client_id=CLIENT_ID, client_secret=CLIENT_SECRET, ssl_verify_path=SSL_LOCAL_PATH) -> Optional[str]:
    """Retrieves OAuth token."""
    verify_path = ssl_verify_path if Path(ssl_verify_path).exists() else True

    logging.info("Attempting to get OAuth token...")
    payload = {'grant_type': 'client_credentials', 'client_id': client_id, 'client_secret': client_secret}
    try:
        response = requests.post(oauth_url, data=payload, timeout=30, verify=verify_path)
        response.raise_for_status()
        token_data = response.json()
        oauth_token = token_data.get('access_token')
        if not oauth_token:
            logging.error("Error: 'access_token' not found in OAuth response.")
            return None
        logging.info("OAuth token obtained successfully.")
        return oauth_token
    except requests.exceptions.RequestException as e:
        logging.error(f"Error getting OAuth token: {e}", exc_info=True)
        return None

def get_openai_client(base_url=BASE_URL) -> Optional[OpenAI]:
    """Initializes and returns the OpenAI client."""
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT:
        return _OPENAI_CLIENT
    if not OpenAI:
        logging.error("OpenAI library not available.")
        return None
    if not _setup_ssl_from_nas():
        logging.warning("Proceeding without explicit SSL setup. API calls might fail.")

    api_key = _get_oauth_token()
    if not api_key:
        logging.error("Aborting client creation due to OAuth token failure.")
        return None
    try:
        _OPENAI_CLIENT = OpenAI(api_key=api_key, base_url=base_url)
        logging.info("OpenAI client created successfully.")
        return _OPENAI_CLIENT
    except Exception as e:
        logging.error(f"Error creating OpenAI client: {e}", exc_info=True)
        return None

# --- API Call ---
def call_gpt_chat_completion(client, model, messages, max_tokens, temperature, tools=None, tool_choice=None):
    """Makes the API call with retry logic, supporting tool calls."""
    last_exception = None
    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            logging.debug(f"Making API call (Attempt {attempt + 1}/{API_RETRY_ATTEMPTS})...")
            completion_kwargs = {
                "model": model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "stream": False,
            }
            if tools and tool_choice:
                completion_kwargs["tools"] = tools
                completion_kwargs["tool_choice"] = tool_choice
                logging.debug("Making API call with tool choice...")
            else:
                completion_kwargs["response_format"] = {"type": "json_object"}
                logging.debug("Making API call with JSON response format...")

            response = client.chat.completions.create(**completion_kwargs)
            logging.debug("API call successful.")
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
            logging.warning(f"API Error on attempt {attempt + 1}: {e}")
            last_exception = e
            time.sleep(API_RETRY_DELAY * (attempt + 1))
        except Exception as e:
            logging.warning(f"Non-API Error on attempt {attempt + 1}: {e}", exc_info=True)
            last_exception = e
            time.sleep(API_RETRY_DELAY)

    logging.error(f"API call failed after {API_RETRY_ATTEMPTS} attempts.")
    if last_exception:
        raise last_exception
    else:
        raise Exception("API call failed for unknown reasons.")

def parse_gpt_json_response(response_content_str: str, expected_keys: List[str]) -> Optional[Dict]:
    """Parses JSON response string from GPT and validates expected keys."""
    try:
        if response_content_str.strip().startswith("```json"):
            response_content_str = response_content_str.strip()[7:-3].strip()
        elif response_content_str.strip().startswith("```"):
            response_content_str = response_content_str.strip()[3:-3].strip()

        data = json.loads(response_content_str)
        if not isinstance(data, dict):
            raise ValueError("Response is not a JSON object.")

        missing_keys = [key for key in expected_keys if key not in data]
        if missing_keys:
            raise ValueError(f"Missing expected keys in response: {', '.join(missing_keys)}")

        logging.debug("GPT JSON response parsed successfully.")
        return data
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding GPT JSON response: {e}")
        logging.error(f"Raw response string: {response_content_str[:500]}...")
        return None
    except ValueError as e:
        logging.error(f"Error validating GPT JSON response: {e}")
        logging.error(f"Raw response string: {response_content_str[:500]}...")
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
    """Processes a single chapter segment using GPT to get summary and tags."""
    messages = _build_chapter_prompt(segment_text, prev_summary, prev_tags, is_final_segment)
    prompt_tokens_est = sum(count_tokens(msg["content"]) for msg in messages)
    logging.debug(f"Estimated prompt tokens for segment: {prompt_tokens_est}")

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

        parsed_data = parse_gpt_json_response(response_content_json_str, expected_keys=["summary", "tags"])

        if usage_info:
            prompt_tokens = usage_info.prompt_tokens
            completion_tokens = usage_info.completion_tokens
            total_tokens = usage_info.total_tokens
            prompt_cost = (prompt_tokens / 1000) * PROMPT_TOKEN_COST
            completion_cost = (completion_tokens / 1000) * COMPLETION_TOKEN_COST
            total_cost = prompt_cost + completion_cost
            logging.info(f"API Usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}, Cost: ${total_cost:.4f}")
        else:
            logging.debug("Usage information not available.")

        return parsed_data

    except Exception as e:
        logging.error(f"Error processing chapter segment: {e}", exc_info=True)
        return None

def get_chapter_summary_and_tags(chapter_text: str, client: OpenAI, model_name: str = MODEL_NAME_CHAT) -> Tuple[Optional[str], Optional[List[str]]]:
    """Generates summary and tags for the chapter text, handling segmentation."""
    total_tokens = count_tokens(chapter_text)
    total_chars = len(chapter_text)
    
    processing_limit = GPT_INPUT_TOKEN_LIMIT - MAX_COMPLETION_TOKENS_CHAPTER - TOKEN_BUFFER
    
    # Log token analysis
    logging.info("-" * 60)
    logging.info("TOKEN ANALYSIS:")
    logging.info(f"  Estimation method: chars/3.5 ratio (empirically calibrated)")
    logging.info(f"  Total chapter tokens: {total_tokens:,}")
    logging.info(f"  Total chapter characters: {total_chars:,}")
    logging.info(f"  Avg chars per token: {total_chars/total_tokens:.2f}" if total_tokens > 0 else "N/A")
    logging.info(f"  GPT input limit: {GPT_INPUT_TOKEN_LIMIT:,}")
    logging.info(f"  Reserved for completion: {MAX_COMPLETION_TOKENS_CHAPTER:,}")
    logging.info(f"  Buffer: {TOKEN_BUFFER:,}")
    logging.info(f"  Available for input: {processing_limit:,}")
    logging.info(f"  Fits in single call: {'YES' if total_tokens <= processing_limit else 'NO'}")
    logging.info("-" * 60)
    
    final_chapter_details = None

    if total_tokens <= processing_limit:
        logging.info("Processing chapter summary/tags in a single call.")
        # Retry logic for single call processing
        result = None
        for retry in range(3):  # Try up to 3 times
            try:
                result = process_chapter_segment_for_details(
                    chapter_text, client, model_name, MAX_COMPLETION_TOKENS_CHAPTER, TEMPERATURE, is_final_segment=True
                )
                if result:
                    break  # Success
                elif retry < 2:
                    logging.warning(f"Single call processing failed, retrying ({retry + 2}/3)...")
                    time.sleep(5 * (retry + 1))
            except Exception as e:
                logging.warning(f"Exception in single call processing, attempt {retry + 1}/3: {e}")
                if retry < 2:
                    time.sleep(5 * (retry + 1))
        
        if result:
            final_chapter_details = {'summary': result.get('summary'), 'tags': result.get('tags')}
        else:
            logging.error("Failed to process chapter in single call after 3 attempts.")
    else:
        logging.info(f"Chapter exceeds token limit ({total_tokens:,} > {processing_limit:,})")
        logging.info("SEGMENTATION REQUIRED:")
        
        num_segments = (total_tokens // processing_limit) + 1
        segment_len_approx = len(chapter_text) // num_segments
        
        logging.warning(f"⚠️ Large chapter requires segmentation:")
        logging.info(f"  Segments needed (by tokens): {num_segments}")
        logging.info(f"  Target chars per segment: {segment_len_approx:,}")
        # Calculate actual tokens for a sample segment
        sample_segment_tokens = count_tokens(chapter_text[:min(segment_len_approx, len(chapter_text))])
        logging.info(f"  Estimated tokens per segment: {sample_segment_tokens:,}")
        
        segments = [chapter_text[i:i + segment_len_approx] for i in range(0, len(chapter_text), segment_len_approx)]
        
        # Verify actual segment sizes
        logging.info(f"  Actual segments created: {len(segments)}")
        
        # Sanity check for excessive segmentation
        if len(segments) > 10:
            logging.error(f"❌ EXCESSIVE SEGMENTATION DETECTED: {len(segments)} segments!")
            logging.error(f"   This usually indicates a data issue or incorrect token counting.")
            logging.error(f"   Chapter has {len(chapter_text):,} chars, estimated at {total_tokens:,} tokens.")
            logging.error(f"   Please verify the chapter content is not corrupted or duplicated.")
        
        for i, seg in enumerate(segments[:3]):  # Show first 3 segments
            seg_tokens = count_tokens(seg)
            logging.info(f"    Segment {i+1}: {len(seg):,} chars, {seg_tokens:,} tokens")
        if len(segments) > 3:
            logging.info(f"    ... and {len(segments) - 3} more segments")
        
        current_summary = None
        current_tags = None
        final_result = None

        for i, segment_text in enumerate(tqdm(segments, desc="Processing Segments")):
            logging.debug(f"Processing segment {i + 1}/{len(segments)}...")
            is_final = (i == len(segments) - 1)
            
            # Retry logic for segment processing
            segment_result = None
            for retry in range(3):  # Try up to 3 times
                try:
                    segment_result = process_chapter_segment_for_details(
                        segment_text, client, model_name, MAX_COMPLETION_TOKENS_CHAPTER, TEMPERATURE,
                        prev_summary=current_summary, prev_tags=current_tags, is_final_segment=is_final
                    )
                    if segment_result:
                        break  # Success, exit retry loop
                    elif retry < 2:
                        logging.warning(f"Segment {i + 1} failed, retrying ({retry + 2}/3)...")
                        time.sleep(5 * (retry + 1))  # Exponential backoff
                except Exception as e:
                    logging.warning(f"Exception processing segment {i + 1}, attempt {retry + 1}/3: {e}")
                    if retry < 2:
                        time.sleep(5 * (retry + 1))

            if segment_result:
                current_summary = segment_result.get('summary')
                current_tags = segment_result.get('tags')
                if is_final:
                    final_result = segment_result
                logging.debug(f"Segment {i + 1} processed.")
            else:
                logging.error(f"Error processing segment {i + 1} after 3 attempts. Continuing with partial data.")
                # Don't abort - try to continue with what we have
                if i == len(segments) - 1 and current_summary:
                    # If it's the final segment and we have previous summaries, use them
                    final_result = {'summary': current_summary, 'tags': current_tags}
                    logging.warning(f"Using summary from segment {i} as final result.")

        if final_result:
            logging.info("Successfully processed all segments for chapter summary/tags.")
            final_chapter_details = {'summary': final_result.get('summary'), 'tags': final_result.get('tags')}
        else:
            logging.error("Failed to get final result after processing segments.")

    if final_chapter_details:
        return final_chapter_details.get('summary'), final_chapter_details.get('tags')
    else:
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
    last_page = pages[-1]
    chapter_name = first_page.get('chapter_name', f'Chapter {chapter_num}')
    chapter_filename = first_page.get('filename', 'unknown.pdf')
    
    # Get page ranges
    pdf_page_numbers = [p.get('page_number', 0) for p in pages]
    original_page_numbers = [p.get('original_page_number', p.get('page_number', 0)) for p in pages]
    
    # Log comprehensive chapter info
    logging.info("=" * 80)
    logging.info(f"CHAPTER {chapter_num}: {chapter_name}")
    logging.info(f"  Filename: {chapter_filename}")
    logging.info(f"  Pages in chapter PDF: {min(pdf_page_numbers)}-{max(pdf_page_numbers)} ({len(pages)} pages)")
    logging.info(f"  Original PDF pages: {min(original_page_numbers)}-{max(original_page_numbers)}")
    logging.info("=" * 80)
    
    # Concatenate all page content for GPT processing
    content_parts = []
    total_page_chars = 0
    page_char_counts = []
    
    for i, page in enumerate(pages):
        page_content = page.get('content', '')
        if page_content:
            page_chars = len(page_content)
            total_page_chars += page_chars
            page_char_counts.append(page_chars)
            content_parts.append(page_content)
            
            # Log details for first few pages and any unusually large pages
            if i < 3 or page_chars > 50000:
                page_num = page.get('page_number', 'unknown')
                orig_page = page.get('original_page_number', page_num)
                if page_chars > 50000:
                    logging.warning(f"  ⚠️ Page {page_num} (orig: {orig_page}) has {page_chars:,} chars - unusually large!")
                else:
                    logging.debug(f"  Page {page_num} (orig: {orig_page}): {page_chars:,} chars")
    
    # Calculate statistics
    if page_char_counts:
        avg_chars_per_page = sum(page_char_counts) / len(page_char_counts)
        max_page_chars = max(page_char_counts)
        min_page_chars = min(page_char_counts)
        
        logging.info(f"  Page statistics:")
        logging.info(f"    - Total pages: {len(pages)}")
        logging.info(f"    - Avg chars/page: {avg_chars_per_page:,.0f}")
        logging.info(f"    - Min chars/page: {min_page_chars:,}")
        logging.info(f"    - Max chars/page: {max_page_chars:,}")
        
        # Sanity check
        if avg_chars_per_page > 20000:
            logging.error(f"  ❌ ABNORMAL PAGE SIZE: Average {avg_chars_per_page:,.0f} chars/page!")
            logging.error(f"     Normal pages are typically 2,000-5,000 chars")
            logging.error(f"     This suggests the input JSON may have concatenated content")
    
    concatenated_content = "\n\n".join(content_parts)
    chapter_token_count = count_tokens(concatenated_content)
    char_count = len(concatenated_content)
    logging.info(f"  Content size: {char_count:,} characters → {chapter_token_count:,} tokens")
    
    # Additional sanity check
    if char_count > 500000:  # ~150 pages of normal text
        logging.error(f"  ❌ CHAPTER TOO LARGE: {char_count:,} characters!")
        logging.error(f"     This is equivalent to ~{char_count/3000:.0f} normal pages")
        logging.error(f"     Likely cause: Input JSON has duplicate or concatenated content")
    
    # Generate summary and tags via GPT (ONCE for the whole chapter)
    chapter_summary, chapter_tags = None, None
    if client:
        try:
            chapter_summary, chapter_tags = get_chapter_summary_and_tags(
                concatenated_content, client, model_name=MODEL_NAME_CHAT
            )
            if chapter_summary is None or chapter_tags is None:
                logging.warning(f"  Failed to generate summary/tags for chapter {chapter_num}.")
        except Exception as e:
            logging.error(f"  Exception generating summary/tags for chapter {chapter_num}: {e}")
    else:
        logging.warning("  OpenAI client not available. Skipping summary/tag generation.")
    
    # Apply chapter summary/tags to each page record
    enriched_pages = []
    for page in pages:
        enriched_page = page.copy()  # Keep all original fields
        
        # Add chapter-level enrichments
        enriched_page['chapter_summary'] = chapter_summary
        enriched_page['chapter_tags'] = chapter_tags
        enriched_page['chapter_token_count'] = chapter_token_count
        
        # Calculate page-specific token count
        page_content = page.get('content', '')
        enriched_page['page_token_count'] = count_tokens(page_content)
        
        # Ensure citation fields are clear
        enriched_page['pdf_filename'] = page.get('filename')  # Split PDF name
        enriched_page['pdf_page_number'] = page.get('page_number')  # Page in split PDF
        # original_page_number and page_reference should already be in the record
        
        enriched_pages.append(enriched_page)
    
    logging.info(f"  Enriched {len(enriched_pages)} pages for Chapter {chapter_num}")
    return enriched_pages

def process_unassigned_pages(pages: List[Dict]) -> List[Dict]:
    """
    Processes pages that don't have a chapter assignment.
    Returns enriched page records without chapter summaries.
    """
    if not pages:
        return []
    
    logging.info(f"Processing {len(pages)} unassigned pages")
    
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
    
    logging.info("--- Starting Stage 1: Chapter Processing (Page-Level Records) ---")
    
    # Setup SSL early for both tiktoken and OpenAI
    logging.info("Setting up SSL certificate...")
    _setup_ssl_from_nas()
    
    share_name = NAS_PARAMS["share"]
    output_path_relative = os.path.join(NAS_OUTPUT_PATH, OUTPUT_FILENAME).replace('\\', '/')
    
    # --- Load Input JSON from NAS ---
    logging.info(f"Loading input JSON from NAS: {share_name}/{NAS_INPUT_JSON_PATH}")
    input_json_bytes = read_from_nas(share_name, NAS_INPUT_JSON_PATH)
    
    if not input_json_bytes:
        logging.error(f"Failed to read input JSON from {share_name}/{NAS_INPUT_JSON_PATH}")
        return
    
    try:
        input_data = json.loads(input_json_bytes.decode('utf-8'))
        if not isinstance(input_data, list):
            logging.error("Input JSON is not a list. Expected array of page records.")
            return
        logging.info(f"Loaded {len(input_data)} page records from input JSON")
        
        # Diagnostic: Check the structure of the input data
        if input_data:
            first_record = input_data[0]
            logging.info("Sample record structure:")
            logging.info(f"  Keys: {list(first_record.keys())}")
            
            # Check content size in first few records
            logging.info("First 5 records content analysis:")
            for i, record in enumerate(input_data[:5]):
                content = record.get('content', '')
                page_num = record.get('page_number', 'unknown')
                orig_page = record.get('original_page_number', page_num)
                chapter = record.get('chapter_number', 'unassigned')
                filename = record.get('filename', 'unknown')
                content_len = len(content)
                
                logging.info(f"  Record {i}: Chapter {chapter}, File: {filename}, Page {page_num} (orig: {orig_page})")
                logging.info(f"    Content length: {content_len:,} chars")
                if content_len > 50000:
                    logging.warning(f"    ⚠️ UNUSUALLY LARGE: {content_len:,} chars!")
                    # Show first 200 chars to check for duplication
                    preview = content[:200].replace('\n', ' ')
                    logging.info(f"    Preview: {preview}...")
            
            # Check overall statistics
            all_content_lengths = [len(r.get('content', '')) for r in input_data]
            avg_content = sum(all_content_lengths) / len(all_content_lengths) if all_content_lengths else 0
            max_content = max(all_content_lengths) if all_content_lengths else 0
            
            logging.info(f"Overall input statistics:")
            logging.info(f"  Total records: {len(input_data)}")
            logging.info(f"  Avg content/record: {avg_content:,.0f} chars")
            logging.info(f"  Max content/record: {max_content:,} chars")
            
            if avg_content > 20000:
                logging.error("❌ INPUT DATA ISSUE: Average content per page is abnormally large!")
                logging.error("   This suggests the EY prep or Chapter Assignment Tool may have concatenated content")
                logging.error("   Normal pages should be 2,000-5,000 chars each")
        
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding input JSON: {e}")
        return
    
    # --- Initialize OpenAI Client ---
    client = None
    if OpenAI:
        client = get_openai_client()
        if client:
            logging.info("OpenAI client initialized successfully.")
        else:
            logging.warning("Failed to initialize OpenAI client. Will proceed without summaries/tags.")
    else:
        logging.warning("OpenAI library not installed. Summaries and tags will not be generated.")
    
    # --- Group Pages by Chapter ---
    chapters = group_pages_by_chapter(input_data)
    
    # Find unassigned pages (those without chapter_number)
    unassigned_pages = [r for r in input_data if r.get('chapter_number') is None]
    
    logging.info(f"Found {len(chapters)} chapters and {len(unassigned_pages)} unassigned pages")
    
    # Diagnostic: Show chapter distribution
    logging.info("Chapter distribution:")
    for chapter_num in sorted(chapters.keys())[:10]:  # Show first 10 chapters
        chapter_pages = chapters[chapter_num]
        chapter_name = chapter_pages[0].get('chapter_name', f'Chapter {chapter_num}')
        total_chars = sum(len(p.get('content', '')) for p in chapter_pages)
        logging.info(f"  Chapter {chapter_num} ({chapter_name}): {len(chapter_pages)} pages, {total_chars:,} total chars")
        if total_chars > 500000:
            logging.warning(f"    ⚠️ Chapter {chapter_num} is abnormally large!")
    
    if len(chapters) > 10:
        logging.info(f"  ... and {len(chapters) - 10} more chapters")
    
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
    logging.info(f"Saving {len(all_enriched_pages)} enriched page records to NAS")
    
    try:
        output_json = json.dumps(all_enriched_pages, indent=2, ensure_ascii=False)
        output_bytes = output_json.encode('utf-8')
        
        if write_to_nas(share_name, output_path_relative, output_bytes):
            logging.info(f"Successfully saved output to: {share_name}/{output_path_relative}")
        else:
            logging.error("Failed to write output to NAS")
    except Exception as e:
        logging.error(f"Error saving output: {e}")
    
    # --- Upload Log File to NAS ---
    try:
        log_file_name = f"stage1_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path_relative = os.path.join(NAS_LOG_PATH, log_file_name).replace('\\', '/')
        
        # Close logging handlers to flush content
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)
        
        # Read log content and upload to NAS
        with open(temp_log_path, 'rb') as f:
            log_content = f.read()
        
        if write_to_nas(share_name, log_path_relative, log_content):
            print(f"Log file uploaded to NAS: {share_name}/{log_path_relative}")
        else:
            print(f"Failed to upload log file to NAS")
        
        # Clean up temp log file
        os.remove(temp_log_path)
    except Exception as e:
        print(f"Error handling log file: {e}")
    
    # --- Final Summary ---
    print("--- Stage 1 Summary ---")
    print(f"Input: {len(input_data)} page records")
    print(f"Output: {len(all_enriched_pages)} enriched page records")
    print(f"Chapters processed: {len(chapters)}")
    if unassigned_pages:
        print(f"Unassigned pages processed: {len(unassigned_pages)}")
    print(f"Output file: {share_name}/{output_path_relative}")
    print("--- Stage 1 Completed ---")

# ==============================================================================
# Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    run_stage1()