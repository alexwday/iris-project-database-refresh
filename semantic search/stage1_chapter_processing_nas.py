# -*- coding: utf-8 -*-
"""
Stage 1: Chapter Processing (NAS-Integrated Version)

Purpose:
Processes markdown files representing individual chapters from a NAS directory.
For each chapter, it extracts metadata (number, name, page range), calculates
token count, generates a summary and tags using an LLM, and assembles the
data for the next stage.

Input: Markdown files in NAS_INPUT_MD_PATH on the NAS drive.
       Filename pattern: <number>_<name>.md (e.g., '1_Introduction.md')
Output: A JSON file on the NAS containing a list of chapter dictionaries.
        (e.g., 'stage1_chapter_data.json')
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

# --- pysmb imports for NAS access ---
from smb.SMBConnection import SMBConnection
from smb import smb_structs

# --- Dependencies Check ---
try:
    import tiktoken
except ImportError:
    tiktoken = None
    print("WARNING: tiktoken not installed. Token counts will be estimated (chars/4). `pip install tiktoken`")

try:
    import natsort
except ImportError:
    natsort = None
    print("INFO: natsort not installed. Chapters might not sort naturally. `pip install natsort`")

try:
    from openai import OpenAI, APIError
except ImportError:
    OpenAI = None
    APIError = None
    print("ERROR: openai library not installed. GPT features unavailable. `pip install openai`")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x # Make tqdm optional, replace with identity function if not found
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
# Path on NAS where markdown files are stored (relative to share root)
NAS_INPUT_MD_PATH = "semantic_search/1_chapters_md"  # TODO: Adjust path as needed
# Path on NAS where output will be saved (relative to share root)
NAS_OUTPUT_PATH = "semantic_search/pipeline_output/stage1"
# Path on NAS where logs will be saved
NAS_LOG_PATH = "semantic_search/pipeline_output/logs"
OUTPUT_FILENAME = "stage1_chapter_data.json"

# --- CA Bundle Configuration ---
# Path on NAS where the SSL certificate is stored (relative to share root)
NAS_SSL_CERT_PATH = "certificates/rbc-ca-bundle.cer"  # TODO: Adjust to match your NAS location
SSL_LOCAL_PATH = "/tmp/rbc-ca-bundle.cer"  # Temp path for cert

# --- Document ID ---
DOCUMENT_ID = "EY_GUIDE_2024_PLACEHOLDER"  # TODO: Set appropriate document ID

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
# NAS Helper Functions (from catalog search pattern)
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

def download_from_nas(share_name, nas_path_relative, local_temp_dir):
    """Downloads a file from NAS to a local temporary directory."""
    local_file_path = os.path.join(local_temp_dir, os.path.basename(nas_path_relative))
    logging.debug(f"Attempting to download from NAS: {share_name}/{nas_path_relative}")
    
    content_bytes = read_from_nas(share_name, nas_path_relative)
    if content_bytes is None:
        return None
    
    try:
        with open(local_file_path, 'wb') as f:
            f.write(content_bytes)
        logging.debug(f"Downloaded to: {local_file_path}")
        return local_file_path
    except Exception as e:
        logging.error(f"Failed to write downloaded file to {local_file_path}: {e}")
        return None

def list_nas_directory(share_name, dir_path_relative):
    """Lists files in a NAS directory."""
    conn = None
    try:
        conn = create_nas_connection()
        if not conn:
            return []
        
        files = conn.listPath(share_name, dir_path_relative)
        # Filter out '.' and '..' entries
        files = [f for f in files if f.filename not in ['.', '..']]
        return files
    except Exception as e:
        logging.error(f"Failed to list NAS directory '{share_name}/{dir_path_relative}': {e}")
        return []
    finally:
        if conn:
            conn.close()

# ==============================================================================
# Logging Setup (Modified for NAS)
# ==============================================================================

def setup_logging():
    """Setup logging to write to NAS."""
    # Create a temporary local log file first
    temp_log = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log')
    temp_log_path = temp_log.name
    temp_log.close()
    
    # Configure logging to write to temp file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler(temp_log_path),
            logging.StreamHandler()
        ]
    )
    
    # Log the temp path for later upload
    logging.info(f"Temporary log file: {temp_log_path}")
    return temp_log_path

# ==============================================================================
# Utility Functions (Original, unchanged)
# ==============================================================================

# --- Tokenizer ---
_TOKENIZER = None
if tiktoken:
    try:
        _TOKENIZER = tiktoken.get_encoding("cl100k_base")
        logging.info("Using 'cl100k_base' tokenizer via tiktoken.")
    except Exception as e:
        logging.warning(f"Failed to initialize tiktoken tokenizer: {e}. Falling back to estimate.")
        _TOKENIZER = None

def count_tokens(text: str) -> int:
    """Counts tokens using tiktoken if available, otherwise estimates (chars/4)."""
    if not text:
        return 0
    if _TOKENIZER:
        try:
            return len(_TOKENIZER.encode(text))
        except Exception as e:
            return len(text) // 4
    else:
        return len(text) // 4

# --- API Client (Modified for NAS SSL) ---
_SSL_CONFIGURED = False
_OPENAI_CLIENT = None

def _setup_ssl_from_nas() -> bool:
    """Downloads SSL cert from NAS and sets environment variables."""
    global _SSL_CONFIGURED
    if _SSL_CONFIGURED:
        return True
    
    logging.info("Setting up SSL certificate from NAS...")
    try:
        # Download cert from NAS
        cert_bytes = read_from_nas(NAS_PARAMS["share"], NAS_SSL_CERT_PATH)
        if cert_bytes is None:
            logging.warning(f"SSL certificate not found on NAS at {NAS_SSL_CERT_PATH}. API calls may fail.")
            _SSL_CONFIGURED = True
            return True
        
        # Write to local temp location
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

# --- File/Path Utils ---
def cleanup_filename(name: str) -> str:
    """Cleans a string for use as a filename."""
    name = str(name)
    name = re.sub(r'[\\/:?*<>|"\']', "", name).strip()
    name = re.sub(r"[\s_]+", "_", name)
    return name[:50].strip("_")

# --- Page Tag Extraction ---
PAGE_NUMBER_TAG_PATTERN = re.compile(r'<!--\s*PageNumber="(\d+)"\s*-->')
AZURE_TAG_PATTERN = re.compile(
    r'<!--\s*Page(Footer|Number|Break|Header)=?(".*?"|\d+)?\s*-->\s*\n?'
)

def clean_azure_tags(text: str) -> str:
    """Removes Azure Document Intelligence specific HTML comment tags from text."""
    return AZURE_TAG_PATTERN.sub("", text)

def extract_page_mapping(content: str) -> list[tuple[int, int]]:
    """
    Extracts a mapping of character positions to page numbers from tags.
    Returns a list of (character_position, page_number) tuples, sorted by position.
    """
    mapping = []
    raw_matches = []
    for match in PAGE_NUMBER_TAG_PATTERN.finditer(content):
        pos = match.start()
        page_num = int(match.group(1))
        raw_matches.append((pos, page_num))

    if not raw_matches: return []

    raw_matches.sort(key=lambda x: (x[0], -x[1]))

    if raw_matches:
        mapping.append(raw_matches[0])
        for i in range(1, len(raw_matches)):
            if raw_matches[i][0] > mapping[-1][0]:
                mapping.append(raw_matches[i])

    if mapping:
        last_entry_pos, last_entry_page = mapping[-1]
        if last_entry_pos < len(content):
            mapping.append((len(content), last_entry_page))

    return mapping

def get_page_range_from_mapping(page_mapping: list[tuple[int, int]], last_processed_end_page: int) -> Tuple[int, int]:
    """Determines start and end page from mapping, using fallback."""
    if not page_mapping:
        start_page = last_processed_end_page + 1
        end_page = start_page
        logging.warning(f"No page tags found. Inferring start page: {start_page}")
    else:
        start_page = page_mapping[0][1]
        end_page = page_mapping[-1][1]
        end_page = max(start_page, end_page)
        logging.debug(f"Tags found. Derived page range: {start_page}-{end_page}")

    return start_page, end_page

# --- Chapter Info Extraction ---
def extract_chapter_info_from_filename(filename: str) -> tuple[Optional[int], str]:
    """
    Extracts chapter number and base name from filename (e.g., '1_Intro.md').
    Returns (chapter_number, base_name_for_title).
    """
    basename = os.path.basename(filename)
    match = re.match(r"(\d+)[_-](.+)", basename)
    if match:
        try:
            chapter_number = int(match.group(1))
            base_name = os.path.splitext(match.group(2))[0]
            title_guess = base_name.replace("_", " ").replace("-", " ")
            return chapter_number, title_guess
        except ValueError:
            pass

    title_guess = os.path.splitext(basename)[0]
    return None, title_guess

# --- GPT Prompting for Chapter Details ---
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
    logging.info(f"Total estimated tokens for chapter: {total_tokens}")

    processing_limit = GPT_INPUT_TOKEN_LIMIT - MAX_COMPLETION_TOKENS_CHAPTER - TOKEN_BUFFER
    final_chapter_details = None

    if total_tokens <= processing_limit:
        logging.info("Processing chapter summary/tags in a single call.")
        result = process_chapter_segment_for_details(
            chapter_text, client, model_name, MAX_COMPLETION_TOKENS_CHAPTER, TEMPERATURE, is_final_segment=True
        )
        if result:
            final_chapter_details = {'summary': result.get('summary'), 'tags': result.get('tags')}
        else:
            logging.error("Failed to process chapter in single call.")
    else:
        logging.info(f"Chapter exceeds token limit ({total_tokens} > {processing_limit}). Processing in segments.")
        num_segments = (total_tokens // processing_limit) + 1
        segment_len_approx = len(chapter_text) // num_segments
        segments = [chapter_text[i:i + segment_len_approx] for i in range(0, len(chapter_text), segment_len_approx)]

        logging.info(f"Divided chapter into {len(segments)} segments.")
        current_summary = None
        current_tags = None
        final_result = None

        for i, segment_text in enumerate(tqdm(segments, desc="Processing Segments")):
            logging.debug(f"Processing segment {i + 1}/{len(segments)}...")
            is_final = (i == len(segments) - 1)
            segment_result = process_chapter_segment_for_details(
                segment_text, client, model_name, MAX_COMPLETION_TOKENS_CHAPTER, TEMPERATURE,
                prev_summary=current_summary, prev_tags=current_tags, is_final_segment=is_final
            )

            if segment_result:
                current_summary = segment_result.get('summary')
                current_tags = segment_result.get('tags')
                if is_final:
                    final_result = segment_result
                logging.debug(f"Segment {i + 1} processed.")
            else:
                logging.error(f"Error processing segment {i + 1}. Aborting chapter summary generation.")
                return None, None

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
# Main Stage 1 Logic (Modified for NAS)
# ==============================================================================

def process_chapter_file(md_file_content: bytes, file_name: str, client: OpenAI, last_processed_end_page: int) -> Optional[Dict]:
    """Processes a single chapter markdown file content."""
    logging.info(f"Processing Chapter File: {file_name}")

    try:
        # 1. Extract chapter number and guess title from filename
        chapter_number, title_guess = extract_chapter_info_from_filename(file_name)
        if chapter_number is None:
            logging.warning(f"Could not extract chapter number from filename '{file_name}'. Skipping.")
            return None

        # 2. Decode content
        raw_content = md_file_content.decode('utf-8')

        # 3. Extract chapter name
        first_line = raw_content.split('\n', 1)[0].strip()
        chapter_name = re.sub(r"^\s*#+\s*", "", first_line).strip()
        if not chapter_name:
            chapter_name = title_guess
        logging.debug(f"  Chapter Name: '{chapter_name}'")

        # 4. Extract page mapping and determine range
        page_mapping = extract_page_mapping(raw_content)
        chapter_page_start, chapter_page_end = get_page_range_from_mapping(page_mapping, last_processed_end_page)
        logging.debug(f"  Page Range: {chapter_page_start}-{chapter_page_end}")

        # 5. Clean content for token counting
        cleaned_content_for_count = clean_azure_tags(raw_content)
        chapter_token_count = count_tokens(cleaned_content_for_count)
        logging.debug(f"  Token Count: {chapter_token_count}")

        # 6. Generate Summary and Tags via GPT
        chapter_summary, chapter_tags = None, None
        if client:
            chapter_summary, chapter_tags = get_chapter_summary_and_tags(raw_content, client, model_name=MODEL_NAME_CHAT)
            if chapter_summary is None or chapter_tags is None:
                logging.warning(f"  Failed to generate summary/tags for chapter {chapter_number}.")
        else:
            logging.warning("  OpenAI client not available. Skipping summary/tag generation.")

        # 7. Assemble chapter data dictionary
        chapter_data = {
            "document_id": DOCUMENT_ID,
            "chapter_number": chapter_number,
            "chapter_name": chapter_name,
            "chapter_page_start": chapter_page_start,
            "chapter_page_end": chapter_page_end,
            "chapter_token_count": chapter_token_count,
            "chapter_summary": chapter_summary,
            "chapter_tags": chapter_tags,
            "raw_content": raw_content,
            "_source_filename": file_name,
        }
        return chapter_data

    except Exception as e:
        logging.error(f"Error processing file {file_name}: {e}", exc_info=True)
        return None

def run_stage1():
    """Main function to execute Stage 1 processing with NAS integration and resumability."""
    # Setup logging
    temp_log_path = setup_logging()
    
    logging.info("--- Starting Stage 1: Chapter Processing (NAS-Integrated) ---")
    
    share_name = NAS_PARAMS["share"]
    output_path_relative = os.path.join(NAS_OUTPUT_PATH, OUTPUT_FILENAME).replace('\\', '/')
    
    # --- Load Existing Data (for Resumability) ---
    existing_data = []
    processed_filenames = set()
    
    # Try to read existing output from NAS
    existing_json_bytes = read_from_nas(share_name, output_path_relative)
    if existing_json_bytes:
        try:
            existing_data = json.loads(existing_json_bytes.decode('utf-8'))
            if isinstance(existing_data, list):
                processed_filenames = {chap.get("_source_filename") for chap in existing_data if chap.get("_source_filename")}
                logging.info(f"Loaded {len(existing_data)} existing chapter records from NAS. Found {len(processed_filenames)} previously processed filenames.")
            else:
                logging.warning("Existing output file does not contain a valid list. Starting fresh.")
                existing_data = []
        except json.JSONDecodeError:
            logging.error("Error decoding JSON from existing output file. Starting fresh.", exc_info=True)
            existing_data = []
    else:
        logging.info("No existing output file found on NAS. Starting fresh.")
    
    # --- Find Input Files on NAS ---
    logging.info(f"Looking for markdown files in NAS path: {share_name}/{NAS_INPUT_MD_PATH}")
    nas_files = list_nas_directory(share_name, NAS_INPUT_MD_PATH)
    
    # Filter for .md files
    markdown_files = [f for f in nas_files if f.filename.lower().endswith('.md') and not f.isDirectory]
    
    if not markdown_files:
        logging.warning(f"No markdown files found in {share_name}/{NAS_INPUT_MD_PATH}.")
        # Save existing data back if it was loaded
        if existing_data:
            json_bytes = json.dumps(existing_data, indent=2, ensure_ascii=False).encode('utf-8')
            write_to_nas(share_name, output_path_relative, json_bytes)
            logging.info(f"No new files to process. Saved {len(existing_data)} existing records back to NAS.")
        return existing_data
    
    # Sort files naturally if possible
    file_names = [f.filename for f in markdown_files]
    if natsort:
        file_names = natsort.natsorted(file_names)
        logging.info(f"Found and naturally sorted {len(file_names)} markdown files.")
    else:
        file_names.sort()
        logging.info(f"Found {len(file_names)} markdown files (standard sort).")
    
    # --- Initialize OpenAI Client ---
    client = get_openai_client()
    if not client:
        logging.warning("OpenAI client initialization failed. Summary/Tag generation will be skipped for new chapters.")
    
    # --- Process Files ---
    processed_this_run_count = 0
    skipped_count = 0
    failed_this_run_count = 0
    
    # Determine the last page number from existing data
    last_processed_end_page = 0
    if existing_data:
        try:
            sorted_existing = sorted(existing_data, key=lambda x: x.get('chapter_number', 0))
            if sorted_existing:
                last_processed_end_page = sorted_existing[-1].get("chapter_page_end", 0)
        except Exception as e:
            logging.warning(f"Could not reliably determine last page from existing data: {e}")
    
    logging.info(f"Starting processing loop. Last known end page: {last_processed_end_page}")
    
    # Create temporary directory for any local operations
    with tempfile.TemporaryDirectory() as temp_dir:
        for file_name in tqdm(file_names, desc="Processing Chapters"):
            if file_name in processed_filenames:
                logging.debug(f"Skipping already processed file: {file_name}")
                skipped_count += 1
                continue
            
            logging.info(f"Processing new or previously failed file: {file_name}")
            
            # Read file content from NAS
            file_path_relative = os.path.join(NAS_INPUT_MD_PATH, file_name).replace('\\', '/')
            md_content = read_from_nas(share_name, file_path_relative)
            
            if md_content is None:
                logging.error(f"Failed to read {file_name} from NAS. Skipping.")
                failed_this_run_count += 1
                continue
            
            # Process the chapter
            chapter_result = process_chapter_file(md_content, file_name, client, last_processed_end_page)
            
            if chapter_result:
                # Success: Append, Update State, and Save Incrementally
                existing_data.append(chapter_result)
                processed_filenames.add(file_name)
                processed_this_run_count += 1
                last_processed_end_page = chapter_result.get("chapter_page_end", last_processed_end_page)
                
                # Incremental Save to NAS
                try:
                    temp_sorted_data = existing_data
                    if natsort:
                        try:
                            temp_sorted_data = sorted(existing_data, key=lambda x: x.get('chapter_number', float('inf')))
                        except Exception as sort_e:
                            logging.warning(f"Could not re-sort data before incremental save: {sort_e}")
                            temp_sorted_data = existing_data
                    
                    json_bytes = json.dumps(temp_sorted_data, indent=2, ensure_ascii=False).encode('utf-8')
                    if write_to_nas(share_name, output_path_relative, json_bytes):
                        logging.debug(f"Incrementally saved {len(temp_sorted_data)} records after processing {file_name}")
                        existing_data = temp_sorted_data
                    else:
                        logging.error(f"Failed to save incremental update to NAS after processing {file_name}")
                
                except Exception as e:
                    logging.error(f"Error during incremental save after processing {file_name}: {e}", exc_info=True)
            
            else:
                failed_this_run_count += 1
                logging.warning(f"Processing failed for {file_name}. It will be retried on the next run.")
    
    # --- Final Sort and Save ---
    if natsort:
        try:
            existing_data = sorted(existing_data, key=lambda x: x.get('chapter_number', float('inf')))
            logging.info("Performed final sort of chapter data.")
            json_bytes = json.dumps(existing_data, indent=2, ensure_ascii=False).encode('utf-8')
            write_to_nas(share_name, output_path_relative, json_bytes)
            logging.info(f"Saved final sorted data with {len(existing_data)} records.")
        except Exception as final_sort_e:
            logging.warning(f"Could not perform final sort or save: {final_sort_e}")
    else:
        logging.info("natsort not available, skipping final sort.")
    
    # --- Upload Log File to NAS ---
    try:
        log_file_name = f"stage1_chapter_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    print(f"Input files found    : {len(file_names)}")
    print(f"Skipped (already done): {skipped_count}")
    print(f"Processed this run   : {processed_this_run_count}")
    print(f"Failed this run      : {failed_this_run_count}")
    print(f"Total records in file: {len(existing_data)}")
    print(f"Output JSON file     : {share_name}/{output_path_relative}")
    print("--- Stage 1 Finished ---")
    
    return existing_data

# ==============================================================================
# Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    run_stage1()