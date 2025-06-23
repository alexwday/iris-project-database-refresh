# -*- coding: utf-8 -*-
"""
Stage 2: Process Documents

Purpose:
Processes markdown files identified by Stage 1 from NAS storage.
For each document, it extracts metadata (number, name, page range), calculates
token count, generates a summary and tags using an LLM, and assembles the
data for the next stage.

Input: 
- JSON output from Stage 1 (1C_nas_files_to_process.json) per document_id
- Markdown files stored on NAS in document_id folders
Output: 
- JSON file per document_id containing a list of chapter dictionaries
- (e.g., '2A_chapter_data.json')
"""

import os
import json
import traceback
import re
import time
import logging
import requests
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
from smb.SMBConnection import SMBConnection
from smb import smb_structs
import io
from datetime import datetime, timezone
import socket

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
# Configuration
# ==============================================================================

# --- NAS Configuration ---
# Network attached storage connection parameters
NAS_PARAMS = {
        "ip": "your_nas_ip",
        "share": "your_share_name",
        "user": "your_nas_user",
        "password": "your_nas_password",
        "port": 445
}
# Base path on the NAS share containing the root folders for different document IDs
NAS_BASE_INPUT_PATH = "path/to/your/base_input_folder"
# Base path on the NAS share where output JSON files will be stored
NAS_OUTPUT_FOLDER_PATH = "path/to/your/output_folder"

# --- Processing Configuration ---
# Document IDs configuration - each line contains document_id and detail level
DOCUMENT_IDS = """
external_ey,detailed
external_ias,standard
# external_ifrs,detailed
external_ifric,standard
external_sic,concise
# external_pwc,detailed
"""

def load_document_ids():
        """Parse document IDs configuration - works for all stages"""
        document_ids = []
        for line in DOCUMENT_IDS.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(',')
                if len(parts) == 2:
                    document_id = parts[0].strip()
                    detail_level = parts[1].strip()
                    document_ids.append({
                        'document_id': document_id,
                        'detail_level': detail_level
                    })
                else:
                    print(f"Warning: Invalid config line ignored: {line}")
        return document_ids

# --- Directory Paths ---
LOG_DIR = "pipeline_output/logs"

# --- API Configuration ---
# TODO: Load securely (e.g., environment variables) or replace placeholders
BASE_URL = os.environ.get("OPENAI_API_BASE", "https://api.example.com/v1") # Use your actual base URL
MODEL_NAME_CHAT = os.environ.get("OPENAI_MODEL_CHAT", "gpt-4-turbo-nonp")
OAUTH_URL = os.environ.get("OAUTH_URL", "https://api.example.com/oauth/token")
CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID", "your_client_id") # Placeholder
CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET", "your_client_secret") # Placeholder
SSL_SOURCE_PATH = os.environ.get("SSL_SOURCE_PATH", "/path/to/your/rbc-ca-bundle.cer") # Adjust path
SSL_LOCAL_PATH = "/tmp/rbc-ca-bundle.cer" # Temp path for cert

# --- API Parameters ---
MAX_COMPLETION_TOKENS_CHAPTER = 4000 # Max tokens for chapter summary/tags response
TEMPERATURE = 0.3 # Lower temperature for more factual responses
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY = 5 # seconds
GPT_INPUT_TOKEN_LIMIT = 80000 # Approx limit for GPT input before needing segmentation
TOKEN_BUFFER = 2000 # Buffer tokens when calculating segmentation limit

# --- Token Cost (Optional) ---
PROMPT_TOKEN_COST = 0.01    # Cost per 1K prompt tokens
COMPLETION_TOKEN_COST = 0.03 # Cost per 1K completion tokens

# --- pysmb Configuration ---
smb_structs.SUPPORT_SMB2 = True
smb_structs.MAX_PAYLOAD_SIZE = 65536
CLIENT_HOSTNAME = socket.gethostname()

# --- Logging Setup ---
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
log_file = Path(LOG_DIR) / 'stage2_process_documents.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

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
            logging.debug(f"Successfully connected to NAS: {NAS_PARAMS['ip']}:{NAS_PARAMS['port']} on share '{NAS_PARAMS['share']}'")
            return conn
        except Exception as e:
            logging.error(f"Exception creating NAS connection: {e}")
            return None

def read_file_from_nas(share_name, file_path_relative):
        """Read a file from NAS into a string."""
        conn = None
        try:
            conn = create_nas_connection()
            if not conn:
                return None

            file_obj = io.BytesIO()
            file_attributes, filesize = conn.retrieveFile(share_name, file_path_relative, file_obj)
            file_obj.seek(0)
            file_content = file_obj.read().decode('utf-8')
            
            logging.debug(f"Successfully read file from NAS: {share_name}/{file_path_relative} ({filesize} bytes)")
            return file_content
        except Exception as e:
            logging.error(f"Failed to read file from NAS '{share_name}/{file_path_relative}': {e}")
            return None
        finally:
            if conn:
                conn.close()

def read_json_from_nas(share_name, file_path_relative):
        """Read a JSON file from NAS and parse it."""
        content = read_file_from_nas(share_name, file_path_relative)
        if content is None:
            return None
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON from NAS file '{share_name}/{file_path_relative}': {e}")
            return None

def write_json_to_nas(share_name, nas_path_relative, data):
        """Writes JSON data to a specified file path on the NAS using pysmb."""
        conn = None
        logging.info(f"Attempting to write to NAS path: {share_name}/{nas_path_relative}")
        try:
            conn = create_nas_connection()
            if not conn:
                return False

            # Ensure directory exists
            dir_path = os.path.dirname(nas_path_relative).replace('\\', '/')
            if dir_path:
                ensure_nas_dir_exists(conn, share_name, dir_path)

            json_string = json.dumps(data, indent=4, ensure_ascii=False)
            data_bytes = json_string.encode('utf-8')
            file_obj = io.BytesIO(data_bytes)

            bytes_written = conn.storeFile(share_name, nas_path_relative, file_obj)
            logging.info(f"Successfully wrote {bytes_written} bytes to: {share_name}/{nas_path_relative}")
            return True
        except Exception as e:
            logging.error(f"Unexpected error writing to NAS '{share_name}/{nas_path_relative}': {e}")
            return False
        finally:
            if conn:
                conn.close()

def ensure_nas_dir_exists(conn, share_name, dir_path):
        """Ensures a directory exists on the NAS, creating it if necessary."""
        if not conn:
            logging.error("Cannot ensure NAS directory: No connection.")
            return False
        
        path_parts = dir_path.strip('/').split('/')
        current_path = ''
        try:
            for part in path_parts:
                if not part: continue
                current_path = os.path.join(current_path, part).replace('\\', '/')
                try:
                    conn.listPath(share_name, current_path)
                except Exception:
                    logging.debug(f"Creating directory on NAS: {current_path}")
                    conn.createDirectory(share_name, current_path)
            return True
        except Exception as e:
            logging.error(f"Failed to ensure/create NAS directory '{dir_path}': {e}")
            return False

# ==============================================================================
# Utility Functions (Self-Contained)
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

# --- API Client ---
_SSL_CONFIGURED = False
_OPENAI_CLIENT = None

def _setup_ssl(source_path=SSL_SOURCE_PATH, local_path=SSL_LOCAL_PATH) -> bool:
    """Copies SSL cert locally and sets environment variables."""
    global _SSL_CONFIGURED
    if _SSL_CONFIGURED:
        return True
    if not Path(source_path).is_file():
         logging.warning(f"SSL source certificate not found at {source_path}. API calls may fail if HTTPS is required and cert is not already trusted.")
         _SSL_CONFIGURED = True # Mark as "configured" to avoid retrying
         return True

    logging.info("Setting up SSL certificate...")
    try:
        source = Path(source_path)
        local = Path(local_path)
        local.parent.mkdir(parents=True, exist_ok=True)
        with open(source, "rb") as source_file, open(local, "wb") as dest_file:
            dest_file.write(source_file.read())
        os.environ["SSL_CERT_FILE"] = str(local)
        os.environ["REQUESTS_CA_BUNDLE"] = str(local)
        logging.info(f"SSL certificate configured successfully at: {local}")
        _SSL_CONFIGURED = True
        return True
    except Exception as e:
        logging.error(f"Error setting up SSL certificate: {e}", exc_info=True)
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
    if not _setup_ssl():
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

            # --- Tool Call Handling ---
            if response_message.tool_calls:
                tool_call = response_message.tool_calls[0]
                if tool_choice and isinstance(tool_choice, dict):
                    expected_tool_name = tool_choice.get("function", {}).get("name")
                    if expected_tool_name and tool_call.function.name != expected_tool_name:
                        raise ValueError(f"Expected tool '{expected_tool_name}' but received '{tool_call.function.name}'")
                function_args_json = tool_call.function.arguments
                return function_args_json, usage_info
            # --- Standard Content Handling ---
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
        # Handle potential markdown code blocks
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
def create_directory(directory: str):
    """Creates the specified directory if it does not already exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)

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
                    "description": "A detailed summary of the chapter segment, following the structure outlined in the prompt (Purpose, Key Topics/Standards, Context/Applicability, Key Outcomes/Decisions)."
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of 5-15 specific, granular topic tags relevant for retrieval by accounting professionals (e.g., standard names/paragraphs, core concepts, procedures, key terms, applicability)."
                }
            },
            "required": ["summary", "tags"]
        }
    }
}

def _build_chapter_prompt(segment_text, prev_summary=None, prev_tags=None, is_final_segment=False):
    """Builds the messages list for the chapter/segment processing call using CO-STAR and XML."""
    system_prompt = """<role>You are an expert financial reporting specialist with deep knowledge of IFRS and US GAAP.</role>
<source_material>You are analyzing segments of a chapter from a comprehensive financial reporting technical guidance manual.</source_material>
<task>Your primary task is to extract key information and generate a highly detailed, structured summary and a set of specific, granular topic tags for the provided text segment. This output will be used to build a knowledge base for accurate retrieval by accounting professionals. You will provide the output using the available 'extract_chapter_details' tool.</task>
<guardrails>Base your analysis strictly on the provided text segment and any previous context given. Do not infer information not explicitly present or heavily implied. Focus on factual extraction and objective summarization. Ensure tags are precise and directly relevant to accounting standards, concepts, or procedures mentioned.</guardrails>"""

    user_prompt_elements = ["<prompt>"]
    user_prompt_elements.append("<context>You are processing a text segment from a chapter within a financial reporting technical guidance manual (likely IFRS or US GAAP focused). The ultimate goal is to populate a knowledge base for efficient and accurate information retrieval by accounting professionals performing research.</context>")
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
# Main Stage 2 Logic
# ==============================================================================

def process_document_file(file_info: Dict, document_id: str, client: OpenAI, last_processed_end_page: int) -> Optional[Dict]:
    """Processes a single document markdown file from NAS."""
    file_name = file_info.get('file_name')
    file_path = file_info.get('file_path')
    
    logging.info(f"Processing Document File: {file_name}")

    try:
        # 1. Extract chapter number and guess title from filename
        chapter_number, title_guess = extract_chapter_info_from_filename(file_name)
        if chapter_number is None:
            logging.warning(f"Could not extract chapter number from filename '{file_name}'. Skipping.")
            return None

        # 2. Read raw content from NAS
        raw_content = read_file_from_nas(NAS_PARAMS["share"], file_path)
        if raw_content is None:
            logging.error(f"Failed to read file content from NAS: {file_path}")
            return None

        # 3. Extract chapter name (use first line if present, else fallback to title guess)
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
            "document_id": document_id,
            "chapter_number": chapter_number,
            "chapter_name": chapter_name,
            "chapter_page_start": chapter_page_start,
            "chapter_page_end": chapter_page_end,
            "chapter_token_count": chapter_token_count,
            "chapter_summary": chapter_summary,
            "chapter_tags": chapter_tags,
            "raw_content": raw_content,
            "_source_filename": file_name,
            # Add file metadata from Stage 1
            "file_size": file_info.get('file_size'),
            "date_created": file_info.get('date_created'),
            "date_last_modified": file_info.get('date_last_modified'),
        }
        return chapter_data

    except Exception as e:
        logging.error(f"Error processing file {file_name}: {e}", exc_info=True)
        return None

def check_skip_flag(document_id: str) -> bool:
    """Check if skip flag exists for this document_id."""
    nas_output_dir_relative = os.path.join(NAS_OUTPUT_FOLDER_PATH, document_id).replace('\\', '/')
    skip_flag_relative_path = os.path.join(nas_output_dir_relative, '_SKIP_SUBSEQUENT_STAGES.flag').replace('\\', '/')
    
    conn = None
    try:
        conn = create_nas_connection()
        if not conn:
            return False
        conn.getAttributes(NAS_PARAMS["share"], skip_flag_relative_path)
        return True  # Flag exists
    except:
        return False  # Flag doesn't exist
    finally:
        if conn:
            conn.close()

def run_stage2():
    """Main function to execute Stage 2 processing."""
    logging.info("--- Starting Stage 2: Process Documents ---")
    
    # Get document IDs
    document_configs = load_document_ids()
    logging.info(f"Processing {len(document_configs)} document IDs:")
    for config in document_configs:
        logging.info(f"   - {config['document_id']} (detail level: {config['detail_level']})")

    # Initialize OpenAI Client
    client = get_openai_client()
    if not client:
        logging.warning("OpenAI client initialization failed. Summary/Tag generation will be skipped for new chapters.")

    # Process each document ID
    for document_config in document_configs:
        document_id = document_config['document_id']
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing Document ID: {document_id}")
        logging.info(f"{'='*60}")

        # Check skip flag
        if check_skip_flag(document_id):
            logging.info(f"Skip flag found for {document_id}. Skipping processing.")
            continue

        # Read Stage 1 output for files to process
        nas_output_dir_relative = os.path.join(NAS_OUTPUT_FOLDER_PATH, document_id).replace('\\', '/')
        process_file_path = os.path.join(nas_output_dir_relative, '1C_nas_files_to_process.json').replace('\\', '/')
        
        files_to_process = read_json_from_nas(NAS_PARAMS["share"], process_file_path)
        if files_to_process is None:
            logging.error(f"Failed to read files to process for {document_id}. Skipping.")
            continue

        if not files_to_process:
            logging.info(f"No files to process for {document_id}.")
            continue

        logging.info(f"Found {len(files_to_process)} files to process for {document_id}")

        # Filter for markdown files only
        md_files = [f for f in files_to_process if f.get('file_name', '').lower().endswith('.md')]
        if not md_files:
            logging.info(f"No markdown files found to process for {document_id}.")
            continue

        logging.info(f"Processing {len(md_files)} markdown files for {document_id}")

        # Sort files naturally if possible
        if natsort:
            md_files = natsort.natsorted(md_files, key=lambda f: f.get('file_name', ''))
            logging.info(f"Naturally sorted {len(md_files)} markdown files.")
        else:
            md_files.sort(key=lambda f: f.get('file_name', ''))
            logging.info(f"Standard sorted {len(md_files)} markdown files.")

        # Process files
        processed_chapters = []
        last_processed_end_page = 0
        processed_count = 0
        failed_count = 0

        for file_info in tqdm(md_files, desc=f"Processing {document_id}"):
            file_name = file_info.get('file_name')
            logging.info(f"Processing file: {file_name}")
            
            chapter_result = process_document_file(file_info, document_id, client, last_processed_end_page)

            if chapter_result:
                processed_chapters.append(chapter_result)
                processed_count += 1
                last_processed_end_page = chapter_result.get("chapter_page_end", last_processed_end_page)
                logging.debug(f"Successfully processed {file_name}")
            else:
                failed_count += 1
                logging.warning(f"Failed to process {file_name}")

        # Sort processed chapters by chapter number
        if natsort:
            try:
                processed_chapters = sorted(processed_chapters, key=lambda x: x.get('chapter_number', float('inf')))
                logging.info("Performed final sort of chapter data.")
            except Exception as e:
                logging.warning(f"Could not perform final sort: {e}")

        # Save output for this document_id
        output_file_path = os.path.join(nas_output_dir_relative, '2A_chapter_data.json').replace('\\', '/')
        if write_json_to_nas(NAS_PARAMS["share"], output_file_path, processed_chapters):
            logging.info(f"Successfully saved {len(processed_chapters)} chapters to {output_file_path}")
        else:
            logging.error(f"Failed to save chapter data for {document_id}")

        # Summary for this document_id
        logging.info(f"--- Summary for {document_id} ---")
        logging.info(f"Files found: {len(md_files)}")
        logging.info(f"Successfully processed: {processed_count}")
        logging.info(f"Failed: {failed_count}")
        logging.info(f"Total chapters saved: {len(processed_chapters)}")

    logging.info("--- Stage 2 Completed ---")

# ==============================================================================
# Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    run_stage2()