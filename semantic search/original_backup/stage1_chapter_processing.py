# -*- coding: utf-8 -*-
"""
Stage 1: Chapter Processing

Purpose:
Processes markdown files representing individual chapters from an input directory.
For each chapter, it extracts metadata (number, name, page range), calculates
token count, generates a summary and tags using an LLM, and assembles the
data for the next stage.

Input: Markdown files in INPUT_MD_DIR (e.g., '1_chapters_md/').
       Filename pattern: <number>_<name>.md (e.g., '1_Introduction.md')
Output: A JSON file in OUTPUT_DIR containing a list of chapter dictionaries.
        (e.g., 'stage1_chapter_data.json')
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

# --- Directory Paths ---
# TODO: Adjust these paths as needed for your environment
INPUT_MD_DIR = "1_chapters_md"  # Directory with source chapter .md files (e.g., 1_Introduction.md)
OUTPUT_DIR = "pipeline_output/stage1" # Directory to save the output JSON
OUTPUT_FILENAME = "stage1_chapter_data.json"
LOG_DIR = "pipeline_output/logs"

# --- Document ID ---
# TODO: Set this to uniquely identify the document being processed
DOCUMENT_ID = "EY_GUIDE_2024_PLACEHOLDER"

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

# --- Logging Setup ---
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
log_file = Path(LOG_DIR) / 'stage1_chapter_processing.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

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
            # logging.debug(f"tiktoken encode failed ('{str(e)[:50]}...'). Falling back to estimate.")
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
         # Allow proceeding without error, maybe system trusts the cert already
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
    # Check if SSL needs verification path
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
        # Continue anyway, maybe cert is already trusted or not needed

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
                # Default to JSON format if no tools specified
                completion_kwargs["response_format"] = {"type": "json_object"}
                logging.debug("Making API call with JSON response format...")

            response = client.chat.completions.create(**completion_kwargs)
            logging.debug("API call successful.")
            response_message = response.choices[0].message
            usage_info = response.usage

            # --- Tool Call Handling ---
            if response_message.tool_calls:
                tool_call = response_message.tool_calls[0]
                # Basic validation (can be enhanced)
                if tool_choice and isinstance(tool_choice, dict):
                    expected_tool_name = tool_choice.get("function", {}).get("name")
                    if expected_tool_name and tool_call.function.name != expected_tool_name:
                        raise ValueError(f"Expected tool '{expected_tool_name}' but received '{tool_call.function.name}'")
                function_args_json = tool_call.function.arguments
                return function_args_json, usage_info # Return JSON string from tool arguments
            # --- Standard Content Handling ---
            elif response_message.content:
                return response_message.content, usage_info # Return content string
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
            # Be slightly lenient for optional-like fields if needed, but log warning
            # Example: if 'tags' could sometimes be missing
            # if not ('tags' in missing_keys and len(missing_keys) == 1):
            #    raise ValueError(f"Missing expected keys: {', '.join(missing_keys)}")
            # else:
            #    logging.warning("Key 'tags' missing, proceeding.")
            #    data['tags'] = [] # Provide default
            raise ValueError(f"Missing expected keys in response: {', '.join(missing_keys)}")

        # Optional: Add type checks here if needed
        # if 'summary' in expected_keys and not isinstance(data.get('summary'), str): ...

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
    return name[:50].strip("_") # Truncate for safety

# --- Page Tag Extraction ---
# Regex to find page number tags like <!-- PageNumber="123" -->.
PAGE_NUMBER_TAG_PATTERN = re.compile(r'<!--\s*PageNumber="(\d+)"\s*-->')
# Regex to find and remove common Azure Document Intelligence tags and potential extra newline.
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

    # Sort primarily by position, secondarily by page number (desc)
    raw_matches.sort(key=lambda x: (x[0], -x[1]))

    if raw_matches:
        mapping.append(raw_matches[0])
        for i in range(1, len(raw_matches)):
            if raw_matches[i][0] > mapping[-1][0]:
                mapping.append(raw_matches[i])

    # Ensure mapping covers the end
    if mapping:
        last_entry_pos, last_entry_page = mapping[-1]
        # Check if the last tag is not already at the very end of the content
        if last_entry_pos < len(content):
            # Add an entry representing the end of the content with the last known page number
             mapping.append((len(content), last_entry_page))
        # If the last tag IS at the end, ensure its page number is used (already handled by sort)

    return mapping

def get_page_range_from_mapping(page_mapping: list[tuple[int, int]], last_processed_end_page: int) -> Tuple[int, int]:
    """Determines start and end page from mapping, using fallback."""
    if not page_mapping:
        # Infer start page based on previous chapter's end.
        start_page = last_processed_end_page + 1
        end_page = start_page  # Assume single page if no tags
        logging.warning(f"No page tags found. Inferring start page: {start_page}")
    else:
        # Use the page number from the first tag found.
        start_page = page_mapping[0][1]
        # Use the page number from the last tag found (mapping includes end marker).
        end_page = page_mapping[-1][1]
        # Ensure end page is not before start page
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
    # Try to find number at the beginning followed by _ or -
    match = re.match(r"(\d+)[_-](.+)", basename)
    if match:
        try:
            chapter_number = int(match.group(1))
            # Remove extension for the base name
            base_name = os.path.splitext(match.group(2))[0]
            # Replace underscores/hyphens with spaces for a cleaner title guess
            title_guess = base_name.replace("_", " ").replace("-", " ")
            return chapter_number, title_guess
        except ValueError:
            pass # Fall through if number conversion fails

    # Fallback: if no pattern match, return None for number and use whole filename (no ext)
    title_guess = os.path.splitext(basename)[0]
    return None, title_guess

# --- GPT Prompting for Chapter Details ---
# Tool schema definition (similar to script 7)
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
    # (Using the detailed prompt structure from 7_generate_chapter_details.py)
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
            client=client, # Added missing client argument
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

# Modified to accept model_name explicitly
def get_chapter_summary_and_tags(chapter_text: str, client: OpenAI, model_name: str = MODEL_NAME_CHAT) -> Tuple[Optional[str], Optional[List[str]]]:
    """Generates summary and tags for the chapter text, handling segmentation."""
    total_tokens = count_tokens(chapter_text)
    logging.info(f"Total estimated tokens for chapter: {total_tokens}")

    processing_limit = GPT_INPUT_TOKEN_LIMIT - MAX_COMPLETION_TOKENS_CHAPTER - TOKEN_BUFFER
    final_chapter_details = None

    if total_tokens <= processing_limit:
        logging.info("Processing chapter summary/tags in a single call.")
        # Pass model_name explicitly
        result = process_chapter_segment_for_details(
            chapter_text, client, model_name, MAX_COMPLETION_TOKENS_CHAPTER, TEMPERATURE, is_final_segment=True
        )
        if result:
            final_chapter_details = {'summary': result.get('summary'), 'tags': result.get('tags')}
        else:
            logging.error("Failed to process chapter in single call.")
    else:
        logging.info(f"Chapter exceeds token limit ({total_tokens} > {processing_limit}). Processing in segments.")
        # Basic segmentation by splitting text (can be improved with paragraph awareness)
        # Estimate number of segments needed
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
            # Pass model_name explicitly
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
                return None, None # Failed

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
# Main Stage 1 Logic
# ==============================================================================

def process_chapter_file(md_file_path: str, client: OpenAI, last_processed_end_page: int) -> Optional[Dict]:
    """Processes a single chapter markdown file."""
    file_basename = os.path.basename(md_file_path)
    logging.info(f"Processing Chapter File: {file_basename}")

    try:
        # 1. Extract chapter number and guess title from filename
        chapter_number, title_guess = extract_chapter_info_from_filename(file_basename)
        if chapter_number is None:
            logging.warning(f"Could not extract chapter number from filename '{file_basename}'. Skipping.")
            return None

        # 2. Read raw content
        with open(md_file_path, "r", encoding="utf-8") as f:
            raw_content = f.read()

        # 3. Extract chapter name (use first line if present, else fallback to title guess)
        first_line = raw_content.split('\n', 1)[0].strip()
        # Remove potential markdown heading characters
        chapter_name = re.sub(r"^\s*#+\s*", "", first_line).strip()
        if not chapter_name:
            chapter_name = title_guess # Use filename part if first line is empty/missing
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
            # Pass MODEL_NAME_CHAT explicitly
            chapter_summary, chapter_tags = get_chapter_summary_and_tags(raw_content, client, model_name=MODEL_NAME_CHAT)
            if chapter_summary is None or chapter_tags is None:
                 logging.warning(f"  Failed to generate summary/tags for chapter {chapter_number}.")
                 # Decide whether to proceed without summary/tags or fail
                 # For now, proceed with None values
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
            "chapter_summary": chapter_summary, # Can be None
            "chapter_tags": chapter_tags,       # Can be None
            "raw_content": raw_content,         # Pass raw content to next stage
            "_source_filename": file_basename,  # Keep track of original file
        }
        return chapter_data

    except Exception as e:
        logging.error(f"Error processing file {file_basename}: {e}", exc_info=True)
        return None


import sys # Added for sys.exit

def run_stage1():
    """Main function to execute Stage 1 processing with resumability."""
    logging.info("--- Starting Stage 1: Chapter Processing ---")
    create_directory(OUTPUT_DIR)
    output_filepath = Path(OUTPUT_DIR) / OUTPUT_FILENAME

    # --- Load Existing Data (for Resumability) ---
    existing_data = []
    processed_filenames = set()
    if output_filepath.exists():
        try:
            with open(output_filepath, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            if not isinstance(existing_data, list):
                 logging.warning(f"Existing output file {output_filepath} does not contain a valid list. Starting fresh.")
                 existing_data = []
            else:
                 # Use _source_filename for robust checking
                 processed_filenames = {chap.get("_source_filename") for chap in existing_data if chap.get("_source_filename")}
                 logging.info(f"Loaded {len(existing_data)} existing chapter records from {output_filepath}. Found {len(processed_filenames)} previously processed filenames.")
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from existing output file {output_filepath}. Starting fresh.", exc_info=True)
            existing_data = []
            processed_filenames = set()
        except Exception as e:
            logging.error(f"Error loading existing output file {output_filepath}: {e}. Starting fresh.", exc_info=True)
            existing_data = []
            processed_filenames = set()

    # --- Find and Sort Input Files ---
    input_path = Path(INPUT_MD_DIR)
    if not input_path.is_dir():
        logging.error(f"Input directory not found: {INPUT_MD_DIR}")
        # If existing data was loaded, still save it back? Or just exit?
        # Let's exit cleanly if input dir is missing.
        sys.exit(f"Error: Input directory '{INPUT_MD_DIR}' not found.")

    markdown_files = list(input_path.glob("*.md"))
    if not markdown_files:
        logging.warning(f"No markdown files found in {INPUT_MD_DIR}.")
        # Save existing data back if it was loaded, otherwise create empty file
        try:
            with open(output_filepath, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            logging.info(f"No new files to process. Saved {len(existing_data)} existing records back to {output_filepath}")
        except Exception as e:
            logging.error(f"Error saving output JSON to {output_filepath}: {e}", exc_info=True)
        return existing_data # Return existing data

    # Sort files naturally if possible
    if natsort:
        markdown_files = natsort.natsorted(markdown_files, key=lambda p: p.name)
        logging.info(f"Found and naturally sorted {len(markdown_files)} markdown files.")
    else:
        markdown_files.sort(key=lambda p: p.name)
        logging.info(f"Found {len(markdown_files)} markdown files (standard sort).")

    # --- Initialize OpenAI Client ---
    client = get_openai_client()
    if not client:
        logging.warning("OpenAI client initialization failed. Summary/Tag generation will be skipped for new chapters.")

    # --- Process Files ---
    # 'existing_data' will now accumulate results during the run
    processed_this_run_count = 0
    skipped_count = 0
    failed_this_run_count = 0
    # Determine the last page number from existing data to continue inference
    last_processed_end_page = 0
    if existing_data:
        # Sort existing data by chapter number to find the true last page
        try:
            sorted_existing = sorted(existing_data, key=lambda x: x.get('chapter_number', 0))
            if sorted_existing:
                last_processed_end_page = sorted_existing[-1].get("chapter_page_end", 0)
        except Exception as e:
             logging.warning(f"Could not reliably determine last page from existing data: {e}. Page inference might be inaccurate.")

    logging.info(f"Starting processing loop. Last known end page: {last_processed_end_page}")

    for md_file in tqdm(markdown_files, desc="Processing Chapters"):
        file_basename = md_file.name
        if file_basename in processed_filenames:
            logging.debug(f"Skipping already processed file: {file_basename}")
            skipped_count += 1
            continue # Skip this file

        logging.info(f"Processing new or previously failed file: {file_basename}")
        chapter_result = process_chapter_file(str(md_file), client, last_processed_end_page)

        if chapter_result:
            # --- Success Case: Append, Update State, and Save Incrementally ---
            existing_data.append(chapter_result)
            processed_filenames.add(file_basename) # Add to set to prevent reprocessing if script restarts mid-run after failure
            processed_this_run_count += 1
            # Update last_processed_end_page based on the *newly* processed chapter
            last_processed_end_page = chapter_result.get("chapter_page_end", last_processed_end_page)

            # --- Incremental Save ---
            try:
                # Re-sort before saving each time to maintain order in the file
                temp_sorted_data = existing_data # Default if natsort fails or isn't available
                if natsort:
                    try:
                        # Sort a copy to avoid modifying existing_data if sort fails mid-way
                        temp_sorted_data = sorted(existing_data, key=lambda x: x.get('chapter_number', float('inf')))
                    except Exception as sort_e:
                        logging.warning(f"Could not re-sort data before incremental save: {sort_e}. Saving in current order.")
                        temp_sorted_data = existing_data # Fallback to unsorted

                with open(output_filepath, "w", encoding="utf-8") as f:
                    json.dump(temp_sorted_data, f, indent=2, ensure_ascii=False)
                logging.debug(f"Incrementally saved {len(temp_sorted_data)} records after processing {file_basename}")
                # Update existing_data with the sorted version if sorting was successful
                existing_data = temp_sorted_data

            except Exception as e:
                logging.error(f"Error during incremental save to {output_filepath} after processing {file_basename}: {e}", exc_info=True)
                # Logged the error, but continue processing other files.
                # The successfully processed data is still in 'existing_data' in memory.

        else:
            # --- Failure Case ---
            failed_this_run_count += 1
            logging.warning(f"Processing failed for {file_basename}. It will be retried on the next run.")
            # Don't update last_processed_end_page on failure

    # --- Final Sort (Guarantee Order) ---
    # Although we sort incrementally, do a final sort for guaranteed order in the final file state.
    if natsort:
        try:
            existing_data = sorted(existing_data, key=lambda x: x.get('chapter_number', float('inf')))
            logging.info("Performed final sort of chapter data.")
            # Save the final sorted list one last time
            with open(output_filepath, "w", encoding="utf-8") as f:
                 json.dump(existing_data, f, indent=2, ensure_ascii=False)
            logging.info(f"Saved final sorted data with {len(existing_data)} records.")
        except Exception as final_sort_e:
            logging.warning(f"Could not perform final sort or save: {final_sort_e}. File may not be perfectly sorted.")
    else:
        logging.info("natsort not available, skipping final sort. Data order depends on processing sequence.")


    # --- Final Summary ---
    # Data is already saved incrementally. Just log the final state.
    final_record_count = 0
    if output_filepath.exists():
         try:
             with open(output_filepath, "r", encoding="utf-8") as f:
                 final_data_check = json.load(f)
                 if isinstance(final_data_check, list):
                     final_record_count = len(final_data_check)
         except Exception:
             logging.warning("Could not verify final record count from output file.")


    logging.info("--- Stage 1 Summary ---")
    logging.info(f"Input files found    : {len(markdown_files)}")
    logging.info(f"Skipped (already done): {skipped_count}")
    logging.info(f"Processed this run   : {processed_this_run_count}")
    logging.info(f"Failed this run      : {failed_this_run_count}")
    logging.info(f"Total records in file: {final_record_count} (approx. if verification failed)")
    logging.info(f"Output JSON file     : {output_filepath}")
    logging.info("--- Stage 1 Finished ---")

    return existing_data # Return the final in-memory list

# ==============================================================================
# Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    run_stage1()
