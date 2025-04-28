import os
import json
import requests
import smbclient
import logging
import time
from pathlib import Path
from datetime import datetime
from openai import AzureOpenAI, RateLimitError, APIError, AuthenticationError
from requests.exceptions import RequestException
import backoff # For retry logic

# ==============================================================================
# --- Configuration ---
# ==============================================================================
# NOTE: For production, externalize these settings (e.g., config file, env vars)

# --- NAS Configuration ---
# Network attached storage connection parameters (matching Stage 1 format)
NAS_PARAMS = {
    "ip": "your_nas_ip",             # Replace with actual NAS IP
    "share": "your_share_name",      # Replace with actual share name
    "user": "your_nas_user",         # Replace with actual SMB username (ensure consistency)
    "password": "your_nas_password"  # Replace with actual SMB password (ensure consistency)
}
# Base path on the NAS share where Stage 1/2 output files were stored (matching Stage 1/2)
NAS_OUTPUT_FOLDER_PATH = "path/to/your/output_folder"   # e.g., path/to/your/output_folder (relative to share root) - RENAMED FOR CONSISTENCY

# --- Processing Configuration ---
DOCUMENT_SOURCE = "internal_cheatsheets" # Matches the source identifier used in Stage 1/2 and DB (ensure this is correct for vision)

# --- Azure AD / OAuth Configuration for GPT API (Matching original structure) ---
OAUTH_CONFIG = {
    "token_url": "YOUR_OAUTH_TOKEN_ENDPOINT_URL", # Replace with actual Token Endpoint URL
    "client_id": "your_client_id",        # Replace with actual Client ID
    "client_secret": "your_client_secret",  # Replace with actual Client Secret
    "scope": "api://your_api_scope/.default" # Replace with actual API scope, e.g., api://your-guid/.default
}
# TOKEN_ENDPOINT is no longer needed as it's part of OAUTH_CONFIG


# --- GPT API Configuration (Matching original structure) ---
GPT_CONFIG = {
    "base_url": "https://your-aoai-endpoint.openai.azure.com/", # Replace with actual AOAI endpoint (Azure uses base_url)
    "api_version": "2024-02-01", # Or your desired API version
    # Model for Markdown Synthesis (can be different from summarization model)
    "markdown_synthesis_model": "gpt-4o", # Or your preferred model
    # Model for Usage/Description Summarization (using tool calling)
    "summarization_model": "gpt-4" # Or your preferred model
}

# --- CA Bundle Configuration ---
# Corrected to use NAS_OUTPUT_FOLDER_PATH
CA_BUNDLE_NAS_PATH = os.path.join(NAS_OUTPUT_FOLDER_PATH, "rbc-ca-bundle.cer") # Path on NAS
LOCAL_CA_BUNDLE_PATH = "rbc-ca-bundle.cer" # Local path to save the bundle

# --- System Prompt Template (Copied from catalog search - small) ---
# Using CO-STAR framework and XML tags, embedded directly
SYSTEM_PROMPT_TEMPLATE = """<CONTEXT>
<PROJECT_CONTEXT>
This project processes diverse documents (extracted from sources like '{document_source}') to create structured catalog entries for a database. These entries contain 'usage' and 'description' fields intended for an agentic RAG (Retrieval-Augmented Generation) system. The 'usage' field allows the AI agent to assess document relevance for retrieval, while the 'description' field provides a concise summary for human users browsing the catalog.
</PROJECT_CONTEXT>
</CONTEXT>

You are an expert technical writer specializing in analyzing documents and generating structured summaries optimized for both AI agent retrieval and human understanding.

<OBJECTIVE>
Your goal is to generate two distinct fields based *only* on the provided <DOCUMENT_CONTENT>:
1.  `usage`: A comprehensive, structured summary intended primarily for an AI agent. It must detail the document's core purpose, key topics, main arguments, important entities (people, places, organizations, concepts), relationships between entities, potential applications or use cases discussed, and any specific terminology, standards, or identifiers mentioned. This detail is crucial for enabling an AI agent to accurately assess the document's relevance to a query based *only* on this `usage` field in the catalog. The level of detail should align with the specified `detail_level`.
2.  `description`: A concise (1-2 sentence) human-readable summary suitable for displaying to end-users browsing the catalog. This should capture the absolute essence of the document.
</OBJECTIVE>

<STYLE>
Analytical, factual, objective, structured, and informative. Use clear and precise language. For the 'usage' field, consider using bullet points or structured text where appropriate to enhance readability for the AI agent, especially at higher detail levels.
</STYLE>

<TONE>
Neutral and professional.
</TONE>

<AUDIENCE>
- Primary: An AI retrieval agent (consuming the `usage` field).
- Secondary: End-users browsing a document catalog (reading the `description` field).
</AUDIENCE>

<TASK>
Analyze the provided document content and generate the `usage` and `description` fields according to the specifications.

<DOCUMENT_CONTENT>
{markdown_content}
</DOCUMENT_CONTENT>

<INSTRUCTIONS>
1.  Carefully read and analyze the entire <DOCUMENT_CONTENT>.
2.  Generate the `usage` string according to the <OBJECTIVE>, focusing on extracting information that aids agentic retrieval. Adapt the length and detail level based on the provided `detail_level`: '{detail_level}'.
    - 'concise': Provide a brief overview of key topics and purpose.
    - 'standard': Offer a balanced summary of topics, entities, and use cases.
    - 'detailed': Require an exhaustive analysis covering all aspects mentioned in the <OBJECTIVE> for `usage`.
3.  Generate the `description` string as a concise 1-2 sentence summary for humans, capturing the document's core essence. This field's length should *not* change based on `detail_level`.
4.  **CRITICAL:** Base both fields *exclusively* on information present within the <DOCUMENT_CONTENT>. Do not infer, add external knowledge, or hallucinate information not explicitly stated in the text.
5.  Format your response strictly as specified in <RESPONSE_FORMAT>. Do not include any preamble, conversational text, or explanations outside the required JSON structure.
</INSTRUCTIONS>
</TASK>

<RESPONSE_FORMAT>
You MUST call the `generate_catalog_fields` tool. Provide the generated `usage` and `description` strings as arguments within a JSON object.

Example JSON for the tool call arguments:
{{
  "usage": "Comprehensive, structured summary based on the document content and detail level...",
  "description": "Concise 1-2 sentence summary."
}}
</RESPONSE_FORMAT>
"""

# --- Tool Definition for Summarization GPT (Copied from catalog search - small) ---
# Renamed variable to match original, structure is the same
GPT_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "generate_catalog_fields", # Tool name matches original
        "description": "Generates the detailed 'usage' summary (for AI retrieval) and the concise 'description' summary (for humans) based on document content.",
        "parameters": {
            "type": "object",
            "properties": {
                "usage": { # Key matches original
                    "type": "string",
                    "description": "A comprehensive, structured summary detailing the document's purpose, topics, entities, relationships, and use cases, optimized for AI agent retrieval assessment. Detail level varies."
                },
                "description": { # Key matches original
                    "type": "string",
                    "description": "A very concise (1-2 sentence) human-readable summary of the document's essence for catalog display."
                }
            },
            "required": ["usage", "description"] # Required keys match original
        }
    }
}

# ==============================================================================
# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Variables ---
access_token = None
token_expiry_time = 0
gpt_client = None

# --- Helper Functions ---

def initialize_smb_client():
    """Sets up smbclient credentials using configured NAS_PARAMS."""
    try:
        smbclient.ClientConfig(username=NAS_PARAMS["user"], password=NAS_PARAMS["password"])
        logging.info("SMB client configured successfully.")
        return True
    except Exception as e:
        logging.error(f"Failed to configure SMB client: {e}")
        return False

def setup_ca_bundle():
    """Downloads CA bundle from NAS and sets environment variables."""
    try:
        if not os.path.exists(LOCAL_CA_BUNDLE_PATH):
            logging.info(f"Downloading CA bundle from {CA_BUNDLE_NAS_PATH}...")
            with smbclient.open_file(CA_BUNDLE_NAS_PATH, mode='rb') as nas_f, open(LOCAL_CA_BUNDLE_PATH, 'wb') as local_f:
                local_f.write(nas_f.read())
            logging.info(f"CA bundle saved to {LOCAL_CA_BUNDLE_PATH}")
        else:
            logging.info(f"Using existing local CA bundle: {LOCAL_CA_BUNDLE_PATH}")

        os.environ['REQUESTS_CA_BUNDLE'] = LOCAL_CA_BUNDLE_PATH
        os.environ['SSL_CERT_FILE'] = LOCAL_CA_BUNDLE_PATH
        logging.info("REQUESTS_CA_BUNDLE and SSL_CERT_FILE environment variables set.")
        return True
    except Exception as e:
        logging.error(f"Failed to setup CA bundle from {CA_BUNDLE_NAS_PATH}: {e}")
        return False

def get_access_token():
    """Gets a new OAuth2 access token using client credentials flow."""
    global access_token, token_expiry_time
    current_time = time.time()

    if access_token and current_time < token_expiry_time - 60: # Refresh 60s before expiry
        logging.debug("Using cached access token.")
        return access_token

    logging.info("Requesting new OAuth access token...")
    payload = {
        'client_id': OAUTH_CONFIG['client_id'],
        'client_secret': OAUTH_CONFIG['client_secret'],
        'scope': OAUTH_CONFIG['scope'],
        'grant_type': 'client_credentials'
    }
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}

    try:
        # Use token_url directly from OAUTH_CONFIG
        response = requests.post(OAUTH_CONFIG['token_url'], data=payload, headers=headers, verify=LOCAL_CA_BUNDLE_PATH)
        response.raise_for_status()
        token_data = response.json()
        access_token = token_data['access_token']
        # Calculate expiry time (expires_in is typically in seconds)
        token_expiry_time = current_time + token_data.get('expires_in', 3600) - 60 # Add buffer
        logging.info("Successfully obtained new access token.")
        return access_token
    except RequestException as e:
        logging.error(f"Failed to get access token: {e}")
        if hasattr(e, 'response') and e.response is not None:
             logging.error(f"Token endpoint response status: {e.response.status_code}")
             logging.error(f"Token endpoint response body: {e.response.text}")
        raise AuthenticationError("Failed to obtain OAuth token.") from e
    except KeyError:
        logging.error("Access token or expires_in not found in token response.")
        raise AuthenticationError("Invalid token response format.")

def initialize_gpt_client():
    """Initializes the AzureOpenAI client with the current access token."""
    global gpt_client
    try:
        token = get_access_token()
        # Use base_url for AzureOpenAI endpoint
        gpt_client = AzureOpenAI(
            base_url=GPT_CONFIG['base_url'],
            api_version=GPT_CONFIG['api_version'],
            azure_ad_token=token
        )
        logging.info("Azure OpenAI client initialized.")
    except AuthenticationError as e:
        logging.error(f"Authentication failed during client initialization: {e}")
        gpt_client = None # Ensure client is None if init fails
    except Exception as e:
        logging.error(f"Unexpected error initializing Azure OpenAI client: {e}")
        gpt_client = None

# Decorator for retry logic on specific OpenAI errors
def retry_on_openai_error(details):
    logging.warning(f"OpenAI API error: {details.get('exception')}. Retrying in {details.get('wait'):.1f} seconds...")

@backoff.on_exception(backoff.expo,
                      (RateLimitError, APIError),
                      max_tries=5,
                      on_backoff=retry_on_openai_error)
def call_gpt_markdown_synthesis(page_vision_data, page_number):
    """Calls GPT to synthesize Markdown from vision data for a single page."""
    if not gpt_client:
        initialize_gpt_client()
        if not gpt_client: # Still None after trying to initialize
             raise APIError("GPT client not initialized.", request=None) # Or a custom exception

    # Combine vision pass results into a structured input string
    input_text_parts = [f"Vision Model Analysis for Page {page_number}:\n"]
    for pass_name, result in page_vision_data.items():
        input_text_parts.append(f"--- {pass_name.upper().replace('_', ' ')} ---")
        input_text_parts.append(str(result)) # Ensure result is string
        input_text_parts.append("") # Add newline for separation

    input_text_parts.append("---")
    input_text_parts.append("Synthesize the above multi-pass vision model analysis into a single, coherent Markdown document representing this page's content. Preserve structure like tables and lists where possible. Focus on accurately representing the information conveyed visually and textually.")
    combined_input = "\n".join(input_text_parts)

    system_prompt = "You are an expert technical writer specializing in interpreting multi-modal analysis results. Your task is to synthesize vision model outputs describing an infographic page into a comprehensive and accurate Markdown representation of that page."

    logging.info(f"Sending request to GPT for Markdown synthesis (Page {page_number})...")
    try:
        completion = gpt_client.chat.completions.create(
            model=GPT_CONFIG['markdown_synthesis_model'], # Use model from GPT_CONFIG
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": combined_input}
            ],
            temperature=0.2, # Lower temperature for more factual synthesis
            max_tokens=3000 # Adjust as needed for expected page complexity
        )
        markdown_content = completion.choices[0].message.content
        logging.info(f"Received Markdown synthesis from GPT for Page {page_number}.")
        return markdown_content
    except (RateLimitError, APIError) as e:
        logging.error(f"OpenAI API error during Markdown synthesis (Page {page_number}): {e}")
        raise # Re-raise for backoff decorator
    except Exception as e:
        logging.error(f"Unexpected error during GPT Markdown synthesis (Page {page_number}): {e}")
        return f"Error: Failed to generate Markdown for page {page_number} - {str(e)}"

# --- Tool Definition for Summarization GPT (Copied from catalog search - small) ---
# (Definition moved before the function that uses it)
GPT_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "generate_catalog_fields", # Tool name matches original
        "description": "Generates the detailed 'usage' summary (for AI retrieval) and the concise 'description' summary (for humans) based on document content.",
        "parameters": {
            "type": "object",
            "properties": {
                "usage": { # Key matches original
                    "type": "string",
                    "description": "A comprehensive, structured summary detailing the document's purpose, topics, entities, relationships, and use cases, optimized for AI agent retrieval assessment. Detail level varies."
                },
                "description": { # Key matches original
                    "type": "string",
                    "description": "A very concise (1-2 sentence) human-readable summary of the document's essence for catalog display."
                }
            },
            "required": ["usage", "description"] # Required keys match original
        }
    }
}

@backoff.on_exception(backoff.expo,
                      (RateLimitError, APIError),
                      max_tries=5,
                      on_backoff=retry_on_openai_error)
def call_gpt_summarization(api_client, markdown_content, detail_level='standard', document_source='unknown'):
    """
    Calls the custom GPT model to generate summaries using tool calling.
    (Implementation copied and adapted from catalog search - small)

    Args:
        api_client: The initialized OpenAI client (renamed from gpt_client for clarity).
        markdown_content: The text content of the document to summarize.
        detail_level (str): The desired level of detail ('concise', 'standard', 'detailed').
        document_source (str): The source identifier for context in the prompt.

    Returns:
        tuple: (description, usage) strings, or (None, None) on failure.
    """
    # Ensure client is initialized (check passed api_client)
    if not api_client:
        logging.error("GPT client not provided to call_gpt_summarization.")
        # Attempt re-initialization as a fallback? Or just fail? Let's fail for now.
        # initialize_gpt_client() # This uses the global 'gpt_client'
        # if not gpt_client:
        #      raise APIError("GPT client not initialized.", request=None)
        # api_client = gpt_client # If re-init worked, use the global one
        return None, None # Fail if client wasn't passed in correctly

    logging.info(f"Calling GPT model for summarization (Detail Level: {detail_level}, Source: {document_source})...")
    try:
        # Format the system prompt with dynamic content (using the template)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            markdown_content=markdown_content,
            detail_level=detail_level,
            document_source=document_source
        )

        messages = [
            # System prompt now contains all instructions, context, and content
            {"role": "system", "content": system_prompt}
            # No separate user message needed as content is embedded in system prompt
        ]

        response = api_client.chat.completions.create(
            model=GPT_CONFIG['summarization_model'], # Use model from GPT_CONFIG
            messages=messages,
            tools=[GPT_TOOL_DEFINITION], # Use the copied tool definition
            tool_choice={"type": "function", "function": {"name": GPT_TOOL_DEFINITION['function']['name']}}, # Force tool use
            max_tokens=2048, # Match original script's setting
            temperature=0.2  # Match original script's setting
        )

        # --- Process Response ---
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            # Expecting only one tool call in this setup
            tool_call = tool_calls[0]
            # Check against the tool name
            if tool_call.function.name == GPT_TOOL_DEFINITION['function']['name']:
                try:
                    function_args = json.loads(tool_call.function.arguments)
                    # Extract using the correct keys
                    usage = function_args.get('usage')
                    description = function_args.get('description')

                    if description is not None and usage is not None: # Check for presence
                        logging.info("GPT successfully returned summaries via tool call.")
                        # Return in the order: description, usage (matching original script's return order)
                        return description, usage
                    else:
                        logging.error(f"GPT tool call arguments missing 'usage' or 'description'. Arguments: {function_args}")
                        return None, None
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse GPT tool call arguments JSON: {e}. Arguments: {tool_call.function.arguments}")
                    return None, None
                except Exception as e:
                    logging.error(f"Unexpected error processing GPT tool call arguments: {e}")
                    return None, None
            else:
                logging.error(f"GPT called unexpected tool: {tool_call.function.name}")
                return None, None
        else:
            # Handle case where the model didn't use the tool
            logging.error(f"GPT did not use the required tool. Response content: {response_message.content}")
            return None, None

    except (RateLimitError, APIError) as e: # Keep backoff decorator handling these
        logging.error(f"OpenAI API error during summarization: {e}")
        raise # Re-raise for backoff decorator
    except Exception as e:
        # Catch other potential API errors or issues
        logging.error(f"Failed to call GPT model for summarization: {type(e).__name__} - {e}")
        return None, None

# --- Main Processing Logic ---

# ==============================================================================
# --- Main Processing Logic ---
# ==============================================================================

def main():
    logging.info("--- Starting Stage 3: Generate Markdown and Summaries ---")
    logging.info(f"--- Document Source: {DOCUMENT_SOURCE} ---")

    if not initialize_smb_client():
        logging.error("Exiting due to SMB client initialization failure.")
        sys.exit(1)

    if not setup_ca_bundle():
        logging.error("Failed to set up CA bundle. Exiting.")
        return

    # Define paths using consistent variable name NAS_OUTPUT_FOLDER_PATH
    source_base_dir_relative = os.path.join(NAS_OUTPUT_FOLDER_PATH, DOCUMENT_SOURCE).replace('\\', '/')
    source_base_dir_smb = f"//{NAS_PARAMS['ip']}/{NAS_PARAMS['share']}/{source_base_dir_relative}"

    stage1_metadata_file = os.path.join(source_base_dir_smb, "1C_nas_files_to_process.json").replace('\\', '/')
    stage2_input_dir_smb = os.path.join(source_base_dir_smb, "2A_vision_outputs").replace('\\', '/') # Input dir from Stage 2
    stage3_md_output_dir_smb = os.path.join(source_base_dir_smb, "3C_generated_markdown").replace('\\', '/') # MD output dir for Stage 3
    stage3_catalog_output_file = os.path.join(source_base_dir_smb, "3A_catalog_entries.json").replace('\\', '/') # Catalog output file
    stage3_content_output_file = os.path.join(source_base_dir_smb, "3B_content_entries.json").replace('\\', '/') # Content output file
    skip_flag_path = os.path.join(source_base_dir_smb, "_SKIP_SUBSEQUENT_STAGES.flag").replace('\\', '/')
    refresh_flag_path = os.path.join(source_base_dir_smb, "_FULL_REFRESH.flag").replace('\\', '/')

    logging.info(f"Stage 1 Metadata File: {stage1_metadata_file}")
    logging.info(f"Stage 2 Input Dir (Vision Outputs): {stage2_input_dir_smb}")
    logging.info(f"Stage 3 Markdown Output Dir: {stage3_md_output_dir_smb}")
    logging.info(f"Stage 3 Catalog Output File: {stage3_catalog_output_file}")
    logging.info(f"Stage 3 Content Output File: {stage3_content_output_file}")

    # Check skip flag
    if smbclient.path.exists(skip_flag_path):
        logging.info(f"'{skip_flag_path}' found. Skipping Stage 3.")
        logging.info(f"'{skip_flag_path}' found. Skipping Stage 3.")
        return

    # Ensure Stage 3 Markdown output directory exists on NAS
    try:
        if not smbclient.path.exists(stage3_md_output_dir_smb): # Use correct variable
            smbclient.makedirs(stage3_md_output_dir_smb, exist_ok=True)
            logging.info(f"Created NAS directory: {stage3_md_output_dir_smb}")
    except Exception as e:
        logging.error(f"Failed to create NAS directory {stage3_md_output_dir_smb}: {e}")
        return

    # Handle Full Refresh
    is_full_refresh = smbclient.path.exists(refresh_flag_path)
    if is_full_refresh:
        logging.warning("'_FULL_REFRESH.flag' found. Deleting existing Stage 3 output files.")
        for f_path in [stage3_catalog_output_file, stage3_content_output_file]:
            try:
                if smbclient.path.exists(f_path):
                    smbclient.remove(f_path)
                    logging.info(f"Deleted existing file: {f_path}")
            except Exception as e:
                logging.error(f"Failed to delete {f_path} during full refresh: {e}")
        # Optionally delete existing generated markdown files too
        try:
            if smbclient.path.exists(stage3_md_output_dir_smb): # Use correct variable
                # This might be slow/complex with smbclient, consider carefully
                # For now, just log a warning if the directory exists during full refresh
                logging.warning(f"Full refresh mode: Existing markdown directory '{stage3_md_output_dir_smb}' found. Contents will be overwritten file-by-file, but old files not corresponding to current inputs will remain.")
        except Exception as e:
             logging.error(f"Error checking markdown dir {stage3_md_output_dir_smb}: {e}")


    # Load original file metadata
    try:
        with smbclient.open_file(stage1_metadata_file, mode='r', encoding='utf-8') as f:
            original_metadata_list = json.load(f)
        # Create a lookup dictionary by filename
        metadata_lookup = {Path(item['filename']).stem: item for item in original_metadata_list}
        logging.info(f"Loaded metadata for {len(metadata_lookup)} files from {stage1_metadata_file}")
    except Exception as e:
        logging.error(f"Failed to load or parse metadata file {stage1_metadata_file}: {e}")
        return

    # Load existing checkpoint data if not full refresh
    catalog_entries = []
    content_entries = []
    processed_stems = set() # Keep track of processed base filenames (stems)

    if not is_full_refresh:
        try:
            if smbclient.path.exists(stage3_catalog_output_file):
                with smbclient.open_file(stage3_catalog_output_file, mode='r', encoding='utf-8') as f:
                    catalog_entries = json.load(f)
                processed_stems.update(Path(entry.get('processed_md_path', '')).stem for entry in catalog_entries if entry.get('processed_md_path'))
                logging.info(f"Loaded {len(catalog_entries)} existing catalog entries for checkpointing.")
        except Exception as e:
            logging.warning(f"Could not load existing catalog entries from {stage3_catalog_output_file}: {e}. Starting fresh.")
            catalog_entries = []
            processed_stems = set()

        # Content entries are less critical for checkpointing based on processed_stems, but load if needed
        try:
            if smbclient.path.exists(stage3_content_output_file):
                 with smbclient.open_file(stage3_content_output_file, mode='r', encoding='utf-8') as f:
                     content_entries = json.load(f) # Load but don't necessarily use for checkpointing logic here
                 logging.info(f"Loaded {len(content_entries)} existing content entries.")
        except Exception as e:
             logging.warning(f"Could not load existing content entries from {stage3_content_output_file}: {e}.")
             content_entries = [] # Reset if loading fails


    # Find vision output *subdirectories* to process from Stage 2
    try:
        # List items in the Stage 2 output directory
        stage2_items = smbclient.listdir(stage2_input_dir_smb)
        # Filter for directories (these should correspond to file stems)
        stage2_subdirs = [
            item for item in stage2_items
            if smbclient.path.isdir(os.path.join(stage2_input_dir_smb, item).replace('\\', '/'))
        ]
        logging.info(f"Found {len(stage2_subdirs)} potential input subdirectories in {stage2_input_dir_smb}")
    except Exception as e:
        logging.error(f"Failed to list subdirectories in {stage2_input_dir_smb}: {e}")
        return

    # Initialize GPT client once before the loop if possible
    initialize_gpt_client()

    # Process each subdirectory (representing one original file)
    for file_stem in stage2_subdirs: # Iterate through directory names (stems)
        start_time = time.time()
        # Construct paths based on the subdirectory name (file_stem)
        json_filename = f"{file_stem}.json"
        json_filepath_nas = os.path.join(stage2_input_dir_smb, file_stem, json_filename).replace('\\', '/')
        markdown_filename = f"{file_stem}.md"
        markdown_filepath_nas = os.path.join(stage3_md_output_dir_smb, markdown_filename).replace('\\', '/') # Use correct MD output dir

        # Check if already processed (using stem)
        if file_stem in processed_stems:
            logging.info(f"Skipping '{file_stem}' as it appears to be already processed (based on stem).")
            continue

        logging.info(f"--- Processing vision output for stem: {file_stem} ---")
        logging.info(f"   Input JSON: {json_filepath_nas}")
        logging.info(f"   Output MD: {markdown_filepath_nas}")

        # Check if the expected JSON file exists within the subdirectory
        try:
            if not smbclient.path.exists(json_filepath_nas):
                logging.warning(f"   Expected JSON file not found at {json_filepath_nas}. Skipping stem '{file_stem}'.")
                continue
        except Exception as e:
             logging.error(f"   Error checking existence of {json_filepath_nas}: {e}. Skipping stem '{file_stem}'.")
             continue

        # 1. Load vision output data from the JSON file inside the subdirectory
        try:
            with smbclient.open_file(json_filepath_nas, mode='r', encoding='utf-8') as f:
                all_pages_vision_output = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load or parse {json_filepath_nas}: {e}")
            continue # Skip to next file stem

        # 2. Synthesize Markdown page by page
        final_markdown_parts = []
        synthesis_successful = True
        # Sort pages based on the key "page_N"
        sorted_page_keys = sorted(all_pages_vision_output.keys(), key=lambda x: int(x.split('_')[-1]))

        for page_key in sorted_page_keys:
            page_number = int(page_key.split('_')[-1])
            page_vision_data = all_pages_vision_output[page_key]
            try:
                page_markdown = call_gpt_markdown_synthesis(page_vision_data, page_number)
                if page_markdown.startswith("Error:"):
                    logging.error(f"Markdown synthesis failed for page {page_number} of stem '{file_stem}'. Aborting file.")
                    synthesis_successful = False
                    break # Stop processing this file
                final_markdown_parts.append(page_markdown)
            except Exception as e: # Catch errors from backoff failure etc.
                 logging.error(f"Critical error during Markdown synthesis for page {page_number} of stem '{file_stem}': {e}")
                 synthesis_successful = False
                 break # Stop processing this file

        if not synthesis_successful:
            continue # Skip to the next file stem

        # 3. Combine and Save Final Markdown
        final_markdown = f"\n\n---\n\n<!-- Page Separator -->\n\n---\n\n".join(final_markdown_parts)
        try:
            with smbclient.open_file(markdown_filepath_nas, mode='w', encoding='utf-8') as f:
                f.write(final_markdown)
            logging.info(f"Saved synthesized Markdown to {markdown_filepath_nas}")
        except Exception as e:
            logging.error(f"Failed to save final Markdown to {markdown_filepath_nas}: {e}")
            continue # Skip file if cannot save MD

        # 4. Generate Usage/Description Summaries using the refactored function
        try:
            # Define detail level (can be made dynamic later if needed)
            current_detail_level = 'standard'
            # Call with initialized client (gpt_client global), markdown, detail level, and source
            description_summary, usage_summary = call_gpt_summarization(
                gpt_client, final_markdown, current_detail_level, DOCUMENT_SOURCE
            )
            # Check if None was returned (indicates an error in call_gpt_summarizer)
            if description_summary is None or usage_summary is None:
                 logging.error(f"Summarization failed for stem '{file_stem}' (check logs from call_gpt_summarization). Skipping DB entry creation.")
                 continue # Skip to next file stem
        except Exception as e: # Catch errors from backoff failure etc.
             logging.error(f"Critical error during summarization call for stem '{file_stem}': {e}")
             continue # Skip to next file stem

        # 5. Prepare and Append DB Entries
        original_meta = metadata_lookup.get(file_stem) # Lookup using the stem
        if not original_meta:
            logging.warning(f"Could not find original metadata for stem '{file_stem}'. Skipping DB entry creation.")
            continue

        # Create Catalog Entry - Use data from original_meta and summaries
        catalog_entry = {
            "document_source": DOCUMENT_SOURCE,
            "document_type": original_meta.get("document_type", "infographic"), # Get type from meta or default
            "document_name": original_meta.get("file_name"), # Use 'file_name' from Stage 1 metadata
            "description": description_summary,
            "usage": usage_summary,
            # Use correct keys from Stage 1 metadata JSON
            "file_creation_date_utc": original_meta.get("date_created"),
            "file_last_modified_date_utc": original_meta.get("date_last_modified"),
            "file_size_bytes": original_meta.get("file_size"),
            "nas_link": None, # Construct this if needed, currently not in Stage 1 meta
            "nas_path": original_meta.get("file_path"), # Use 'file_path' from Stage 1 metadata
            "processed_md_path": markdown_filepath_nas # Link to the generated MD
        }
        catalog_entries.append(catalog_entry)

        # Create Content Entry - Use data from original_meta and summaries/markdown
        content_entry = {
            "document_source": DOCUMENT_SOURCE,
            "document_type": original_meta.get("document_type", "infographic"),
            "document_name": original_meta.get("file_name"), # Use 'file_name' from Stage 1 metadata
            "section_id": 0, # Use 0 for the whole document
            "section_name": file_stem, # Use base filename (stem) as section name
            "section_summary": usage_summary, # Use usage summary for section summary
            "content": final_markdown,
            "creation_timestamp": datetime.utcnow().isoformat() # Add timestamp
        }
        content_entries.append(content_entry)

        # 6. Save Checkpoint Files (after EACH successfully processed file stem)
        try:
            # Save Catalog Entries
            with smbclient.open_file(stage3_catalog_output_file, mode='w', encoding='utf-8') as f:
                json.dump(catalog_entries, f, indent=2)
            # Save Content Entries
            with smbclient.open_file(stage3_content_output_file, mode='w', encoding='utf-8') as f:
                json.dump(content_entries, f, indent=2)
            processing_time = time.time() - start_time
            logging.info(f"Successfully processed and checkpointed stem '{file_stem}' ({processing_time:.2f}s).")
        except Exception as e:
            logging.error(f"CRITICAL: Failed to save checkpoint files after processing stem '{file_stem}': {e}")
            # Decide how to handle - potentially stop the process?
            # For now, log and continue, but data might be lost on restart if script fails later.

    logging.info("--- Finished Stage 3 Processing Loop ---")


if __name__ == "__main__":
    main()
