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
    "user": "your_smb_user",         # Replace with actual SMB username
    "password": "your_smb_password"  # Replace with actual SMB password
}
# Base path on the NAS share where output JSON files will be stored
NAS_BASE_OUTPUT_FOLDER = "//nas_ip/share/path/to/your/output_folder"   # e.g., //192.168.1.100/share/processed

# --- Processing Configuration ---
DOCUMENT_SOURCE = "internal_cheatsheets" # Matches the source identifier used in Stage 1/2 and DB

# --- Azure AD / OAuth Configuration for GPT API ---
TENANT_ID = "your_tenant_id" # Replace with actual Tenant ID
CLIENT_ID = "your_client_id" # Replace with actual Client ID
CLIENT_SECRET = "your_client_secret" # Replace with actual Client Secret
TOKEN_ENDPOINT = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
SCOPE = "api://your_api_scope/.default" # Replace with actual API scope, e.g., api://your-guid/.default

# --- GPT API Configuration ---
AZURE_OPENAI_ENDPOINT = "https://your-aoai-endpoint.openai.azure.com/" # Replace with actual AOAI endpoint
API_VERSION = "2024-02-01" # Or your desired API version
# Model for Markdown Synthesis (can be different from summarization model)
MARKDOWN_SYNTHESIS_MODEL = "gpt-4o" # Or your preferred model
# Model for Usage/Description Summarization (using tool calling)
SUMMARIZATION_MODEL = "gpt-4" # Or your preferred model

# --- CA Bundle Configuration ---
CA_BUNDLE_NAS_PATH = os.path.join(NAS_BASE_OUTPUT_FOLDER, "rbc-ca-bundle.cer") # Path on NAS
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
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'scope': SCOPE,
        'grant_type': 'client_credentials'
    }
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}

    try:
        response = requests.post(TOKEN_ENDPOINT, data=payload, headers=headers, verify=LOCAL_CA_BUNDLE_PATH)
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
        gpt_client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=API_VERSION,
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
            model=MARKDOWN_SYNTHESIS_MODEL,
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
            model=SUMMARIZATION_MODEL, # Use configured model name
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

    source_output_dir_nas = os.path.join(NAS_BASE_OUTPUT_FOLDER, DOCUMENT_SOURCE)
    stage1_metadata_file = os.path.join(source_output_dir_nas, "1C_nas_files_to_process.json")
    stage2_input_dir_nas = os.path.join(source_output_dir_nas, "2A_vision_outputs")
    stage3_md_output_dir_nas = os.path.join(source_output_dir_nas, "3C_generated_markdown")
    stage3_catalog_output_file = os.path.join(source_output_dir_nas, "3A_catalog_entries.json")
    stage3_content_output_file = os.path.join(source_output_dir_nas, "3B_content_entries.json")
    skip_flag_path = os.path.join(source_output_dir_nas, "_SKIP_SUBSEQUENT_STAGES.flag")
    refresh_flag_path = os.path.join(source_output_dir_nas, "_FULL_REFRESH.flag")

    # Check skip flag
    if smbclient.path.exists(skip_flag_path):
        logging.info(f"'{skip_flag_path}' found. Skipping Stage 3.")
        return

    # Ensure Stage 3 Markdown output directory exists on NAS
    try:
        if not smbclient.path.exists(stage3_md_output_dir_nas):
            smbclient.makedirs(stage3_md_output_dir_nas, exist_ok=True)
            logging.info(f"Created NAS directory: {stage3_md_output_dir_nas}")
    except Exception as e:
        logging.error(f"Failed to create NAS directory {stage3_md_output_dir_nas}: {e}")
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
        # try:
        #     if smbclient.path.exists(stage3_md_output_dir_nas):
        #         # This might be slow/complex with smbclient, consider carefully
        #         logging.warning(f"Manual deletion of contents in {stage3_md_output_dir_nas} might be needed for full refresh.")
        # except Exception as e:
        #      logging.error(f"Error checking/deleting markdown dir {stage3_md_output_dir_nas}: {e}")


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


    # Find vision output JSON files to process
    try:
        vision_output_files = [f for f in smbclient.listdir(stage2_input_dir_nas) if f.endswith('.json')]
        logging.info(f"Found {len(vision_output_files)} JSON files in {stage2_input_dir_nas}")
    except Exception as e:
        logging.error(f"Failed to list files in {stage2_input_dir_nas}: {e}")
        return

    # Initialize GPT client once before the loop if possible
    initialize_gpt_client()

    # Process each vision output file
    for json_filename in vision_output_files:
        start_time = time.time()
        file_stem = Path(json_filename).stem
        json_filepath_nas = os.path.join(stage2_input_dir_nas, json_filename)
        markdown_filename = f"{file_stem}.md"
        markdown_filepath_nas = os.path.join(stage3_md_output_dir_nas, markdown_filename)

        # Check if already processed (using stem)
        if file_stem in processed_stems:
            logging.info(f"Skipping '{json_filename}' as it appears to be already processed (based on stem).")
            continue

        logging.info(f"--- Processing vision output: {json_filename} ---")

        # 1. Load vision output data
        try:
            with smbclient.open_file(json_filepath_nas, mode='r', encoding='utf-8') as f:
                all_pages_vision_output = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load or parse {json_filepath_nas}: {e}")
            continue # Skip to next file

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
                    logging.error(f"Markdown synthesis failed for page {page_number} of {json_filename}. Aborting file.")
                    synthesis_successful = False
                    break # Stop processing this file
                final_markdown_parts.append(page_markdown)
            except Exception as e: # Catch errors from backoff failure etc.
                 logging.error(f"Critical error during Markdown synthesis for page {page_number} of {json_filename}: {e}")
                 synthesis_successful = False
                 break # Stop processing this file

        if not synthesis_successful:
            continue # Skip to the next JSON file

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
                 logging.error(f"Summarization failed for {json_filename} (check logs from call_gpt_summarization). Skipping DB entry creation.")
                 continue # Skip to next file
        except Exception as e: # Catch errors from backoff failure etc.
             logging.error(f"Critical error during summarization call for {json_filename}: {e}")
             continue # Skip to next file

        # 5. Prepare and Append DB Entries
        original_meta = metadata_lookup.get(file_stem)
        if not original_meta:
            logging.warning(f"Could not find original metadata for {file_stem}. Skipping DB entry creation.")
            continue

        # Create Catalog Entry
        catalog_entry = {
            "document_source": DOCUMENT_SOURCE,
            "document_type": original_meta.get("document_type", "infographic"), # Get type from meta or default
            "document_name": original_meta.get("filename"),
            "description": description_summary,
            "usage": usage_summary,
            "file_creation_date_utc": original_meta.get("creation_time_utc"),
            "file_last_modified_date_utc": original_meta.get("modified_time_utc"),
            "file_size_bytes": original_meta.get("size"),
            "nas_link": original_meta.get("nas_link"), # Construct this if not present
            "nas_path": original_meta.get("nas_path"),
            "processed_md_path": markdown_filepath_nas # Link to the generated MD
        }
        catalog_entries.append(catalog_entry)

        # Create Content Entry
        content_entry = {
            "document_source": DOCUMENT_SOURCE,
            "document_type": original_meta.get("document_type", "infographic"),
            "document_name": original_meta.get("filename"),
            "section_id": 0, # Use 0 for the whole document
            "section_name": file_stem, # Use base filename as section name
            "section_summary": usage_summary, # Use usage summary for section summary
            "content": final_markdown,
            "creation_timestamp": datetime.utcnow().isoformat()
        }
        content_entries.append(content_entry)

        # 6. Save Checkpoint Files (after EACH file)
        try:
            with smbclient.open_file(stage3_catalog_output_file, mode='w', encoding='utf-8') as f:
                json.dump(catalog_entries, f, indent=2)
            with smbclient.open_file(stage3_content_output_file, mode='w', encoding='utf-8') as f:
                json.dump(content_entries, f, indent=2)
            processing_time = time.time() - start_time
            logging.info(f"Successfully processed and checkpointed '{json_filename}' ({processing_time:.2f}s).")
        except Exception as e:
            logging.error(f"CRITICAL: Failed to save checkpoint files after processing {json_filename}: {e}")
            # Decide how to handle - potentially stop the process?
            # For now, log and continue, but data might be lost on restart.

    logging.info("--- Finished Stage 3 ---")


if __name__ == "__main__":
    main()
