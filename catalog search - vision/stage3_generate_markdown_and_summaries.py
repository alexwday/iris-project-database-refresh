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

# --- Configuration ---
# TODO: Externalize configuration
NAS_BASE_OUTPUT_FOLDER = "//nas_ip/share/path/to/your/output_folder"   # e.g., //192.168.1.100/share/processed
DOCUMENT_SOURCE = "infographics_source" # Matches the source identifier used in Stage 1/2 and DB
SMB_USER = os.getenv("SMB_USER", "your_smb_user")
SMB_PASSWORD = os.getenv("SMB_PASSWORD", "your_smb_password")

# Azure AD / OAuth Configuration for GPT API
TENANT_ID = os.getenv("AZURE_TENANT_ID", "your_tenant_id")
CLIENT_ID = os.getenv("AZURE_CLIENT_ID", "your_client_id")
CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET", "your_client_secret")
TOKEN_ENDPOINT = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
SCOPE = "api://your_api_scope/.default" # e.g., api://your-guid/.default

# GPT API Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-aoai-endpoint.openai.azure.com/")
API_VERSION = "2024-02-01" # Or your desired API version
# Model for Markdown Synthesis (can be different from summarization model)
MARKDOWN_SYNTHESIS_MODEL = os.getenv("MARKDOWN_SYNTHESIS_MODEL", "gpt-4o") # Or your preferred model
# Model for Usage/Description Summarization (using tool calling)
SUMMARIZATION_MODEL = os.getenv("SUMMARIZATION_MODEL", "gpt-4") # Or your preferred model

# CA Bundle Configuration
CA_BUNDLE_NAS_PATH = os.path.join(NAS_BASE_OUTPUT_FOLDER, "rbc-ca-bundle.cer") # Path on NAS
LOCAL_CA_BUNDLE_PATH = "rbc-ca-bundle.cer" # Local path to save the bundle

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Variables ---
access_token = None
token_expiry_time = 0
gpt_client = None

# --- Helper Functions ---

def register_smb_credentials():
    """Registers SMB credentials if provided."""
    if SMB_USER and SMB_PASSWORD:
        try:
            smbclient.ClientConfig(username=SMB_USER, password=SMB_PASSWORD)
            logging.info("SMB credentials registered.")
        except Exception as e:
            logging.error(f"Failed to register SMB credentials: {e}")
            raise e

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

# Define the tool for the summarization call
summarization_tool = {
    "type": "function",
    "function": {
        "name": "generate_catalog_fields",
        "description": "Generates structured usage and description fields for cataloging a document.",
        "parameters": {
            "type": "object",
            "properties": {
                "usage": {
                    "type": "string",
                    "description": "Detailed, structured summary for AI agent retrieval assessment. Describe the document's purpose, key topics, data types, intended audience, and potential use cases. Be specific and comprehensive."
                },
                "description": {
                    "type": "string",
                    "description": "Concise 1-2 sentence summary suitable for human users browsing a catalog."
                }
            },
            "required": ["usage", "description"]
        }
    }
}

@backoff.on_exception(backoff.expo,
                      (RateLimitError, APIError),
                      max_tries=5,
                      on_backoff=retry_on_openai_error)
def call_gpt_summarization(full_markdown_content, filename):
    """Calls GPT using tool calling to generate usage and description summaries."""
    if not gpt_client:
        initialize_gpt_client()
        if not gpt_client:
             raise APIError("GPT client not initialized.", request=None)

    system_prompt = "You are a technical writer tasked with creating catalog entries for documents. Analyze the provided Markdown content, which represents a processed infographic. Generate two fields: 'usage' (a detailed summary for AI agent assessment) and 'description' (a brief summary for humans)."
    user_prompt = f"Generate the catalog fields for the following document content derived from '{filename}':\n\n{full_markdown_content}"

    logging.info(f"Sending request to GPT for summarization ({filename})...")
    try:
        completion = gpt_client.chat.completions.create(
            model=SUMMARIZATION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            tools=[summarization_tool],
            tool_choice={"type": "function", "function": {"name": "generate_catalog_fields"}},
            temperature=0.5, # Slightly higher temp for summarization creativity
        )

        response_message = completion.choices[0].message
        if response_message.tool_calls and response_message.tool_calls[0].function.name == "generate_catalog_fields":
            tool_call = response_message.tool_calls[0]
            function_args = json.loads(tool_call.function.arguments)
            usage = function_args.get("usage")
            description = function_args.get("description")
            logging.info(f"Received summaries from GPT for {filename}.")
            return usage, description
        else:
            logging.error(f"GPT did not call the expected tool for {filename}. Response: {response_message}")
            return "Error: Tool not called", "Error: Tool not called"

    except (RateLimitError, APIError) as e:
        logging.error(f"OpenAI API error during summarization ({filename}): {e}")
        raise # Re-raise for backoff decorator
    except json.JSONDecodeError as e:
         logging.error(f"Failed to parse tool arguments from GPT response for {filename}: {e}")
         logging.error(f"Raw arguments: {tool_call.function.arguments}")
         return "Error: Invalid tool arguments", "Error: Invalid tool arguments"
    except Exception as e:
        logging.error(f"Unexpected error during GPT summarization ({filename}): {e}")
        return f"Error: Summarization failed - {str(e)}", f"Error: Summarization failed - {str(e)}"

# --- Main Processing Logic ---

def main():
    logging.info("--- Starting Stage 3: Generate Markdown and Summaries ---")
    register_smb_credentials()

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

        # 4. Generate Usage/Description Summaries
        try:
            usage_summary, description_summary = call_gpt_summarization(final_markdown, json_filename)
            if usage_summary.startswith("Error:") or description_summary.startswith("Error:"):
                 logging.error(f"Summarization failed for {json_filename}. Skipping DB entry creation.")
                 continue # Skip to next file
        except Exception as e:
             logging.error(f"Critical error during summarization for {json_filename}: {e}")
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
