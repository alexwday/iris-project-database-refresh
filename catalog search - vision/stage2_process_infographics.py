import os
import json
import base64
import requests
import smbclient
import fitz  # PyMuPDF
from PIL import Image
import io
import logging
import time
from pathlib import Path
import tempfile
import sys # Added for sys.exit

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
    "user": "your_nas_user",         # Replace with actual SMB username (ensure consistency)
    "password": "your_nas_password"  # Replace with actual SMB password (ensure consistency)
}
# Base path on the NAS share where Stage 1 output files were stored (matching Stage 1)
NAS_OUTPUT_FOLDER_PATH = "path/to/your/output_folder"   # e.g., path/to/your/output_folder (relative to share root) - RENAMED FOR CONSISTENCY

# --- Processing Configuration ---
DOCUMENT_SOURCE = "internal_cheatsheets" # Matches the source identifier used in Stage 1 and DB (ensure this is correct for vision)

# --- Vision Model Configuration ---
VISION_API_URL = 'https://your-endpoint-url/v1/chat/completions' # Replace with your actual endpoint
VISION_MODEL_NAME = 'qwen-2-vl' # Adjust if needed
MAX_TOKENS = 1500 # Max tokens for vision model response
IMAGE_DPI = 150 # DPI for converting PDF pages to images

# --- Vision Prompts (CO-STAR Format) ---
VISION_PROMPTS = {
    "pass1_holistic": """<prompt>
<Context>You are analyzing a single page from a potentially multi-page infographic document. This is the first pass, aiming for a high-level understanding.</Context>
<Objective>Describe the overall topic, visual style (e.g., professional, playful, technical), and the primary message or conclusion this infographic page seems designed to convey. Identify the immediate takeaway.</Objective>
<Style>Concise, descriptive, and analytical.</Style>
<Tone>Objective and informative.</Tone>
<Audience>An analyst who needs a quick overview before detailed examination.</Audience>
<Response>Provide the description in a single paragraph.</Response>
</prompt>""",
    "pass2_text": """<prompt>
<Context>You are analyzing a single page from an infographic. This pass focuses *only* on extracting text content.</Context>
<Objective>Extract *all* readable text from this page verbatim. Include titles, headings, subheadings, paragraph text, list items, text within charts or diagrams, labels, captions, callouts, and any other visible text.</Objective>
<Style>Exact transcription. Preserve relative formatting like bullet points or numbered lists where possible using Markdown.</Style>
<Tone>Neutral and precise.</Tone>
<Audience>A system that will process this raw text.</Audience>
<Response>Output the extracted text. Use Markdown for basic formatting like lists if appropriate.</Response>
</prompt>""",
    "pass3_data_viz": """<prompt>
<Context>You are analyzing a single page from an infographic, focusing on quantitative information.</Context>
<Objective>Identify all charts (e.g., bar, line, pie), graphs, and data tables. For each, describe its type, what data it presents, key trends or values shown, and axis/legend labels. Additionally, extract any standalone statistics, percentages, or numerical figures presented elsewhere on the page.</Objective>
<Style>Detailed and structured. Clearly separate descriptions for each visual element or standalone figure.</Style>
<Tone>Analytical and factual.</Tone>
<Audience>An analyst needing to understand the data presented.</Audience>
<Response>Use bullet points or numbered lists to detail each identified data visualization or numerical figure and its associated description/values.</Response>
</prompt>""",
    "pass4_layout": """<prompt>
<Context>You are analyzing the design and organization of a single infographic page.</Context>
<Objective>Analyze the layout and visual organization. Describe how information is segmented (e.g., sections, columns, blocks), the reading path suggested by the design, and how visual elements (lines, arrows, color, size, placement) are used to group related information, create emphasis, or guide the viewer's eye.</Objective>
<Style>Descriptive and interpretive, focusing on design principles.</Style>
<Tone>Observational and analytical.</Tone>
<Audience>A designer or analyst studying the infographic's structure.</Audience>
<Response>Provide the analysis in a structured paragraph or bullet points, highlighting key structural features and flow elements.</Response>
</prompt>""",
    "pass5_diagrams_symbols": """<prompt>
<Context>You are analyzing explanatory diagrams and symbolic visuals on a single infographic page.</Context>
<Objective>Identify and describe any process flows, timelines, organizational charts, mind maps, or other explanatory diagrams. Detail their components, connections, and the process/structure depicted. Also, identify key icons, illustrations, or symbolic graphics (excluding data charts), describing what they represent and their likely purpose or contribution to the message.</Objective>
<Style>Detailed and descriptive. Clearly distinguish between different types of diagrams or symbols.</Style>
<Tone>Informative and interpretive.</Tone>
<Audience>An analyst needing to understand non-data visual explanations.</Audience>
<Response>Use bullet points or numbered lists to detail each identified diagram or symbolic visual and its description/purpose.</Response>
</prompt>""",
    "pass6_context_attribution": """<prompt>
<Context>You are extracting metadata and source information from a single infographic page.</Context>
<Objective>Extract information typically found in headers, footers, margins, or source citations. This includes document titles, page numbers, author/organization, logos, publication dates, data sources, footnotes, disclaimers, or contact information.</Objective>
<Style>Factual extraction.</Style>
<Tone>Neutral and precise.</Tone>
<Audience>A system or archivist needing provenance information.</Audience>
<Response>List each piece of extracted information clearly, labeling what it is (e.g., "Source:", "Date:", "Page Number:").</Response>
</prompt>"""
}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout) # Log to stdout

# ==============================================================================
# --- Helper Functions (Adopted from small/stage2 for consistency) ---
# ==============================================================================

def initialize_smb_client():
    """Sets up smbclient credentials using configured NAS_PARAMS."""
    try:
        smbclient.ClientConfig(username=NAS_PARAMS["user"], password=NAS_PARAMS["password"])
        logging.info("SMB client configured successfully.")
        return True
    except Exception as e:
        logging.error(f"[ERROR] Failed to configure SMB client: {e}")
        return False

def create_nas_directory(smb_dir_path):
    """Creates a directory on the NAS if it doesn't exist."""
    try:
        # Construct the full SMB path format if not already provided
        if not smb_dir_path.startswith(f"//{NAS_PARAMS['ip']}/{NAS_PARAMS['share']}/"):
             relative_path = smb_dir_path.replace(os.sep, '/') # Ensure forward slashes
             smb_full_path = f"//{NAS_PARAMS['ip']}/{NAS_PARAMS['share']}/{relative_path}"
        else:
             smb_full_path = smb_dir_path

        if not smbclient.path.exists(smb_full_path):
            logging.info(f"   Creating NAS directory: {smb_full_path}")
            smbclient.makedirs(smb_full_path, exist_ok=True)
            logging.info(f"   Successfully created directory.")
        else:
            # logging.info(f"   NAS directory already exists: {smb_full_path}") # Optional: reduce verbosity
            pass
        return True
    except Exception as e: # General exception will catch SMB errors too
        logging.error(f"   [ERROR] Unexpected error creating/accessing NAS directory '{smb_full_path}': {e}")
        return False

def write_to_nas(smb_path, content_bytes):
    """Writes bytes to a file path on the NAS using smbclient."""
    logging.info(f"   Attempting to write to NAS path: {smb_path}")
    try:
        # Ensure the directory exists first
        dir_path = os.path.dirname(smb_path)
        if not create_nas_directory(dir_path): # Use the helper to create dir
            return False # Failed to create directory

        with smbclient.open_file(smb_path, mode='wb') as f: # Write in binary mode
            f.write(content_bytes)
        logging.info(f"   Successfully wrote {len(content_bytes)} bytes to: {smb_path}")
        return True
    except Exception as e: # General exception will catch SMB errors too
        logging.error(f"   [ERROR] Unexpected error writing to NAS '{smb_path}': {e}")
        return False

def download_from_nas(smb_path, local_temp_dir):
    """Downloads a file from NAS to a local temporary directory."""
    local_file_path = os.path.join(local_temp_dir, os.path.basename(smb_path))
    logging.info(f"   Attempting to download from NAS: {smb_path} to {local_file_path}")
    try:
        with smbclient.open_file(smb_path, mode='rb') as nas_f:
            with open(local_file_path, 'wb') as local_f:
                local_f.write(nas_f.read())
        logging.info(f"   Successfully downloaded to: {local_file_path}")
        return local_file_path
    except Exception as e: # General exception will catch SMB errors too
        logging.error(f"   [ERROR] Unexpected error downloading from NAS '{smb_path}': {e}")
        return None

# --- Vision Specific Helpers ---

def encode_image(image_path):
    """Encodes an image file to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"Error encoding image {image_path}: {e}")
        return None

def call_vision_api(image_path, prompt_text):
    """Calls the local vision model API for a given image and prompt."""
    base64_image = encode_image(image_path)
    if not base64_image:
        return None

    headers = {'Content-Type': 'application/json'}
    payload = {
        'model': VISION_MODEL_NAME,
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt_text},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{base64_image}'}}
                ]
            }
        ],
        'max_tokens': MAX_TOKENS
    }

    logging.info(f"Sending request to Vision API for {Path(image_path).name} with prompt type: {prompt_text[:50]}...") # Log prompt start
    try:
        response = requests.post(VISION_API_URL, headers=headers, json=payload, timeout=120) # Added timeout
        response.raise_for_status()
        result = response.json()
        response_text = result.get('choices', [{}])[0].get('message', {}).get('content', 'Error: No content found')
        logging.info(f"Received response from Vision API for {Path(image_path).name}.")
        return response_text
    except requests.exceptions.Timeout:
        logging.error(f"Timeout error calling Vision API for {image_path}.")
        return "Error: API call timed out."
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling Vision API for {image_path}: {e}")
        # Log response body if possible and useful
        if hasattr(e, 'response') and e.response is not None:
             logging.error(f"Response status: {e.response.status_code}")
             logging.error(f"Response body: {e.response.text}")
        return f"Error: API call failed - {str(e)}"
    except (KeyError, IndexError) as e:
        logging.error(f"Unexpected response format from Vision API for {image_path}: {e}")
        logging.error(f"Raw response: {json.dumps(result, indent=2)}")
        return "Error: Unexpected API response format."
    except Exception as e:
        logging.error(f"An unexpected error occurred during Vision API call for {image_path}: {e}")
        return "Error: An unexpected error occurred."


# --- Main Processing Logic ---

# ==============================================================================
# --- Main Processing Logic ---
# ==============================================================================

def main():
    logging.info("\n" + "="*60)
    logging.info(f"--- Running Stage 2: Process Infographics with Vision Model ---")
    logging.info(f"--- Document Source: {DOCUMENT_SOURCE} ---")
    logging.info("="*60 + "\n")

    # --- Initialize Clients ---
    logging.info("[1] Initializing SMB Client...")
    if not initialize_smb_client():
        sys.exit(1) # Exit if SMB client fails
    logging.info("-" * 60)

    # --- Define Paths (Following small/stage2 pattern) ---
    logging.info("[2] Defining NAS Paths...")
    # Base output directory from Stage 1
    stage1_output_dir_relative = os.path.join(NAS_OUTPUT_FOLDER_PATH, DOCUMENT_SOURCE).replace('\\', '/')
    stage1_output_dir_smb = f"//{NAS_PARAMS['ip']}/{NAS_PARAMS['share']}/{stage1_output_dir_relative}"
    # Input JSON file from Stage 1
    files_to_process_json_smb = os.path.join(stage1_output_dir_smb, '1C_nas_files_to_process.json').replace('\\', '/')
    # Base output directory for Stage 2 results
    stage2_output_dir_relative = os.path.join(stage1_output_dir_relative, '2A_vision_outputs').replace('\\', '/') # Changed folder name
    stage2_output_dir_smb = f"//{NAS_PARAMS['ip']}/{NAS_PARAMS['share']}/{stage2_output_dir_relative}"

    logging.info(f"   Stage 1 Output Dir (SMB): {stage1_output_dir_smb}")
    logging.info(f"   Input JSON File (SMB): {files_to_process_json_smb}")
    logging.info(f"   Stage 2 Output Base Dir (SMB): {stage2_output_dir_smb}")

    # Ensure base Stage 2 output directory exists
    if not create_nas_directory(stage2_output_dir_smb): # Use helper
        logging.error("[CRITICAL ERROR] Could not create base Stage 2 output directory on NAS. Exiting.")
        sys.exit(1)
    logging.info("-" * 60)

    # --- Check for Skip Flag from Stage 1 ---
    logging.info("[3] Checking for skip flag from Stage 1...")
    skip_flag_file_name = '_SKIP_SUBSEQUENT_STAGES.flag'
    skip_flag_smb_path = os.path.join(stage1_output_dir_smb, skip_flag_file_name).replace('\\', '/')
    logging.info(f"   Checking for flag file: {skip_flag_smb_path}")
    should_skip = False
    try:
        # Ensure SMB client is configured (should be from step [1])
        if smbclient.path.exists(skip_flag_smb_path):
            logging.info(f"   Skip flag file found. Stage 1 indicated no files to process.")
            should_skip = True
        else:
            logging.info(f"   Skip flag file not found. Proceeding with Stage 2.")
    except Exception as e:
        logging.error(f"   [WARNING] Unexpected error checking for skip flag file '{skip_flag_smb_path}': {e}")
        logging.warning(f"   Proceeding with Stage 2 assuming no skip flag.")
        # Continue execution if flag check fails, maybe log warning
    logging.info("-" * 60)

    if should_skip:
        logging.info("\n" + "="*60)
        logging.info(f"--- Stage 2 Skipped (Skip flag found) ---")
        logging.info("="*60 + "\n")
        return # Exit main function

    # --- Load Files to Process List ---
    logging.info(f"[4] Loading list of files to process from: {os.path.basename(files_to_process_json_smb)}...")
    files_to_process = []
    try:
        # Read the JSON file content from NAS
        with smbclient.open_file(files_to_process_json_smb, mode='r', encoding='utf-8') as f:
            files_to_process = json.load(f)
        logging.info(f"   Successfully loaded {len(files_to_process)} file entries.")
        if not files_to_process:
             logging.info("   List is empty. No files to process in Stage 2.")
             logging.info("\n" + "="*60)
             logging.info(f"--- Stage 2 Completed (No files to process) ---")
             logging.info("="*60 + "\n")
             # Consider creating skip flag here if Stage 1 didn't? (Stage 1 should handle this)
             return # Exit this function early
    except json.JSONDecodeError as e:
        logging.error(f"   [CRITICAL ERROR] Failed to parse JSON from '{files_to_process_json_smb}': {e}")
        sys.exit(1) # Exit script if critical file cannot be read/parsed
    except smbclient.SambaClientError as e:
        if "NT_STATUS_NO_SUCH_FILE" in str(e):
             logging.error(f"   [CRITICAL ERROR] Input file not found: {files_to_process_json_smb}. Make sure Stage 1 ran successfully.")
        else:
             logging.error(f"   [CRITICAL ERROR] Failed to read input file {files_to_process_json_smb} from NAS: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"   [CRITICAL ERROR] An unexpected error occurred loading {files_to_process_json_smb}: {e}")
        sys.exit(1)
    logging.info("-" * 60)


    # --- Process Each File ---
    logging.info(f"[5] Processing {len(files_to_process)} files...")
    processed_count = 0
    error_count = 0

    # Create a single temporary directory for all downloads/chunks
    with tempfile.TemporaryDirectory() as local_temp_dir:
        logging.info(f"   Using temporary directory: {local_temp_dir}")

        for i, file_info in enumerate(files_to_process):
            start_time = time.time()
            file_has_error = False
            local_pdf_path = None # Ensure defined in outer scope for cleanup
            logging.info(f"\n--- Processing file {i+1}/{len(files_to_process)} ---")
            try:
                # Use .get for safer dictionary access
                file_name = file_info.get('file_name')
                file_path_relative = file_info.get('file_path') # Path relative to share root from Stage 1
                # Note: Stage 1 JSON uses 'file_path', not 'nas_path'. Adjusting keys.
                # nas_file_path = file_info.get("nas_path") # Old key
                # base_filename = file_info.get("filename") # Old key

                if not file_name or not file_path_relative:
                    logging.warning(f"   [ERROR] Skipping entry due to missing 'file_name' or 'file_path': {file_info}")
                    error_count += 1
                    continue

                logging.info(f"   File Name: {file_name}")
                logging.info(f"   Relative NAS Path: {file_path_relative}")

                # Construct full SMB path for the input file
                input_file_smb_path = f"//{NAS_PARAMS['ip']}/{NAS_PARAMS['share']}/{file_path_relative}"

                # Construct output paths (with per-file subfolder)
                file_name_base = Path(file_name).stem # Use pathlib for robust stem extraction
                file_output_subfolder_relative = os.path.join(stage2_output_dir_relative, file_name_base).replace('\\', '/')
                file_output_subfolder_smb = f"//{NAS_PARAMS['ip']}/{NAS_PARAMS['share']}/{file_output_subfolder_relative}"
                # Output JSON will be inside the subfolder
                output_json_filename = f"{file_name_base}.json"
                output_json_smb_path = os.path.join(file_output_subfolder_smb, output_json_filename).replace('\\', '/')

                logging.info(f"   Output Subfolder (SMB): {file_output_subfolder_smb}")
                logging.info(f"   Output JSON Path (SMB): {output_json_smb_path}")

                # Ensure the specific output subfolder exists for this file
                if not create_nas_directory(file_output_subfolder_smb): # Use helper
                    logging.error(f"   [ERROR] Failed to create output subfolder for {file_name}. Skipping.")
                    error_count += 1
                    continue

                # Check if output already exists (for potential resume capability)
                # TODO: Add more robust checkpointing if needed (e.g., check timestamp or content hash)
                if smbclient.path.exists(output_json_smb_path):
                    logging.info(f"   Output file {output_json_smb_path} already exists. Skipping.")
                    continue # Skip to next file

                # 1. Download PDF from NAS using helper
                local_pdf_path = download_from_nas(input_file_smb_path, local_temp_dir)
                if not local_pdf_path:
                    logging.error(f"   [ERROR] Failed to download {file_name} from NAS. Skipping.")
                    error_count += 1
                    continue # Skip to next file

                all_pages_vision_output = {}
                pdf_processing_error = False

                # 2. Process PDF: Split, Convert, Analyze
                try:
                    pdf_document = fitz.open(local_pdf_path)
                    num_pages = len(pdf_document)
                    logging.info(f"   Opened PDF '{file_name}' with {num_pages} pages.")

                    for page_num in range(num_pages):
                        page_index = page_num + 1 # Use 1-based index for logging/keys
                        logging.info(f"   Processing Page {page_index}/{num_pages}...")
                        page_start_time = time.time()
                        page = pdf_document.load_page(page_num)
                        local_page_jpeg_path = None # Ensure defined for cleanup

                        try:
                            # Convert page to JPEG using configured DPI
                            pix = page.get_pixmap(dpi=IMAGE_DPI)
                            img_bytes = pix.tobytes("jpeg")
                            local_page_jpeg_path = os.path.join(local_temp_dir, f"{file_name_base}_page_{page_index}.jpg")

                            try:
                                with open(local_page_jpeg_path, "wb") as img_file:
                                    img_file.write(img_bytes)
                            except Exception as e:
                                 logging.error(f"   [ERROR] Failed to save page {page_index} as JPEG: {e}")
                                 # Decide how to handle: skip page or fail file?
                                 pdf_processing_error = True # Mark error for the file
                                 break # Stop processing pages for this file

                            # Run multi-pass vision analysis for the page
                            page_results = {}
                            api_call_failed = False
                            for pass_name, prompt in VISION_PROMPTS.items():
                                logging.info(f"      Running {pass_name} for page {page_index}...")
                                vision_response = call_vision_api(local_page_jpeg_path, prompt)
                                if vision_response is None or "Error:" in vision_response:
                                     logging.error(f"      [ERROR] Vision API call failed for {pass_name} on page {page_index}. Response: {vision_response}")
                                     page_results[pass_name] = vision_response if vision_response is not None else "Error: Failed to get response"
                                     api_call_failed = True
                                     # Decide: break inner loop (passes) or continue? Let's continue for now.
                                else:
                                     page_results[pass_name] = vision_response
                                # Optional: Add a small delay between API calls if needed
                                # time.sleep(0.5)

                            all_pages_vision_output[f"page_{page_index}"] = page_results
                            page_end_time = time.time()
                            logging.info(f"   Finished Page {page_index} ({page_end_time - page_start_time:.2f}s). API errors: {api_call_failed}")

                            if api_call_failed:
                                # Decide if one failed API call should fail the whole file
                                # pdf_processing_error = True
                                # break # Option: Stop processing pages if one API call fails
                                pass # Continue processing other pages for now

                        finally:
                            # Clean up temporary page image inside the page loop
                            if local_page_jpeg_path and os.path.exists(local_page_jpeg_path):
                                try:
                                    os.remove(local_page_jpeg_path)
                                except OSError as e:
                                    logging.warning(f"   [WARNING] Could not remove temporary page file {local_page_jpeg_path}: {e}")

                    pdf_document.close()

                except Exception as e:
                    logging.error(f"   [ERROR] Failed to process PDF {local_pdf_path}: {e}")
                    pdf_processing_error = True # Mark error for the file

                if pdf_processing_error:
                    file_has_error = True # Mark the file as having an error
                    # No results to save if PDF processing failed critically
                else:
                    # 3. Save results to NAS using helper
                    if all_pages_vision_output:
                        logging.info(f"   Saving vision analysis JSON output to NAS...")
                        try:
                            output_data_str = json.dumps(all_pages_vision_output, indent=2)
                            output_data_bytes = output_data_str.encode('utf-8')
                            if not write_to_nas(output_json_smb_path, output_data_bytes): # Use helper
                                logging.error(f"   [ERROR] Failed to write JSON file for {file_name} to NAS.")
                                file_has_error = True # Mark error even if vision analysis worked
                        except Exception as e:
                            logging.error(f"   [ERROR] Failed to serialize or save results JSON to {output_json_smb_path}: {e}")
                            file_has_error = True
                    else:
                         logging.warning(f"   No vision output generated for {file_name}. Nothing to save.")
                         # This might be an error condition depending on expectations
                         # file_has_error = True


            except Exception as e:
                logging.error(f"   [ERROR] Unexpected error processing file {file_info.get('file_name', 'N/A')}: {e}")
                file_has_error = True
            finally:
                # --- Cleanup ---
                if local_pdf_path and os.path.exists(local_pdf_path):
                    try:
                        os.remove(local_pdf_path)
                        # logging.info(f"   Cleaned up temporary PDF: {local_pdf_path}") # Optional verbosity
                    except OSError as e:
                        logging.warning(f"   [WARNING] Failed to remove temporary PDF file {local_pdf_path}: {e}")

                # --- Update Counters ---
                if file_has_error:
                    error_count += 1
                    logging.info(f"--- Finished file {i+1} (ERROR) ---")
                else:
                    processed_count += 1
                    logging.info(f"--- Finished file {i+1} (Success) ---")
                end_time = time.time()
                logging.info(f"--- Time taken: {end_time - start_time:.2f} seconds ---")


    # --- Final Summary ---
    logging.info("\n" + "="*60)
    logging.info(f"--- Stage 2 Processing Summary ---")
    logging.info(f"   Total files attempted: {len(files_to_process)}")
    logging.info(f"   Successfully processed: {processed_count}")
    logging.info(f"   Errors encountered: {error_count}")
    logging.info("="*60 + "\n")

    if error_count > 0:
        logging.warning(f"[WARNING] {error_count} files encountered errors during processing. Check logs above.")
        # Optionally exit with error code if any file failed
        # sys.exit(1)

    logging.info(f"--- Stage 2 Completed ---")


if __name__ == "__main__":
    main()
