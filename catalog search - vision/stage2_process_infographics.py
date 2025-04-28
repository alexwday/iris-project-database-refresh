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

# --- Configuration ---
# TODO: Externalize configuration
NAS_BASE_INPUT_FOLDER = "//nas_ip/share/path/to/your/base_input_folder" # e.g., //192.168.1.100/share/documents
NAS_BASE_OUTPUT_FOLDER = "//nas_ip/share/path/to/your/output_folder"   # e.g., //192.168.1.100/share/processed
DOCUMENT_SOURCE = "infographics_source" # Matches the source identifier used in Stage 1 and DB
SMB_USER = os.getenv("SMB_USER", "your_smb_user")
SMB_PASSWORD = os.getenv("SMB_PASSWORD", "your_smb_password")

VISION_API_URL = 'https://your-endpoint-url/v1/chat/completions' # Replace with your actual endpoint
VISION_MODEL_NAME = 'qwen-2-vl' # Adjust if needed
MAX_TOKENS = 1500 # Max tokens for vision model response

# Define the 6 CO-STAR prompts using XML structure
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def register_smb_credentials():
    """Registers SMB credentials if provided."""
    if SMB_USER and SMB_PASSWORD:
        try:
            smbclient.ClientConfig(username=SMB_USER, password=SMB_PASSWORD)
            logging.info("SMB credentials registered.")
        except Exception as e:
            logging.error(f"Failed to register SMB credentials: {e}")
            # Decide if this is fatal or if anonymous access might work
            # raise e # Uncomment to make it fatal

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

def main():
    logging.info("--- Starting Stage 2: Process Infographics ---")
    register_smb_credentials()

    source_output_dir_nas = os.path.join(NAS_BASE_OUTPUT_FOLDER, DOCUMENT_SOURCE)
    stage1_output_file = os.path.join(source_output_dir_nas, "1C_nas_files_to_process.json")
    stage2_output_dir_nas = os.path.join(source_output_dir_nas, "2A_vision_outputs")
    skip_flag_path = os.path.join(source_output_dir_nas, "_SKIP_SUBSEQUENT_STAGES.flag")

    # Check skip flag
    if smbclient.path.exists(skip_flag_path):
        logging.info(f"'{skip_flag_path}' found. Skipping Stage 2.")
        return

    # Ensure Stage 2 output directory exists on NAS
    try:
        if not smbclient.path.exists(stage2_output_dir_nas):
            smbclient.makedirs(stage2_output_dir_nas, exist_ok=True)
            logging.info(f"Created NAS directory: {stage2_output_dir_nas}")
    except Exception as e:
        logging.error(f"Failed to create NAS directory {stage2_output_dir_nas}: {e}")
        return # Cannot proceed without output directory

    # Load list of files to process
    try:
        with smbclient.open_file(stage1_output_file, mode='r', encoding='utf-8') as f:
            files_to_process = json.load(f)
        logging.info(f"Loaded {len(files_to_process)} files to process from {stage1_output_file}")
    except smbclient.SambaClientError as e:
        if "NT_STATUS_NO_SUCH_FILE" in str(e):
             logging.error(f"Input file not found: {stage1_output_file}. Make sure Stage 1 ran successfully.")
        else:
             logging.error(f"Failed to read input file {stage1_output_file} from NAS: {e}")
        return
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON from {stage1_output_file}: {e}")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred loading {stage1_output_file}: {e}")
        return

    if not files_to_process:
        logging.info("No files listed in '1C_nas_files_to_process.json'. Stage 2 complete.")
        # Consider creating the skip flag here if it wasn't created by Stage 1
        # try:
        #     with smbclient.open_file(skip_flag_path, mode='w') as f:
        #         f.write('Skipping subsequent stages as no files were processed in Stage 2.')
        #     logging.info(f"Created skip flag: {skip_flag_path}")
        # except Exception as e:
        #     logging.error(f"Failed to create skip flag {skip_flag_path}: {e}")
        return

    # Create a local temporary directory for processing
    with tempfile.TemporaryDirectory() as local_temp_dir:
        logging.info(f"Using temporary directory: {local_temp_dir}")

        for file_info in files_to_process:
            nas_file_path = file_info.get("nas_path")
            base_filename = file_info.get("filename")
            if not nas_file_path or not base_filename:
                logging.warning(f"Skipping entry due to missing 'nas_path' or 'filename': {file_info}")
                continue

            logging.info(f"--- Processing: {base_filename} ---")
            local_pdf_path = os.path.join(local_temp_dir, base_filename)
            output_json_filename = f"{Path(base_filename).stem}.json"
            output_json_path_nas = os.path.join(stage2_output_dir_nas, output_json_filename)

            # Check if output already exists (for potential resume capability)
            # TODO: Add more robust checkpointing if needed
            if smbclient.path.exists(output_json_path_nas):
                logging.info(f"Output file {output_json_path_nas} already exists. Skipping.")
                continue

            # 1. Download PDF from NAS
            try:
                logging.info(f"Downloading {nas_file_path} to {local_pdf_path}")
                with smbclient.open_file(nas_file_path, mode='rb') as nas_f, open(local_pdf_path, 'wb') as local_f:
                    local_f.write(nas_f.read())
            except Exception as e:
                logging.error(f"Failed to download {nas_file_path}: {e}")
                continue # Skip to next file

            all_pages_vision_output = {}
            start_time = time.time()

            # 2. Process PDF: Split, Convert, Analyze
            try:
                pdf_document = fitz.open(local_pdf_path)
                num_pages = len(pdf_document)
                logging.info(f"Opened PDF '{base_filename}' with {num_pages} pages.")

                for page_num in range(num_pages):
                    page_index = page_num + 1 # Use 1-based index for logging/keys
                    logging.info(f"Processing Page {page_index}/{num_pages}...")
                    page = pdf_document.load_page(page_num)

                    # Convert page to JPEG
                    pix = page.get_pixmap(dpi=150) # Adjust DPI as needed
                    img_bytes = pix.tobytes("jpeg")
                    local_page_jpeg_path = os.path.join(local_temp_dir, f"{Path(base_filename).stem}_page_{page_index}.jpg")

                    try:
                        with open(local_page_jpeg_path, "wb") as img_file:
                            img_file.write(img_bytes)
                    except Exception as e:
                         logging.error(f"Failed to save page {page_index} as JPEG: {e}")
                         # Decide how to handle: skip page or fail file?
                         continue # Skip this page

                    # Run multi-pass vision analysis for the page
                    page_results = {}
                    for pass_name, prompt in VISION_PROMPTS.items():
                        logging.info(f"  Running {pass_name} for page {page_index}...")
                        vision_response = call_vision_api(local_page_jpeg_path, prompt)
                        page_results[pass_name] = vision_response if vision_response is not None else "Error: Failed to get response"
                        # Optional: Add a small delay between API calls if needed
                        # time.sleep(1)

                    all_pages_vision_output[f"page_{page_index}"] = page_results

                    # Clean up temporary page image
                    try:
                        os.remove(local_page_jpeg_path)
                    except OSError as e:
                        logging.warning(f"Could not remove temporary page file {local_page_jpeg_path}: {e}")

                pdf_document.close()

            except Exception as e:
                logging.error(f"Failed to process PDF {local_pdf_path}: {e}")
                # Clean up downloaded PDF before skipping
                try:
                    os.remove(local_pdf_path)
                except OSError as e_del:
                    logging.warning(f"Could not remove temporary PDF file {local_pdf_path}: {e_del}")
                continue # Skip to next file

            # 3. Save results to NAS
            try:
                output_data = json.dumps(all_pages_vision_output, indent=2)
                with smbclient.open_file(output_json_path_nas, mode='w', encoding='utf-8') as f:
                    f.write(output_data)
                processing_time = time.time() - start_time
                logging.info(f"Successfully processed '{base_filename}' ({processing_time:.2f}s) and saved results to {output_json_path_nas}")
            except Exception as e:
                logging.error(f"Failed to save results JSON to {output_json_path_nas}: {e}")
                # Consider cleanup or retry logic

            # 4. Clean up downloaded PDF
            try:
                os.remove(local_pdf_path)
            except OSError as e:
                logging.warning(f"Could not remove temporary PDF file {local_pdf_path}: {e}")

    logging.info("--- Finished Stage 2 ---")

if __name__ == "__main__":
    main()
