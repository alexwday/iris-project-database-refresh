# -*- coding: utf-8 -*-
"""
Stage 2: Process Documents with Vision Model (Vision variant for catalog search)

This script is a modified version of stage2_process_documents.py that uses
a vision model instead of Azure Document Intelligence to process PDF documents.

It takes the list of files identified in Stage 1 (from '1C_nas_files_to_process.json')
and processes each file using a vision model with multiple analysis passes,
then synthesizes the results into markdown format compatible with Stage 3.

The output format and location exactly matches the standard catalog search pipeline:
- Saves to the same '2B_outputs' folder as the original stage 2
- Structured JSON with document_name, total_pages, and pages array
- Each page contains page_number and markdown_content fields

This is a complete drop-in replacement for stage2_process_documents.py when
processing visual/infographic documents. Stage 3 can process the output without
any modifications.
"""

import os
import sys
import json
import tempfile
import time
import warnings
import logging
from datetime import datetime, timezone
import base64
import requests
import fitz  # PyMuPDF for PDF handling
from PIL import Image
import io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Use pysmb instead of smbclient ---
from smb.SMBConnection import SMBConnection
from smb import smb_structs
import socket # For gethostname
# --- End pysmb import ---

from openai import OpenAI  # For markdown synthesis

# Suppress common PDF warnings that don't affect processing
warnings.filterwarnings("ignore", message=".*wrong pointing object.*")
warnings.filterwarnings("ignore", message=".*invalid pdf header.*")
warnings.filterwarnings("ignore", message=".*PdfReadWarning.*")

# Configure pypdf logging to reduce noise
logging.getLogger("pypdf").setLevel(logging.ERROR)

# ==============================================================================
# --- Configuration ---
# ==============================================================================

# --- Vision Model Configuration ---
VISION_API_URL = 'https://your-endpoint-url/v1/chat/completions' # Replace with your actual endpoint
VISION_MODEL_NAME = 'qwen-2-vl' # Adjust if needed
MAX_TOKENS = 1500 # Max tokens for vision model response
IMAGE_DPI = 150 # DPI for converting PDF pages to images

# --- GPT Configuration for Markdown Synthesis ---
GPT_CONFIG = {
    "base_url": "https://your-aoai-endpoint.openai.azure.com/", # Replace with actual endpoint
    "api_key": "YOUR_API_KEY",  # Replace with actual API key
    "markdown_synthesis_model": "gpt-4o", # Model for synthesizing markdown from vision passes
}

# --- NAS Configuration (Should match Stage 1 or be loaded) ---
# Network attached storage connection parameters
NAS_PARAMS = {
    "ip": "your_nas_ip",
    "share": "your_share_name",
    "user": "your_nas_user",
    "password": "your_nas_password",
    "port": 445 # Default SMB port (can be 139)
}
# Base path on the NAS share where Stage 1 output files were stored
NAS_OUTPUT_FOLDER_PATH = "path/to/your/output_folder" # Relative path from share root

# --- Processing Configuration ---
# Document sources configuration - each line contains source name and detail level
DOCUMENT_SOURCES = """
internal_cheatsheets,detailed
internal_esg,standard
# internal_policies,concise
financial_reports,detailed
marketing_materials,concise
# technical_docs,detailed
"""

def load_document_sources():
    """Parse document sources configuration - works for all stages"""
    sources = []
    for line in DOCUMENT_SOURCES.strip().split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            parts = line.split(',')
            if len(parts) == 2:
                source_name = parts[0].strip()
                detail_level = parts[1].strip()
                sources.append({
                    'name': source_name,
                    'detail_level': detail_level
                })
            else:
                print(f"Warning: Invalid config line ignored: {line}")
    return sources

# Concurrent Processing Configuration
MAX_CONCURRENT_PAGES = 3  # Number of pages to process simultaneously with vision model

# --- pysmb Configuration ---
smb_structs.SUPPORT_SMB2 = True # Enable SMB2/3 support if available
smb_structs.MAX_PAYLOAD_SIZE = 65536 # Can sometimes help with large directories
CLIENT_HOSTNAME = socket.gethostname() # Get local machine name for SMB connection

# --- Vision Prompts (CO-STAR Format) - From original vision preprocessing ---
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

# ==============================================================================
# --- Helper Functions (pysmb versions) ---
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
            print("   [ERROR] Failed to connect to NAS.")
            return None
        return conn
    except Exception as e:
        print(f"   [ERROR] Exception creating NAS connection: {e}")
        return None

def ensure_nas_dir_exists(conn, share_name, dir_path):
    """Ensures a directory exists on NAS, creating it and parent directories if needed."""
    if not dir_path or dir_path == '/' or dir_path == '.':
        return True
    
    try:
        conn.listPath(share_name, dir_path, timeout=30)
        return True
    except Exception:
        parent_dir = os.path.dirname(dir_path).replace('\\', '/')
        if parent_dir and parent_dir != dir_path:
            if not ensure_nas_dir_exists(conn, share_name, parent_dir):
                return False
        
        try:
            conn.createDirectory(share_name, dir_path, timeout=30)
            print(f"   Created NAS directory: {share_name}/{dir_path}")
            return True
        except Exception as e:
            print(f"   [ERROR] Failed to create directory {dir_path}: {e}")
            return False

def write_to_nas(share_name, nas_path_relative, content_bytes):
    """Writes bytes to a file path on the NAS using pysmb."""
    conn = None
    print(f"   Attempting to write to NAS path: {share_name}/{nas_path_relative}")
    try:
        conn = create_nas_connection()
        if not conn:
            return False

        dir_path = os.path.dirname(nas_path_relative).replace('\\', '/')
        if dir_path and not ensure_nas_dir_exists(conn, share_name, dir_path):
             print(f"   [ERROR] Failed to ensure output directory exists: {dir_path}")
             return False

        with io.BytesIO(content_bytes) as file_obj:
            bytes_written = conn.storeFile(share_name, nas_path_relative, file_obj, timeout=60)
        
        print(f"   Successfully wrote {bytes_written} bytes to NAS.")
        return True
    except Exception as e:
        print(f"   [ERROR] Failed to write to NAS: {e}")
        return False
    finally:
        if conn:
            try:
                conn.close()
            except:
                pass

def read_json_from_nas(share_name, nas_path_relative):
    """Reads and parses JSON content from a file on NAS."""
    conn = None
    try:
        conn = create_nas_connection()
        if not conn:
            return None
        
        with io.BytesIO() as file_obj:
            conn.retrieveFile(share_name, nas_path_relative, file_obj, timeout=60)
            file_obj.seek(0)
            json_str = file_obj.read().decode('utf-8')
            return json.loads(json_str)
    except Exception as e:
        print(f"   [ERROR] Failed to read JSON from NAS: {e}")
        return None
    finally:
        if conn:
            try:
                conn.close()
            except:
                pass

def download_file_from_nas(share_name, nas_path_relative, local_path):
    """Downloads a file from NAS to a local path."""
    conn = None
    try:
        conn = create_nas_connection()
        if not conn:
            return False
        
        with open(local_path, 'wb') as local_file:
            conn.retrieveFile(share_name, nas_path_relative, local_file, timeout=120)
        
        print(f"   Downloaded {nas_path_relative} from NAS to {local_path}")
        return True
    except Exception as e:
        print(f"   [ERROR] Failed to download from NAS: {e}")
        return False
    finally:
        if conn:
            try:
                conn.close()
            except:
                pass

# ==============================================================================
# --- Vision Processing Functions ---
# ==============================================================================

def encode_image(image_path):
    """Encodes an image file to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"   [ERROR] Error encoding image {image_path}: {e}")
        return None

def call_vision_api(image_path, prompt_text):
    """Calls the vision model API for a given image and prompt."""
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

    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            response = requests.post(VISION_API_URL, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            response_text = result.get('choices', [{}])[0].get('message', {}).get('content', 'Error: No content found')
            return response_text

        except requests.exceptions.Timeout:
            print(f"   [WARNING] Timeout on attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return "Error: API call timed out after retries."
        
        except Exception as e:
            print(f"   [ERROR] Vision API error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return f"Error: API call failed - {str(e)}"
    
    return "Error: Failed to get response after all retries."

def synthesize_vision_to_markdown(page_vision_data, page_number):
    """
    Synthesizes vision model analysis passes into coherent markdown.
    Adapted from original stage3_generate_markdown_and_summaries.py
    """
    try:
        # Initialize OpenAI client for markdown synthesis
        client = OpenAI(
            base_url=GPT_CONFIG["base_url"],
            api_key=GPT_CONFIG["api_key"]
        )
        
        # Combine vision pass results into a structured input string
        input_text_parts = [f"Vision Model Analysis for Page {page_number}:\n"]
        for pass_name, result in page_vision_data.items():
            input_text_parts.append(f"--- {pass_name.upper().replace('_', ' ')} ---")
            input_text_parts.append(str(result))
            input_text_parts.append("")
        
        input_text_parts.append("---")
        input_text_parts.append(
            "Synthesize the above multi-pass vision model analysis into a single, coherent Markdown document "
            "representing this page's content. Preserve structure like tables and lists where possible. "
            "Focus on accurately representing the information conveyed visually and textually."
        )
        combined_input = "\n".join(input_text_parts)
        
        system_prompt = (
            "You are an expert technical writer specializing in interpreting multi-modal analysis results. "
            "Your task is to synthesize vision model outputs describing an infographic page into a "
            "comprehensive and accurate Markdown representation of that page."
        )
        
        completion = client.chat.completions.create(
            model=GPT_CONFIG['markdown_synthesis_model'],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": combined_input}
            ],
            temperature=0.2,
            max_tokens=3000
        )
        
        markdown_content = completion.choices[0].message.content
        return markdown_content
        
    except Exception as e:
        print(f"   [ERROR] Failed to synthesize markdown for page {page_number}: {e}")
        return f"Error: Failed to generate Markdown for page {page_number} - {str(e)}"

def process_pdf_page(pdf_path, page_num, temp_dir):
    """Process a single PDF page through all vision passes and synthesize to markdown."""
    try:
        # Open PDF and extract page
        pdf_document = fitz.open(pdf_path)
        page = pdf_document.load_page(page_num)
        
        # Convert page to image
        pix = page.get_pixmap(dpi=IMAGE_DPI)
        img_bytes = pix.tobytes("jpeg")
        
        # Save image temporarily
        image_path = os.path.join(temp_dir, f"page_{page_num + 1}.jpg")
        with open(image_path, "wb") as img_file:
            img_file.write(img_bytes)
        
        # Run all vision passes
        vision_results = {}
        for pass_name, prompt in VISION_PROMPTS.items():
            print(f"      Running {pass_name} for page {page_num + 1}...")
            response = call_vision_api(image_path, prompt)
            vision_results[pass_name] = response
        
        # Clean up image
        try:
            os.remove(image_path)
        except:
            pass
        
        pdf_document.close()
        
        # Synthesize vision results to markdown
        print(f"      Synthesizing markdown for page {page_num + 1}...")
        markdown_content = synthesize_vision_to_markdown(vision_results, page_num + 1)
        
        return {
            "page_number": page_num + 1,
            "markdown_content": markdown_content,
            "vision_passes": vision_results  # Keep raw passes for debugging if needed
        }
        
    except Exception as e:
        print(f"   [ERROR] Failed to process page {page_num + 1}: {e}")
        return {
            "page_number": page_num + 1,
            "markdown_content": None,
            "error": str(e)
        }

def process_pages_batch(pdf_path, page_numbers, temp_dir, max_workers=3):
    """Process multiple PDF pages concurrently."""
    page_results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_page = {
            executor.submit(process_pdf_page, pdf_path, page_num, temp_dir): page_num
            for page_num in page_numbers
        }
        
        for future in as_completed(future_to_page):
            page_num = future_to_page[future]
            try:
                result = future.result(timeout=300)  # 5 minute timeout per page
                page_results.append(result)
            except Exception as e:
                print(f"   [ERROR] Page {page_num + 1} processing failed: {e}")
                page_results.append({
                    "page_number": page_num + 1,
                    "markdown_content": None,
                    "error": str(e)
                })
    
    # Sort results by page number
    page_results.sort(key=lambda x: x["page_number"])
    return page_results

# ==============================================================================
# --- Main Processing Logic ---
# ==============================================================================

def main():
    print("\n" + "="*60)
    print("--- Running Stage 2: Process Documents with Vision Model ---")
    print("="*60 + "\n")
    
    # Load document sources configuration
    sources = load_document_sources()
    if not sources:
        print("[ERROR] No document sources configured to process!")
        sys.exit(1)
    
    print(f"[1] Processing {len(sources)} document source(s)...")
    print("-" * 60)
    
    overall_processed = 0
    overall_errors = 0
    
    for source_config in sources:
        document_source = source_config['name']
        detail_level = source_config['detail_level']
        
        print(f"\n--- Processing source: {document_source} (detail: {detail_level}) ---")
        
        # Define paths
        share_name = NAS_PARAMS["share"]
        stage1_output_dir_relative = os.path.join(NAS_OUTPUT_FOLDER_PATH, document_source).replace('\\', '/')
        files_to_process_json_relative = os.path.join(stage1_output_dir_relative, '1C_nas_files_to_process.json').replace('\\', '/')
        stage2_output_dir_relative = os.path.join(stage1_output_dir_relative, '2B_outputs').replace('\\', '/')
        
        print(f"   Stage 1 Output Dir: {stage1_output_dir_relative}")
        print(f"   Stage 2 Output Dir: {stage2_output_dir_relative}")
        
        # Load files to process from Stage 1
        print(f"[2] Loading Stage 1 file list...")
        files_to_process = read_json_from_nas(share_name, files_to_process_json_relative)
        
        if not files_to_process:
            print(f"   [WARNING] No files to process from Stage 1 for source '{document_source}'. Skipping.")
            continue
        
        print(f"   Found {len(files_to_process)} file(s) to process.")
        print("-" * 60)
        
        # Process each file
        processed_count = 0
        error_count = 0
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for i, file_info in enumerate(files_to_process):
                start_time = time.time()
                file_name = file_info.get('file_name', 'Unknown')
                file_path_nas = file_info.get('file_path_nas', '')
                
                print(f"\n--- Processing file {i+1}/{len(files_to_process)}: {file_name} ---")
                
                # Define output path
                output_json_filename = os.path.splitext(file_name)[0] + '.json'
                output_json_relative_path = os.path.join(stage2_output_dir_relative, output_json_filename).replace('\\', '/')
                
                file_has_error = False
                local_file_path = None
                
                try:
                    # Download file from NAS
                    local_file_path = os.path.join(temp_dir, file_name)
                    print(f"   Downloading from NAS: {file_path_nas}")
                    
                    if not download_file_from_nas(share_name, file_path_nas, local_file_path):
                        print(f"   [ERROR] Failed to download {file_name} from NAS. Skipping.")
                        error_count += 1
                        continue
                    
                    # Process PDF
                    structured_result = None
                    
                    if file_name.lower().endswith('.pdf'):
                        try:
                            reader = fitz.open(local_file_path)
                            page_count = len(reader)
                            reader.close()
                            
                            print(f"   PDF detected with {page_count} pages. Using vision processing.")
                            
                            # Process pages in batches
                            page_numbers = list(range(page_count))
                            page_results = process_pages_batch(
                                local_file_path, 
                                page_numbers, 
                                temp_dir,
                                max_workers=MAX_CONCURRENT_PAGES
                            )
                            
                            # Filter out vision_passes from final output (keep them for debugging if needed)
                            cleaned_pages = []
                            for page in page_results:
                                cleaned_pages.append({
                                    "page_number": page["page_number"],
                                    "markdown_content": page.get("markdown_content", "")
                                })
                            
                            # Create structured result matching catalog search format
                            structured_result = {
                                "document_name": file_name,
                                "total_pages": page_count,
                                "pages": cleaned_pages
                            }
                            
                            # Check for failed pages
                            failed_pages = [p for p in page_results if p.get('markdown_content') is None]
                            if failed_pages:
                                print(f"   [WARNING] {len(failed_pages)} pages failed to process.")
                            else:
                                print(f"   All {page_count} pages processed successfully.")
                                
                        except Exception as e:
                            print(f"   [ERROR] Failed to process PDF {file_name}: {e}")
                            file_has_error = True
                    
                    else:
                        # For non-PDF files, we might need different handling
                        print(f"   [WARNING] Non-PDF file detected: {file_name}. Vision processing not implemented for this type.")
                        file_has_error = True
                    
                    # Save results to NAS
                    if structured_result and not file_has_error:
                        print(f"   Saving structured JSON output to NAS...")
                        
                        json_bytes = json.dumps(structured_result, indent=4).encode('utf-8')
                        if not write_to_nas(share_name, output_json_relative_path, json_bytes):
                            print(f"   [ERROR] Failed to write JSON file for {file_name} to NAS.")
                            file_has_error = True
                        else:
                            print(f"   Successfully saved JSON with {structured_result['total_pages']} pages.")
                    
                except Exception as e:
                    print(f"   [ERROR] Unexpected error processing file {file_name}: {e}")
                    file_has_error = True
                
                finally:
                    # Cleanup
                    if local_file_path and os.path.exists(local_file_path):
                        try:
                            os.remove(local_file_path)
                        except OSError as e:
                            print(f"   [WARNING] Failed to remove temporary file: {e}")
                    
                    # Update counters
                    if file_has_error:
                        error_count += 1
                        print(f"--- Finished file {i+1} (ERROR) ---")
                    else:
                        processed_count += 1
                        print(f"--- Finished file {i+1} (Success) ---")
                    
                    end_time = time.time()
                    print(f"--- Time taken: {end_time - start_time:.2f} seconds ---")
        
        # Summary for this source
        print(f"\n--- Summary for source '{document_source}' ---")
        print(f"   Files processed successfully: {processed_count}")
        print(f"   Files with errors: {error_count}")
        print(f"   Total files: {len(files_to_process)}")
        
        overall_processed += processed_count
        overall_errors += error_count
    
    # Final summary
    print("\n" + "="*60)
    print("--- Stage 2 Completed ---")
    print(f"Total files processed successfully: {overall_processed}")
    print(f"Total files with errors: {overall_errors}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()