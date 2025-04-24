# -*- coding: utf-8 -*-
"""
Stage 2: Process Documents with Azure Document Intelligence

This script takes the list of files identified in Stage 1
(from '1C_nas_files_to_process.json') and processes each file
using the Azure Document Intelligence 'prebuilt-layout' model
to extract content in Markdown format.

It handles large PDF files (>2000 pages) by splitting them into
1000-page chunks, processing each chunk individually, and then
recombining the Markdown output.

Outputs for each processed file (Markdown and full analysis JSON)
are stored in a structured directory on the NAS.
"""

import os
import sys
import json
import tempfile
import time
from datetime import datetime, timezone
import smbclient
from pypdf import PdfReader, PdfWriter # For PDF handling
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat # Corrected import

# ==============================================================================
# --- Configuration ---
# ==============================================================================

# --- Azure Document Intelligence Configuration ---
# IMPORTANT: Replace with your actual endpoint and key, or load from env/config
AZURE_DI_ENDPOINT = "YOUR_DI_ENDPOINT"
AZURE_DI_KEY = "YOUR_DI_KEY"

# --- NAS Configuration (Should match Stage 1 or be loaded) ---
# IMPORTANT: Replace placeholder values if not loaded from a shared config.
NAS_PARAMS = {
    "ip": "your_nas_ip",          # Replace with NAS IP address
    "share": "your_share_name",   # Replace with NAS share name
    "user": "your_nas_user",      # Replace with NAS username
    "password": "your_nas_password" # Replace with NAS password
}
# Base path on the NAS share where Stage 1 output files were stored.
NAS_OUTPUT_FOLDER_PATH = "path/to/your/output_folder" # From Stage 1

# --- Processing Configuration (Should match Stage 1) ---
# Define the specific document source processed in Stage 1.
DOCUMENT_SOURCE = 'internal_esg' # From Stage 1
DB_TABLE_NAME = 'apg_catalog'    # From Stage 1 (Potentially needed for context, maybe not)

# PDF Processing Configuration
PDF_PAGE_LIMIT = 2000
PDF_CHUNK_SIZE = 1000

# ==============================================================================
# --- Helper Functions ---
# ==============================================================================

def initialize_smb_client():
    """Sets up smbclient credentials."""
    try:
        smbclient.ClientConfig(username=NAS_PARAMS["user"], password=NAS_PARAMS["password"])
        print("SMB client configured successfully.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to configure SMB client: {e}")
        return False

def create_nas_directory(smb_dir_path):
    """Creates a directory on the NAS if it doesn't exist."""
    try:
        if not smbclient.path.exists(smb_dir_path):
            print(f"   Creating NAS directory: {smb_dir_path}")
            smbclient.makedirs(smb_dir_path, exist_ok=True)
            print(f"   Successfully created directory.")
        else:
            # print(f"   NAS directory already exists: {smb_dir_path}") # Optional: reduce verbosity
            pass
        return True
    except smbclient.SambaClientError as e:
        print(f"   [ERROR] SMB Error creating/accessing directory '{smb_dir_path}': {e}")
        return False
    except Exception as e:
        print(f"   [ERROR] Unexpected error creating/accessing NAS directory '{smb_dir_path}': {e}")
        return False

def write_to_nas(smb_path, content_bytes):
    """Writes bytes to a file path on the NAS using smbclient."""
    print(f"   Attempting to write to NAS path: {smb_path}")
    try:
        # Ensure the directory exists first (redundant if create_nas_directory called before, but safe)
        dir_path = os.path.dirname(smb_path)
        if not create_nas_directory(dir_path):
             return False # Failed to create directory

        with smbclient.open_file(smb_path, mode='wb') as f: # Write in binary mode
            f.write(content_bytes)
        print(f"   Successfully wrote {len(content_bytes)} bytes to: {smb_path}")
        return True
    except smbclient.SambaClientError as e:
        print(f"   [ERROR] SMB Error writing to '{smb_path}': {e}")
        return False
    except Exception as e:
        print(f"   [ERROR] Unexpected error writing to NAS '{smb_path}': {e}")
        return False

def download_from_nas(smb_path, local_temp_dir):
    """Downloads a file from NAS to a local temporary directory."""
    local_file_path = os.path.join(local_temp_dir, os.path.basename(smb_path))
    print(f"   Attempting to download from NAS: {smb_path} to {local_file_path}")
    try:
        with smbclient.open_file(smb_path, mode='rb') as nas_f:
            with open(local_file_path, 'wb') as local_f:
                local_f.write(nas_f.read())
        print(f"   Successfully downloaded to: {local_file_path}")
        return local_file_path
    except smbclient.SambaClientError as e:
        print(f"   [ERROR] SMB Error downloading from '{smb_path}': {e}")
        return None
    except Exception as e:
        print(f"   [ERROR] Unexpected error downloading from NAS '{smb_path}': {e}")
        return None

def analyze_document_with_di(di_client, local_file_path, output_format=DocumentContentFormat.MARKDOWN): # Corrected usage
    """Analyzes a local document using Azure Document Intelligence layout model."""
    print(f"   Analyzing local file with DI: {local_file_path}")
    try:
        # Read the file content first
        with open(local_file_path, "rb") as f:
            document_bytes = f.read() # Read bytes only once

        # Try passing bytes directly as the second argument, as allowed by the SDK signature.
        poller = di_client.begin_analyze_document(
            "prebuilt-layout",       # model_id (positional)
            document_bytes,          # analyze_request (positional, as bytes)
            output_content_format=output_format # kwargs
        )
        result = poller.result()
        print(f"   DI analysis successful.")
        return result
    except Exception as e:
        # Print the specific error type for better debugging
        print(f"   [ERROR] Document Intelligence analysis failed for {local_file_path}: {type(e).__name__} - {e}")
        return None

def split_pdf(local_pdf_path, chunk_size, temp_dir):
    """Splits a PDF into chunks of a specified size."""
    chunk_paths = []
    base_name = os.path.splitext(os.path.basename(local_pdf_path))[0]
    print(f"   Splitting PDF: {os.path.basename(local_pdf_path)} into {chunk_size}-page chunks...")
    try:
        reader = PdfReader(local_pdf_path)
        total_pages = len(reader.pages)
        for i in range(0, total_pages, chunk_size):
            writer = PdfWriter()
            start_page = i
            end_page = min(i + chunk_size, total_pages)
            print(f"      Creating chunk {len(chunk_paths) + 1}: pages {start_page + 1}-{end_page}")
            for page_num in range(start_page, end_page):
                writer.add_page(reader.pages[page_num])

            chunk_filename = f"{base_name}_chunk_{len(chunk_paths) + 1}.pdf"
            chunk_path = os.path.join(temp_dir, chunk_filename)
            with open(chunk_path, "wb") as chunk_file:
                writer.write(chunk_file)
            chunk_paths.append(chunk_path)
            print(f"      Saved chunk to: {chunk_path}")
        print(f"   Successfully split into {len(chunk_paths)} chunks.")
        return chunk_paths
    except Exception as e:
        print(f"   [ERROR] Failed to split PDF {local_pdf_path}: {e}")
        return [] # Return empty list on failure

# ==============================================================================
# --- Main Execution Logic ---
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print(f"--- Running Stage 2: Process Documents with Document Intelligence ---")
    print(f"--- Document Source: {DOCUMENT_SOURCE} ---")
    print("="*60 + "\n")

    # --- Initialize Clients ---
    print("[1] Initializing Clients...")
    if not initialize_smb_client():
        sys.exit(1)

    try:
        di_client = DocumentIntelligenceClient(
            endpoint=AZURE_DI_ENDPOINT, credential=AzureKeyCredential(AZURE_DI_KEY)
        )
        print("Document Intelligence client initialized successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Document Intelligence client: {e}")
        sys.exit(1)
    print("-" * 60)

    # --- Define Paths ---
    print("[2] Defining NAS Paths...")
    # Base output directory from Stage 1
    stage1_output_dir_relative = os.path.join(NAS_OUTPUT_FOLDER_PATH, DOCUMENT_SOURCE).replace('\\', '/')
    stage1_output_dir_smb = f"//{NAS_PARAMS['ip']}/{NAS_PARAMS['share']}/{stage1_output_dir_relative}"
    # Input JSON file from Stage 1
    files_to_process_json_smb = os.path.join(stage1_output_dir_smb, '1C_nas_files_to_process.json').replace('\\', '/')
    # Base output directory for Stage 2 results
    stage2_output_dir_relative = os.path.join(stage1_output_dir_relative, '2A_processed_files').replace('\\', '/')
    stage2_output_dir_smb = f"//{NAS_PARAMS['ip']}/{NAS_PARAMS['share']}/{stage2_output_dir_relative}"

    print(f"   Stage 1 Output Dir (SMB): {stage1_output_dir_smb}")
    print(f"   Input JSON File (SMB): {files_to_process_json_smb}")
    print(f"   Stage 2 Output Base Dir (SMB): {stage2_output_dir_smb}")

    # Ensure base Stage 2 output directory exists
    if not create_nas_directory(stage2_output_dir_smb):
        print("[CRITICAL ERROR] Could not create base Stage 2 output directory on NAS. Exiting.")
        sys.exit(1)
    print("-" * 60)

    # --- Load Files to Process List ---
    print(f"[3] Loading list of files to process from: {os.path.basename(files_to_process_json_smb)}...")
    files_to_process = []
    try:
        # Read the JSON file content from NAS
        with smbclient.open_file(files_to_process_json_smb, mode='r', encoding='utf-8') as f:
            files_to_process = json.load(f)
        print(f"   Successfully loaded {len(files_to_process)} file entries.")
        if not files_to_process:
             print("   List is empty. No files to process in Stage 2.")
             print("\n" + "="*60)
             print(f"--- Stage 2 Completed (No files to process) ---")
             print("="*60 + "\n")
             sys.exit(0) # Successful exit, nothing to do
    except smbclient.SambaClientError as e:
        print(f"   [CRITICAL ERROR] SMB Error reading '{files_to_process_json_smb}': {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"   [CRITICAL ERROR] Failed to parse JSON from '{files_to_process_json_smb}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"   [CRITICAL ERROR] Unexpected error reading or parsing '{files_to_process_json_smb}': {e}")
        sys.exit(1)
    print("-" * 60)

    # --- Process Each File ---
    print(f"[4] Processing {len(files_to_process)} files...")
    processed_count = 0
    error_count = 0

    # Create a single temporary directory for all downloads/chunks
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"   Using temporary directory: {temp_dir}")

        for i, file_info in enumerate(files_to_process):
            start_time = time.time()
            file_has_error = False
            print(f"\n--- Processing file {i+1}/{len(files_to_process)} ---")
            try:
                file_name = file_info.get('file_name')
                file_path_relative = file_info.get('file_path') # Path relative to share root

                if not file_name or not file_path_relative:
                    print("   [ERROR] Skipping entry due to missing 'file_name' or 'file_path'.")
                    error_count += 1
                    continue

                print(f"   File Name: {file_name}")
                print(f"   Relative NAS Path: {file_path_relative}")

                # Construct full SMB path for the input file
                input_file_smb_path = f"//{NAS_PARAMS['ip']}/{NAS_PARAMS['share']}/{file_path_relative}"

                # Construct output paths
                file_name_base = os.path.splitext(file_name)[0]
                file_output_subfolder_smb = os.path.join(stage2_output_dir_smb, file_name_base).replace('\\', '/')
                output_md_smb_path = os.path.join(file_output_subfolder_smb, f"{file_name_base}.md").replace('\\', '/')
                output_json_smb_path = os.path.join(file_output_subfolder_smb, f"{file_name_base}.json").replace('\\', '/') # Base name for non-chunked

                print(f"   Output Subfolder (SMB): {file_output_subfolder_smb}")
                print(f"   Output Markdown Path (SMB): {output_md_smb_path}")
                print(f"   Output JSON Path (SMB): {output_json_smb_path}")

                # Ensure the specific output subfolder exists for this file
                if not create_nas_directory(file_output_subfolder_smb):
                    print(f"   [ERROR] Failed to create output subfolder for {file_name}. Skipping.")
                    error_count += 1
                    continue

                # Download file from NAS to temporary local directory
                local_file_path = download_from_nas(input_file_smb_path, temp_dir)
                if not local_file_path:
                    print(f"   [ERROR] Failed to download {file_name} from NAS. Skipping.")
                    error_count += 1
                    continue # Skip to next file

                # --- Document Intelligence Processing ---
                combined_markdown = ""
                results_json_list = [] # To store DI results (full or chunked)

                # Check if PDF and needs splitting
                is_pdf = file_name.lower().endswith('.pdf')
                needs_splitting = False
                page_count = 0
                local_chunk_paths = []

                if is_pdf:
                    try:
                        reader = PdfReader(local_file_path)
                        page_count = len(reader.pages)
                        print(f"   PDF detected with {page_count} pages.")
                        if page_count > PDF_PAGE_LIMIT:
                            needs_splitting = True
                            print(f"   PDF exceeds page limit ({PDF_PAGE_LIMIT}). Splitting required.")
                        else:
                             print(f"   PDF within page limit. Processing as single document.")
                    except Exception as pdf_err:
                        print(f"   [WARNING] Could not read PDF metadata for {file_name}: {pdf_err}. Attempting to process as single document.")
                        is_pdf = False # Treat as non-pdf if metadata read fails

                if is_pdf and needs_splitting:
                    # Split PDF into chunks
                    local_chunk_paths = split_pdf(local_file_path, PDF_CHUNK_SIZE, temp_dir)
                    if not local_chunk_paths:
                        print(f"   [ERROR] Failed to split PDF {file_name}. Skipping analysis.")
                        file_has_error = True
                    else:
                        # Process each chunk
                        all_chunks_processed = True
                        for chunk_index, chunk_path in enumerate(local_chunk_paths):
                            print(f"      Processing chunk {chunk_index + 1}/{len(local_chunk_paths)}...")
                            analyze_result = analyze_document_with_di(di_client, chunk_path)
                            if analyze_result and analyze_result.content:
                                combined_markdown += analyze_result.content + "\n\n" # Add separator between chunks
                                # Manually create dictionary from AnalyzeResult, mirroring example structure
                                result_dict = {'content': analyze_result.content, 'pages': []}
                                if hasattr(analyze_result, 'pages'):
                                    for page in analyze_result.pages:
                                        page_data = {
                                            'page_number': page.page_number,
                                            'width': page.width if hasattr(page, 'width') else None,
                                            'height': page.height if hasattr(page, 'height') else None,
                                            'unit': page.unit if hasattr(page, 'unit') else None,
                                            'angle': page.angle if hasattr(page, 'angle') else None
                                            # Add other page attributes if needed
                                        }
                                        result_dict['pages'].append(page_data)
                                if hasattr(analyze_result, 'tables') and analyze_result.tables:
                                    result_dict['tables_count'] = len(analyze_result.tables)
                                if hasattr(analyze_result, 'paragraphs') and analyze_result.paragraphs:
                                    result_dict['paragraphs_count'] = len(analyze_result.paragraphs)
                                # Add other relevant attributes if needed
                                results_json_list.append(result_dict)
                                print(f"      Chunk {chunk_index + 1} processed successfully.")
                            else:
                                print(f"      [ERROR] Failed to process chunk {chunk_index + 1} for {file_name}.")
                                file_has_error = True
                                all_chunks_processed = False
                                # Decide whether to break or continue processing other chunks
                                # break # Option: Stop processing this file if one chunk fails
                        if not all_chunks_processed:
                             print(f"   [ERROR] Not all chunks of {file_name} were processed successfully.")
                        else:
                             print(f"   All chunks of {file_name} processed.")

                else: # Process non-PDF or small PDF
                    analyze_result = analyze_document_with_di(di_client, local_file_path)
                    if analyze_result and analyze_result.content:
                        combined_markdown = analyze_result.content
                        # Manually create dictionary from AnalyzeResult, mirroring example structure
                        result_dict = {'content': analyze_result.content, 'pages': []}
                        if hasattr(analyze_result, 'pages'):
                            for page in analyze_result.pages:
                                page_data = {
                                    'page_number': page.page_number,
                                    'width': page.width if hasattr(page, 'width') else None,
                                    'height': page.height if hasattr(page, 'height') else None,
                                    'unit': page.unit if hasattr(page, 'unit') else None,
                                    'angle': page.angle if hasattr(page, 'angle') else None
                                    # Add other page attributes if needed
                                }
                                result_dict['pages'].append(page_data)
                        if hasattr(analyze_result, 'tables') and analyze_result.tables:
                            result_dict['tables_count'] = len(analyze_result.tables)
                        if hasattr(analyze_result, 'paragraphs') and analyze_result.paragraphs:
                            result_dict['paragraphs_count'] = len(analyze_result.paragraphs)
                        # Add other relevant attributes if needed
                        results_json_list.append(result_dict)
                    else:
                        print(f"   [ERROR] Failed to process document {file_name}.")
                        file_has_error = True

                # --- Save Results to NAS ---
                if combined_markdown and not file_has_error:
                    print(f"   Saving combined Markdown output to NAS...")
                    md_bytes = combined_markdown.encode('utf-8')
                    if not write_to_nas(output_md_smb_path, md_bytes):
                        print(f"   [ERROR] Failed to write Markdown file for {file_name} to NAS.")
                        file_has_error = True # Mark error even if DI worked

                if results_json_list and not file_has_error:
                    print(f"   Saving analysis JSON output(s) to NAS...")
                    if needs_splitting:
                        # Save JSON for each chunk
                        for chunk_index, result_json in enumerate(results_json_list):
                            chunk_json_smb_path = os.path.join(file_output_subfolder_smb, f"{file_name_base}_chunk_{chunk_index + 1}.json").replace('\\', '/')
                            # Add default handler for non-serializable objects, like in the example
                            json_bytes = json.dumps(result_json, indent=4, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o)).encode('utf-8')
                            if not write_to_nas(chunk_json_smb_path, json_bytes):
                                print(f"   [ERROR] Failed to write JSON chunk {chunk_index + 1} for {file_name} to NAS.")
                                # Don't necessarily mark file_has_error, maybe just log warning?
                    else:
                        # Save single JSON for non-split files
                        # Add default handler for non-serializable objects, like in the example
                        json_bytes = json.dumps(results_json_list[0], indent=4, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o)).encode('utf-8')
                        if not write_to_nas(output_json_smb_path, json_bytes):
                            print(f"   [ERROR] Failed to write JSON file for {file_name} to NAS.")
                            # Don't necessarily mark file_has_error, maybe just log warning?

            except Exception as e:
                print(f"   [ERROR] Unexpected error processing file {file_info.get('file_name', 'N/A')}: {e}")
                file_has_error = True
            finally:
                # --- Cleanup ---
                if local_file_path and os.path.exists(local_file_path):
                    try:
                        os.remove(local_file_path)
                        # print(f"   Cleaned up temporary file: {local_file_path}") # Optional verbosity
                    except OSError as e:
                        print(f"   [WARNING] Failed to remove temporary file {local_file_path}: {e}")
                if local_chunk_paths:
                    for chunk_path in local_chunk_paths:
                         if os.path.exists(chunk_path):
                            try:
                                os.remove(chunk_path)
                                # print(f"   Cleaned up temporary chunk: {chunk_path}") # Optional verbosity
                            except OSError as e:
                                print(f"   [WARNING] Failed to remove temporary chunk {chunk_path}: {e}")

                # --- Update Counters ---
                if file_has_error:
                    error_count += 1
                    print(f"--- Finished file {i+1} (ERROR) ---")
                else:
                    processed_count += 1
                    print(f"--- Finished file {i+1} (Success) ---")
                end_time = time.time()
                print(f"--- Time taken: {end_time - start_time:.2f} seconds ---")


    # --- Final Summary ---
    print("\n" + "="*60)
    print(f"--- Stage 2 Processing Summary ---")
    print(f"   Total files attempted: {len(files_to_process)}")
    print(f"   Successfully processed: {processed_count}")
    print(f"   Errors encountered: {error_count}")
    print("="*60 + "\n")

    if error_count > 0:
        print(f"[WARNING] {error_count} files encountered errors during processing. Check logs above.")
        # Optionally exit with error code if any file failed
        # sys.exit(1)

    print(f"--- Stage 2 Completed ---")
