c# -*- coding: utf-8 -*-
"""
Stage 2: Process Documents with Azure Document Intelligence (using pysmb)

This script takes the list of files identified in Stage 1
(from '1C_nas_files_to_process.json') and processes each file
using the Azure Document Intelligence 'prebuilt-layout' model
to extract content in Markdown format.

It handles large PDF files (>2000 pages) by splitting them into
1000-page chunks, processing each chunk individually, and then
recombining the Markdown output.

Outputs for each processed file (Markdown and full analysis JSON)
are stored in a structured directory on the NAS using pysmb.
"""

import os
import sys
import json
import tempfile
import time
from datetime import datetime, timezone
# --- Use pysmb instead of smbclient ---
from smb.SMBConnection import SMBConnection
from smb import smb_structs
import io # For reading/writing strings/bytes to NAS
import socket # For gethostname
# --- End pysmb import ---
from pypdf import PdfReader, PdfWriter # For PDF handling
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat
# Removed httpx import

# Removed unused psycopg2 import

# ==============================================================================
# --- Configuration ---
# ==============================================================================

# --- Azure Document Intelligence Configuration ---
# Azure service connection parameters
AZURE_DI_ENDPOINT = "YOUR_DI_ENDPOINT"
AZURE_DI_KEY = "YOUR_DI_KEY"

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

# --- Processing Configuration (Should match Stage 1) ---
# Define the specific document source processed in Stage 1.
DOCUMENT_SOURCE = 'internal_esg' # From Stage 1
# DB_TABLE_NAME = 'apg_catalog'    # From Stage 1 (Not used in Stage 2)

# PDF Processing Configuration
PDF_PAGE_LIMIT = 2000
PDF_CHUNK_SIZE = 1000

# --- CA Bundle Configuration ---
CA_BUNDLE_FILENAME = 'rbc-ca-bundle.cer' # Added CA bundle filename (ensure this exists on NAS)

# --- Proxy Configuration (Fill in if needed) ---
PROXY_CONFIG = {
    "use_proxy": True, # Set to True to enable proxy settings
    "url": "http://your_proxy_server:port", # e.g., http://proxy.example.com:8080
    "username": "YOUR_PROXY_USERNAME", # Set to None if no authentication needed
    "password": "YOUR_PROXY_PASSWORD"  # Set to None if no authentication needed
}

# --- pysmb Configuration ---
# Increase timeout for potentially slow NAS operations
smb_structs.SUPPORT_SMB2 = True # Enable SMB2/3 support if available
smb_structs.MAX_PAYLOAD_SIZE = 65536 # Can sometimes help with large directories
CLIENT_HOSTNAME = socket.gethostname() # Get local machine name for SMB connection

# ==============================================================================
# --- Helper Functions (pysmb versions) ---
# ==============================================================================

# ... (Helper functions remain unchanged) ...
def create_nas_connection():
    """Creates and returns an authenticated SMBConnection object."""
    try:
        conn = SMBConnection(
            NAS_PARAMS["user"],
            NAS_PARAMS["password"],
            CLIENT_HOSTNAME, # Local machine name
            NAS_PARAMS["ip"], # Remote server name (can be IP)
            use_ntlm_v2=True,
            is_direct_tcp=(NAS_PARAMS["port"] == 445) # Use direct TCP if port 445
        )
        connected = conn.connect(NAS_PARAMS["ip"], NAS_PARAMS["port"], timeout=60) # Increased timeout
        if not connected:
            print("   [ERROR] Failed to connect to NAS.")
            return None
        # print(f"   Successfully connected to NAS: {NAS_PARAMS['ip']}:{NAS_PARAMS['port']} on share '{NAS_PARAMS['share']}'") # Reduce verbosity
        return conn
    except Exception as e:
        print(f"   [ERROR] Exception creating NAS connection: {e}")
        return None

def ensure_nas_dir_exists(conn, share_name, dir_path_relative):
    """Ensures a directory exists on the NAS, creating it if necessary."""
    if not conn:
        print("   [ERROR] Cannot ensure NAS directory: No connection.")
        return False
    
    # pysmb needs paths relative to the share, using '/' as separator
    path_parts = dir_path_relative.strip('/').split('/')
    current_path = ''
    try:
        for part in path_parts:
            if not part: continue
            current_path = os.path.join(current_path, part).replace('\\', '/')
            try:
                # Check if it exists by trying to list it
                conn.listPath(share_name, current_path)
                # print(f"      Directory exists: {current_path}") # Reduce verbosity
            except Exception: # If listPath fails, assume it doesn't exist
                print(f"      Creating directory on NAS: {share_name}/{current_path}")
                conn.createDirectory(share_name, current_path)
        return True
    except Exception as e:
        print(f"   [ERROR] Failed to ensure/create NAS directory '{share_name}/{dir_path_relative}': {e}")
        return False

def write_to_nas(share_name, nas_path_relative, content_bytes):
    """Writes bytes to a file path on the NAS using pysmb."""
    conn = None
    print(f"   Attempting to write to NAS path: {share_name}/{nas_path_relative}")
    try:
        conn = create_nas_connection()
        if not conn:
            return False

        # Ensure the directory exists before writing the file
        dir_path = os.path.dirname(nas_path_relative).replace('\\', '/')
        if dir_path and not ensure_nas_dir_exists(conn, share_name, dir_path):
             print(f"   [ERROR] Failed to ensure output directory exists: {dir_path}")
             return False

        # Use BytesIO for pysmb storeFile
        file_obj = io.BytesIO(content_bytes)

        # Store the file
        bytes_written = conn.storeFile(share_name, nas_path_relative, file_obj)
        print(f"   Successfully wrote {bytes_written} bytes to: {share_name}/{nas_path_relative}")
        return True
    except Exception as e:
        print(f"   [ERROR] Unexpected error writing to NAS '{share_name}/{nas_path_relative}': {e}")
        return False
    finally:
        if conn:
            conn.close()

def read_from_nas(share_name, nas_path_relative):
    """Reads content (as bytes) from a file path on the NAS using pysmb."""
    conn = None
    print(f"   Attempting to read from NAS path: {share_name}/{nas_path_relative}")
    try:
        conn = create_nas_connection()
        if not conn:
            return None

        file_obj = io.BytesIO()
        file_attributes, filesize = conn.retrieveFile(share_name, nas_path_relative, file_obj)
        file_obj.seek(0)
        content_bytes = file_obj.read()
        print(f"   Successfully read {filesize} bytes from: {share_name}/{nas_path_relative}")
        return content_bytes
    except Exception as e:
        print(f"   [ERROR] Unexpected error reading from NAS '{share_name}/{nas_path_relative}': {e}")
        return None
    finally:
        if conn:
            conn.close()

def download_from_nas(share_name, nas_path_relative, local_temp_dir):
    """Downloads a file from NAS to a local temporary directory using pysmb with read-only access."""
    local_file_path = os.path.join(local_temp_dir, os.path.basename(nas_path_relative))
    print(f"   Attempting to download from NAS: {share_name}/{nas_path_relative} to {local_file_path}")
    conn = None
    
    # FIX: Add retry logic for file-in-use errors
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            conn = create_nas_connection()
            if not conn:
                return None

            # FIX: Use retrieveFileFromOffset with read-only intent
            # This method allows more control over the file access
            file_obj = io.BytesIO()
            
            # First get file attributes to check accessibility
            try:
                file_attrs = conn.getAttributes(share_name, nas_path_relative)
                print(f"   File size: {file_attrs.file_size} bytes")
            except Exception as attr_err:
                print(f"   [WARNING] Could not get file attributes: {attr_err}")
            
            # Attempt to retrieve the file
            # Using retrieveFileFromOffset which can be more tolerant of locked files
            file_attributes, filesize = conn.retrieveFileFromOffset(
                share_name, 
                nas_path_relative, 
                file_obj,
                offset=0,  # Start from beginning
                max_length=-1  # Read entire file
            )
            
            # Write the retrieved content to local file
            file_obj.seek(0)
            with open(local_file_path, 'wb') as local_f:
                local_f.write(file_obj.read())
                
            print(f"   Successfully downloaded {filesize} bytes to: {local_file_path}")
            return local_file_path
            
        except Exception as e:
            error_str = str(e).lower()
            # Check for sharing violation or access denied errors
            if any(err in error_str for err in ['sharing violation', 'access denied', 'locked', 'in use']):
                print(f"   [Attempt {attempt + 1}/{max_retries}] File appears to be in use: {e}")
                if attempt < max_retries - 1:
                    print(f"   Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"   [ERROR] Failed to download after {max_retries} attempts - file may be locked.")
            else:
                print(f"   [ERROR] Unexpected error downloading from NAS '{share_name}/{nas_path_relative}': {e}")
            
            # Clean up potentially partially downloaded file
            if os.path.exists(local_file_path):
                try:
                    os.remove(local_file_path)
                except OSError:
                    pass
            return None
        finally:
            if conn:
                conn.close()
    
    return None  # If all retries failed

def check_nas_path_exists(share_name, nas_path_relative):
    """Checks if a file or directory exists on the NAS using pysmb."""
    conn = None
    # print(f"   Checking existence of NAS path: {share_name}/{nas_path_relative}") # Verbose
    try:
        conn = create_nas_connection()
        if not conn:
            return False # Cannot check if connection failed

        # Use getAttributes to check existence - works for files and dirs
        # It will raise an OperationFailure exception if the path does not exist.
        conn.getAttributes(share_name, nas_path_relative)
        # print(f"   Path exists: {share_name}/{nas_path_relative}") # Verbose
        return True # Path exists if no exception was raised
    except Exception as e:
        # Check if the error message indicates "No such file" or similar
        # pysmb often raises OperationFailure with specific NTSTATUS codes.
        # STATUS_OBJECT_NAME_NOT_FOUND (0xC0000034) is common.
        err_str = str(e).lower()
        if "no such file" in err_str or "object_name_not_found" in err_str or "0xc0000034" in err_str:
            # print(f"   Path does not exist: {share_name}/{nas_path_relative}") # Verbose
            return False # Expected outcome if the file/path doesn't exist
        else:
            # Log other unexpected errors during the check
            print(f"   [WARNING] Unexpected error checking existence of NAS path '{share_name}/{nas_path_relative}': {type(e).__name__} - {e}")
            return False # Assume not found on other errors
    finally:
        if conn:
            conn.close()

# --- Other Helper Functions (Unchanged) ---

def analyze_document_with_di(di_client, local_file_path, output_format=DocumentContentFormat.MARKDOWN, max_retries=3, retry_delay=5):
    """
    Analyzes a local document using Azure Document Intelligence layout model,
    with retry logic for transient errors like SSLEOFError.
    """
    print(f"   Analyzing local file with DI: {local_file_path}")
    last_exception = None
    for attempt in range(max_retries):
        try:
            print(f"      DI Analysis Attempt {attempt + 1}/{max_retries}...")
            with open(local_file_path, "rb") as f:
                document_bytes = f.read()

            poller = di_client.begin_analyze_document(
                "prebuilt-layout",
                document_bytes,
                output_content_format=output_format
            )
            result = poller.result() # This is where the network call happens and might fail
            print(f"   DI analysis successful on attempt {attempt + 1}.")
            return result
        except Exception as e:
            last_exception = e
            print(f"   [Attempt {attempt + 1} ERROR] DI analysis failed for {local_file_path}: {type(e).__name__} - {e}")
            # Check if it's the last attempt
            if attempt < max_retries - 1:
                print(f"      Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"   [ERROR] DI analysis failed after {max_retries} attempts.")
                break # Exit loop after final attempt fails

    # If loop finishes without returning, it means all retries failed
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
        return []

# ==============================================================================
# --- Main Processing Function ---
# ==============================================================================

def main_processing_stage2(di_client, files_to_process_json_relative, stage2_output_dir_relative):
    """Loads input files, processes them with DI, and saves results using pysmb."""
    print(f"--- Starting Main Processing for Stage 2 ---")
    share_name = NAS_PARAMS["share"] # Convenience variable

    # --- Load Files to Process List ---
    print(f"[4] Loading list of files to process from: {share_name}/{files_to_process_json_relative}...")
    files_to_process = []
    try:
        # Read the JSON file content from NAS using pysmb helper
        json_bytes = read_from_nas(share_name, files_to_process_json_relative)
        if json_bytes is None:
             print(f"   [CRITICAL ERROR] Failed to read '{files_to_process_json_relative}' from NAS.")
             sys.exit(1) # Exit if critical file cannot be read

        files_to_process = json.loads(json_bytes.decode('utf-8'))
        print(f"   Successfully loaded {len(files_to_process)} file entries.")
        if not files_to_process:
             print("   List is empty. No files to process in Stage 2.")
             print("\n" + "="*60)
             print(f"--- Stage 2 Completed (No files to process) ---")
             print("="*60 + "\n")
             return # Exit this function early
    except json.JSONDecodeError as e:
        print(f"   [CRITICAL ERROR] Failed to parse JSON from '{files_to_process_json_relative}': {e}")
        sys.exit(1) # Exit if critical file cannot be parsed
    except Exception as e:
        print(f"   [CRITICAL ERROR] Unexpected error reading or parsing '{files_to_process_json_relative}': {e}")
        sys.exit(1) # Exit on other unexpected errors
    print("-" * 60)

    # --- Process Each File ---
    print(f"[5] Processing {len(files_to_process)} files...") # Renumbered step
    processed_count = 0
    error_count = 0

    # Create a single temporary directory for all downloads/chunks
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"   Using temporary directory: {temp_dir}")

        for i, file_info in enumerate(files_to_process):
            start_time = time.time()
            file_has_error = False
            local_file_path = None # Ensure defined in outer scope for finally block
            local_chunk_paths = [] # Ensure defined in outer scope for finally block

            print(f"\n--- Processing file {i+1}/{len(files_to_process)} ---")
            try:
                file_name = file_info.get('file_name')
                # This path is relative to the *base input path* from Stage 1 config
                file_path_from_base = file_info.get('file_path')

                if not file_name or not file_path_from_base:
                    print("   [ERROR] Skipping entry due to missing 'file_name' or 'file_path'.")
                    error_count += 1
                    continue

                # Construct the full path relative to the *share root* for pysmb
                input_file_relative_path = file_path_from_base # Path from Stage 1 JSON is already relative to share

                print(f"   File Name: {file_name}")
                print(f"   Relative NAS Input Path: {share_name}/{input_file_relative_path}")

                # Construct output paths (relative to share)
                file_name_base = os.path.splitext(file_name)[0]
                # Output subfolder relative to share
                file_output_subfolder_relative = os.path.join(stage2_output_dir_relative, file_name_base).replace('\\', '/')
                output_md_relative_path = os.path.join(file_output_subfolder_relative, f"{file_name_base}.md").replace('\\', '/')
                output_json_relative_path = os.path.join(file_output_subfolder_relative, f"{file_name_base}.json").replace('\\', '/') # Base name for non-chunked

                print(f"   Output Subfolder (Relative): {share_name}/{file_output_subfolder_relative}")
                print(f"   Output Markdown Path (Relative): {share_name}/{output_md_relative_path}")
                print(f"   Output JSON Path (Relative): {share_name}/{output_json_relative_path}")

                # Ensure the specific output subfolder exists for this file (needs connection)
                conn_output_check = create_nas_connection()
                if not conn_output_check:
                     print(f"   [ERROR] Failed to connect to NAS to create output subfolder for {file_name}. Skipping.")
                     error_count += 1
                     continue
                try: # Wrap ensure_nas_dir_exists in try/finally to ensure connection close
                    if not ensure_nas_dir_exists(conn_output_check, share_name, file_output_subfolder_relative):
                        print(f"   [ERROR] Failed to create output subfolder for {file_name}. Skipping.")
                        error_count += 1
                        continue # Skip to next file
                finally:
                    conn_output_check.close() # Close connection after check/create

                # Download file from NAS to temporary local directory using pysmb helper
                local_file_path = download_from_nas(share_name, input_file_relative_path, temp_dir)
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
                                            'angle': page.angle if hasattr(page, 'angle') else None,
                                            'spans': []  # Add spans information
                                        }
                                        # Extract spans information if available
                                        if hasattr(page, 'spans') and page.spans:
                                            for span in page.spans:
                                                span_data = {
                                                    'offset': span.offset if hasattr(span, 'offset') else None,
                                                    'length': span.length if hasattr(span, 'length') else None
                                                }
                                                page_data['spans'].append(span_data)
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
                                    'angle': page.angle if hasattr(page, 'angle') else None,
                                    'spans': []  # Add spans information
                                }
                                # Extract spans information if available
                                if hasattr(page, 'spans') and page.spans:
                                    for span in page.spans:
                                        span_data = {
                                            'offset': span.offset if hasattr(span, 'offset') else None,
                                            'length': span.length if hasattr(span, 'length') else None
                                        }
                                        page_data['spans'].append(span_data)
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
                    if not write_to_nas(share_name, output_md_relative_path, md_bytes):
                        print(f"   [ERROR] Failed to write Markdown file for {file_name} to NAS.")
                        file_has_error = True # Mark error even if DI worked

                if results_json_list and not file_has_error:
                    print(f"   Saving analysis JSON output(s) to NAS...")
                    # Simple JSON serialization helper
                    def json_serializer(obj):
                        if isinstance(obj, (datetime, timezone)): # Example: handle datetime
                            return obj.isoformat()
                        # Add more type handlers if needed
                        try:
                            return str(obj) # Fallback to string representation
                        except Exception:
                            return None # Or some placeholder for unslizable objects

                    if needs_splitting:
                        # Save JSON for each chunk
                        for chunk_index, result_json in enumerate(results_json_list):
                            chunk_json_relative_path = os.path.join(file_output_subfolder_relative, f"{file_name_base}_chunk_{chunk_index + 1}.json").replace('\\', '/')
                            json_bytes = json.dumps(result_json, indent=4, default=json_serializer).encode('utf-8')
                            if not write_to_nas(share_name, chunk_json_relative_path, json_bytes):
                                print(f"   [ERROR] Failed to write JSON chunk {chunk_index + 1} for {file_name} to NAS.")
                                # Decide if this constitutes a full file error
                    else:
                        # Save single JSON for non-split files
                        json_bytes = json.dumps(results_json_list[0], indent=4, default=json_serializer).encode('utf-8')
                        if not write_to_nas(share_name, output_json_relative_path, json_bytes):
                            print(f"   [ERROR] Failed to write JSON file for {file_name} to NAS.")
                            # Decide if this constitutes a full file error

            except Exception as e:
                print(f"   [ERROR] Unexpected error processing file {file_info.get('file_name', 'N/A')}: {e}")
                file_has_error = True
            finally:
                # --- Cleanup ---
                if local_file_path and os.path.exists(local_file_path):
                    try:
                        os.remove(local_file_path)
                    except OSError as e:
                        print(f"   [WARNING] Failed to remove temporary file {local_file_path}: {e}")
                if local_chunk_paths:
                    for chunk_path in local_chunk_paths:
                         if os.path.exists(chunk_path):
                            try:
                                os.remove(chunk_path)
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

    print(f"--- Stage 2 Completed ---")
    print(f"--- End of Main Processing for Stage 2 ---")

# ==============================================================================
# --- Script Entry Point ---
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print(f"--- Running Stage 2: Process Documents with Document Intelligence (using pysmb) ---")
    print(f"--- Document Source: {DOCUMENT_SOURCE} ---")
    print("="*60 + "\n")

    # --- Setup Custom CA Bundle and Initialize Clients ---
    temp_cert_file_path = None # Store path instead of file object
    original_requests_ca_bundle = os.environ.get('REQUESTS_CA_BUNDLE') # Store original env var value
    original_ssl_cert_file = os.environ.get('SSL_CERT_FILE') # Store original env var value
    original_http_proxy = os.environ.get('HTTP_PROXY') # Store original proxy values
    original_https_proxy = os.environ.get('HTTPS_PROXY')
    proxy_set_by_script = False # Flag to track if we set the proxy vars

    di_client = None # Initialize DI client
    # http_transport = None # Removed httpx transport variable
    initialization_error = False # Flag to track if setup fails
    should_skip = False # Flag for skipping based on Stage 1 flag

    try:
        # --- Download and Set Custom CA Bundle ---
        print("[1] Setting up Custom CA Bundle...")
        share_name = NAS_PARAMS["share"]
        # Construct relative path to CA bundle within the NAS_OUTPUT_FOLDER_PATH
        # Assuming bundle is at the root of NAS_OUTPUT_FOLDER_PATH, adjust if needed
        ca_bundle_relative_path = os.path.join(NAS_OUTPUT_FOLDER_PATH, CA_BUNDLE_FILENAME).replace('\\', '/')
        print(f"   Looking for CA bundle at: {share_name}/{ca_bundle_relative_path}")

        # Read CA bundle content from NAS using pysmb helper
        ca_bundle_bytes = read_from_nas(share_name, ca_bundle_relative_path)

        if ca_bundle_bytes:
            # Create a temporary file to store the certificate
            with tempfile.NamedTemporaryFile(delete=False, suffix=".cer", mode='wb') as temp_cert_file:
                temp_cert_file.write(ca_bundle_bytes)
                temp_cert_file_path = temp_cert_file.name # Store the path for cleanup
                print(f"   Downloaded CA bundle to temporary file: {temp_cert_file_path}")

            # Set the environment variables
            os.environ['REQUESTS_CA_BUNDLE'] = temp_cert_file_path
            os.environ['SSL_CERT_FILE'] = temp_cert_file_path
            print(f"   Set REQUESTS_CA_BUNDLE environment variable.")
            print(f"   Set SSL_CERT_FILE environment variable.")
        else:
            print(f"   [WARNING] CA Bundle file not found or could not be read from {share_name}/{ca_bundle_relative_path}. Proceeding without custom CA bundle.")

        # --- Set Proxy Environment Variables (if configured) ---
        if PROXY_CONFIG.get("use_proxy"):
            print("[2] Setting up Proxy Configuration...")
            proxy_url_base = PROXY_CONFIG.get("url")
            proxy_user = PROXY_CONFIG.get("username")
            proxy_pass = PROXY_CONFIG.get("password")
            proxy_string = proxy_url_base # Start with base URL

            if proxy_user and proxy_pass and proxy_url_base:
                # Construct URL with authentication if user/pass provided
                protocol, rest = proxy_url_base.split("://", 1)
                proxy_string = f"{protocol}://{proxy_user}:{proxy_pass}@{rest}"
                print("   Configuring proxy with authentication.")
            elif proxy_url_base:
                print("   Configuring proxy without authentication.")
            else:
                print("   [WARNING] Proxy use enabled but proxy URL is not configured. Skipping proxy setup.")
                proxy_string = None

            # Restore setting environment variables
            if proxy_string:
                os.environ['HTTP_PROXY'] = proxy_string
                os.environ['HTTPS_PROXY'] = proxy_string
                proxy_set_by_script = True
                print(f"   Set HTTP_PROXY and HTTPS_PROXY environment variables.")
        else:
            print("[2] Proxy configuration disabled. Skipping proxy setup.")

        # --- Initialize DI Client (Relying on Environment Variables) ---
        # Removed explicit httpx transport initialization
        print("[3] Initializing Document Intelligence Client...") # Renumbered
        try:
            # Initialize DI client directly. It should pick up the proxy and CA bundle
            # environment variables set earlier (HTTPS_PROXY, SSL_CERT_FILE/REQUESTS_CA_BUNDLE).
            # Do NOT disable SSL verification here (remove connection_verify=False).
            di_client = DocumentIntelligenceClient(
                endpoint=AZURE_DI_ENDPOINT,
                credential=AzureKeyCredential(AZURE_DI_KEY)
                # No transport argument, let the SDK handle it.
                # No connection_verify=False, let it use the CA bundle from env vars.
            )
            print("   Document Intelligence client initialized (relying on environment variables for proxy/CA).")

        except Exception as e:
            print(f"[CRITICAL ERROR] Failed to initialize Document Intelligence client: {e}")
            initialization_error = True # Set flag
            # Don't exit yet, let finally block run

        if not initialization_error: # Proceed only if DI client initialized
            print("-" * 60)

            # --- Define Paths (Relative to Share) ---
            print("[4] Defining NAS Paths (Relative)...") # Renumbered
            # share_name is already defined
            # Base output directory from Stage 1 (relative to share)
            stage1_output_dir_relative = os.path.join(NAS_OUTPUT_FOLDER_PATH, DOCUMENT_SOURCE).replace('\\', '/')
            # Input JSON file from Stage 1 (relative to share)
            files_to_process_json_relative = os.path.join(stage1_output_dir_relative, '1C_nas_files_to_process.json').replace('\\', '/')
            # Base output directory for Stage 2 results (relative to share)
            stage2_output_dir_relative = os.path.join(stage1_output_dir_relative, '2A_processed_files').replace('\\', '/')

            print(f"   Stage 1 Output Dir (Relative): {share_name}/{stage1_output_dir_relative}")
            print(f"   Input JSON File (Relative): {share_name}/{files_to_process_json_relative}")
            print(f"   Stage 2 Output Base Dir (Relative): {share_name}/{stage2_output_dir_relative}")

            # Ensure base Stage 2 output directory exists using pysmb helper
            conn_base_check = create_nas_connection()
            if not conn_base_check:
                 print("[CRITICAL ERROR] Failed to connect to NAS to check/create base Stage 2 output directory.")
                 initialization_error = True # Set flag
            else:
                try:
                    if not ensure_nas_dir_exists(conn_base_check, share_name, stage2_output_dir_relative):
                        print("[CRITICAL ERROR] Could not create base Stage 2 output directory on NAS.")
                        initialization_error = True # Set flag
                    else:
                        print(f"   Base Stage 2 output directory ensured.")
                finally:
                    conn_base_check.close() # Ensure connection is closed
            print("-" * 60)

            if not initialization_error: # Proceed only if paths defined and dir ensured
                # --- Check for Skip Flag from Stage 1 ---
                print("[5] Checking for skip flag from Stage 1...") # Renumbered
                skip_flag_file_name = '_SKIP_SUBSEQUENT_STAGES.flag'
                skip_flag_relative_path = os.path.join(stage1_output_dir_relative, skip_flag_file_name).replace('\\', '/')
                print(f"   Checking for flag file: {share_name}/{skip_flag_relative_path}")
                try:
                    # Use pysmb helper to check existence
                    if check_nas_path_exists(share_name, skip_flag_relative_path):
                        print(f"   Skip flag file found. Stage 1 indicated no files to process.")
                        should_skip = True
                    else:
                        print(f"   Skip flag file not found. Proceeding with Stage 2.")
                except Exception as e:
                    # check_nas_path_exists already logs warnings for non-"not found" errors
                    print(f"   Proceeding with Stage 2 despite potential error checking skip flag.")
                    # Continue execution if flag check fails unexpectedly
                print("-" * 60)

                # --- Execute Main Processing if Not Skipped ---
                if should_skip:
                    print("\n" + "="*60)
                    print(f"--- Stage 2 Skipped (No files to process from Stage 1) ---")
                    print("="*60 + "\n")
                else:
                    # Call the main processing function only if not skipping and DI client is valid
                    main_processing_stage2(di_client, files_to_process_json_relative, stage2_output_dir_relative)

    # --- Cleanup (Executes regardless of success/failure in the try block) ---
    finally:
        print("\n--- Cleaning up Stage 2 ---")

        # Close the httpx transport - No longer needed as transport is not explicitly created
        # if http_transport:
        #     try:
        #         http_transport.close()
        #         print("   Closed httpx transport.")
        #     except Exception as e:
        #         print(f"   [WARNING] Error closing httpx transport: {e}")

        # Clean up the temporary certificate file
        if temp_cert_file_path and os.path.exists(temp_cert_file_path):
            try:
                os.remove(temp_cert_file_path)
                print(f"   Removed temporary CA bundle file: {temp_cert_file_path}")
            except OSError as e:
                 print(f"   [WARNING] Failed to remove temporary CA bundle file {temp_cert_file_path}: {e}")

        # Restore original environment variables
        # Restore REQUESTS_CA_BUNDLE
        current_requests_bundle = os.environ.get('REQUESTS_CA_BUNDLE')
        if original_requests_ca_bundle is None:
            # If it didn't exist originally, remove it if we set it
            if current_requests_bundle == temp_cert_file_path:
                 print("   Unsetting REQUESTS_CA_BUNDLE environment variable.")
                 if 'REQUESTS_CA_BUNDLE' in os.environ:
                     del os.environ['REQUESTS_CA_BUNDLE']
        else:
            # If it existed originally, restore its value if it changed
            if current_requests_bundle != original_requests_ca_bundle:
                 print(f"   Restoring original REQUESTS_CA_BUNDLE environment variable.")
                 os.environ['REQUESTS_CA_BUNDLE'] = original_requests_ca_bundle

        # Restore SSL_CERT_FILE
        current_ssl_cert = os.environ.get('SSL_CERT_FILE')
        if original_ssl_cert_file is None:
            # If it didn't exist originally, remove it if we set it
            if current_ssl_cert == temp_cert_file_path:
                 print("   Unsetting SSL_CERT_FILE environment variable.")
                 if 'SSL_CERT_FILE' in os.environ:
                     del os.environ['SSL_CERT_FILE']
        else:
            # If it existed originally, restore its value if it changed
            if current_ssl_cert != original_ssl_cert_file:
                 print(f"   Restoring original SSL_CERT_FILE environment variable.")
                 os.environ['SSL_CERT_FILE'] = original_ssl_cert_file

        # Restore Proxy Variables if they were set by the script
        if proxy_set_by_script:
            print("   Restoring original proxy environment variables...")
            # Restore HTTP_PROXY
            if original_http_proxy is None:
                if 'HTTP_PROXY' in os.environ:
                    del os.environ['HTTP_PROXY']
                    print("   Unset HTTP_PROXY.")
            else:
                os.environ['HTTP_PROXY'] = original_http_proxy
                print("   Restored original HTTP_PROXY.")
            # Restore HTTPS_PROXY
            if original_https_proxy is None:
                if 'HTTPS_PROXY' in os.environ:
                    del os.environ['HTTPS_PROXY']
                    print("   Unset HTTPS_PROXY.")
            else:
                os.environ['HTTPS_PROXY'] = original_https_proxy
                print("   Restored original HTTPS_PROXY.")

        print("--- Cleanup Complete ---")

    # Exit with error code if initialization failed
    if initialization_error:
        sys.exit(1)

    # Script ends after finally block
