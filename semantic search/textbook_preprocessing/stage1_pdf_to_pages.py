# -*- coding: utf-8 -*-
"""
Stage 1: PDF to Pages Processing

Purpose:
Processes PDF files from NAS storage by splitting them page-by-page and using 
Azure Document Intelligence to extract markdown content for each page.
Outputs a simple JSON structure with all pages for later chapter organization.

Input: PDF files in document source subfolders on NAS
Output: JSON file per document with page-by-page markdown content

Based on catalog search pipeline style and approach.
"""

import os
import sys
import json
import tempfile
import time
import socket
from smb.SMBConnection import SMBConnection
from smb import smb_structs
import io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# --- Dependencies Check ---
try:
    from pypdf import PdfReader, PdfWriter
except ImportError:
    try:
        from PyPDF2 import PdfReader, PdfWriter
        print("INFO: Using PyPDF2 instead of pypdf. Consider upgrading to pypdf.")
    except ImportError:
        print("ERROR: Neither pypdf nor PyPDF2 found. Install with: pip install pypdf")
        sys.exit(1)

try:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential
except ImportError:
    print("ERROR: Azure Document Intelligence library not found. Install with: pip install azure-ai-documentintelligence")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x
    print("INFO: tqdm not installed. Progress bars disabled. `pip install tqdm`")

# ==============================================================================
# Configuration
# ==============================================================================

# --- NAS Configuration ---
NAS_PARAMS = {
        "ip": "your_nas_ip",
        "share": "your_share_name", 
        "user": "your_nas_user",
        "password": "your_nas_password",
        "port": 445
}

# Base path on the NAS share containing document source subfolders
NAS_BASE_INPUT_PATH = "path/to/your/base_input_folder"
# Base path on the NAS share where output JSON files will be stored
NAS_OUTPUT_FOLDER_PATH = "path/to/your/output_folder"

# --- Document Sources Configuration ---
# Document sources to process - each line contains source name
DOCUMENT_SOURCES = """
external_ey
external_ias
# external_ifrs
external_ifric
external_sic
# external_pwc
"""

def load_document_sources():
        """Parse document sources configuration."""
        sources = []
        for line in DOCUMENT_SOURCES.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                sources.append(line)
        return sources

# --- Azure Document Intelligence Configuration ---
AZURE_DI_ENDPOINT = os.environ.get("AZURE_DI_ENDPOINT", "https://your-di-endpoint.cognitiveservices.azure.com/")
AZURE_DI_KEY = os.environ.get("AZURE_DI_KEY", "your-di-key")

# --- Processing Configuration ---
MAX_CONCURRENT_PAGES = 3  # Number of pages to process concurrently
DI_RETRY_ATTEMPTS = 3
DI_RETRY_DELAY = 5  # seconds

# --- pysmb Configuration ---
smb_structs.SUPPORT_SMB2 = True
smb_structs.MAX_PAYLOAD_SIZE = 65536
CLIENT_HOSTNAME = socket.gethostname()

# ==============================================================================
# NAS Helper Functions (from catalog search)
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
        print(f"   Successfully connected to NAS: {NAS_PARAMS['ip']}:{NAS_PARAMS['port']} on share '{NAS_PARAMS['share']}'")
        return conn
    except Exception as e:
        print(f"   [ERROR] Exception creating NAS connection: {e}")
        return None

def ensure_nas_dir_exists(conn, share_name, dir_path):
    """Ensures a directory exists on the NAS, creating it if necessary."""
    if not conn:
        print("   [ERROR] Cannot ensure NAS directory: No connection.")
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
                print(f"      Creating directory on NAS: {current_path}")
                conn.createDirectory(share_name, current_path)
        return True
    except Exception as e:
        print(f"   [ERROR] Failed to ensure/create NAS directory '{dir_path}': {e}")
        return False

def list_nas_files(share_name, folder_path):
    """Lists files in a specific directory on NAS."""
    files_list = []
    conn = None
    try:
        conn = create_nas_connection()
        if not conn:
            return None
        
        try:
            items = conn.listPath(share_name, folder_path)
            for item in items:
                if item.filename in ['.', '..']:
                    continue
                if item.filename.startswith('.') or item.filename.startswith('~$'):
                    continue
                if not item.isDirectory:
                    file_path = os.path.join(folder_path, item.filename).replace('\\', '/')
                    files_list.append({
                        'file_name': item.filename,
                        'file_path': file_path,
                        'file_size': item.file_size
                    })
            return files_list
        except Exception as e:
            print(f"   [ERROR] Failed to list files in '{share_name}/{folder_path}': {e}")
            return None
    finally:
        if conn:
            conn.close()

def download_file_from_nas(share_name, file_path, local_path):
    """Downloads a file from NAS to local path."""
    conn = None
    try:
        conn = create_nas_connection()
        if not conn:
            return False
        
        # Ensure local directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        with open(local_path, 'wb') as local_file:
            file_attributes, filesize = conn.retrieveFile(share_name, file_path, local_file)
        
        print(f"   Downloaded {filesize} bytes from NAS: {file_path}")
        return True
    except Exception as e:
        print(f"   [ERROR] Failed to download file from NAS '{share_name}/{file_path}': {e}")
        return False
    finally:
        if conn:
            conn.close()

def write_json_to_nas(share_name, nas_path_relative, data):
    """Writes JSON data to a specified file path on the NAS."""
    conn = None
    print(f"   Attempting to write to NAS path: {share_name}/{nas_path_relative}")
    try:
        conn = create_nas_connection()
        if not conn:
            return False

        # Ensure directory exists
        dir_path = os.path.dirname(nas_path_relative).replace('\\', '/')
        if dir_path and not ensure_nas_dir_exists(conn, share_name, dir_path):
            print(f"   [ERROR] Failed to ensure output directory exists: {dir_path}")
            return False

        json_string = json.dumps(data, indent=4, ensure_ascii=False)
        data_bytes = json_string.encode('utf-8')
        file_obj = io.BytesIO(data_bytes)

        bytes_written = conn.storeFile(share_name, nas_path_relative, file_obj)
        print(f"   Successfully wrote {bytes_written} bytes to: {share_name}/{nas_path_relative}")
        return True
    except Exception as e:
        print(f"   [ERROR] Unexpected error writing to NAS '{share_name}/{nas_path_relative}': {e}")
        return False
    finally:
        if conn:
            conn.close()

# ==============================================================================
# Azure Document Intelligence Functions (from catalog search)
# ==============================================================================

def create_di_client():
    """Creates and returns an Azure Document Intelligence client."""
    try:
        client = DocumentIntelligenceClient(
            endpoint=AZURE_DI_ENDPOINT,
            credential=AzureKeyCredential(AZURE_DI_KEY)
        )
        print(f"   Azure DI client created successfully for endpoint: {AZURE_DI_ENDPOINT}")
        return client
    except Exception as e:
        print(f"   [ERROR] Failed to create Azure DI client: {e}")
        return None

def analyze_document_with_di(di_client, local_file_path, max_retries=DI_RETRY_ATTEMPTS, retry_delay=DI_RETRY_DELAY):
    """Analyzes a document using Azure Document Intelligence with retry logic."""
    last_exception = None
    for attempt in range(max_retries):
        try:
            print(f"   [Attempt {attempt + 1}] Analyzing document with Azure DI: {os.path.basename(local_file_path)}")
            
            with open(local_file_path, "rb") as f:
                poller = di_client.begin_analyze_document(
                    "prebuilt-layout", 
                    analyze_request=f,
                    content_type="application/octet-stream"
                )
            
            result = poller.result()
            print(f"   DI analysis successful on attempt {attempt + 1}.")
            return result
        except Exception as e:
            last_exception = e
            print(f"   [Attempt {attempt + 1} ERROR] DI analysis failed for {local_file_path}: {type(e).__name__} - {e}")
            if attempt < max_retries - 1:
                print(f"      Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"   [ERROR] DI analysis failed after {max_retries} attempts.")
                break

    return None

# ==============================================================================
# PDF Processing Functions (from catalog search)
# ==============================================================================

def extract_individual_pages(local_pdf_path, temp_dir):
    """Extracts each page of a PDF as individual PDF files."""
    page_paths = []
    base_name = os.path.splitext(os.path.basename(local_pdf_path))[0]
    print(f"   Extracting individual pages from PDF: {os.path.basename(local_pdf_path)}...")
    try:
        reader = PdfReader(local_pdf_path)
        total_pages = len(reader.pages)
        print(f"   Total pages to extract: {total_pages}")
        
        for page_num in range(total_pages):
            writer = PdfWriter()
            writer.add_page(reader.pages[page_num])
            
            page_filename = f"{base_name}_page_{page_num + 1}.pdf"
            page_path = os.path.join(temp_dir, page_filename)
            
            with open(page_path, "wb") as page_file:
                writer.write(page_file)
            
            page_paths.append({
                'page_number': page_num + 1,
                'file_path': page_path
            })
            
        print(f"   Successfully extracted {len(page_paths)} individual pages.")
        return page_paths
    except Exception as e:
        print(f"   [ERROR] Failed to extract pages from PDF {local_pdf_path}: {e}")
        return []

def process_single_page(di_client, page_info, max_retries=DI_RETRY_ATTEMPTS, retry_delay=DI_RETRY_DELAY):
    """Processes a single PDF page using Azure Document Intelligence."""
    page_num = page_info['page_number']
    page_path = page_info['file_path']
    
    try:
        # Process this individual page
        analyze_result = analyze_document_with_di(di_client, page_path, max_retries=max_retries, retry_delay=retry_delay)
        
        if analyze_result and analyze_result.content:
            # Clean up Azure DI page markers since we track page numbers separately
            content = analyze_result.content
            
            # Remove common Azure DI page markers (same as catalog search)
            import re
            content = re.sub(r'<!--\s*PageNumber:\s*\d+\s*-->', '', content)
            content = re.sub(r'<!--\s*PageFooter:.*?-->', '', content, flags=re.DOTALL)
            content = re.sub(r'<!--\s*PageHeader:.*?-->', '', content, flags=re.DOTALL)
            content = re.sub(r'<!--\s*PageBreak\s*-->', '', content)
            
            # Clean up extra whitespace
            content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
            content = content.strip()
            
            return {
                'page_number': page_num,
                'markdown_content': content,
                'success': True
            }
        else:
            print(f"   [ERROR] No content returned from DI for page {page_num}")
            return {
                'page_number': page_num,
                'markdown_content': None,
                'success': False
            }
            
    except Exception as e:
        print(f"   [ERROR] Failed to process page {page_num}: {e}")
        return {
            'page_number': page_num,
            'markdown_content': None,
            'success': False
        }

def process_pages_batch(di_client, page_files, max_workers=MAX_CONCURRENT_PAGES):
    """Processes multiple pages concurrently."""
    print(f"   Processing {len(page_files)} pages with {max_workers} concurrent workers...")
    
    page_results_dict = {}
    results_lock = Lock()
    
    def process_and_store(page_info):
        result = process_single_page(di_client, page_info)
        with results_lock:
            page_results_dict[result['page_number']] = result
        return result
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_page = {executor.submit(process_and_store, page_info): page_info for page_info in page_files}
        
        # Process completed tasks with progress bar
        completed_count = 0
        for future in tqdm(as_completed(future_to_page), total=len(page_files), desc="Processing Pages"):
            try:
                result = future.result()
                completed_count += 1
                if result['success']:
                    print(f"   ✓ Page {result['page_number']} processed successfully")
                else:
                    print(f"   ✗ Page {result['page_number']} failed to process")
            except Exception as e:
                page_info = future_to_page[future]
                print(f"   [ERROR] Exception processing page {page_info['page_number']}: {e}")
    
    # Sort results by page number
    page_results = [page_results_dict[page_num] for page_num in sorted(page_results_dict.keys())]
    
    # Remove the 'success' field from results for output
    for result in page_results:
        result.pop('success', None)
    
    successful_pages = sum(1 for r in page_results if r['markdown_content'] is not None)
    print(f"   Batch processing completed: {successful_pages}/{len(page_files)} pages successful")
    
    return page_results

# ==============================================================================
# Main Processing Functions
# ==============================================================================

def process_pdf_document(document_source, pdf_file_info, di_client):
    """Processes a single PDF document from NAS."""
    file_name = pdf_file_info['file_name']
    file_path = pdf_file_info['file_path']
    
    print(f"\n--- Processing PDF Document: {file_name} ---")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Download PDF from NAS
            local_pdf_path = os.path.join(temp_dir, file_name)
            print(f"   Downloading PDF from NAS...")
            if not download_file_from_nas(NAS_PARAMS["share"], file_path, local_pdf_path):
                print(f"   [ERROR] Failed to download PDF {file_name} from NAS. Skipping.")
                return None
            
            # Get page count
            try:
                reader = PdfReader(local_pdf_path)
                page_count = len(reader.pages)
                print(f"   PDF detected with {page_count} pages. Using page-by-page processing.")
            except Exception as e:
                print(f"   [ERROR] Failed to read PDF {file_name}: {e}. Skipping.")
                return None
            
            # Extract individual pages
            page_files = extract_individual_pages(local_pdf_path, temp_dir)
            if not page_files:
                print(f"   [ERROR] Failed to extract pages from PDF {file_name}. Skipping.")
                return None
            
            # Process all pages with Azure DI
            page_results = process_pages_batch(di_client, page_files, max_workers=MAX_CONCURRENT_PAGES)
            
            # Create structured result
            structured_result = {
                "document_name": file_name,
                "document_source": document_source,
                "total_pages": page_count,
                "pages": page_results
            }
            
            # Check processing success
            failed_pages = [p for p in page_results if p['markdown_content'] is None]
            if failed_pages:
                print(f"   [WARNING] {len(failed_pages)} pages failed to process for {file_name}.")
            else:
                print(f"   ✓ All {page_count} pages processed successfully for {file_name}.")
            
            # Clean up individual page files
            for page_info in page_files:
                try:
                    if os.path.exists(page_info['file_path']):
                        os.remove(page_info['file_path'])
                except OSError:
                    pass
            
            return structured_result
            
        except Exception as e:
            print(f"   [ERROR] Unexpected error processing PDF {file_name}: {e}")
            return None

def run_stage1():
    """Main function to execute Stage 1 processing."""
    print("\n" + "="*60)
    print("--- Starting Stage 1: PDF to Pages Processing ---")
    print("="*60 + "\n")
    
    # Get document sources
    document_sources = load_document_sources()
    print(f"[0] Processing {len(document_sources)} document sources:")
    for source in document_sources:
        print(f"   - {source}")
    print("-" * 60)
    
    # Initialize Azure DI client
    print("\n[1] Initializing Azure Document Intelligence client...")
    di_client = create_di_client()
    if not di_client:
        print("[CRITICAL ERROR] Failed to initialize Azure DI client. Exiting.")
        sys.exit(1)
    print("-" * 60)
    
    # Process each document source
    for document_source in document_sources:
        print(f"\n{'='*60}")
        print(f"Processing Document Source: {document_source}")
        print(f"{'='*60}")
        
        # Construct NAS paths
        nas_input_path = os.path.join(NAS_BASE_INPUT_PATH, document_source).replace('\\', '/')
        nas_output_path = os.path.join(NAS_OUTPUT_FOLDER_PATH, document_source).replace('\\', '/')
        
        print(f"   NAS Input Path: {NAS_PARAMS['share']}/{nas_input_path}")
        print(f"   NAS Output Path: {NAS_PARAMS['share']}/{nas_output_path}")
        
        # List PDF files in source directory
        print(f"\n[2] Listing PDF files in source directory...")
        files_list = list_nas_files(NAS_PARAMS["share"], nas_input_path)
        if files_list is None:
            print(f"   [ERROR] Failed to list files in {document_source}. Skipping.")
            continue
        
        # Filter for PDF files
        pdf_files = [f for f in files_list if f['file_name'].lower().endswith('.pdf')]
        if not pdf_files:
            print(f"   No PDF files found in {document_source}. Skipping.")
            continue
        
        print(f"   Found {len(pdf_files)} PDF files to process:")
        for pdf_file in pdf_files:
            print(f"      - {pdf_file['file_name']} ({pdf_file['file_size']} bytes)")
        
        # Process each PDF file
        print(f"\n[3] Processing PDF files...")
        processed_documents = []
        successful_count = 0
        failed_count = 0
        
        for pdf_file in pdf_files:
            result = process_pdf_document(document_source, pdf_file, di_client)
            if result:
                processed_documents.append(result)
                successful_count += 1
            else:
                failed_count += 1
        
        # Save results to NAS
        if processed_documents:
            print(f"\n[4] Saving results to NAS...")
            output_file_path = os.path.join(nas_output_path, 'stage1_pages_data.json').replace('\\', '/')
            
            # Combine all documents into single output
            combined_output = {
                "document_source": document_source,
                "total_documents": len(processed_documents),
                "documents": processed_documents
            }
            
            if write_json_to_nas(NAS_PARAMS["share"], output_file_path, combined_output):
                print(f"   ✓ Successfully saved results to: {output_file_path}")
            else:
                print(f"   [ERROR] Failed to save results for {document_source}")
        
        # Summary for this document source
        print(f"\n--- Summary for {document_source} ---")
        print(f"PDF files found: {len(pdf_files)}")
        print(f"Successfully processed: {successful_count}")
        print(f"Failed: {failed_count}")
        if processed_documents:
            total_pages = sum(doc['total_pages'] for doc in processed_documents)
            print(f"Total pages processed: {total_pages}")
        print("-" * 60)
    
    print("\n" + "="*60)
    print("--- Stage 1 Completed Successfully ---")
    print("="*60 + "\n")

# ==============================================================================
# Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    run_stage1()