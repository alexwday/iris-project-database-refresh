# -*- coding: utf-8 -*-
"""
EY Prep: PDF to Markdown Conversion for EY Documents

Purpose:
Processes a single EY PDF file from a NAS directory and converts each page to markdown
using Azure Document Intelligence. Outputs a flat JSON array with document metadata
and page content, removing only PageNumber tags while preserving other Azure tags.

Input: Single PDF file in NAS_INPUT_PATH on the NAS drive (errors if multiple PDFs)
Output: Flat JSON array on the NAS containing page-by-page records
        (e.g., 'semantic_search/prep_output/ey/ey_prep_output.json')
"""

import os
import json
import time
import re
import warnings
import logging
import tempfile
import socket
import io
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- pysmb imports for NAS access ---
from smb.SMBConnection import SMBConnection
from smb import smb_structs

# PDF handling
from pypdf import PdfReader, PdfWriter

# Azure Document Intelligence
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentContentFormat

# Dependencies check
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x
    print("INFO: tqdm not installed. Progress bars disabled. `pip install tqdm`")

# Suppress common PDF warnings
warnings.filterwarnings("ignore", message=".*wrong pointing object.*")
warnings.filterwarnings("ignore", message=".*invalid pdf header.*")
warnings.filterwarnings("ignore", message=".*PdfReadWarning.*")

# Configure pypdf logging to reduce noise
logging.getLogger("pypdf").setLevel(logging.ERROR)

# ==============================================================================
# Configuration (Hardcoded - update these values)
# ==============================================================================

# --- NAS Configuration ---
NAS_PARAMS = {
    "ip": "your_nas_ip",  # TODO: Replace with actual NAS IP
    "share": "your_share_name",  # TODO: Replace with actual share name
    "user": "your_nas_user",  # TODO: Replace with actual NAS username
    "password": "your_nas_password",  # TODO: Replace with actual NAS password
    "port": 445  # Default SMB port (can be 139)
}

# --- Directory Paths (Relative to NAS Share) ---
# Path on NAS where EY PDF file is stored (relative to share root)
NAS_INPUT_PATH = "semantic_search/source_documents/ey"  # TODO: Adjust to your EY PDF location
# Path on NAS where output will be saved (relative to share root)
NAS_OUTPUT_PATH = "semantic_search/prep_output/ey"
# Path on NAS where logs will be saved
NAS_LOG_PATH = "semantic_search/prep_output/ey/logs"
OUTPUT_FILENAME = "ey_prep_output.json"

# --- CA Bundle Configuration ---
# Path on NAS where the SSL certificate is stored (relative to share root)
NAS_SSL_CERT_PATH = "certificates/rbc-ca-bundle.cer"  # TODO: Adjust to match your NAS location
SSL_LOCAL_PATH = "/tmp/rbc-ca-bundle.cer"  # Temp path for cert

# --- Document Configuration ---
DOCUMENT_ID = "EY_GUIDE_2024"  # TODO: Set appropriate document ID for this EY document

# --- Azure Document Intelligence Configuration (Hardcoded) ---
AZURE_DI_ENDPOINT = "YOUR_DI_ENDPOINT"  # TODO: Replace with actual endpoint
AZURE_DI_KEY = "YOUR_DI_KEY"  # TODO: Replace with actual key

# --- Processing Configuration ---
MAX_CONCURRENT_PAGES = 5  # Number of pages to process simultaneously
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY = 5  # seconds

# --- pysmb Configuration ---
smb_structs.SUPPORT_SMB2 = True
smb_structs.MAX_PAYLOAD_SIZE = 65536
CLIENT_HOSTNAME = socket.gethostname()

# ==============================================================================
# NAS Helper Functions (from Stage 1 pattern)
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
            logging.error("Failed to connect to NAS.")
            return None
        logging.debug(f"Successfully connected to NAS: {NAS_PARAMS['ip']}:{NAS_PARAMS['port']}")
        return conn
    except Exception as e:
        logging.error(f"Exception creating NAS connection: {e}")
        return None

def ensure_nas_dir_exists(conn, share_name, dir_path_relative):
    """Ensures a directory exists on the NAS, creating it if necessary."""
    if not conn:
        logging.error("Cannot ensure NAS directory: No connection.")
        return False
    
    path_parts = dir_path_relative.strip('/').split('/')
    current_path = ''
    try:
        for part in path_parts:
            if not part: continue
            current_path = os.path.join(current_path, part).replace('\\', '/')
            try:
                conn.listPath(share_name, current_path)
                logging.debug(f"Directory exists: {current_path}")
            except Exception:
                logging.info(f"Creating directory on NAS: {share_name}/{current_path}")
                conn.createDirectory(share_name, current_path)
        return True
    except Exception as e:
        logging.error(f"Failed to ensure/create NAS directory '{share_name}/{dir_path_relative}': {e}")
        return False

def write_to_nas(share_name, nas_path_relative, content_bytes):
    """Writes bytes to a file path on the NAS using pysmb."""
    conn = None
    logging.info(f"Attempting to write to NAS path: {share_name}/{nas_path_relative}")
    try:
        conn = create_nas_connection()
        if not conn:
            return False

        dir_path = os.path.dirname(nas_path_relative).replace('\\', '/')
        if dir_path and not ensure_nas_dir_exists(conn, share_name, dir_path):
            logging.error(f"Failed to ensure output directory exists: {dir_path}")
            return False

        file_obj = io.BytesIO(content_bytes)
        bytes_written = conn.storeFile(share_name, nas_path_relative, file_obj)
        logging.info(f"Successfully wrote {bytes_written} bytes to: {share_name}/{nas_path_relative}")
        return True
    except Exception as e:
        logging.error(f"Unexpected error writing to NAS '{share_name}/{nas_path_relative}': {e}")
        return False
    finally:
        if conn:
            conn.close()

def read_from_nas(share_name, nas_path_relative):
    """Reads content (as bytes) from a file path on the NAS using pysmb."""
    conn = None
    logging.debug(f"Attempting to read from NAS path: {share_name}/{nas_path_relative}")
    try:
        conn = create_nas_connection()
        if not conn:
            return None

        file_obj = io.BytesIO()
        file_attributes, filesize = conn.retrieveFile(share_name, nas_path_relative, file_obj)
        file_obj.seek(0)
        content_bytes = file_obj.read()
        logging.debug(f"Successfully read {filesize} bytes from: {share_name}/{nas_path_relative}")
        return content_bytes
    except Exception as e:
        logging.error(f"Unexpected error reading from NAS '{share_name}/{nas_path_relative}': {e}")
        return None
    finally:
        if conn:
            conn.close()

def download_from_nas(share_name, nas_path_relative, local_temp_dir):
    """Downloads a file from NAS to a local temporary directory."""
    local_file_path = os.path.join(local_temp_dir, os.path.basename(nas_path_relative))
    logging.debug(f"Attempting to download from NAS: {share_name}/{nas_path_relative}")
    
    content_bytes = read_from_nas(share_name, nas_path_relative)
    if content_bytes is None:
        return None
    
    try:
        with open(local_file_path, 'wb') as f:
            f.write(content_bytes)
        logging.debug(f"Downloaded to: {local_file_path}")
        return local_file_path
    except Exception as e:
        logging.error(f"Failed to write downloaded file to {local_file_path}: {e}")
        return None

def list_nas_directory(share_name, dir_path_relative):
    """Lists files in a NAS directory."""
    conn = None
    try:
        conn = create_nas_connection()
        if not conn:
            return []
        
        files = conn.listPath(share_name, dir_path_relative)
        # Filter out '.' and '..' entries
        files = [f for f in files if f.filename not in ['.', '..']]
        return files
    except Exception as e:
        logging.error(f"Failed to list NAS directory '{share_name}/{dir_path_relative}': {e}")
        return []
    finally:
        if conn:
            conn.close()

# ==============================================================================
# Logging Setup (Modified for NAS)
# ==============================================================================

def setup_logging():
    """Setup logging to write to NAS."""
    # Create a temporary local log file first
    temp_log = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log')
    temp_log_path = temp_log.name
    temp_log.close()
    
    # Configure logging to write to temp file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler(temp_log_path),
            logging.StreamHandler()
        ]
    )
    
    # Suppress Azure SDK verbose logging
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("azure.core").setLevel(logging.WARNING)
    logging.getLogger("azure.ai.documentintelligence").setLevel(logging.WARNING)
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
    
    # Also suppress urllib3 and requests logging from Azure SDK
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("requests.packages.urllib3").setLevel(logging.WARNING)
    
    # Suppress SMB connection details
    logging.getLogger("SMB").setLevel(logging.WARNING)
    logging.getLogger("SMB.SMBConnection").setLevel(logging.WARNING)
    
    # Log the temp path for later upload
    logging.info(f"Temporary log file: {temp_log_path}")
    return temp_log_path

# ==============================================================================
# PageNumber Tag Extraction and Removal
# ==============================================================================

# Pattern to extract PageNumber value (captures any format including roman numerals, alphanumeric, etc.)
PAGE_NUMBER_EXTRACT_PATTERN = re.compile(
    r'<!--\s*PageNumber\s*[=:]\s*["\']?([^"\'>\s]+)["\']?\s*-->', 
    re.IGNORECASE
)

# Broader pattern to remove ALL PageNumber tags after extraction
PAGE_NUMBER_REMOVE_PATTERN = re.compile(
    r'<!--\s*PageNumber[^>]*-->', 
    re.IGNORECASE
)

def extract_and_clean_page_number(content: str) -> tuple[str, str]:
    """
    Extracts page reference from PageNumber tags and removes them from content.
    Returns: (cleaned_content, page_reference)
    """
    if not content:
        return content, None
    
    # First, extract the page reference value
    page_reference = None
    match = PAGE_NUMBER_EXTRACT_PATTERN.search(content)
    if match:
        page_reference = match.group(1).strip()
        logging.debug(f"Extracted page reference: {page_reference}")
    
    # Remove all PageNumber tags
    cleaned_content = PAGE_NUMBER_REMOVE_PATTERN.sub('', content)
    
    # Clean up extra newlines created by tag removal
    cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)
    
    # Remove trailing/leading whitespace
    cleaned_content = cleaned_content.strip()
    
    return cleaned_content, page_reference

# ==============================================================================
# Utility Functions
# ==============================================================================

def extract_individual_pages(local_pdf_path: str, temp_dir: str) -> List[Dict]:
    """Extracts each page of a PDF as individual PDF files."""
    page_files = []
    base_name = os.path.splitext(os.path.basename(local_pdf_path))[0]
    
    try:
        reader = PdfReader(local_pdf_path)
        total_pages = len(reader.pages)
        logging.info(f"Extracting {total_pages} pages from PDF...")
        
        for page_num in range(total_pages):
            writer = PdfWriter()
            writer.add_page(reader.pages[page_num])
            
            page_filename = f"{base_name}_page_{page_num + 1}.pdf"
            page_path = os.path.join(temp_dir, page_filename)
            
            with open(page_path, "wb") as page_file:
                writer.write(page_file)
            
            page_files.append({
                'page_number': page_num + 1,
                'file_path': page_path,
                'original_pdf': local_pdf_path
            })
        
        logging.info(f"Page extraction complete: {len(page_files)} pages ready for processing")
        return page_files
        
    except Exception as e:
        logging.error(f"Failed to extract pages from {local_pdf_path}: {e}")
        return []

def analyze_document_with_di(di_client: DocumentIntelligenceClient, 
                            page_file_path: str, 
                            max_retries: int = 3, 
                            retry_delay: int = 5) -> Optional[Any]:
    """Analyzes a single page PDF using Azure Document Intelligence."""
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            # Only log if there's a retry
            if attempt > 0:
                logging.debug(f"DI Analysis Retry {attempt + 1}/{max_retries} for {os.path.basename(page_file_path)}")
            
            with open(page_file_path, "rb") as f:
                document_bytes = f.read()
            
            poller = di_client.begin_analyze_document(
                "prebuilt-layout",
                document_bytes,
                output_content_format=DocumentContentFormat.MARKDOWN
            )
            result = poller.result()
            
            return result
            
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                logging.debug(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logging.error(f"DI analysis failed for page after {max_retries} attempts: {e}")
    
    return None

def process_single_page(di_client: DocumentIntelligenceClient, 
                       page_info: Dict) -> Dict:
    """Processes a single PDF page, extracts page reference, and removes PageNumber tags."""
    page_num = page_info['page_number']
    page_path = page_info['file_path']
    original_pdf = page_info['original_pdf']
    
    try:
        # Analyze the page with Azure DI
        result = analyze_document_with_di(di_client, page_path)
        
        if result and result.content:
            # Extract page reference and clean PageNumber tags
            cleaned_content, page_reference = extract_and_clean_page_number(result.content)
            
            return {
                'success': True,
                'page_number': page_num,
                'page_reference': page_reference,  # New field with extracted value
                'content': cleaned_content,
                'original_file': os.path.basename(original_pdf)
            }
        else:
            return {
                'success': False,
                'page_number': page_num,
                'page_reference': None,
                'content': None,
                'original_file': os.path.basename(original_pdf),
                'error': 'No content returned from Azure DI'
            }
            
    except Exception as e:
        logging.error(f"Exception processing page {page_num}: {e}")
        return {
            'success': False,
            'page_number': page_num,
            'page_reference': None,
            'content': None,
            'original_file': os.path.basename(original_pdf),
            'error': str(e)
        }

def process_pages_batch_incremental(di_client: DocumentIntelligenceClient,
                                   page_files: List[Dict],
                                   output_file_path: str,
                                   document_id: str,
                                   filename: str,
                                   filepath: str,
                                   max_workers: int = 5) -> Tuple[int, List[int]]:
    """Processes multiple PDF pages concurrently and writes to JSON incrementally."""
    total_pages = len(page_files)
    logging.info(f"Starting Azure DI processing for {total_pages} pages (max {max_workers} concurrent)")
    logging.info(f"Writing results incrementally to: {output_file_path}")
    
    # Dictionary to store results by page number for ordering
    results_dict = {}
    pages_processed = 0
    pages_written = 0
    failed_pages = []
    
    # Start the JSON array
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write('[\n')
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all page processing tasks
        future_to_page = {
            executor.submit(process_single_page, di_client, page_info): page_info
            for page_info in page_files
        }
        
        # Track the next page number to write (ensures sequential order)
        next_page_to_write = 1
        first_record = True
        
        # Collect results as they complete
        for future in tqdm(as_completed(future_to_page), total=len(page_files), desc="Processing pages"):
            page_info = future_to_page[future]
            pages_processed += 1
            
            try:
                result = future.result()
                page_num = result['page_number']
                
                if not result['success']:
                    failed_pages.append(page_num)
                    logging.warning(f"Page {page_num} failed: {result.get('error', 'Unknown error')}")
                    # Store None for failed pages to maintain sequence
                    results_dict[page_num] = None
                else:
                    # Store the successful result
                    results_dict[page_num] = {
                        'document_id': document_id,
                        'filename': filename,
                        'filepath': filepath,
                        'page_number': page_num,
                        'page_reference': result.get('page_reference'),  # Include extracted page reference
                        'content': result['content']
                    }
                
            except Exception as e:
                page_num = page_info['page_number']
                failed_pages.append(page_num)
                logging.error(f"Exception for page {page_num}: {e}")
                results_dict[page_num] = None
            
            # Write any sequential pages that are ready
            with open(output_file_path, 'a', encoding='utf-8') as f:
                while next_page_to_write in results_dict:
                    page_data = results_dict[next_page_to_write]
                    if page_data is not None:  # Only write successful pages
                        if not first_record:
                            f.write(',\n')
                        json.dump(page_data, f, indent=2, ensure_ascii=False)
                        first_record = False
                        pages_written += 1
                    # Remove from dict to free memory
                    del results_dict[next_page_to_write]
                    next_page_to_write += 1
            
            # Progress update every 100 pages
            if pages_processed % 100 == 0:
                buffered_pages = len(results_dict)
                logging.info(f"Progress: {pages_processed}/{total_pages} processed, {pages_written} written, {buffered_pages} buffered")
                
                # Warning if too many pages are buffered in memory
                if buffered_pages > 500:
                    logging.warning(f"High memory usage: {buffered_pages} pages buffered waiting for sequential write")
    
    # Close the JSON array
    with open(output_file_path, 'a', encoding='utf-8') as f:
        f.write('\n]')
    
    successful = pages_written
    logging.info(f"Azure DI processing completed: {successful}/{total_pages} pages successful")
    logging.info(f"Results written to: {output_file_path}")
    
    if failed_pages:
        logging.warning(f"Failed pages: {failed_pages}")
    
    return successful, failed_pages

def process_pdf_file_incremental(local_pdf_path: str,
                                original_nas_path: str,
                                filename: str,
                                di_client: DocumentIntelligenceClient,
                                temp_dir: str,
                                output_file_path: str,
                                document_id: str) -> Dict:
    """Processes a single PDF file and writes pages incrementally to JSON."""
    logging.info(f"Processing PDF: {filename}")
    
    try:
        # Extract individual pages
        page_files = extract_individual_pages(local_pdf_path, temp_dir)
        
        if not page_files:
            return {
                'filename': filename,
                'filepath': original_nas_path,
                'success': False,
                'error': 'Failed to extract pages',
                'total_pages': 0,
                'successful_pages': 0
            }
        
        # Process all pages and write incrementally
        successful_pages, failed_pages = process_pages_batch_incremental(
            di_client, page_files, output_file_path, 
            document_id, filename, original_nas_path,
            MAX_CONCURRENT_PAGES
        )
        
        # Clean up temp page files
        for page_info in page_files:
            try:
                if os.path.exists(page_info['file_path']):
                    os.remove(page_info['file_path'])
            except OSError:
                pass
        
        return {
            'filename': filename,
            'filepath': original_nas_path,
            'success': True,
            'total_pages': len(page_files),
            'successful_pages': successful_pages,
            'failed_pages': failed_pages
        }
        
    except Exception as e:
        logging.error(f"Error processing PDF {filename}: {e}")
        return {
            'filename': filename,
            'filepath': original_nas_path,
            'success': False,
            'error': str(e),
            'total_pages': 0,
            'successful_pages': 0
        }

# ==============================================================================
# Main Processing Function
# ==============================================================================

def run_ey_prep():
    """Main function to execute EY document preprocessing."""
    # Setup logging
    temp_log_path = setup_logging()
    
    logging.info("--- Starting EY Prep: PDF to Markdown Conversion ---")
    
    share_name = NAS_PARAMS["share"]
    output_path_relative = os.path.join(NAS_OUTPUT_PATH, OUTPUT_FILENAME).replace('\\', '/')
    
    # Find PDF files on NAS
    logging.info(f"Looking for PDF files in NAS path: {share_name}/{NAS_INPUT_PATH}")
    nas_files = list_nas_directory(share_name, NAS_INPUT_PATH)
    
    # Filter for PDF files
    pdf_files = [f for f in nas_files if f.filename.lower().endswith('.pdf') and not f.isDirectory]
    
    if not pdf_files:
        logging.error(f"No PDF files found in {share_name}/{NAS_INPUT_PATH}")
        return
    
    if len(pdf_files) > 1:
        logging.error(f"Error: Multiple PDF files found ({len(pdf_files)}). EY prep expects exactly one PDF file.")
        logging.error(f"Files found: {[f.filename for f in pdf_files]}")
        return
    
    logging.info(f"Found 1 PDF file to process: {pdf_files[0].filename}")
    
    # Initialize Azure DI client
    logging.info("Initializing Azure Document Intelligence client...")
    try:
        di_client = DocumentIntelligenceClient(
            endpoint=AZURE_DI_ENDPOINT,
            credential=AzureKeyCredential(AZURE_DI_KEY)
        )
        logging.info("Azure DI client initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize Azure DI client: {e}")
        return
    
    # Create temporary directory for downloads and processing
    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info(f"Using temporary directory: {temp_dir}")
        
        # Create temp output file
        temp_output_path = os.path.join(temp_dir, OUTPUT_FILENAME)
        
        # Process the single PDF
        pdf_file_info = pdf_files[0]
        filename = pdf_file_info.filename
        nas_file_path = os.path.join(NAS_INPUT_PATH, filename).replace('\\', '/')
        
        logging.info(f"Downloading {filename} from NAS...")
        
        # Download PDF from NAS to temp directory
        local_pdf_path = download_from_nas(share_name, nas_file_path, temp_dir)
        
        if not local_pdf_path:
            logging.error(f"Failed to download {filename} from NAS. Exiting.")
            return
        
        # Process this PDF with incremental writing
        pdf_result = process_pdf_file_incremental(
            local_pdf_path, nas_file_path, filename, 
            di_client, temp_dir, temp_output_path, DOCUMENT_ID
        )
        
        if pdf_result['success']:
            logging.info(f"Successfully processed {pdf_result['filename']}: "
                       f"{pdf_result['successful_pages']}/{pdf_result['total_pages']} pages")
            
            # Upload the completed JSON file to NAS
            logging.info(f"Uploading results to NAS: {share_name}/{output_path_relative}")
            try:
                with open(temp_output_path, 'rb') as f:
                    json_bytes = f.read()
                
                if write_to_nas(share_name, output_path_relative, json_bytes):
                    logging.info(f"Successfully saved output to NAS")
                else:
                    logging.error(f"Failed to save output to NAS")
            except Exception as e:
                logging.error(f"Failed to upload output JSON to NAS: {e}")
        else:
            logging.error(f"Failed to process {pdf_result['filename']}: {pdf_result.get('error', 'Unknown error')}")
        
        # Clean up downloaded PDF
        try:
            if os.path.exists(local_pdf_path):
                os.remove(local_pdf_path)
        except OSError:
            pass
    
    # Upload log file to NAS
    try:
        log_file_name = f"ey_prep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path_relative = os.path.join(NAS_LOG_PATH, log_file_name).replace('\\', '/')
        
        # Close logging handlers to flush content
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)
        
        # Read log content and upload to NAS
        with open(temp_log_path, 'rb') as f:
            log_content = f.read()
        
        if write_to_nas(share_name, log_path_relative, log_content):
            print(f"Log file uploaded to NAS: {share_name}/{log_path_relative}")
        else:
            print(f"Failed to upload log file to NAS")
        
        # Clean up temp log file
        os.remove(temp_log_path)
    except Exception as e:
        print(f"Error handling log file: {e}")
    
    # Final summary
    print("--- EY Prep Summary ---")
    print(f"Document ID: {DOCUMENT_ID}")
    if 'pdf_result' in locals() and pdf_result:
        print(f"PDF file processed: {filename}")
        print(f"Total pages: {pdf_result.get('total_pages', 0)}")
        print(f"Successfully processed: {pdf_result.get('successful_pages', 0)}")
        if pdf_result.get('failed_pages'):
            print(f"Failed pages: {len(pdf_result.get('failed_pages', []))}")
    print(f"Output file: {share_name}/{output_path_relative}")
    print("--- EY Prep Completed ---")

# ==============================================================================
# Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    run_ey_prep()