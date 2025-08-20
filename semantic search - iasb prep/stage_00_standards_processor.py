#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IASB Prep: Standards Processor - Azure DI Conversion and JSON Generation

Purpose:
Processes merged IASB standard PDFs through Azure Document Intelligence,
assigns chapter numbers based on standard numbers, and creates stage1_input.json
matching the EY prep output format.

Input: Merged PDFs from stage_00_pdf_merger.py
Output: stage1_input.json with all pages converted to markdown
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

# --- NAS Configuration (matching EY prep) ---
NAS_PARAMS = {
    "ip": "your_nas_ip",  # TODO: Replace with actual NAS IP
    "share": "your_share_name",  # TODO: Replace with actual share name
    "user": "your_nas_user",  # TODO: Replace with actual NAS username
    "password": "your_nas_password",  # TODO: Replace with actual NAS password
    "port": 445  # Default SMB port (can be 139)
}

# --- Directory Paths (Relative to NAS Share) ---
# Input path - merged PDFs from stage_00_pdf_merger
NAS_INPUT_PATH_TEMPLATE = "semantic_search/prep_output/iasb/{standard}/merged"
# Output path for final JSON and chapter PDFs
NAS_OUTPUT_PATH_TEMPLATE = "semantic_search/prep_output/iasb/{standard}"
# Log path
NAS_LOG_PATH_TEMPLATE = "semantic_search/prep_output/iasb/{standard}/logs"

# --- CA Bundle Configuration (matching EY prep) ---
# Path on NAS where the SSL certificate is stored (relative to share root)
NAS_SSL_CERT_PATH = "certificates/rbc-ca-bundle.cer"  # TODO: Adjust to match your NAS location
SSL_LOCAL_PATH = "/tmp/rbc-ca-bundle.cer"  # Temp path for cert

# --- Document Configuration ---
STANDARD_TYPE = "ias"  # TODO: Set to "ias", "ifrs", "ifric", or "sic" for current run
DOCUMENT_ID_TEMPLATE = "{STANDARD}_2024"  # Will be formatted with STANDARD_TYPE.upper()

# --- Azure Document Intelligence Configuration (Hardcoded - matching EY prep) ---
AZURE_DI_ENDPOINT = "YOUR_DI_ENDPOINT"  # TODO: Replace with actual endpoint
AZURE_DI_KEY = "YOUR_DI_KEY"  # TODO: Replace with actual key

# --- Processing Configuration (matching EY prep) ---
MAX_CONCURRENT_PAGES = 5  # Number of pages to process simultaneously
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY = 5  # seconds

# --- pysmb Configuration (matching EY prep) ---
smb_structs.SUPPORT_SMB2 = True
smb_structs.MAX_PAYLOAD_SIZE = 65536
CLIENT_HOSTNAME = socket.gethostname()

# ==============================================================================
# NAS Helper Functions (from EY prep pattern - identical)
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
# Logging Setup (matching EY prep)
# ==============================================================================

def setup_logging(standard_type: str):
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
    
    # Suppress Azure SDK verbose logging (matching EY prep)
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
# PageNumber Tag Extraction and Removal (from EY prep)
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
# PDF Processing Functions
# ==============================================================================

def parse_merged_filename(filename: str) -> Optional[Dict]:
    """
    Parse merged PDF filename to extract standard info.
    Pattern: standard-number-name.pdf
    Example: ias-2-inventories.pdf
    """
    pattern = r'^([a-z]+)-(\d+)-(.+)\.pdf$'
    match = re.match(pattern, filename, re.IGNORECASE)
    
    if not match:
        logging.warning(f"Filename does not match expected pattern: {filename}")
        return None
    
    standard = match.group(1).lower()
    number = int(match.group(2))
    name = match.group(3)
    
    return {
        'filename': filename,
        'standard': standard,
        'number': number,
        'name': name,
        'name_formatted': name.replace('-', ' ').title()
    }

def sort_merged_pdfs(pdf_files: List[str]) -> List[Tuple[str, int, str]]:
    """
    Sort PDF files by standard number and use standard number as chapter number.
    
    Returns:
        List of tuples: (filename, chapter_number, chapter_name)
    """
    parsed_files = []
    
    for filename in pdf_files:
        parsed = parse_merged_filename(filename)
        if parsed:
            parsed_files.append(parsed)
    
    # Sort by standard number
    parsed_files.sort(key=lambda x: x['number'])
    
    # Use actual standard number as chapter number
    result = []
    for parsed in parsed_files:
        # Use the standard number as the chapter number
        chapter_number = parsed['number']
        # Format chapter name: "IAS 2 - Inventories"
        chapter_name = f"{parsed['standard'].upper()} {parsed['number']} - {parsed['name_formatted']}"
        result.append((parsed['filename'], chapter_number, chapter_name))
        logging.info(f"Chapter {chapter_number}: {chapter_name} ({parsed['filename']})")
    
    return result

def extract_individual_pages(local_pdf_path: str, temp_dir: str) -> List[Dict]:
    """Extracts each page of a PDF as individual PDF files (from EY prep)."""
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
    """Analyzes a single page PDF using Azure Document Intelligence (from EY prep)."""
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
                       page_info: Dict,
                       chapter_info: Dict) -> Dict:
    """Processes a single PDF page (modified from EY prep)."""
    page_num = page_info['page_number']
    page_path = page_info['file_path']
    
    try:
        # Analyze the page with Azure DI
        result = analyze_document_with_di(di_client, page_path)
        
        if result and result.content:
            # Extract page reference and clean PageNumber tags
            cleaned_content, page_reference = extract_and_clean_page_number(result.content)
            
            return {
                'success': True,
                'page_number': page_num,
                'page_reference': page_reference,
                'content': cleaned_content,
                'chapter_number': chapter_info['chapter_number'],
                'chapter_name': chapter_info['chapter_name']
            }
        else:
            # Blank page - still create a record to maintain page alignment
            logging.info(f"Page {page_num} appears to be blank (no content from Azure DI)")
            return {
                'success': True,  # Mark as success to include in output
                'page_number': page_num,
                'page_reference': None,
                'content': "",  # Empty content for blank page
                'chapter_number': chapter_info['chapter_number'],
                'chapter_name': chapter_info['chapter_name'],
                'is_blank': True  # Flag to indicate blank page
            }
            
    except Exception as e:
        logging.error(f"Exception processing page {page_num}: {e}")
        return {
            'success': False,
            'page_number': page_num,
            'error': str(e)
        }

def process_chapter_pdf(di_client: DocumentIntelligenceClient,
                       local_pdf_path: str,
                       chapter_number: int,
                       chapter_name: str,
                       temp_dir: str) -> List[Dict]:
    """Process all pages of a chapter PDF."""
    logging.info(f"Processing Chapter {chapter_number}: {chapter_name}")
    
    # Extract individual pages
    page_files = extract_individual_pages(local_pdf_path, temp_dir)
    
    if not page_files:
        logging.error(f"Failed to extract pages from {os.path.basename(local_pdf_path)}")
        return []
    
    chapter_info = {
        'chapter_number': chapter_number,
        'chapter_name': chapter_name,
        'source_filename': os.path.basename(local_pdf_path)
    }
    
    # Process pages concurrently
    all_results = []
    failed_pages = []
    blank_pages = []
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PAGES) as executor:
        # Submit all page processing tasks
        future_to_page = {
            executor.submit(process_single_page, di_client, page_info, chapter_info): page_info
            for page_info in page_files
        }
        
        # Collect results as they complete
        for future in tqdm(as_completed(future_to_page), 
                          total=len(page_files), 
                          desc=f"Chapter {chapter_number}"):
            page_info = future_to_page[future]
            
            try:
                result = future.result()
                if result['success']:
                    all_results.append(result)
                    if result.get('is_blank'):
                        blank_pages.append(page_info['page_number'])
                else:
                    failed_pages.append(page_info['page_number'])
                    logging.warning(f"Failed page {page_info['page_number']}: {result.get('error')}")
            except Exception as e:
                failed_pages.append(page_info['page_number'])
                logging.error(f"Exception for page {page_info['page_number']}: {e}")
    
    # Clean up temp page files
    for page_info in page_files:
        try:
            if os.path.exists(page_info['file_path']):
                os.remove(page_info['file_path'])
        except OSError:
            pass
    
    if blank_pages:
        logging.info(f"Chapter {chapter_number}: {len(blank_pages)} blank pages detected: {blank_pages}")
    
    if failed_pages:
        logging.warning(f"Chapter {chapter_number}: {len(failed_pages)} failed pages: {failed_pages}")
    
    # Sort results by page number
    all_results.sort(key=lambda x: x['page_number'])
    
    logging.info(f"Chapter {chapter_number} complete: {len(all_results)} pages ({len(blank_pages)} blank, {len(failed_pages)} failed)")
    
    return all_results

def create_chapter_pdf(reader: PdfReader, output_path: str) -> bool:
    """Create a chapter PDF file (for compatibility with EY format)."""
    try:
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        
        with open(output_path, 'wb') as f:
            writer.write(f)
        return True
    except Exception as e:
        logging.error(f"Failed to create chapter PDF: {e}")
        return False

# ==============================================================================
# Main Processing Function
# ==============================================================================

def process_all_standards(standard_type: str):
    """Main function to process all merged PDFs for a standard type."""
    share_name = NAS_PARAMS["share"]
    document_id = DOCUMENT_ID_TEMPLATE.format(STANDARD=standard_type.upper())
    
    # Format paths
    nas_input_path = NAS_INPUT_PATH_TEMPLATE.format(standard=standard_type)
    nas_output_path = NAS_OUTPUT_PATH_TEMPLATE.format(standard=standard_type)
    
    logging.info(f"Document ID: {document_id}")
    logging.info(f"Looking for merged PDFs in: {share_name}/{nas_input_path}")
    
    # List merged PDFs from NAS
    nas_files = list_nas_directory(share_name, nas_input_path)
    pdf_files = [f.filename for f in nas_files 
                 if f.filename.lower().endswith('.pdf') and not f.isDirectory]
    
    if not pdf_files:
        logging.error(f"No PDF files found in {share_name}/{nas_input_path}")
        return
    
    logging.info(f"Found {len(pdf_files)} merged PDFs to process")
    
    # Sort PDFs and assign chapter numbers
    sorted_chapters = sort_merged_pdfs(pdf_files)
    
    if not sorted_chapters:
        logging.error("No valid PDFs to process after sorting")
        return
    
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
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info(f"Using temporary directory: {temp_dir}")
        
        # Create output subdirectory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = f"chapters_{timestamp}"
        output_dir_relative = os.path.join(nas_output_path, output_folder).replace('\\', '/')
        
        # Collect all JSON records
        all_json_records = []
        
        # Process each chapter
        for filename, chapter_number, chapter_name in sorted_chapters:
            logging.info(f"\nProcessing {filename} as Chapter {chapter_number}")
            
            # Download PDF from NAS
            nas_file_path = os.path.join(nas_input_path, filename).replace('\\', '/')
            local_pdf_path = download_from_nas(share_name, nas_file_path, temp_dir)
            
            if not local_pdf_path:
                logging.error(f"Failed to download {filename}")
                continue
            
            # Process through Azure DI
            chapter_results = process_chapter_pdf(
                di_client, 
                local_pdf_path, 
                chapter_number, 
                chapter_name,
                temp_dir
            )
            
            if not chapter_results:
                logging.warning(f"No results for {filename}")
                continue
            
            # Create chapter PDF filename (matching EY format)
            parsed = parse_merged_filename(filename)
            if parsed:
                # Format: 01_ias_2_inventories.pdf
                chapter_pdf_name = f"{chapter_number:02d}_{parsed['standard']}_{parsed['number']}_{parsed['name']}.pdf"
            else:
                chapter_pdf_name = f"{chapter_number:02d}_{filename}"
            
            chapter_pdf_path = os.path.join(output_dir_relative, chapter_pdf_name).replace('\\', '/')
            
            # Create JSON records for this chapter (including blank pages for alignment)
            for page_idx, page_result in enumerate(chapter_results, start=1):
                record = {
                    'document_id': document_id,
                    'filename': chapter_pdf_name,
                    'filepath': chapter_pdf_path,
                    'page_number': page_idx,  # Sequential within chapter
                    'page_reference': page_result.get('page_reference'),
                    'content': page_result.get('content', ''),  # Empty string for blank pages
                    'chapter_number': chapter_number,
                    'chapter_name': chapter_name,
                    'source_filename': filename,  # Original merged PDF
                    'source_page_number': page_result['page_number']  # Page in merged PDF
                }
                
                # Optionally include blank page flag
                if page_result.get('is_blank'):
                    record['is_blank'] = True
                    
                all_json_records.append(record)
            
            # Upload chapter PDF to NAS (copy of merged PDF with new name)
            try:
                with open(local_pdf_path, 'rb') as f:
                    pdf_bytes = f.read()
                
                if write_to_nas(share_name, chapter_pdf_path, pdf_bytes):
                    logging.info(f"Uploaded chapter PDF: {chapter_pdf_name}")
                else:
                    logging.error(f"Failed to upload chapter PDF: {chapter_pdf_name}")
            except Exception as e:
                logging.error(f"Error uploading chapter PDF: {e}")
            
            # Clean up local PDF
            try:
                if os.path.exists(local_pdf_path):
                    os.remove(local_pdf_path)
            except OSError:
                pass
        
        # Create and upload stage1_input.json
        if all_json_records:
            logging.info(f"\nCreating stage1_input.json with {len(all_json_records)} records")
            
            # Create JSON in temp directory
            temp_json_path = os.path.join(temp_dir, "stage1_input.json")
            with open(temp_json_path, 'w', encoding='utf-8') as f:
                json.dump(all_json_records, f, indent=2, ensure_ascii=False)
            
            # Upload to NAS
            json_nas_path = os.path.join(output_dir_relative, "stage1_input.json").replace('\\', '/')
            
            try:
                with open(temp_json_path, 'rb') as f:
                    json_bytes = f.read()
                
                if write_to_nas(share_name, json_nas_path, json_bytes):
                    logging.info(f"Successfully uploaded stage1_input.json")
                    logging.info(f"Output location: {share_name}/{output_dir_relative}")
                else:
                    logging.error("Failed to upload stage1_input.json")
            except Exception as e:
                logging.error(f"Error uploading JSON: {e}")
        else:
            logging.error("No records to write to stage1_input.json")

# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    """Main entry point."""
    # Setup logging
    temp_log_path = setup_logging(STANDARD_TYPE)
    
    logging.info("=" * 80)
    logging.info(f"Starting IASB Standards Processor for {STANDARD_TYPE.upper()}")
    logging.info("=" * 80)
    
    try:
        # Process all standards
        process_all_standards(STANDARD_TYPE)
        
    except Exception as e:
        logging.error(f"Unexpected error in main processing: {e}", exc_info=True)
    
    # Upload log file to NAS
    try:
        share_name = NAS_PARAMS["share"]
        nas_log_path = NAS_LOG_PATH_TEMPLATE.format(standard=STANDARD_TYPE)
        log_file_name = f"{STANDARD_TYPE}_processor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path_relative = os.path.join(nas_log_path, log_file_name).replace('\\', '/')
        
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
    
    print("=" * 80)
    print(f"IASB Standards Processor completed for {STANDARD_TYPE.upper()}")
    print("Check logs for details")
    print("=" * 80)

if __name__ == "__main__":
    main()