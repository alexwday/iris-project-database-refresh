#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IASB Prep: PDF Merger for Standards Documents

Purpose:
Merges related PDF files (base, B-, C- prefixed) for IASB standards (IAS, IFRS, IFRIC, SIC)
into consolidated PDFs based on standard number and name.

Input: Multiple PDF files with naming pattern: [prefix-]standard-number-name.pdf
Output: Merged PDFs without prefix in filename
"""

import os
import re
import logging
import tempfile
import socket
import io
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict

# --- pysmb imports for NAS access ---
from smb.SMBConnection import SMBConnection
from smb import smb_structs

# PDF handling
from pypdf import PdfReader, PdfWriter

# Dependencies check
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x
    print("INFO: tqdm not installed. Progress bars disabled. `pip install tqdm`")

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
# Input path template - will be formatted with standard type
NAS_INPUT_PATH_TEMPLATE = "semantic_search/source_documents/iasb/{standard}"  # {standard} = ias, ifrs, ifric, or sic
# Output path template
NAS_OUTPUT_PATH_TEMPLATE = "semantic_search/prep_output/iasb/{standard}/merged"
# Log path template
NAS_LOG_PATH_TEMPLATE = "semantic_search/prep_output/iasb/{standard}/logs"

# --- CA Bundle Configuration (matching EY prep) ---
# Path on NAS where the SSL certificate is stored (relative to share root)
NAS_SSL_CERT_PATH = "certificates/rbc-ca-bundle.cer"  # TODO: Adjust to match your NAS location
SSL_LOCAL_PATH = "/tmp/rbc-ca-bundle.cer"  # Temp path for cert

# --- Processing Configuration ---
STANDARD_TYPE = "ias"  # TODO: Set to "ias", "ifrs", "ifric", or "sic for current run

# --- pysmb Configuration (matching EY prep) ---
smb_structs.SUPPORT_SMB2 = True
smb_structs.MAX_PAYLOAD_SIZE = 65536
CLIENT_HOSTNAME = socket.gethostname()

# ==============================================================================
# NAS Helper Functions (from EY prep pattern)
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
            except Exception:
                conn.createDirectory(share_name, current_path)
        return True
    except Exception as e:
        logging.error(f"Failed to ensure/create NAS directory '{share_name}/{dir_path_relative}': {e}")
        return False

def write_to_nas(share_name, nas_path_relative, content_bytes):
    """Writes bytes to a file path on the NAS using pysmb."""
    conn = None
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
    try:
        conn = create_nas_connection()
        if not conn:
            return None

        file_obj = io.BytesIO()
        file_attributes, filesize = conn.retrieveFile(share_name, nas_path_relative, file_obj)
        file_obj.seek(0)
        content_bytes = file_obj.read()
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
    
    content_bytes = read_from_nas(share_name, nas_path_relative)
    if content_bytes is None:
        return None
    
    try:
        with open(local_file_path, 'wb') as f:
            f.write(content_bytes)
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
# Logging Setup (Modified for NAS - matching EY prep)
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
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(temp_log_path),
            logging.StreamHandler()
        ]
    )
    
    # Suppress all unnecessary logging
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    logging.getLogger("SMB").setLevel(logging.ERROR)
    logging.getLogger("SMB.SMBConnection").setLevel(logging.ERROR)
    
    return temp_log_path

# ==============================================================================
# PDF Merging Functions
# ==============================================================================

def parse_filename(filename: str) -> Optional[Dict]:
    """
    Parse PDF filename to extract components.
    
    Pattern: [prefix-]standard-number-name.pdf
    Example: B-ias-2-inventories.pdf
    
    Returns:
        Dict with keys: prefix, standard, number, name, base_key
    """
    # Pattern to match: optional prefix, standard, number, name
    pattern = r'^(B-|C-)?([a-z]+)-(\d+)-(.+)\.pdf$'
    match = re.match(pattern, filename, re.IGNORECASE)
    
    if not match:
        logging.warning(f"Filename does not match expected pattern: {filename}")
        return None
    
    prefix = match.group(1) or ''  # Empty string if no prefix
    standard = match.group(2).lower()
    number = int(match.group(3))
    name = match.group(4)
    
    # Base key for grouping (without prefix)
    base_key = f"{standard}-{number}-{name}"
    
    return {
        'filename': filename,
        'prefix': prefix.rstrip('-') if prefix else '',  # Remove trailing dash
        'standard': standard,
        'number': number,
        'name': name,
        'base_key': base_key,
        'base_filename': f"{base_key}.pdf"
    }

def group_files_by_standard(pdf_files: List[str], standard_type: str) -> Tuple[Dict[str, List[Dict]], List[str], List[str]]:
    """
    Group PDF files by their standard-number-name combination.
    
    Args:
        pdf_files: List of PDF filenames
        standard_type: The standard type to filter for (ias, ifrs, ifric, sic)
    
    Returns:
        Tuple of:
        - Dictionary mapping base_key to list of parsed file info, sorted by prefix priority
        - List of skipped files (wrong standard)
        - List of invalid files (wrong pattern)
    """
    groups = defaultdict(list)
    skipped_files = []
    invalid_files = []
    
    for filename in pdf_files:
        parsed = parse_filename(filename)
        if not parsed:
            invalid_files.append(filename)
            continue
        
        # Filter for the specified standard type
        if parsed['standard'] != standard_type.lower():
            skipped_files.append(filename)
            continue
        
        groups[parsed['base_key']].append(parsed)
    
    # Sort each group by prefix priority: no prefix (base) -> B -> C
    prefix_priority = {'': 0, 'B': 1, 'C': 2}
    for base_key in groups:
        groups[base_key].sort(key=lambda x: prefix_priority.get(x['prefix'], 999))
    
    return dict(groups), skipped_files, invalid_files

def merge_pdf_group(pdf_paths: List[str], output_path: str) -> Tuple[bool, int]:
    """
    Merge multiple PDFs into a single file in the specified order.
    
    Args:
        pdf_paths: List of full paths to PDFs to merge (in order)
        output_path: Path for the merged output PDF
    
    Returns:
        Tuple of (success, total_pages)
    """
    try:
        writer = PdfWriter()
        total_pages = 0
        
        for pdf_path in pdf_paths:
            reader = PdfReader(pdf_path)
            num_pages = len(reader.pages)
            
            for page_num in range(num_pages):
                writer.add_page(reader.pages[page_num])
                total_pages += 1
        
        # Write merged PDF
        with open(output_path, 'wb') as output_file:
            writer.write(output_file)
        
        return True, total_pages
        
    except Exception as e:
        logging.error(f"Failed to merge PDFs: {e}")
        return False, 0

def process_all_pdfs(nas_input_path: str, nas_output_path: str, standard_type: str):
    """
    Main processing function to merge all PDFs for a given standard type.
    
    Args:
        nas_input_path: NAS input directory path
        nas_output_path: NAS output directory path
        standard_type: Type of standard (ias, ifrs, ifric, sic)
    """
    share_name = NAS_PARAMS["share"]
    
    # List all files in the input directory
    logging.info(f"Scanning input directory: {nas_input_path}")
    nas_files = list_nas_directory(share_name, nas_input_path)
    
    # Filter for PDF files
    pdf_files = [f.filename for f in nas_files 
                 if f.filename.lower().endswith('.pdf') and not f.isDirectory]
    
    if not pdf_files:
        logging.error(f"No PDF files found in input directory")
        return
    
    logging.info(f"Found {len(pdf_files)} PDF files in input directory")
    
    # Group files by standard-number-name
    file_groups, skipped_files, invalid_files = group_files_by_standard(pdf_files, standard_type)
    
    # Log file categorization
    logging.info("=" * 60)
    logging.info(f"File Analysis for {standard_type.upper()} Processing:")
    logging.info(f"  Total input files: {len(pdf_files)}")
    logging.info(f"  {standard_type.upper()} files to process: {sum(len(g) for g in file_groups.values())}")
    logging.info(f"  Standards to merge: {len(file_groups)}")
    
    if skipped_files:
        logging.info(f"  Skipped (wrong standard): {len(skipped_files)}")
        for f in skipped_files[:5]:  # Show first 5
            logging.info(f"    - {f}")
        if len(skipped_files) > 5:
            logging.info(f"    ... and {len(skipped_files) - 5} more")
    
    if invalid_files:
        logging.warning(f"  Invalid filename pattern: {len(invalid_files)}")
        for f in invalid_files:
            logging.warning(f"    - {f}")
    
    logging.info("=" * 60)
    
    if not file_groups:
        logging.error(f"No {standard_type.upper()} files found to merge")
        return
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info(f"Using temporary directory: {temp_dir}")
        
        # Download all PDFs to temp directory
        logging.info("Downloading PDFs from NAS...")
        local_files = {}
        for filename in pdf_files:
            nas_file_path = os.path.join(nas_input_path, filename).replace('\\', '/')
            local_path = download_from_nas(share_name, nas_file_path, temp_dir)
            if local_path:
                local_files[filename] = local_path
            else:
                logging.error(f"Failed to download {filename}")
        
        # Track processing statistics
        total_input_files = sum(len(g) for g in file_groups.values())
        files_processed = []
        files_failed = []
        
        # Process each group
        merged_files = []
        logging.info("\nStarting merge process...")
        
        for base_key, file_infos in tqdm(file_groups.items(), desc="Merging PDF groups"):
            # Get local paths for this group in order
            local_paths = []
            group_files = []
            
            for file_info in file_infos:
                group_files.append(file_info['filename'])
                if file_info['filename'] in local_files:
                    local_paths.append(local_files[file_info['filename']])
                    files_processed.append(file_info['filename'])
                else:
                    logging.error(f"Missing download for: {file_info['filename']}")
                    files_failed.append(file_info['filename'])
            
            if not local_paths:
                logging.error(f"Skipping {base_key}: no files available")
                continue
            
            # Log group details
            prefixes = [f['prefix'] or 'base' for f in file_infos]
            logging.info(f"Merging {base_key}: {len(local_paths)} files ({', '.join(prefixes)})")
            
            # Output filename is the base filename (no prefix)
            output_filename = file_infos[0]['base_filename']
            temp_output_path = os.path.join(temp_dir, output_filename)
            
            # Merge the PDFs
            success, total_pages = merge_pdf_group(local_paths, temp_output_path)
            if success:
                merged_files.append({
                    'filename': output_filename,
                    'local_path': temp_output_path,
                    'file_count': len(local_paths),
                    'source_files': group_files,
                    'standard_number': file_infos[0]['number'],
                    'total_pages': total_pages
                })
                logging.info(f"  → Created {output_filename} ({total_pages} pages)")
            else:
                logging.error(f"  → Failed to merge {base_key}")
                files_failed.extend(group_files)
        
        # Upload merged PDFs to NAS
        logging.info(f"\nUploading {len(merged_files)} merged PDFs to NAS...")
        successful_uploads = 0
        failed_uploads = []
        
        for merged_file in tqdm(merged_files, desc="Uploading"):
            nas_output_file = os.path.join(nas_output_path, merged_file['filename']).replace('\\', '/')
            
            try:
                with open(merged_file['local_path'], 'rb') as f:
                    pdf_bytes = f.read()
                
                if write_to_nas(share_name, nas_output_file, pdf_bytes):
                    successful_uploads += 1
                else:
                    failed_uploads.append(merged_file['filename'])
            except Exception as e:
                logging.error(f"Error uploading {merged_file['filename']}: {e}")
                failed_uploads.append(merged_file['filename'])
        
        # Final validation and summary
        logging.info("\n" + "=" * 60)
        logging.info("PROCESSING SUMMARY")
        logging.info("=" * 60)
        logging.info(f"Input Files:")
        logging.info(f"  Total PDFs found: {len(pdf_files)}")
        logging.info(f"  {standard_type.upper()} files: {total_input_files}")
        logging.info(f"  Files processed: {len(files_processed)}")
        logging.info(f"  Files failed: {len(files_failed)}")
        
        logging.info(f"\nOutput Files:")
        logging.info(f"  Standards merged: {len(merged_files)}")
        logging.info(f"  Successfully uploaded: {successful_uploads}")
        logging.info(f"  Failed uploads: {len(failed_uploads)}")
        
        # Validation check
        if len(files_processed) != total_input_files:
            logging.warning(f"\n⚠️ WARNING: Not all input files were processed!")
            logging.warning(f"   Expected: {total_input_files}, Processed: {len(files_processed)}")
            if files_failed:
                logging.warning(f"   Failed files:")
                for f in files_failed:
                    logging.warning(f"     - {f}")
        else:
            logging.info(f"\n✓ All {total_input_files} input files were successfully processed")
        
        if failed_uploads:
            logging.error(f"\n❌ Failed to upload:")
            for f in failed_uploads:
                logging.error(f"   - {f}")
        
        logging.info("=" * 60)

# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    """Main entry point for PDF merger."""
    # Setup logging
    temp_log_path = setup_logging(STANDARD_TYPE)
    
    print("=" * 60)
    print(f"IASB PDF Merger - {STANDARD_TYPE.upper()} Standards")
    print("=" * 60)
    
    # Format paths with standard type
    nas_input_path = NAS_INPUT_PATH_TEMPLATE.format(standard=STANDARD_TYPE)
    nas_output_path = NAS_OUTPUT_PATH_TEMPLATE.format(standard=STANDARD_TYPE)
    nas_log_path = NAS_LOG_PATH_TEMPLATE.format(standard=STANDARD_TYPE)
    
    logging.info(f"Configuration:")
    logging.info(f"  Standard type: {STANDARD_TYPE.upper()}")
    logging.info(f"  Input path: {nas_input_path}")
    logging.info(f"  Output path: {nas_output_path}")
    
    try:
        # Process all PDFs
        process_all_pdfs(nas_input_path, nas_output_path, STANDARD_TYPE)
        
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
    
    # Upload log file to NAS
    try:
        share_name = NAS_PARAMS["share"]
        log_file_name = f"{STANDARD_TYPE}_merger_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path_relative = os.path.join(nas_log_path, log_file_name).replace('\\', '/')
        
        # Close logging handlers to flush content
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)
        
        # Read log content and upload to NAS
        with open(temp_log_path, 'rb') as f:
            log_content = f.read()
        
        if write_to_nas(share_name, log_path_relative, log_content):
            print(f"Log saved to: {log_path_relative}")
        else:
            print(f"Failed to save log file")
        
        # Clean up temp log file
        os.remove(temp_log_path)
    except Exception as e:
        print(f"Error saving log: {e}")
    
    print("\nProcessing complete. Check log for details.")
    print("=" * 60)

if __name__ == "__main__":
    main()