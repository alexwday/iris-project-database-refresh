#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 5: CSV Export Pipeline with Master Database Management
Reads Stage 4 output, validates it, and manages a master CSV database on NAS.

Key Features:
- Maintains a master CSV database file on NAS
- Automatically updates/replaces data for specific document_ids
- Creates the master CSV if it doesn't exist
- Filters and replaces existing document data when updating
- Creates timestamped backup copies
- Validates data against PostgreSQL schema requirements
- Handles NULL values appropriately for auto-generated fields

Input: JSON file from Stage 4 output (stage4_embedded_chunks.json)
Output: 
  - Master CSV database at configured path (master_database.csv)
  - Timestamped backup copies in backups/ subdirectory

Usage:
  python stage_05_csv_export.py [options]
  
Options:
  --document-id ID    Document ID to update in master CSV
  --master-csv PATH   Path to master CSV file on NAS (relative to share)
  --verbose          Enable verbose logging
  
Environment Variables:
  STAGE5_DOCUMENT_ID  Document ID (if not provided via CLI)
  STAGE5_MASTER_CSV   Master CSV path (if not provided via CLI)
"""

import os
import json
import csv
import logging
import tempfile
import socket
import io
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

# --- pysmb imports for NAS access ---
from smb.SMBConnection import SMBConnection
from smb import smb_structs

# --- Dependencies Check ---
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x
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
    "port": 445,  # Default SMB port (can be 139)
}

# --- Directory Paths (Relative to NAS Share) ---
NAS_INPUT_PATH = "semantic_search/pipeline_output/stage4/stage4_embedded_chunks.json"
NAS_OUTPUT_PATH = "semantic_search/pipeline_output/stage5"
NAS_LOG_PATH = "semantic_search/pipeline_output/logs"

# --- Master CSV Configuration ---
MASTER_CSV_PATH = "semantic_search/pipeline_output/stage5/master_database.csv"  # Master CSV on NAS
DOCUMENT_ID = None  # Will be extracted from input data or set via config

# --- pysmb Configuration ---
smb_structs.SUPPORT_SMB2 = True
smb_structs.MAX_PAYLOAD_SIZE = 65536
CLIENT_HOSTNAME = socket.gethostname()

# --- Logging Level Control ---
VERBOSE_LOGGING = False

# ==============================================================================
# PostgreSQL Schema Definition (must match stage3_database_schema.sql)
# ==============================================================================

# Define the exact order and names of columns in the database
DATABASE_COLUMNS = [
    "id",                       # SERIAL PRIMARY KEY - leave empty, auto-generated
    "document_id",              # VARCHAR(255) NOT NULL
    "filename",                 # VARCHAR(255) NOT NULL
    "filepath",                 # TEXT
    "source_filename",          # VARCHAR(255)
    "chapter_number",           # INTEGER
    "chapter_name",             # VARCHAR(500)
    "chapter_summary",          # TEXT
    "chapter_page_count",       # INTEGER
    "section_number",           # INTEGER
    "section_summary",          # TEXT
    "section_start_page",       # INTEGER
    "section_end_page",         # INTEGER
    "section_page_count",       # INTEGER
    "section_start_reference",  # VARCHAR(50) - NEW
    "section_end_reference",    # VARCHAR(50) - NEW
    "chunk_number",             # INTEGER NOT NULL
    "chunk_content",            # TEXT NOT NULL
    "chunk_start_page",         # INTEGER
    "chunk_end_page",           # INTEGER
    "chunk_start_reference",    # VARCHAR(50) - NEW
    "chunk_end_reference",      # VARCHAR(50) - NEW
    "embedding",                # VECTOR(2000)
    "extra1",                   # TEXT - leave empty
    "extra2",                   # TEXT - leave empty
    "extra3",                   # TEXT - leave empty
    "created_at",               # TIMESTAMP - leave empty, auto-generated
    "last_modified"             # TIMESTAMP - leave empty, auto-generated
]

# Mapping from JSON fields to database columns
JSON_TO_DB_MAPPING = {
    "document_id": "document_id",
    "filename": "filename",
    "filepath": "filepath",
    "source_filename": "source_filename",
    "chapter_number": "chapter_number",
    "chapter_name": "chapter_name",
    "chapter_summary": "chapter_summary",
    "chapter_page_count": "chapter_page_count",
    "section_number": "section_number",
    "section_summary": "section_summary",
    "section_start_page": "section_start_page",
    "section_end_page": "section_end_page",
    "section_page_count": "section_page_count",
    "section_start_reference": "section_start_reference",
    "section_end_reference": "section_end_reference",
    "chunk_number": "chunk_number",
    "chunk_content": "chunk_content",
    "chunk_start_page": "chunk_start_page",
    "chunk_end_page": "chunk_end_page",
    "chunk_start_reference": "chunk_start_reference",
    "chunk_end_reference": "chunk_end_reference",
    "embedding": "embedding"
}

# ==============================================================================
# Configuration Validation
# ==============================================================================

def validate_configuration():
    """Validates that configuration values have been properly set."""
    errors = []
    
    if "your_nas_ip" in NAS_PARAMS["ip"]:
        errors.append("NAS IP address not configured")
    if "your_share_name" in NAS_PARAMS["share"]:
        errors.append("NAS share name not configured")
    if "your_nas_user" in NAS_PARAMS["user"]:
        errors.append("NAS username not configured")
    if "your_nas_password" in NAS_PARAMS["password"]:
        errors.append("NAS password not configured")
    
    if errors:
        print("❌ Configuration errors detected:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease update the configuration values in the script before running.")
        return False
    return True

# ==============================================================================
# NAS Helper Functions
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
            is_direct_tcp=(NAS_PARAMS["port"] == 445),
        )
        connected = conn.connect(NAS_PARAMS["ip"], NAS_PARAMS["port"], timeout=60)
        if not connected:
            logging.error("Failed to connect to NAS")
            return None
        return conn
    except Exception as e:
        logging.error(f"Exception creating NAS connection: {e}")
        return None

def ensure_nas_dir_exists(conn, share_name, dir_path_relative):
    """Ensures a directory exists on the NAS, creating it if necessary."""
    if not conn:
        return False
    
    path_parts = dir_path_relative.strip("/").split("/")
    current_path = ""
    try:
        for part in path_parts:
            if not part:
                continue
            current_path = os.path.join(current_path, part).replace("\\", "/")
            try:
                conn.listPath(share_name, current_path)
            except Exception:
                conn.createDirectory(share_name, current_path)
        return True
    except Exception as e:
        logging.error(f"Failed to ensure NAS directory: {e}")
        return False

def write_to_nas(share_name, nas_path_relative, content_bytes):
    """Writes bytes to a file path on the NAS using pysmb."""
    conn = None
    try:
        conn = create_nas_connection()
        if not conn:
            return False
        
        dir_path = os.path.dirname(nas_path_relative).replace("\\", "/")
        if dir_path and not ensure_nas_dir_exists(conn, share_name, dir_path):
            return False
        
        file_obj = io.BytesIO(content_bytes)
        bytes_written = conn.storeFile(share_name, nas_path_relative, file_obj)
        
        if bytes_written == 0 and len(content_bytes) > 0:
            logging.error(f"No bytes written to {nas_path_relative}")
            return False
        
        return True
    except Exception as e:
        logging.error(f"Error writing to NAS: {e}")
        return False
    finally:
        if conn:
            conn.close()

def read_from_nas(share_name, nas_path_relative):
    """Reads content (as bytes) from a file path on the NAS using pysmb."""
    conn = None
    file_obj = None
    try:
        conn = create_nas_connection()
        if not conn:
            return None
        
        file_obj = io.BytesIO()
        _, _ = conn.retrieveFile(share_name, nas_path_relative, file_obj)
        file_obj.seek(0)
        content_bytes = file_obj.read()
        return content_bytes
    except Exception as e:
        logging.error(f"Error reading from NAS: {e}")
        return None
    finally:
        if file_obj:
            try:
                file_obj.close()
            except Exception:
                pass
        if conn:
            conn.close()

def file_exists_on_nas(share_name, nas_path_relative):
    """Check if a file exists on the NAS."""
    conn = None
    try:
        conn = create_nas_connection()
        if not conn:
            return False
        
        # Try to get file attributes
        try:
            attributes = conn.getAttributes(share_name, nas_path_relative)
            return attributes.file_size > 0  # File exists and has content
        except:
            return False
    except Exception as e:
        logging.error(f"Error checking file existence on NAS: {e}")
        return False
    finally:
        if conn:
            conn.close()

def load_master_csv_from_nas(share_name, master_path):
    """Load the master CSV from NAS and return as list of dictionaries."""
    try:
        csv_bytes = read_from_nas(share_name, master_path)
        if csv_bytes is None:
            return None
        
        csv_content = csv_bytes.decode('utf-8')
        csv_reader = csv.DictReader(io.StringIO(csv_content))
        
        # Convert to list of dictionaries
        rows = []
        for row in csv_reader:
            rows.append(dict(row))
        
        return rows
    except Exception as e:
        logging.error(f"Error loading master CSV: {e}")
        return None

def filter_rows_by_document_id(rows, document_id):
    """Filter out rows with matching document_id."""
    if not document_id:
        return rows
    
    filtered = []
    removed_count = 0
    
    for row in rows:
        if row.get('document_id') != document_id:
            filtered.append(row)
        else:
            removed_count += 1
    
    log_progress(f"  Removed {removed_count} existing rows for document_id: {document_id}")
    return filtered

# ==============================================================================
# Logging Setup
# ==============================================================================

def setup_logging():
    """Setup logging with controlled verbosity."""
    temp_log = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log")
    temp_log_path = temp_log.name
    temp_log.close()
    
    # Clear any existing handlers to prevent duplication
    logging.root.handlers = []
    
    log_level = logging.DEBUG if VERBOSE_LOGGING else logging.WARNING
    
    # Only add file handler to root logger (no console handler)
    root_file_handler = logging.FileHandler(temp_log_path)
    root_file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[root_file_handler],
    )
    
    # Progress logger handles all console output
    progress_logger = logging.getLogger("progress")
    progress_logger.setLevel(logging.INFO)
    progress_logger.propagate = False
    
    # Console handler for progress messages
    progress_console_handler = logging.StreamHandler()
    progress_console_handler.setFormatter(logging.Formatter("%(message)s"))
    progress_logger.addHandler(progress_console_handler)
    
    # File handler for progress messages
    progress_file_handler = logging.FileHandler(temp_log_path)
    progress_file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    progress_logger.addHandler(progress_file_handler)
    
    # If verbose logging is enabled, also show warnings/errors on console
    if VERBOSE_LOGGING:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        console_handler.setLevel(logging.WARNING)
        logging.root.addHandler(console_handler)
    
    return temp_log_path

def log_progress(message, end="\n"):
    """Log a progress message that always shows."""
    progress_logger = logging.getLogger("progress")
    
    if end == "":
        sys.stdout.write(message)
        sys.stdout.flush()
        for handler in progress_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.stream.write(f"{datetime.now().isoformat()} - {message}\n")
                handler.flush()
    else:
        progress_logger.info(message)

# ==============================================================================
# Data Validation Functions
# ==============================================================================

def validate_chunk(chunk: Dict, index: int) -> List[str]:
    """
    Validate a single chunk against schema requirements.
    
    Returns list of validation errors (empty if valid).
    """
    errors = []
    
    # Check required fields
    required_fields = ["document_id", "filename", "chunk_number", "chunk_content"]
    for field in required_fields:
        if field not in chunk or chunk[field] is None or chunk[field] == "":
            errors.append(f"Chunk {index}: Missing or empty required field '{field}'")
    
    # Check embedding field
    if "embedding" in chunk and chunk["embedding"] is not None:
        if not isinstance(chunk["embedding"], list):
            errors.append(f"Chunk {index}: Embedding must be a list")
        elif len(chunk["embedding"]) != 2000:
            errors.append(f"Chunk {index}: Embedding has {len(chunk['embedding'])} dimensions, expected 2000")
    
    # Check integer fields
    integer_fields = ["chapter_number", "section_number", "chunk_number", 
                     "chapter_page_count", "section_page_count",
                     "section_start_page", "section_end_page",
                     "chunk_start_page", "chunk_end_page"]
    for field in integer_fields:
        if field in chunk and chunk[field] is not None:
            if not isinstance(chunk[field], int):
                try:
                    # Try to convert to int
                    int(chunk[field])
                except (ValueError, TypeError):
                    errors.append(f"Chunk {index}: Field '{field}' must be an integer")
    
    return errors

def format_embedding_for_postgres(embedding: Optional[List[float]]) -> str:
    """
    Format embedding array for PostgreSQL vector type.
    PostgreSQL expects format: [0.1,0.2,0.3,...]
    """
    if embedding is None or not embedding:
        return ""  # Return empty string for NULL
    
    # Format as PostgreSQL array literal
    formatted = "[" + ",".join(str(float(x)) for x in embedding) + "]"
    return formatted

def chunk_to_csv_row(chunk: Dict) -> List[Any]:
    """
    Convert a chunk dictionary to a CSV row matching database schema.
    """
    row = []
    
    for column in DATABASE_COLUMNS:
        if column == "id":
            # Auto-generated by PostgreSQL
            row.append("")
        elif column == "created_at" or column == "last_modified":
            # Auto-generated timestamps
            row.append("")
        elif column in ["extra1", "extra2", "extra3"]:
            # Placeholder fields
            row.append("")
        elif column == "embedding":
            # Special formatting for embedding vector
            embedding = chunk.get("embedding")
            row.append(format_embedding_for_postgres(embedding))
        else:
            # Map from JSON field name to database column
            json_field = None
            for json_key, db_col in JSON_TO_DB_MAPPING.items():
                if db_col == column:
                    json_field = json_key
                    break
            
            if json_field:
                value = chunk.get(json_field)
                # Convert None to empty string for CSV
                if value is None:
                    row.append("")
                else:
                    row.append(value)
            else:
                # Column not in mapping
                row.append("")
    
    return row

# ==============================================================================
# Main Processing
# ==============================================================================

def process_chunks_to_csv(chunks: List[Dict]) -> tuple[List[List[Any]], List[str], Optional[str]]:
    """
    Process chunks and convert to CSV rows.
    
    Returns:
        Tuple of (csv_rows, validation_errors, document_id)
    """
    csv_rows = []
    all_errors = []
    document_id = None
    
    log_progress(f"\n📊 Processing {len(chunks)} chunks for CSV export...")
    
    # Extract document_id from first chunk
    if chunks and chunks[0].get('document_id'):
        document_id = chunks[0]['document_id']
        log_progress(f"  Document ID: {document_id}")
    
    for i, chunk in enumerate(tqdm(chunks, desc="Validating and converting chunks")):
        # Validate chunk
        errors = validate_chunk(chunk, i + 1)
        if errors:
            all_errors.extend(errors)
        
        # Convert to CSV row regardless of validation errors
        # (we'll report errors but still export what we can)
        csv_row = chunk_to_csv_row(chunk)
        csv_rows.append(csv_row)
    
    return csv_rows, all_errors, document_id

def merge_with_master_csv(new_rows: List[List[Any]], document_id: Optional[str], master_csv_path: str = None) -> List[List[Any]]:
    """
    Merge new rows with existing master CSV, replacing rows with matching document_id.
    
    Returns:
        Combined list of CSV rows
    """
    # Use provided path or default
    master_path = master_csv_path or MASTER_CSV_PATH
    
    log_progress(f"\n🔄 Managing master CSV database...")
    
    # Check if master CSV exists
    if file_exists_on_nas(NAS_PARAMS["share"], master_path):
        log_progress(f"  Master CSV exists, loading from: {master_path}")
        
        # Load existing master CSV
        existing_rows = load_master_csv_from_nas(NAS_PARAMS["share"], master_path)
        
        if existing_rows is None:
            log_progress("  ⚠️ Failed to load master CSV, treating as new")
            return new_rows
        
        log_progress(f"  Loaded {len(existing_rows)} existing rows")
        
        # Filter out rows with matching document_id
        if document_id:
            filtered_rows = filter_rows_by_document_id(existing_rows, document_id)
        else:
            filtered_rows = existing_rows
            log_progress("  ⚠️ No document_id found, appending all new rows")
        
        # Convert filtered dictionary rows back to list format
        merged_rows = []
        for row_dict in filtered_rows:
            row_list = []
            for column in DATABASE_COLUMNS:
                row_list.append(row_dict.get(column, ""))
            merged_rows.append(row_list)
        
        # Add new rows
        merged_rows.extend(new_rows)
        
        log_progress(f"  Total rows in updated master: {len(merged_rows)}")
        return merged_rows
    
    else:
        log_progress(f"  Master CSV does not exist, creating new at: {master_path}")
        return new_rows

def main():
    """Main processing function for Stage 5"""
    
    # Use local variables instead of modifying globals
    document_id_to_use = DOCUMENT_ID
    master_csv_path_to_use = MASTER_CSV_PATH
    verbose_logging_to_use = VERBOSE_LOGGING
    
    # Parse command-line arguments if running as script
    try:
        parser = argparse.ArgumentParser(description="Stage 5: CSV Export Pipeline with Master Database Management")
        parser.add_argument("--document-id", help="Document ID to update in master CSV")
        parser.add_argument("--master-csv", help="Path to master CSV file on NAS (relative to share)")
        parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
        args = parser.parse_args()
        
        # Apply command-line arguments
        if args.document_id:
            document_id_to_use = args.document_id
            log_progress(f"Using document ID from CLI: {document_id_to_use}")
        
        if args.master_csv:
            master_csv_path_to_use = args.master_csv
            log_progress(f"Using master CSV path from CLI: {master_csv_path_to_use}")
        
        if args.verbose:
            verbose_logging_to_use = True
    except:
        # In notebook environment, argparse will fail - just use defaults
        pass
    
    # Check environment variables as fallback
    if document_id_to_use is None and os.environ.get("STAGE5_DOCUMENT_ID"):
        document_id_to_use = os.environ.get("STAGE5_DOCUMENT_ID")
        log_progress(f"Using document ID from environment: {document_id_to_use}")
    
    if os.environ.get("STAGE5_MASTER_CSV"):
        master_csv_path_to_use = os.environ.get("STAGE5_MASTER_CSV")
        log_progress(f"Using master CSV path from environment: {master_csv_path_to_use}")
    
    # Validate configuration
    if not validate_configuration():
        return 1
    
    # Setup logging
    temp_log_path = setup_logging()
    log_progress("\n" + "="*60)
    log_progress("🚀 Starting Stage 5: CSV Export Pipeline")
    log_progress("="*60)
    
    # Load input data from NAS
    log_progress(f"\n📥 Loading chunks from NAS: {NAS_INPUT_PATH}")
    try:
        input_bytes = read_from_nas(NAS_PARAMS["share"], NAS_INPUT_PATH)
        if input_bytes is None:
            log_progress("❌ Failed to read input file from NAS")
            return 1
        
        chunks = json.loads(input_bytes.decode('utf-8'))
        log_progress(f"✅ Loaded {len(chunks)} chunks from input file")
    except Exception as e:
        log_progress(f"❌ Error loading input file: {e}")
        logging.error(f"Error loading input file: {e}", exc_info=True)
        return 1
    
    if not chunks:
        log_progress("⚠️ No chunks found in input file")
        return 1
    
    # Process chunks to CSV
    csv_rows, validation_errors, document_id = process_chunks_to_csv(chunks)
    
    # Use document_id from data if not set via CLI or env
    if document_id and document_id_to_use is None:
        document_id_to_use = document_id
    
    # Report validation errors
    if validation_errors:
        log_progress(f"\n⚠️ Found {len(validation_errors)} validation issues:")
        for error in validation_errors[:10]:  # Show first 10 errors
            log_progress(f"  - {error}")
        if len(validation_errors) > 10:
            log_progress(f"  ... and {len(validation_errors) - 10} more")
    else:
        log_progress("\n✅ All chunks passed validation")
    
    # Merge with master CSV
    all_csv_rows = merge_with_master_csv(csv_rows, document_id_to_use, master_csv_path_to_use)
    
    # Write merged CSV to memory buffer
    log_progress(f"\n💾 Writing master CSV with {len(all_csv_rows)} total rows...")
    csv_buffer = io.StringIO()
    csv_writer = csv.writer(csv_buffer, quoting=csv.QUOTE_MINIMAL)
    
    # Write header row
    csv_writer.writerow(DATABASE_COLUMNS)
    
    # Write data rows
    for row in all_csv_rows:
        csv_writer.writerow(row)
    
    # Get CSV content as bytes
    csv_content = csv_buffer.getvalue()
    csv_bytes = csv_content.encode('utf-8')
    
    # Save master CSV to NAS
    log_progress(f"💾 Saving master CSV to NAS: {master_csv_path_to_use}")
    if write_to_nas(NAS_PARAMS["share"], master_csv_path_to_use, csv_bytes):
        log_progress(f"✅ Successfully saved master CSV with {len(all_csv_rows)} total rows")
    else:
        log_progress("❌ Failed to write master CSV to NAS")
        return 1
    
    # Also save a timestamped backup copy
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_filename = f"iris_semantic_search_{timestamp}.csv"
    backup_path = os.path.join(NAS_OUTPUT_PATH, "backups", backup_filename).replace("\\", "/")
    
    log_progress(f"\n📦 Creating backup copy: {backup_filename}")
    if write_to_nas(NAS_PARAMS["share"], backup_path, csv_bytes):
        log_progress(f"✅ Backup saved successfully")
    else:
        log_progress("⚠️ Failed to save backup (master CSV was saved successfully)")
    
    # Upload log file to NAS
    try:
        with open(temp_log_path, 'r') as f:
            log_content = f.read()
        nas_log_filename = f"stage5_csv_export_{timestamp}.log"
        nas_log_path = os.path.join(NAS_LOG_PATH, nas_log_filename).replace("\\", "/")
        
        if write_to_nas(NAS_PARAMS["share"], nas_log_path, log_content.encode('utf-8')):
            log_progress(f"📝 Log file uploaded to NAS: {nas_log_path}")
        else:
            log_progress("⚠️ Could not upload log file to NAS")
    except Exception as e:
        log_progress(f"⚠️ Error uploading log: {e}")
    
    # Print summary statistics
    log_progress("\n" + "="*60)
    log_progress("📊 Stage 5 Processing Summary:")
    log_progress(f"  Total chunks processed: {len(chunks)}")
    log_progress(f"  New CSV rows added: {len(csv_rows)}")
    log_progress(f"  Total rows in master: {len(all_csv_rows)}")
    log_progress(f"  Document ID: {document_id_to_use if document_id_to_use else 'Not specified'}")
    log_progress(f"  Master file size: {len(csv_bytes):,} bytes")
    
    # Check for embeddings
    chunks_with_embeddings = sum(1 for c in chunks if c.get('embedding') is not None)
    log_progress(f"\n  Chunks with embeddings: {chunks_with_embeddings}")
    log_progress(f"  Chunks without embeddings: {len(chunks) - chunks_with_embeddings}")
    
    if validation_errors:
        log_progress(f"\n⚠️ Validation issues found: {len(validation_errors)}")
        log_progress("  Check the log file for details")
    
    log_progress("="*60)
    log_progress("✅ Stage 5 processing completed successfully!")
    log_progress(f"📄 Master CSV updated at: {master_csv_path_to_use}")
    if document_id_to_use:
        log_progress(f"🔄 Document '{document_id_to_use}' has been updated in the master database")
    log_progress("="*60 + "\n")
    
    # Clean up temp log file
    try:
        os.unlink(temp_log_path)
    except:
        pass
    
    return 0

if __name__ == "__main__":
    exit(main())