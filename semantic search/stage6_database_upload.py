#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 6: Database Upload Pipeline
Loads CSV file from Stage 5, connects to PostgreSQL database, clears existing data
for the specified document_id, and inserts new records using COPY FROM for efficiency.

Key Features:
- Reads CSV file from NAS (Stage 5 output)
- Connects to PostgreSQL using psycopg2
- Clears existing data for document_id before insertion
- Uses COPY FROM for efficient bulk loading
- Verifies insertion count
- Handles embedding vectors properly

Input: CSV file from Stage 5 output (iris_semantic_search_YYYY-MM-DD_HH-MM-SS.csv)
Output: Populated iris_semantic_search table in PostgreSQL database
"""

import os
import csv
import json
import logging
import tempfile
import socket
import io
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

# --- pysmb imports for NAS access ---
from smb.SMBConnection import SMBConnection
from smb import smb_structs

# --- Dependencies Check ---
try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    psycopg2 = None
    execute_values = None
    print("ERROR: psycopg2 library not installed. Database operations unavailable. `pip install psycopg2-binary`")

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
NAS_INPUT_PATH = "semantic_search/pipeline_output/stage5"
NAS_LOG_PATH = "semantic_search/pipeline_output/logs"

# --- Document ID ---
# TODO: Ensure this matches the DOCUMENT_ID used in previous stages
# This is CRITICAL for clearing the correct data before insertion.
DOCUMENT_ID = "EY_GUIDE_2024"  # TODO: Update with actual document ID

# --- Database Configuration ---
# TODO: Load securely (e.g., environment variables) or replace placeholders
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "iris_db")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "password")  # Be cautious storing passwords directly

# --- Database Operations ---
TARGET_TABLE = "iris_semantic_search"
INSERT_BATCH_SIZE = 1000  # Number of rows to insert at once

# --- pysmb Configuration ---
smb_structs.SUPPORT_SMB2 = True
smb_structs.MAX_PAYLOAD_SIZE = 65536
CLIENT_HOSTNAME = socket.gethostname()

# --- Logging Level Control ---
VERBOSE_LOGGING = False

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
    if "localhost" in DB_HOST and os.environ.get("DB_HOST") is None:
        errors.append("Database host not configured (using default localhost)")
    if "password" in DB_PASSWORD and os.environ.get("DB_PASSWORD") is None:
        errors.append("Database password not configured (using default)")
    
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

def list_nas_files(share_name, path_relative):
    """List files in a NAS directory."""
    conn = None
    try:
        conn = create_nas_connection()
        if not conn:
            return None
        
        files = conn.listPath(share_name, path_relative)
        # Filter out . and .. entries
        files = [f for f in files if f.filename not in ['.', '..']]
        return files
    except Exception as e:
        logging.error(f"Error listing NAS files: {e}")
        return None
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

def write_to_nas(share_name, nas_path_relative, content_bytes):
    """Writes bytes to a file path on the NAS using pysmb."""
    conn = None
    try:
        conn = create_nas_connection()
        if not conn:
            return False
        
        # Ensure directory exists
        dir_path = os.path.dirname(nas_path_relative).replace("\\", "/")
        if dir_path:
            path_parts = dir_path.strip("/").split("/")
            current_path = ""
            for part in path_parts:
                if not part:
                    continue
                current_path = os.path.join(current_path, part).replace("\\", "/")
                try:
                    conn.listPath(share_name, current_path)
                except:
                    conn.createDirectory(share_name, current_path)
        
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
# Database Operations
# ==============================================================================

def get_db_connection():
    """Establishes and returns a PostgreSQL database connection."""
    if not psycopg2:
        logging.error("psycopg2 library is not available.")
        return None
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        log_progress(f"✅ Database connection established to {DB_NAME} on {DB_HOST}:{DB_PORT}")
        return conn
    except psycopg2.Error as e:
        logging.error(f"Database connection error: {e}", exc_info=True)
        log_progress(f"❌ Failed to connect to database: {e}")
        return None

def clear_existing_data(conn, table: str, doc_id: str) -> bool:
    """Deletes existing rows for the given document_id from the table."""
    if not conn:
        return False
    
    delete_sql = f"DELETE FROM {table} WHERE document_id = %s;"
    try:
        with conn.cursor() as cur:
            cur.execute(delete_sql, (doc_id,))
            deleted_count = cur.rowcount
            conn.commit()
            log_progress(f"🗑️  Deleted {deleted_count} existing rows for document_id '{doc_id}'")
            return True
    except psycopg2.Error as e:
        logging.error(f"Error deleting data for document_id '{doc_id}': {e}", exc_info=True)
        log_progress(f"❌ Failed to delete existing data: {e}")
        conn.rollback()
        return False

def upload_csv_to_database(conn, table: str, csv_content: str) -> Tuple[int, int]:
    """
    Upload CSV content to database using COPY FROM for efficiency.
    
    Returns:
        Tuple of (inserted_count, failed_count)
    """
    if not conn:
        return 0, 0
    
    try:
        # Process CSV to exclude auto-generated columns
        # The CSV has all columns, but we need to skip: id (col 0), created_at (col 26), last_modified (col 27)
        csv_reader = csv.reader(io.StringIO(csv_content))
        header = next(csv_reader)  # Read header
        
        # Create new CSV without auto-generated columns
        # Columns to keep: 1-25 (skip 0, 26, 27)
        modified_csv = io.StringIO()
        csv_writer = csv.writer(modified_csv)
        
        # Write modified header (excluding id, created_at, last_modified)
        modified_header = header[1:26]  # Skip id (0) and timestamps (26, 27)
        csv_writer.writerow(modified_header)
        
        # Write data rows with same exclusions
        for row in csv_reader:
            modified_row = row[1:26]  # Skip id (0) and timestamps (26, 27)
            csv_writer.writerow(modified_row)
        
        # Reset to beginning for COPY FROM
        modified_csv.seek(0)
        
        with conn.cursor() as cur:
            # Use COPY FROM to load the CSV data
            # Now the CSV only has the columns we want to insert
            cur.copy_expert(
                f"""COPY {table} (
                    document_id, filename, filepath, source_filename,
                    chapter_number, chapter_name, chapter_summary, chapter_page_count,
                    section_number, section_summary, section_start_page, section_end_page,
                    section_page_count, section_start_reference, section_end_reference,
                    chunk_number, chunk_content, chunk_start_page, chunk_end_page,
                    chunk_start_reference, chunk_end_reference, embedding,
                    extra1, extra2, extra3
                ) FROM STDIN WITH (
                    FORMAT csv,
                    HEADER true,
                    NULL '',
                    QUOTE '"',
                    ESCAPE '"'
                )""",
                modified_csv
            )
            
            inserted_count = cur.rowcount
            conn.commit()
            
            log_progress(f"✅ Successfully inserted {inserted_count} records using COPY FROM")
            return inserted_count, 0
            
    except psycopg2.Error as e:
        logging.error(f"Database error during COPY FROM: {e}", exc_info=True)
        log_progress(f"❌ Failed to insert records: {e}")
        conn.rollback()
        
        # Try to parse the error to get more details
        if "invalid input syntax for type vector" in str(e):
            log_progress("⚠️  Error appears to be related to embedding format. Check that embeddings are in [x,y,z] format.")
        
        return 0, 0
    except Exception as e:
        logging.error(f"Unexpected error during upload: {e}", exc_info=True)
        log_progress(f"❌ Unexpected error: {e}")
        conn.rollback()
        return 0, 0

def verify_insertion(conn, table: str, doc_id: str, expected_count: int) -> bool:
    """Verifies the number of rows inserted for the document_id."""
    if not conn:
        return False
    
    count_sql = f"SELECT COUNT(*) FROM {table} WHERE document_id = %s;"
    try:
        with conn.cursor() as cur:
            cur.execute(count_sql, (doc_id,))
            actual_count = cur.fetchone()[0]
            log_progress(f"📊 Verification: Found {actual_count} rows for document_id '{doc_id}'")
            
            if actual_count == expected_count:
                log_progress("✅ Verification successful: Row count matches expected count")
                return True
            else:
                log_progress(f"⚠️  Row count mismatch! Expected {expected_count}, Found {actual_count}")
                return False
    except psycopg2.Error as e:
        logging.error(f"Database error during verification: {e}", exc_info=True)
        log_progress(f"❌ Verification failed: {e}")
        return False

def get_sample_records(conn, table: str, doc_id: str, limit: int = 5) -> List[Dict]:
    """Retrieve sample records to verify data integrity."""
    if not conn:
        return []
    
    sample_sql = f"""
        SELECT 
            id, document_id, filename, chunk_number, 
            LENGTH(chunk_content) as content_length,
            CASE WHEN embedding IS NOT NULL THEN 'Present' ELSE 'NULL' END as embedding_status
        FROM {table} 
        WHERE document_id = %s 
        ORDER BY id 
        LIMIT %s;
    """
    
    try:
        with conn.cursor() as cur:
            cur.execute(sample_sql, (doc_id, limit))
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            
            samples = []
            for row in rows:
                samples.append(dict(zip(columns, row)))
            
            return samples
    except psycopg2.Error as e:
        logging.error(f"Error retrieving sample records: {e}", exc_info=True)
        return []

# ==============================================================================
# Main Processing
# ==============================================================================

def find_latest_csv_file() -> Optional[str]:
    """Find the most recent CSV file in the Stage 5 output directory."""
    try:
        files = list_nas_files(NAS_PARAMS["share"], NAS_INPUT_PATH)
        if not files:
            return None
        
        # Filter for CSV files matching our pattern
        csv_files = []
        for f in files:
            if f.filename.startswith("iris_semantic_search_") and f.filename.endswith(".csv"):
                csv_files.append(f.filename)
        
        if not csv_files:
            return None
        
        # Sort by filename (which includes timestamp) and get the latest
        csv_files.sort(reverse=True)
        return csv_files[0]
        
    except Exception as e:
        logging.error(f"Error finding CSV files: {e}")
        return None

def main():
    """Main processing function for Stage 6"""
    
    # Validate configuration
    if not validate_configuration():
        return 1
    
    # Setup logging
    temp_log_path = setup_logging()
    log_progress("\n" + "="*60)
    log_progress("🚀 Starting Stage 6: Database Upload Pipeline")
    log_progress("="*60)
    
    # Find the latest CSV file
    log_progress("\n📂 Looking for latest CSV file in NAS...")
    csv_filename = find_latest_csv_file()
    
    if not csv_filename:
        log_progress("❌ No CSV files found in Stage 5 output directory")
        return 1
    
    log_progress(f"📄 Found CSV file: {csv_filename}")
    
    # Load CSV from NAS
    csv_path = os.path.join(NAS_INPUT_PATH, csv_filename).replace("\\", "/")
    log_progress(f"📥 Loading CSV from NAS: {csv_path}")
    
    try:
        csv_bytes = read_from_nas(NAS_PARAMS["share"], csv_path)
        if csv_bytes is None:
            log_progress("❌ Failed to read CSV file from NAS")
            return 1
        
        csv_content = csv_bytes.decode('utf-8')
        
        # Count rows properly using csv reader (excluding header)
        csv_reader = csv.reader(io.StringIO(csv_content))
        header = next(csv_reader)  # Skip header
        row_count = sum(1 for _ in csv_reader)
        log_progress(f"✅ Loaded CSV with {row_count} data rows")
        
    except Exception as e:
        log_progress(f"❌ Error loading CSV file: {e}")
        logging.error(f"Error loading CSV: {e}", exc_info=True)
        return 1
    
    # Database operations
    conn = None
    success = False
    
    try:
        # Connect to database
        log_progress("\n🔌 Connecting to PostgreSQL database...")
        conn = get_db_connection()
        if not conn:
            raise ConnectionError("Failed to establish database connection")
        
        # Clear existing data
        log_progress(f"\n🗑️  Clearing existing data for document_id '{DOCUMENT_ID}'...")
        if not clear_existing_data(conn, TARGET_TABLE, DOCUMENT_ID):
            raise RuntimeError(f"Failed to clear existing data for document_id {DOCUMENT_ID}")
        
        # Upload CSV data
        log_progress(f"\n📤 Uploading {row_count} rows to table '{TARGET_TABLE}'...")
        inserted_count, failed_count = upload_csv_to_database(conn, TARGET_TABLE, csv_content)
        
        if inserted_count == 0:
            raise RuntimeError("Failed to insert any records into the database")
        
        if failed_count > 0:
            log_progress(f"⚠️  Warning: {failed_count} records failed to insert")
        
        # Verify insertion
        log_progress("\n🔍 Verifying insertion...")
        if verify_insertion(conn, TARGET_TABLE, DOCUMENT_ID, inserted_count):
            success = True
            
            # Get sample records
            samples = get_sample_records(conn, TARGET_TABLE, DOCUMENT_ID)
            if samples:
                log_progress("\n📋 Sample records:")
                for i, sample in enumerate(samples, 1):
                    log_progress(f"  {i}. ID: {sample['id']}, File: {sample['filename']}, "
                               f"Chunk: {sample['chunk_number']}, Content: {sample['content_length']} chars, "
                               f"Embedding: {sample['embedding_status']}")
        else:
            success = False
            log_progress("❌ Verification failed - counts don't match")
            
    except Exception as e:
        log_progress(f"❌ Error during database operations: {e}")
        logging.error(f"Database operation error: {e}", exc_info=True)
        success = False
    finally:
        if conn:
            conn.close()
            log_progress("\n🔌 Database connection closed")
    
    # Upload log file to NAS
    try:
        with open(temp_log_path, 'r') as f:
            log_content = f.read()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        nas_log_filename = f"stage6_database_upload_{timestamp}.log"
        nas_log_path = os.path.join(NAS_LOG_PATH, nas_log_filename).replace("\\", "/")
        
        if write_to_nas(NAS_PARAMS["share"], nas_log_path, log_content.encode('utf-8')):
            log_progress(f"\n📝 Log file uploaded to NAS: {nas_log_path}")
        else:
            log_progress("\n⚠️ Could not upload log file to NAS")
    except Exception as e:
        log_progress(f"\n⚠️ Error uploading log: {e}")
    
    # Print summary
    log_progress("\n" + "="*60)
    log_progress("📊 Stage 6 Processing Summary:")
    
    if success:
        log_progress(f"✅ Successfully uploaded data to PostgreSQL")
        log_progress(f"  Table: {TARGET_TABLE}")
        log_progress(f"  Document ID: {DOCUMENT_ID}")
        log_progress(f"  Rows inserted: {inserted_count}")
        log_progress(f"  CSV file: {csv_filename}")
    else:
        log_progress(f"❌ Stage 6 finished with errors")
        log_progress(f"  Check the log file for details")
    
    log_progress("="*60)
    
    if success:
        log_progress("✅ Stage 6 completed successfully!")
    else:
        log_progress("❌ Stage 6 failed - check logs for details")
    
    log_progress("="*60 + "\n")
    
    # Clean up temp log file
    try:
        os.unlink(temp_log_path)
    except:
        pass
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())