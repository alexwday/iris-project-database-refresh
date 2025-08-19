#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 6: Database Upload Pipeline (FIXED VERSION)
Loads CSV file from Stage 5, connects to PostgreSQL database, clears existing data
for the specified document_id, and inserts new records with proper vector handling.

FIXES:
- Properly handles pgvector type casting using staging table approach
- Registers pgvector extension before operations
- Uses staging table with TEXT embedding column, then casts to vector type

Key Features:
- Reads CSV file from NAS (Stage 5 output)
- Connects to PostgreSQL using psycopg2 and pgvector
- Creates temporary staging table for CSV import
- Uses COPY FROM to load into staging table
- INSERT from staging with proper vector casting
- Verifies insertion count

Input: CSV file from Stage 5 output (iris_semantic_search_YYYY-MM-DD_HH-MM-SS.csv)
Output: Properly populated iris_semantic_search table with working vector embeddings
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
    from pgvector.psycopg2 import register_vector
except ImportError:
    register_vector = None
    print("ERROR: pgvector library not installed. Vector operations unavailable. `pip install pgvector`")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x
    print("INFO: tqdm not installed. Progress bars disabled. `pip install tqdm`")

# ==============================================================================
# Configuration (Copy from original stage6_database_upload.py)
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
DOCUMENT_ID = "EY_GUIDE_2024"  # TODO: Update with actual document ID

# --- Database Configuration ---
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "iris_db")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "password")

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
# Logging Setup
# ==============================================================================

def setup_logging(log_file: Optional[str] = None):
    """Configure logging to both file and console."""
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    level = logging.DEBUG if VERBOSE_LOGGING else logging.INFO
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers,
        force=True
    )

def log_progress(message: str):
    """Print progress message to console and log."""
    print(message)
    logging.info(message.strip())

# ==============================================================================
# Database Connection with pgvector
# ==============================================================================

def connect_to_database() -> Optional[psycopg2.extensions.connection]:
    """
    Establish database connection and register pgvector extension.
    """
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        
        # CRITICAL: Register pgvector extension
        if register_vector:
            register_vector(conn)
            log_progress("✅ pgvector extension registered successfully")
        else:
            log_progress("⚠️  WARNING: pgvector not available - vector operations may fail")
        
        return conn
    except psycopg2.Error as e:
        logging.error(f"Database connection failed: {e}", exc_info=True)
        log_progress(f"❌ Failed to connect to database: {e}")
        return None

# ==============================================================================
# Fixed Upload Function with Staging Table
# ==============================================================================

def upload_csv_with_staging(conn, table: str, csv_content: str) -> Tuple[int, int]:
    """
    Upload CSV content using staging table approach for proper vector casting.
    
    Steps:
    1. Create temporary staging table with TEXT embedding column
    2. COPY CSV data into staging table
    3. INSERT from staging to target table with vector casting
    
    Returns:
        Tuple of (inserted_count, failed_count)
    """
    if not conn:
        return 0, 0
    
    try:
        with conn.cursor() as cur:
            # Step 1: Create staging table with TEXT embedding column
            log_progress("📋 Creating temporary staging table...")
            cur.execute("""
                CREATE TEMP TABLE staging_import (
                    document_id VARCHAR(255),
                    filename VARCHAR(255),
                    filepath TEXT,
                    source_filename VARCHAR(255),
                    chapter_number INTEGER,
                    chapter_name TEXT,
                    chapter_summary TEXT,
                    chapter_page_count INTEGER,
                    section_number INTEGER,
                    section_summary TEXT,
                    section_start_page INTEGER,
                    section_end_page INTEGER,
                    section_page_count INTEGER,
                    section_start_reference VARCHAR(50),
                    section_end_reference VARCHAR(50),
                    chunk_number INTEGER,
                    chunk_content TEXT,
                    chunk_start_page INTEGER,
                    chunk_end_page INTEGER,
                    chunk_start_reference VARCHAR(50),
                    chunk_end_reference VARCHAR(50),
                    embedding TEXT,  -- TEXT type for staging
                    extra1 TEXT,
                    extra2 TEXT,
                    extra3 TEXT
                );
            """)
            log_progress("✅ Staging table created")
            
            # Step 2: Process CSV to exclude auto-generated columns
            csv_reader = csv.reader(io.StringIO(csv_content))
            header = next(csv_reader)  # Read header
            
            logging.info(f"Original CSV header columns: {header}")
            
            # Create modified CSV without id, created_at, last_modified columns
            modified_csv = io.StringIO()
            csv_writer = csv.writer(modified_csv)
            
            # Write header for staging table (columns 1-25 from original, skipping 0, 26, 27)
            staging_header = header[1:26]  # Skip id (0) and timestamps (26, 27)
            csv_writer.writerow(staging_header)
            
            row_count = 0
            for row in csv_reader:
                modified_row = row[1:26]  # Skip id (0) and timestamps (26, 27)
                csv_writer.writerow(modified_row)
                row_count += 1
                
                # Log first row for debugging
                if row_count == 1:
                    logging.info(f"First data row sample: {modified_row[:3]}")
                    if len(modified_row) > 21:  # Embedding column index
                        emb_preview = modified_row[21][:50] if modified_row[21] else "NULL"
                        logging.info(f"First row embedding preview: {emb_preview}...")
            
            log_progress(f"📝 Prepared {row_count} rows for import")
            
            # Reset to beginning for COPY FROM
            modified_csv.seek(0)
            
            # Step 3: COPY data into staging table
            log_progress("📥 Loading data into staging table...")
            cur.copy_expert(
                """COPY staging_import FROM STDIN WITH (
                    FORMAT csv,
                    HEADER true,
                    NULL '',
                    QUOTE '"',
                    ESCAPE '"'
                )""",
                modified_csv
            )
            
            staging_count = cur.rowcount
            log_progress(f"✅ Loaded {staging_count} rows into staging table")
            
            # Step 4: Verify embedding data in staging
            cur.execute("""
                SELECT COUNT(*) as total,
                       COUNT(embedding) as with_embedding,
                       COUNT(CASE WHEN embedding IS NULL OR embedding = '' THEN 1 END) as null_embedding
                FROM staging_import;
            """)
            stats = cur.fetchone()
            log_progress(f"📊 Staging stats - Total: {stats[0]}, With embedding: {stats[1]}, Null embedding: {stats[2]}")
            
            # Step 5: INSERT from staging to target with vector casting
            log_progress(f"🔄 Inserting from staging to {table} with vector casting...")
            
            insert_sql = f"""
                INSERT INTO {table} (
                    document_id, filename, filepath, source_filename,
                    chapter_number, chapter_name, chapter_summary, chapter_page_count,
                    section_number, section_summary, section_start_page, section_end_page,
                    section_page_count, section_start_reference, section_end_reference,
                    chunk_number, chunk_content, chunk_start_page, chunk_end_page,
                    chunk_start_reference, chunk_end_reference, embedding,
                    extra1, extra2, extra3
                )
                SELECT 
                    document_id, filename, filepath, source_filename,
                    chapter_number, chapter_name, chapter_summary, chapter_page_count,
                    section_number, section_summary, section_start_page, section_end_page,
                    section_page_count, section_start_reference, section_end_reference,
                    chunk_number, chunk_content, chunk_start_page, chunk_end_page,
                    chunk_start_reference, chunk_end_reference,
                    CASE 
                        WHEN embedding IS NOT NULL AND embedding != '' 
                        THEN embedding::vector(2000)
                        ELSE NULL
                    END as embedding,
                    extra1, extra2, extra3
                FROM staging_import;
            """
            
            cur.execute(insert_sql)
            inserted_count = cur.rowcount
            
            # Step 6: Verify the insertion worked correctly
            cur.execute(f"""
                SELECT COUNT(*) as total,
                       COUNT(embedding) as with_embedding,
                       COUNT(CASE WHEN embedding IS NULL THEN 1 END) as null_embedding
                FROM {table}
                WHERE document_id = %s;
            """, (DOCUMENT_ID,))
            
            final_stats = cur.fetchone()
            log_progress(f"📊 Final stats in {table} - Total: {final_stats[0]}, With embedding: {final_stats[1]}, Null embedding: {final_stats[2]}")
            
            # Step 7: Test vector similarity on a sample
            cur.execute(f"""
                SELECT id, 
                       embedding IS NOT NULL as has_embedding,
                       CASE 
                           WHEN embedding IS NOT NULL 
                           THEN array_length(embedding::real[], 1)
                           ELSE NULL
                       END as dimensions
                FROM {table}
                WHERE document_id = %s
                LIMIT 5;
            """, (DOCUMENT_ID,))
            
            log_progress("🔍 Sample vector verification:")
            for row in cur.fetchall():
                log_progress(f"   ID {row[0]}: has_embedding={row[1]}, dimensions={row[2]}")
            
            conn.commit()
            log_progress(f"✅ Successfully inserted {inserted_count} records with proper vector casting")
            
            return inserted_count, 0
            
    except psycopg2.Error as e:
        logging.error(f"Database error during staging upload: {e}", exc_info=True)
        log_progress(f"❌ Failed to insert records: {e}")
        
        if hasattr(e, 'diag'):
            if e.diag.message_detail:
                log_progress(f"   Detail: {e.diag.message_detail}")
            if e.diag.message_hint:
                log_progress(f"   Hint: {e.diag.message_hint}")
        
        conn.rollback()
        return 0, 0
        
    except Exception as e:
        logging.error(f"Unexpected error during upload: {e}", exc_info=True)
        log_progress(f"❌ Unexpected error: {e}")
        conn.rollback()
        return 0, 0

# ==============================================================================
# NAS File Operations (kept from original)
# ==============================================================================

def connect_to_nas() -> Optional[SMBConnection]:
    """Establish SMB connection to NAS."""
    try:
        conn = SMBConnection(
            NAS_PARAMS["user"],
            NAS_PARAMS["password"],
            CLIENT_HOSTNAME,
            NAS_PARAMS["ip"],
            use_ntlm_v2=True
        )
        
        connected = conn.connect(NAS_PARAMS["ip"], NAS_PARAMS["port"])
        if connected:
            logging.info("Successfully connected to NAS")
            return conn
        else:
            logging.error("Failed to connect to NAS")
            return None
    except Exception as e:
        logging.error(f"NAS connection error: {e}", exc_info=True)
        return None

def read_file_from_nas(nas_conn: SMBConnection, file_path: str) -> Optional[str]:
    """Read file content from NAS."""
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            nas_conn.retrieveFile(NAS_PARAMS["share"], file_path, temp_file)
            temp_file_path = temp_file.name
        
        with open(temp_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        os.unlink(temp_file_path)
        return content
        
    except Exception as e:
        logging.error(f"Error reading file from NAS: {e}", exc_info=True)
        return None

def find_latest_csv_on_nas(nas_conn: SMBConnection) -> Optional[str]:
    """Find the most recent CSV file in the Stage 5 output directory."""
    try:
        files = nas_conn.listPath(NAS_PARAMS["share"], NAS_INPUT_PATH)
        
        csv_files = [
            f for f in files 
            if f.filename.startswith("iris_semantic_search_") 
            and f.filename.endswith(".csv")
            and not f.isDirectory
        ]
        
        if not csv_files:
            return None
        
        # Sort by filename (which includes timestamp)
        csv_files.sort(key=lambda x: x.filename, reverse=True)
        latest_file = csv_files[0].filename
        
        return f"{NAS_INPUT_PATH}/{latest_file}"
        
    except Exception as e:
        logging.error(f"Error listing NAS directory: {e}", exc_info=True)
        return None

def delete_existing_data(conn, table: str, doc_id: str) -> bool:
    """Delete existing data for document_id from table."""
    if not conn:
        return False
    
    try:
        with conn.cursor() as cur:
            delete_sql = f"DELETE FROM {table} WHERE document_id = %s;"
            cur.execute(delete_sql, (doc_id,))
            deleted_count = cur.rowcount
            conn.commit()
            log_progress(f"🗑️  Deleted {deleted_count} existing rows for document_id '{doc_id}'")
            return True
    except psycopg2.Error as e:
        logging.error(f"Error deleting data: {e}", exc_info=True)
        conn.rollback()
        return False

# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    """Main execution function."""
    start_time = datetime.now()
    
    # Setup logging
    log_file = f"stage6_upload_{start_time.strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(log_file)
    
    log_progress("="*60)
    log_progress("🚀 Starting Stage 6: Database Upload Pipeline (FIXED)")
    log_progress(f"   Timestamp: {start_time}")
    log_progress(f"   Document ID: {DOCUMENT_ID}")
    log_progress("="*60)
    
    # Connect to NAS
    log_progress("\n📁 Connecting to NAS...")
    nas_conn = connect_to_nas()
    if not nas_conn:
        log_progress("❌ Failed to connect to NAS. Exiting.")
        return
    
    # Find latest CSV file
    log_progress("\n🔍 Finding latest CSV file...")
    csv_path = find_latest_csv_on_nas(nas_conn)
    if not csv_path:
        log_progress("❌ No CSV file found. Exiting.")
        nas_conn.close()
        return
    
    log_progress(f"   Found: {csv_path}")
    
    # Read CSV content
    log_progress("\n📖 Reading CSV file from NAS...")
    csv_content = read_file_from_nas(nas_conn, csv_path)
    if not csv_content:
        log_progress("❌ Failed to read CSV file. Exiting.")
        nas_conn.close()
        return
    
    csv_lines = csv_content.count('\n')
    log_progress(f"   Read {csv_lines} lines from CSV")
    
    # Close NAS connection
    nas_conn.close()
    
    # Connect to database
    log_progress("\n🔌 Connecting to database...")
    db_conn = connect_to_database()
    if not db_conn:
        log_progress("❌ Failed to connect to database. Exiting.")
        return
    
    # Delete existing data
    log_progress(f"\n🗑️  Clearing existing data for document_id '{DOCUMENT_ID}'...")
    if not delete_existing_data(db_conn, TARGET_TABLE, DOCUMENT_ID):
        log_progress("❌ Failed to clear existing data. Exiting.")
        db_conn.close()
        return
    
    # Upload new data with staging table approach
    log_progress(f"\n📤 Uploading data to {TARGET_TABLE}...")
    inserted_count, failed_count = upload_csv_with_staging(db_conn, TARGET_TABLE, csv_content)
    
    if inserted_count > 0:
        log_progress(f"\n✅ Upload completed successfully!")
        log_progress(f"   Inserted: {inserted_count} records")
        if failed_count > 0:
            log_progress(f"   Failed: {failed_count} records")
    else:
        log_progress(f"\n❌ Upload failed - no records inserted")
    
    # Close database connection
    db_conn.close()
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    log_progress("\n" + "="*60)
    log_progress("📊 Stage 6 Processing Summary:")
    log_progress(f"   Document ID: {DOCUMENT_ID}")
    log_progress(f"   Records inserted: {inserted_count}")
    log_progress(f"   Duration: {duration}")
    log_progress(f"   End time: {end_time}")
    log_progress("="*60)
    
    if inserted_count > 0:
        log_progress("✅ Stage 6 completed successfully!")
        log_progress("✨ Vector embeddings are now properly stored and searchable!")
    else:
        log_progress("❌ Stage 6 failed - please check the logs")

if __name__ == "__main__":
    main()