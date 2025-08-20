#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 6: Database Upload Pipeline - Master Database Version
Loads the master CSV database file from Stage 5, connects to PostgreSQL database,
clears the ENTIRE table, and inserts all records with proper vector handling.

Key Features:
- Reads master CSV database file from NAS (master_database.csv)
- Clears entire PostgreSQL table (not just specific document_id)
- Uploads all records from master CSV
- Connects to PostgreSQL using psycopg2 and pgvector
- Creates temporary staging table for CSV import
- Uses COPY FROM to load into staging table
- INSERT from staging with proper vector casting
- Verifies insertion count

Input: Master CSV database file from Stage 5 (master_database.csv)
Output: Fully replaced iris_semantic_search table with all records from master CSV
"""

import os
import csv
import json
import logging
import tempfile
import socket
import io
import sys
import argparse
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

# --- Master CSV Configuration ---
MASTER_CSV_FILENAME = "master_database.csv"  # Master database file from Stage 5
MASTER_CSV_PATH = os.path.join(NAS_INPUT_PATH, MASTER_CSV_FILENAME).replace("\\", "/")

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

def setup_logging(log_file: Optional[str] = None, verbose: bool = False):
    """Configure logging to both file and console."""
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    level = logging.DEBUG if verbose else logging.INFO
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
                FROM {table};
            """)
            
            final_stats = cur.fetchone()
            log_progress(f"📊 Final stats in {table} - Total: {final_stats[0]}, With embedding: {final_stats[1]}, Null embedding: {final_stats[2]}")
            
            # Get document ID breakdown
            cur.execute(f"""
                SELECT document_id, COUNT(*) as count
                FROM {table}
                GROUP BY document_id
                ORDER BY document_id;
            """)
            
            doc_stats = cur.fetchall()
            if doc_stats:
                log_progress("\n📚 Document breakdown:")
                for doc_id, count in doc_stats:
                    log_progress(f"   {doc_id}: {count} chunks")
            
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
                LIMIT 5;
            """)
            
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
        logging.info("Successfully connected to NAS")
        return conn
    except Exception as e:
        logging.error(f"Exception creating NAS connection: {e}")
        return None

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

def clear_entire_table(conn, table: str) -> bool:
    """Clear ALL records from the database table."""
    if not conn:
        return False
    
    try:
        with conn.cursor() as cur:
            # Get count before deletion
            cur.execute(f"SELECT COUNT(*) FROM {table};")
            before_count = cur.fetchone()[0]
            
            # TRUNCATE is faster than DELETE for entire table
            truncate_sql = f"TRUNCATE TABLE {table} RESTART IDENTITY;"
            cur.execute(truncate_sql)
            conn.commit()
            
            log_progress(f"🗑️  Cleared entire table: {before_count} records removed")
            return True
    except psycopg2.Error as e:
        logging.error(f"Error clearing table: {e}", exc_info=True)
        conn.rollback()
        return False

# ==============================================================================
# Main Execution
# ==============================================================================

def validate_configuration() -> bool:
    """Validate that all configuration values have been set."""
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

def main():
    """Main execution function."""
    start_time = datetime.now()
    
    # Use local variables instead of modifying globals
    master_csv_path_to_use = MASTER_CSV_PATH
    verbose_logging_to_use = VERBOSE_LOGGING
    
    # Parse command-line arguments if running as script
    try:
        parser = argparse.ArgumentParser(description="Stage 6: Database Upload - Master Database Version")
        parser.add_argument("--master-csv", help="Path to master CSV file on NAS (relative to share)")
        parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
        args = parser.parse_args()
        
        if args.master_csv:
            master_csv_path_to_use = args.master_csv
            
        if args.verbose:
            verbose_logging_to_use = True
    except:
        # In notebook environment, argparse will fail - just use defaults
        pass
    
    # Check environment variables as fallback
    if os.environ.get("STAGE6_MASTER_CSV"):
        master_csv_path_to_use = os.environ.get("STAGE6_MASTER_CSV")
    
    # Validate configuration first
    if not validate_configuration():
        return 1
    
    # Setup logging
    log_file = f"stage6_upload_{start_time.strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(log_file, verbose_logging_to_use)
    
    log_progress("="*60)
    log_progress("🚀 Starting Stage 6: Database Upload Pipeline - Master Database Version")
    log_progress(f"   Timestamp: {start_time}")
    log_progress(f"   Master CSV: {os.path.basename(master_csv_path_to_use)}")
    log_progress("="*60)
    
    # Step 1: Connect to NAS and check for master CSV file
    log_progress("\n📁 Connecting to NAS...")
    log_progress(f"   NAS: {NAS_PARAMS['ip']}:{NAS_PARAMS['port']}")
    log_progress(f"   Share: {NAS_PARAMS['share']}")
    log_progress(f"   Master CSV Path: {master_csv_path_to_use}")
    
    # Check if master CSV exists
    smb_conn = create_nas_connection()
    if not smb_conn:
        log_progress("❌ Failed to connect to NAS. Exiting.")
        return 1
    
    try:
        # Try to get file attributes to check if it exists
        try:
            attributes = smb_conn.getAttributes(NAS_PARAMS["share"], master_csv_path_to_use)
            if attributes.file_size > 0:
                log_progress(f"✅ Found master CSV: {os.path.basename(master_csv_path_to_use)}")
                log_progress(f"   File size: {attributes.file_size:,} bytes")
            else:
                log_progress(f"❌ Master CSV file is empty: {master_csv_path_to_use}")
                return 1
        except:
            log_progress(f"❌ Master CSV file not found: {master_csv_path_to_use}")
            log_progress("   Please run Stage 5 first to create the master database file")
            return 1
    finally:
        smb_conn.close()
    
    # Step 2: Read master CSV content from NAS
    log_progress("\n📥 Reading master CSV content from NAS...")
    csv_bytes = read_from_nas(NAS_PARAMS["share"], master_csv_path_to_use)
    if not csv_bytes:
        log_progress("❌ Failed to read CSV file from NAS. Exiting.")
        return 1
    
    # Decode bytes to string
    try:
        csv_content = csv_bytes.decode('utf-8')
        # Count lines for verification
        line_count = len(csv_content.splitlines())
        log_progress(f"✅ Successfully read CSV: {len(csv_bytes):,} bytes, {line_count:,} lines")
    except UnicodeDecodeError as e:
        log_progress(f"❌ Failed to decode CSV content: {e}")
        return 1
    
    # Connect to database
    log_progress("\n🔌 Connecting to database...")
    db_conn = connect_to_database()
    if not db_conn:
        log_progress("❌ Failed to connect to database. Exiting.")
        return 1
    
    # Clear entire table
    log_progress(f"\n🗑️  Clearing entire table {TARGET_TABLE}...")
    if not clear_entire_table(db_conn, TARGET_TABLE):
        log_progress("❌ Failed to clear table. Exiting.")
        db_conn.close()
        return 1
    
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
    log_progress(f"   Master CSV: {os.path.basename(master_csv_path_to_use)}")
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
    sys.exit(main())