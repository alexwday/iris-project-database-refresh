# -*- coding: utf-8 -*-
"""
Stage 4: Database Population & Verification

Purpose:
Loads the final processed chunk records from Stage 3, connects to the
PostgreSQL database, clears any existing data for the specified document_id
in the target table, inserts the new records, and performs basic verification.

Input:
- JSON file from Stage 3 (e.g., 'pipeline_output/stage3/stage3_final_records.json').
- Database connection details (configured below or via environment variables).
- document_id to identify the dataset being replaced/inserted.

Output:
- Populated 'guidance_sections' table in the PostgreSQL database.
- Log messages indicating success or failure.
"""

import os
import json
import traceback
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

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
    tqdm = lambda x, **kwargs: x # Make tqdm optional
    print("INFO: tqdm not installed. Progress bars disabled. `pip install tqdm`")

# ==============================================================================
# Configuration
# ==============================================================================

# --- Directory Paths ---
# TODO: Adjust these paths as needed
STAGE3_OUTPUT_DIR = "pipeline_output/stage3"
STAGE3_FILENAME = "stage3_final_records.json"
LOG_DIR = "pipeline_output/logs"

# --- Document ID ---
# TODO: Ensure this matches the DOCUMENT_ID used in previous stages
# This is CRITICAL for clearing the correct data before insertion.
DOCUMENT_ID = "EY_GUIDE_2024_PLACEHOLDER"

# --- Database Configuration ---
# TODO: Load securely (e.g., environment variables) or replace placeholders
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "ey_guidance")
DB_USER = os.environ.get("DB_USER", "user")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "password") # Be cautious storing passwords directly

# --- Database Operations ---
TARGET_TABLE = "guidance_sections"
INSERT_BATCH_SIZE = 100 # Adjust batch size based on performance/memory

# --- Logging Setup ---
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
log_file = Path(LOG_DIR) / 'stage4_database_population.log'
# Remove existing handlers if configuring multiple times in a notebook
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)

# ==============================================================================
# Utility Functions (Self-Contained)
# ==============================================================================

# --- Database Connection ---
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
        logging.info(f"Database connection established to {DB_NAME} on {DB_HOST}:{DB_PORT}")
        return conn
    except psycopg2.Error as e:
        logging.error(f"Database connection error: {e}", exc_info=True)
        return None

# --- File/Path Utils ---
def create_directory(directory: str):
    """Creates the specified directory if it does not already exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Database Operations
# ==============================================================================

def clear_existing_data(conn, table: str, doc_id: str) -> bool:
    """Deletes existing rows for the given document_id from the table."""
    if not conn: return False
    delete_sql = f"DELETE FROM {table} WHERE document_id = %s;"
    try:
        with conn.cursor() as cur:
            cur.execute(delete_sql, (doc_id,))
            deleted_count = cur.rowcount
            conn.commit()
            logging.info(f"Deleted {deleted_count} existing rows for document_id '{doc_id}' from table '{table}'.")
            return True
    except psycopg2.Error as e:
        logging.error(f"Error deleting data for document_id '{doc_id}' from {table}: {e}", exc_info=True)
        conn.rollback() # Rollback on error
        return False

def insert_records(conn, table: str, records: List[Dict], batch_size: int = INSERT_BATCH_SIZE) -> Tuple[int, int]:
    """Inserts records into the target table using execute_values for efficiency."""
    if not conn or not records: return 0, 0
    if not execute_values:
        logging.error("psycopg2.extras.execute_values not available. Cannot perform batch insert.")
        return 0, 0

    # Define columns based on the schema (excluding DB-generated ones like id, created_at, text_search_vector)
    # Ensure order matches the values tuple generation below
    columns = [
        "document_id", "chapter_number", "section_number", "part_number", "sequence_number",
        "chapter_name", "chapter_tags", "chapter_summary", "chapter_token_count",
        "section_start_page", "section_end_page", "section_importance_score", "section_token_count",
        "section_hierarchy", "section_title", "section_standard", "section_standard_codes", "section_references",
        "content", "embedding"
    ]
    cols_sql = ", ".join(columns)
    vals_sql = f"%s" # Template for execute_values

    insert_sql = f"INSERT INTO {table} ({cols_sql}) VALUES {vals_sql};"

    # Prepare data tuples in the correct order
    data_tuples = []
    skipped_records = 0
    for record in records:
        # Handle potential None embeddings - replace with NULL for DB if needed, or skip?
        # Assuming embedding column allows NULLs. If not, skip or use a default.
        # psycopg2 typically handles None as NULL.
        if record.get('embedding') is None:
            logging.warning(f"Record sequence {record.get('sequence_number')} has NULL embedding.")
            # embedding_value = None # Or handle as needed
        # Ensure all required columns are present
        try:
            tuple_values = tuple(record.get(col) for col in columns)
            data_tuples.append(tuple_values)
        except KeyError as e:
             logging.error(f"Record sequence {record.get('sequence_number')} missing key {e}. Skipping record.")
             skipped_records += 1
        except Exception as e:
             logging.error(f"Error preparing record sequence {record.get('sequence_number')}: {e}. Skipping record.")
             skipped_records += 1

    inserted_count = 0
    if not data_tuples:
        logging.warning("No valid records prepared for insertion.")
        return 0, skipped_records

    try:
        with conn.cursor() as cur:
            execute_values(cur, insert_sql, data_tuples, page_size=batch_size)
            inserted_count = cur.rowcount # Note: execute_values doesn't directly return count easily, this might be total affected rows if triggers exist. Rely on verification.
            conn.commit()
            # We log the number of tuples attempted, verification step confirms actual count.
            logging.info(f"Attempted to insert {len(data_tuples)} records in batches.")
            inserted_count = len(data_tuples) # Assume success if no exception
    except psycopg2.Error as e:
        logging.error(f"Database error during batch insert: {e}", exc_info=True)
        conn.rollback()
        return 0, skipped_records + len(data_tuples) # All attempted tuples failed in this batch
    except Exception as e:
        logging.error(f"Unexpected error during batch insert: {e}", exc_info=True)
        conn.rollback()
        return 0, skipped_records + len(data_tuples)

    return inserted_count, skipped_records


def verify_insertion(conn, table: str, doc_id: str, expected_count: int) -> bool:
    """Verifies the number of rows inserted for the document_id."""
    if not conn: return False
    count_sql = f"SELECT COUNT(*) FROM {table} WHERE document_id = %s;"
    try:
        with conn.cursor() as cur:
            cur.execute(count_sql, (doc_id,))
            actual_count = cur.fetchone()[0]
            logging.info(f"Verification: Found {actual_count} rows for document_id '{doc_id}'. Expected {expected_count}.")
            if actual_count == expected_count:
                logging.info("Verification successful: Row count matches expected count.")
                return True
            else:
                logging.error(f"Verification failed: Row count mismatch! Expected {expected_count}, Found {actual_count}.")
                return False
    except psycopg2.Error as e:
        logging.error(f"Database error during verification count: {e}", exc_info=True)
        return False
    except Exception as e:
        logging.error(f"Unexpected error during verification: {e}", exc_info=True)
        return False

# ==============================================================================
# Main Stage 4 Logic
# ==============================================================================

def run_stage4():
    """Main function to execute Stage 4 processing."""
    logging.info("--- Starting Stage 4: Database Population & Verification ---")

    # --- Load Stage 3 Data ---
    stage3_output_file = Path(STAGE3_OUTPUT_DIR) / STAGE3_FILENAME
    if not stage3_output_file.exists():
        logging.error(f"Stage 3 output file not found: {stage3_output_file}"); return False
    try:
        with open(stage3_output_file, "r", encoding="utf-8") as f: final_records = json.load(f)
        logging.info(f"Loaded {len(final_records)} final records from {stage3_output_file}")
    except Exception as e:
        logging.error(f"Error loading Stage 3 data: {e}", exc_info=True); return False
    if not final_records: logging.warning("Stage 3 data is empty. Nothing to insert."); return True # Consider empty input a success?

    # --- Database Operations ---
    conn = None
    success = False
    try:
        conn = get_db_connection()
        if not conn: raise ConnectionError("Failed to establish database connection.")

        # 1. Clear Existing Data
        if not clear_existing_data(conn, TARGET_TABLE, DOCUMENT_ID):
            raise RuntimeError(f"Failed to clear existing data for document_id {DOCUMENT_ID}.")

        # 2. Insert New Records
        inserted_count, skipped_count = insert_records(conn, TARGET_TABLE, final_records)
        if skipped_count > 0:
             logging.warning(f"Skipped {skipped_count} records during preparation for insert.")
        if inserted_count == 0 and len(final_records) > skipped_count: # Check if insertion actually failed vs just skipped
             raise RuntimeError("Failed to insert records into the database.")
        logging.info(f"Successfully inserted {inserted_count} records.")

        # 3. Verify Insertion
        expected_final_count = len(final_records) - skipped_count
        if not verify_insertion(conn, TARGET_TABLE, DOCUMENT_ID, expected_final_count):
            # Verification failure is critical, but don't raise error, just log it.
            logging.error("Database insertion verification failed.")
            # Allow script to finish, but indicate potential issue
            success = False # Mark as not fully successful if verification fails
        else:
             success = True # Mark as successful only if verification passes

    except Exception as e:
        logging.error(f"Error during Stage 4 execution: {e}", exc_info=True)
        success = False
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")

    # --- Print Summary ---
    logging.info("--- Stage 4 Summary ---")
    if success:
        logging.info(f"Successfully populated table '{TARGET_TABLE}' for document_id '{DOCUMENT_ID}'.")
    else:
        logging.error(f"Stage 4 finished with errors or verification failure for document_id '{DOCUMENT_ID}'. Check logs.")
    logging.info("--- Stage 4 Finished ---")

    return success

# ==============================================================================
# Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    run_stage4()
