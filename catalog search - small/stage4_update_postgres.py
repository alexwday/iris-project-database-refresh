# -*- coding: utf-8 -*-
"""
Stage 4: Update PostgreSQL Database

This script performs the final stage of the data synchronization process.
It connects to a PostgreSQL database and updates the 'apg_catalog' and
'apg_content' tables based on the JSON files generated in previous stages.

Workflow:
1.  Reads the list of records to delete (identified in Stage 1).
2.  Reads the new/updated catalog entries (generated in Stage 3).
3.  Reads the new/updated content entries (generated in Stage 3).
4.  Connects to the PostgreSQL database.
5.  Performs validation: Counts records for the source before operations.
6.  Deletes records from both tables based on the 'files_to_delete' list,
    using document_source, document_type, and document_name as the key.
7.  Performs validation: Counts records after deletion.
8.  Inserts the new catalog entries into the 'apg_catalog' table.
9.  Inserts the new content entries into the 'apg_content' table.
10. Performs validation: Counts records after insertion.
"""

import psycopg2
import psycopg2.extras # For execute_values
import json
import sys
import os
# --- Use pysmb instead of smbclient ---
from smb.SMBConnection import SMBConnection
from smb import smb_structs
import io # For reading/writing strings/bytes to NAS
import socket # For gethostname
# --- End pysmb import ---
from datetime import datetime

# ==============================================================================
# --- Configuration ---
# ==============================================================================

# --- Database Configuration ---
# Database connection parameters
DB_PARAMS = {
    "host": "localhost",
    "port": "5432",
    "dbname": "maven-finance",
    "user": "your_username",
    "password": "your_password"
}
DB_CATALOG_TABLE = 'apg_catalog'
DB_CONTENT_TABLE = 'apg_content'

# --- NAS Configuration ---
# Network attached storage connection parameters
NAS_PARAMS = {
    "ip": "your_nas_ip",
    "share": "your_share_name",
    "user": "your_nas_user",
    "password": "your_nas_password",
    "port": 445 # Default SMB port (can be 139)
}
# Base path on the NAS share where Stage 1/3 output files were stored
NAS_OUTPUT_FOLDER_PATH = "path/to/your/output_folder"

# --- pysmb Configuration ---
# Increase timeout for potentially slow NAS operations
smb_structs.SUPPORT_SMB2 = True # Enable SMB2/3 support if available
smb_structs.MAX_PAYLOAD_SIZE = 65536 # Can sometimes help with large directories
CLIENT_HOSTNAME = socket.gethostname() # Get local machine name for SMB connection

# --- Processing Configuration ---
# Define the specific document source processed in previous stages.
DOCUMENT_SOURCE = 'internal_esg' # From Stage 1/3

# --- Input Filenames ---
FILES_TO_DELETE_FILENAME = '1D_postgres_files_to_delete.json'
CATALOG_ENTRIES_FILENAME = '3A_catalog_entries.json'
CONTENT_ENTRIES_FILENAME = '3B_content_entries.json'

# ==============================================================================
# --- Helper Functions ---
# ==============================================================================

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
        return conn
    except Exception as e:
        print(f"   [ERROR] Exception creating NAS connection: {e}")
        return None

def check_nas_path_exists(share_name, nas_path_relative):
    """Checks if a file or directory exists on the NAS using pysmb."""
    conn = None
    try:
        conn = create_nas_connection()
        if not conn:
            return False # Cannot check if connection failed

        # Use getAttributes to check existence - works for files and dirs
        conn.getAttributes(share_name, nas_path_relative)
        return True # Path exists if no exception was raised
    except Exception as e:
        # Check if the error message indicates "No such file" or similar
        err_str = str(e).lower()
        if "no such file" in err_str or "object_name_not_found" in err_str or "0xc0000034" in err_str:
            return False # Expected outcome if the file/path doesn't exist
        else:
            print(f"   [WARNING] Unexpected error checking existence of NAS path '{share_name}/{nas_path_relative}': {type(e).__name__} - {e}")
            return False # Assume not found on other errors
    finally:
        if conn:
            conn.close()

def read_json_from_nas(nas_path_relative):
    """Reads and parses JSON data from a file path on the NAS."""
    print(f"   Attempting to read JSON from NAS path: {NAS_PARAMS['share']}/{nas_path_relative}")
    try:
        if not check_nas_path_exists(NAS_PARAMS["share"], nas_path_relative):
            print(f"   [WARNING] JSON file not found at: {NAS_PARAMS['share']}/{nas_path_relative}. Returning empty list.")
            return [] # Assume results file, return empty list

        conn = create_nas_connection()
        if not conn:
            return None

        file_obj = io.BytesIO()
        file_attributes, filesize = conn.retrieveFile(NAS_PARAMS["share"], nas_path_relative, file_obj)
        file_obj.seek(0)
        content_bytes = file_obj.read()
        conn.close()
        
        data = json.loads(content_bytes.decode('utf-8'))
        print(f"   Successfully read and parsed JSON from: {NAS_PARAMS['share']}/{nas_path_relative} ({len(data)} records)")
        return data
    except json.JSONDecodeError as e:
        print(f"   [ERROR] Failed to parse JSON from '{nas_path_relative}': {e}")
        return None # Indicate failure
    except Exception as e:
        print(f"   [ERROR] Unexpected error reading JSON from NAS '{nas_path_relative}': {e}")
        return None # Indicate failure

def get_db_connection():
    """Establishes and returns a PostgreSQL database connection."""
    conn = None
    try:
        print(f"   Connecting to database '{DB_PARAMS['dbname']}' on {DB_PARAMS['host']}...")
        conn = psycopg2.connect(**DB_PARAMS)
        print("   Connection successful.")
        return conn
    except psycopg2.Error as db_err:
        print(f"   [CRITICAL ERROR] Database connection error: {db_err}")
        return None

def count_records(conn, table_name, document_source):
    """Counts records in a table for a specific document source."""
    count = -1 # Default to -1 to indicate error
    try:
        with conn.cursor() as cur:
            query = f"SELECT COUNT(*) FROM {table_name} WHERE document_source = %s;"
            cur.execute(query, (document_source,))
            result = cur.fetchone()
            if result:
                count = result[0]
    except psycopg2.Error as e:
        print(f"   [ERROR] Failed to count records in {table_name} for source '{document_source}': {e}")
        # conn.rollback() # Rollback if count is part of a larger transaction? Maybe not needed here.
    except Exception as e:
         print(f"   [ERROR] Unexpected error counting records in {table_name}: {e}")
    return count

# ==============================================================================
# --- Main Processing Function ---
# ==============================================================================

def main_processing_stage4(delete_list_relative_path, catalog_list_relative_path, content_list_relative_path):
    """Handles the core logic for Stage 4: reading inputs, DB operations."""
    print(f"--- Starting Main Processing for Stage 4 ---")

    # --- Read Input Files from NAS ---
    print("[4] Reading Input JSON Files from NAS...") # Renumbered step
    files_to_delete = read_json_from_nas(delete_list_relative_path)
    catalog_entries = read_json_from_nas(catalog_list_relative_path)
    content_entries = read_json_from_nas(content_list_relative_path)

    if files_to_delete is None or catalog_entries is None or content_entries is None:
        print("[CRITICAL ERROR] Failed to read one or more input JSON files from NAS. Exiting.")
        sys.exit(1)

    # Check if there's nothing to do (no deletions and no insertions)
    if not files_to_delete and not catalog_entries and not content_entries:
        print("   No files to delete and no new entries to insert. Stage 4 has no work.")
        print("\n" + "="*60)
        print(f"--- Stage 4 Completed (No DB operations needed) ---")
        print("="*60 + "\n")
        return # Nothing to do, exit function cleanly

    print(f"   Files to Delete: {len(files_to_delete)} records")
    print(f"   Catalog Entries to Insert: {len(catalog_entries)} records")
    print(f"   Content Entries to Insert: {len(content_entries)} records")
    print("-" * 60)

    # --- Database Operations ---
    conn = None
    initial_catalog_count = -1
    initial_content_count = -1
    after_delete_catalog_count = -1
    after_delete_content_count = -1
    final_catalog_count = -1
    final_content_count = -1

    try:
        # --- Connect to Database ---
        print("[5] Connecting to PostgreSQL Database...") # Renumbered
        conn = get_db_connection()
        if conn is None:
            sys.exit(1) # Exit script if DB connection fails
        print("-" * 60)

        # --- Validation (Before) ---
        print("[6] Performing Pre-Operation Validation...") # Renumbered
        initial_catalog_count = count_records(conn, DB_CATALOG_TABLE, DOCUMENT_SOURCE)
        initial_content_count = count_records(conn, DB_CONTENT_TABLE, DOCUMENT_SOURCE)
        print(f"   Initial count in '{DB_CATALOG_TABLE}' for source '{DOCUMENT_SOURCE}': {initial_catalog_count}")
        print(f"   Initial count in '{DB_CONTENT_TABLE}' for source '{DOCUMENT_SOURCE}': {initial_content_count}")
        print("-" * 60)

        # --- Deletion Phase ---
        print("[7] Deleting Records Marked for Deletion...") # Renumbered
        deleted_count = 0
        if not files_to_delete:
            print("   No records marked for deletion. Skipping deletion phase.")
            after_delete_catalog_count = initial_catalog_count # Set counts for verification later
            after_delete_content_count = initial_content_count
        else:
            try:
                with conn.cursor() as cur:
                    delete_keys = set() # Use a set to avoid duplicate delete operations if keys appear multiple times
                    for item in files_to_delete:
                        key = (
                            item.get('document_source'),
                            item.get('document_type'),
                            item.get('document_name')
                        )
                        # Basic validation: ensure all key components are present
                        if all(k is not None for k in key):
                            delete_keys.add(key)
                        else:
                            print(f"   [WARNING] Skipping deletion for record with missing key components: {item}")

                    if not delete_keys:
                        print("   No valid keys found for deletion after filtering.")
                        after_delete_catalog_count = initial_catalog_count # Set counts for verification later
                        after_delete_content_count = initial_content_count
                    else:
                        print(f"   Attempting to delete records for {len(delete_keys)} unique keys...")
                        for key_tuple in delete_keys:
                            doc_source, doc_type, doc_name = key_tuple
                            print(f"      Deleting: Source='{doc_source}', Type='{doc_type}', Name='{doc_name}'")

                            # Delete from Content Table
                            del_content_sql = f"""
                                DELETE FROM {DB_CONTENT_TABLE}
                                WHERE document_source = %s AND document_type = %s AND document_name = %s;
                            """
                            cur.execute(del_content_sql, (doc_source, doc_type, doc_name))
                            print(f"         {cur.rowcount} rows deleted from {DB_CONTENT_TABLE}")

                            # Delete from Catalog Table
                            del_catalog_sql = f"""
                                DELETE FROM {DB_CATALOG_TABLE}
                                WHERE document_source = %s AND document_type = %s AND document_name = %s;
                            """
                            cur.execute(del_catalog_sql, (doc_source, doc_type, doc_name))
                            print(f"         {cur.rowcount} rows deleted from {DB_CATALOG_TABLE}")
                            deleted_count += 1 # Count unique keys deleted

                conn.commit() # Commit transaction after all deletions for a key set
                print(f"   Deletion phase completed. Committed changes for {deleted_count} unique keys.")

                # --- Validation (After Deletion) ---
                print("\n   Performing Post-Deletion Validation...")
                after_delete_catalog_count = count_records(conn, DB_CATALOG_TABLE, DOCUMENT_SOURCE)
                after_delete_content_count = count_records(conn, DB_CONTENT_TABLE, DOCUMENT_SOURCE)
                print(f"   Count in '{DB_CATALOG_TABLE}' after deletion: {after_delete_catalog_count}")
                print(f"   Count in '{DB_CONTENT_TABLE}' after deletion: {after_delete_content_count}")

            except psycopg2.Error as e:
                print(f"   [ERROR] Database error during deletion: {e}")
                if conn:
                    conn.rollback()
                print("   Rolled back deletion transaction.")
                sys.exit(1) # Exit on deletion error
            except Exception as e:
                print(f"   [ERROR] Unexpected error during deletion: {e}")
                if conn:
                    conn.rollback()
                print("   Rolled back deletion transaction.")
                sys.exit(1) # Exit on deletion error
        print("-" * 60)

        # --- Insertion Phase ---
        print("[8] Inserting New/Updated Records...") # Renumbered
        inserted_catalog_count = 0
        inserted_content_count = 0

        # Insert Catalog Entries
        if not catalog_entries:
            print("   No catalog entries to insert.")
        else:
            print(f"   Inserting {len(catalog_entries)} catalog entries into '{DB_CATALOG_TABLE}'...")
            # Define columns in the order they appear in the table
            catalog_cols = [
                'document_source', 'document_type', 'document_name', 'document_description',
                'document_usage', 'document_usage_embedding', 'document_description_embedding',  # Added embedding fields
                'date_created', 'date_last_modified', 'file_name',
                'file_type', 'file_size', 'file_path', 'file_link'
                # Add 'processed_md_path' if it exists in the table, otherwise omit
            ]
            # Prepare data tuples, ensuring order matches catalog_cols
            catalog_data = []
            for entry in catalog_entries:
                 # Handle potential missing keys gracefully (e.g., default to None)
                 # Ensure date strings are valid ISO format or handle conversion if needed
                 data_tuple = tuple(entry.get(col) for col in catalog_cols)
                 catalog_data.append(data_tuple)

            if catalog_data:
                try:
                    with conn.cursor() as cur:
                        # Construct query using .format() instead of f-string
                        insert_query_template = "INSERT INTO {} ({}) VALUES %s;"
                        insert_query = insert_query_template.format(DB_CATALOG_TABLE, ", ".join(catalog_cols))

                        psycopg2.extras.execute_values(
                            cur, insert_query, catalog_data, template=None, page_size=100
                        )
                        inserted_catalog_count = cur.rowcount # execute_values might not return accurate count directly, this is often total affected
                        conn.commit()
                        print(f"   Successfully inserted {len(catalog_data)} records into {DB_CATALOG_TABLE}. Committed changes.")
                        # Note: cur.rowcount after execute_values might be unreliable depending on version/driver.
                        # Relying on len(catalog_data) for expected count.
                except psycopg2.Error as e:
                    print(f"   [ERROR] Database error during catalog insertion: {e}")
                    if conn:
                        conn.rollback()
                    print("   Rolled back catalog insertion transaction.")
                    sys.exit(1)
                except Exception as e:
                    print(f"   [ERROR] Unexpected error during catalog insertion: {e}")
                    if conn:
                        conn.rollback()
                    print("   Rolled back catalog insertion transaction.")
                    sys.exit(1)
            else:
                 print("   No valid catalog data tuples prepared for insertion.")


        # Insert Content Entries
        if not content_entries:
            print("   No content entries to insert.")
        else:
            print(f"   Inserting {len(content_entries)} content entries into '{DB_CONTENT_TABLE}'...")
            # Define DB columns for insertion (matching apg_content schema, excluding id, created_at)
            content_cols_db = [
                'document_source', 'document_type', 'document_name',
                'section_id', 'section_name', 'section_summary', 'section_content', 'page_number' # Added page_number
            ]
            # Define corresponding keys expected in the JSON data from Stage 3, in the same order as DB columns
            content_cols_json = [
                'document_source', 'document_type', 'document_name',
                'section_id', 'section_name', 'section_summary', 'section_content', 'page_number' # Updated to match new structure
            ]
            # Prepare data tuples using JSON keys in the order of DB columns
            content_data = []
            for entry in content_entries:
                 # Fetch data using JSON keys, maintaining order of DB columns for insertion
                 data_tuple = tuple(entry.get(json_col) for json_col in content_cols_json)
                 content_data.append(data_tuple)

            if content_data:
                try:
                    with conn.cursor() as cur:
                        # Construct query using .format() and DB column names
                        insert_query_template = "INSERT INTO {} ({}) VALUES %s;"
                        insert_query = insert_query_template.format(DB_CONTENT_TABLE, ", ".join(content_cols_db)) # Use DB column names

                        psycopg2.extras.execute_values(
                            cur, insert_query, content_data, template=None, page_size=100
                        )
                        inserted_content_count = cur.rowcount # See note above about rowcount reliability
                        conn.commit()
                        print(f"   Successfully inserted {len(content_data)} records into {DB_CONTENT_TABLE}. Committed changes.")
                except psycopg2.Error as e:
                    print(f"   [ERROR] Database error during content insertion: {e}")
                    if conn:
                        conn.rollback()
                    print("   Rolled back content insertion transaction.")
                    sys.exit(1)
                except Exception as e:
                    print(f"   [ERROR] Unexpected error during content insertion: {e}")
                    if conn:
                        conn.rollback()
                    print("   Rolled back content insertion transaction.")
                    sys.exit(1)
            else:
                 print("   No valid content data tuples prepared for insertion.")

        print("-" * 60)

        # --- Validation (After Insertion) ---
        print("[9] Performing Post-Insertion Validation...") # Renumbered
        final_catalog_count = count_records(conn, DB_CATALOG_TABLE, DOCUMENT_SOURCE)
        final_content_count = count_records(conn, DB_CONTENT_TABLE, DOCUMENT_SOURCE)
        print(f"   Final count in '{DB_CATALOG_TABLE}' for source '{DOCUMENT_SOURCE}': {final_catalog_count}")
        print(f"   Final count in '{DB_CONTENT_TABLE}' for source '{DOCUMENT_SOURCE}': {final_content_count}")

        # --- Final Count Verification ---
        print("\n   Verification:")
        # Catalog verification
        expected_catalog_count = after_delete_catalog_count + len(catalog_entries)
        if final_catalog_count == expected_catalog_count:
            print(f"   OK: Final catalog count ({final_catalog_count}) matches expected count ({after_delete_catalog_count} + {len(catalog_entries)} = {expected_catalog_count}).")
        else:
            print(f"   WARNING: Final catalog count ({final_catalog_count}) does NOT match expected count ({expected_catalog_count}).")

        # Content verification
        expected_content_count = after_delete_content_count + len(content_entries)
        if final_content_count == expected_content_count:
             print(f"   OK: Final content count ({final_content_count}) matches expected count ({after_delete_content_count} + {len(content_entries)} = {expected_content_count}).")
        else:
             print(f"   WARNING: Final content count ({final_content_count}) does NOT match expected count ({expected_content_count}).")

        print("-" * 60)

    except Exception as e:
        # Catch-all for any unexpected errors during the main process
        print(f"[CRITICAL ERROR] An unexpected error occurred during Stage 4 execution: {e}")
        if conn:
            try:
                conn.rollback() # Attempt rollback if connection exists
                print("   Attempted to rollback any pending transaction.")
            except psycopg2.Error as rb_err:
                 print(f"   [ERROR] Failed to rollback transaction: {rb_err}")
        sys.exit(1)

    finally:
        # Ensure the database connection is always closed
        if conn is not None:
            conn.close()
            print("\n[10] Database connection closed.") # Renumbered

    print("\n" + "="*60)
    print(f"--- Stage 4 Completed Successfully ---")
    print("--- Database updates applied ---")
    print("="*60 + "\n")
    print(f"--- End of Main Processing for Stage 4 ---")

# ==============================================================================
# --- Script Entry Point ---
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print(f"--- Running Stage 4: Update PostgreSQL Database ---")
    print(f"--- Document Source: {DOCUMENT_SOURCE} ---")
    print("="*60 + "\n")

    # --- Define NAS Paths (Relative) ---
    print("[1] Defining NAS Input Paths (Relative)...")
    source_base_dir_relative = os.path.join(NAS_OUTPUT_FOLDER_PATH, DOCUMENT_SOURCE).replace('\\', '/')

    delete_list_relative_path = os.path.join(source_base_dir_relative, FILES_TO_DELETE_FILENAME).replace('\\', '/')
    catalog_list_relative_path = os.path.join(source_base_dir_relative, CATALOG_ENTRIES_FILENAME).replace('\\', '/')
    content_list_relative_path = os.path.join(source_base_dir_relative, CONTENT_ENTRIES_FILENAME).replace('\\', '/')

    print(f"   Files to Delete List (Relative): {NAS_PARAMS['share']}/{delete_list_relative_path}")
    print(f"   Catalog Entries List (Relative): {NAS_PARAMS['share']}/{catalog_list_relative_path}")
    print(f"   Content Entries List (Relative): {NAS_PARAMS['share']}/{content_list_relative_path}")
    print("-" * 60)

    # --- Check for Skip Flag from Stage 1 ---
    print("[2] Checking for skip flag from Stage 1...")
    skip_flag_file_name = '_SKIP_SUBSEQUENT_STAGES.flag'
    # Flag file is in the base output dir for the source (same level as 1D*.json, 3A*.json etc.)
    skip_flag_relative_path = os.path.join(source_base_dir_relative, skip_flag_file_name).replace('\\', '/')
    print(f"   Checking for flag file: {NAS_PARAMS['share']}/{skip_flag_relative_path}")
    should_skip = False
    try:
        if check_nas_path_exists(NAS_PARAMS["share"], skip_flag_relative_path):
            print(f"   Skip flag file found. Stage 1 indicated no files to process.")
            should_skip = True
        else:
            print(f"   Skip flag file not found. Proceeding with Stage 4.")
    except Exception as e:
        print(f"   [WARNING] Unexpected error checking for skip flag file '{NAS_PARAMS['share']}/{skip_flag_relative_path}': {e}")
        print(f"   Proceeding with Stage 4.")
        # Continue execution
    print("-" * 60)

    # --- Execute Main Processing if Not Skipped ---
    if should_skip:
        print("\n" + "="*60)
        print(f"--- Stage 4 Skipped (No files to process from Stage 1) ---")
        print("="*60 + "\n")
    else:
        # Call the main processing function only if not skipping
        main_processing_stage4(delete_list_relative_path, catalog_list_relative_path, content_list_relative_path)

    # Script ends naturally here if skipped or after main_processing completes
