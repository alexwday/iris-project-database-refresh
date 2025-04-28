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
import smbclient
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
    "password": "your_nas_password"
}
# Base path on the NAS share where Stage 1/3 output files were stored
NAS_OUTPUT_FOLDER_PATH = "path/to/your/output_folder"

# --- Processing Configuration ---
# Define the specific document source processed in previous stages.
DOCUMENT_SOURCE = 'internal_cheatsheets' # Corrected document source

# --- Input Filenames ---
FILES_TO_DELETE_FILENAME = '1D_postgres_files_to_delete.json'
CATALOG_ENTRIES_FILENAME = '3A_catalog_entries.json'
CONTENT_ENTRIES_FILENAME = '3B_content_entries.json'

# ==============================================================================
# --- Helper Functions ---
# ==============================================================================

def initialize_smb_client():
    """Sets up smbclient credentials."""
    try:
        smbclient.ClientConfig(username=NAS_PARAMS["user"], password=NAS_PARAMS["password"])
        print("SMB client configured successfully.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to configure SMB client: {e}")
        return False

def read_json_from_nas(smb_path):
    """Reads and parses JSON data from a file path on the NAS."""
    print(f"   Attempting to read JSON from NAS path: {smb_path}")
    try:
        if not smbclient.path.exists(smb_path):
            print(f"   [WARNING] JSON file not found at: {smb_path}. Returning empty list.")
            return [] # Assume results file, return empty list

        with smbclient.open_file(smb_path, mode='r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"   Successfully read and parsed JSON from: {smb_path} ({len(data)} records)")
        return data
    except smbclient.SambaClientError as e:
        print(f"   [ERROR] SMB Error reading JSON from '{smb_path}': {e}")
        return None # Indicate failure
    except json.JSONDecodeError as e:
        print(f"   [ERROR] Failed to parse JSON from '{smb_path}': {e}")
        return None # Indicate failure
    except Exception as e:
        print(f"   [ERROR] Unexpected error reading JSON from NAS '{smb_path}': {e}")
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

def main_processing_stage4(delete_list_smb_path, catalog_list_smb_path, content_list_smb_path):
    """Handles the core logic for Stage 4: reading inputs, DB operations."""
    print(f"--- Starting Main Processing for Stage 4 ---")

    # --- Read Input Files from NAS ---
    print("[4] Reading Input JSON Files from NAS...") # Renumbered step
    files_to_delete = read_json_from_nas(delete_list_smb_path)
    catalog_entries = read_json_from_nas(catalog_list_smb_path)
    content_entries = read_json_from_nas(content_list_smb_path)

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
                'document_usage', 'date_created', 'date_last_modified', 'file_name',
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
                'section_id', 'section_name', 'section_summary', 'section_content' # Correct DB columns
            ]
            # Define corresponding keys expected in the JSON data from Stage 3, in the same order as DB columns
            content_cols_json = [
                'document_source', 'document_type', 'document_name',
                'section_id', 'section_name', 'section_summary', 'content' # Correct JSON keys ('content' maps to 'section_content')
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

    # --- Initialize SMB Client ---
    print("[1] Initializing SMB Client...")
    if not initialize_smb_client():
        sys.exit(1)
    print("-" * 60)

    # --- Define NAS Paths ---
    print("[2] Defining NAS Input Paths...")
    source_base_dir_relative = os.path.join(NAS_OUTPUT_FOLDER_PATH, DOCUMENT_SOURCE).replace('\\', '/')
    source_base_dir_smb = f"//{NAS_PARAMS['ip']}/{NAS_PARAMS['share']}/{source_base_dir_relative}"

    delete_list_smb_path = os.path.join(source_base_dir_smb, FILES_TO_DELETE_FILENAME).replace('\\', '/')
    catalog_list_smb_path = os.path.join(source_base_dir_smb, CATALOG_ENTRIES_FILENAME).replace('\\', '/')
    content_list_smb_path = os.path.join(source_base_dir_smb, CONTENT_ENTRIES_FILENAME).replace('\\', '/')

    print(f"   Files to Delete List (SMB): {delete_list_smb_path}")
    print(f"   Catalog Entries List (SMB): {catalog_list_smb_path}")
    print(f"   Content Entries List (SMB): {content_list_smb_path}")
    print("-" * 60)

    # --- Check for Skip Flag from Stage 1 ---
    print("[3] Checking for skip flag from Stage 1...")
    skip_flag_file_name = '_SKIP_SUBSEQUENT_STAGES.flag'
    # Flag file is in the base output dir for the source (same level as 1D*.json, 3A*.json etc.)
    skip_flag_smb_path = os.path.join(source_base_dir_smb, skip_flag_file_name).replace('\\', '/')
    print(f"   Checking for flag file: {skip_flag_smb_path}")
    should_skip = False
    try:
        # Ensure SMB client is configured (should be from step [1])
        if smbclient.path.exists(skip_flag_smb_path):
            print(f"   Skip flag file found. Stage 1 indicated no files to process.")
            should_skip = True
        else:
            print(f"   Skip flag file not found. Proceeding with Stage 4.")
    except smbclient.SambaClientError as e:
        print(f"   [WARNING] SMB Error checking for skip flag file '{skip_flag_smb_path}': {e}")
        print(f"   Proceeding with Stage 4, but there might be an issue accessing NAS.")
        # Continue execution, assuming no skip if flag check fails
    except Exception as e:
        print(f"   [WARNING] Unexpected error checking for skip flag file '{skip_flag_smb_path}': {e}")
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
        main_processing_stage4(delete_list_smb_path, catalog_list_smb_path, content_list_smb_path)

    # Script ends naturally here if skipped or after main_processing completes
