# -*- coding: utf-8 -*-
"""
Stage 1: Extract & Compare - PostgreSQL Catalog vs. NAS Filesystem (using pysmb)

This script performs the first stage of a data synchronization process.
It connects to a PostgreSQL database to retrieve a catalog of files
(expected to be present) for a specific document source. It then connects
to a Network Attached Storage (NAS) device via SMB (using pysmb) to list
the actual files present in the corresponding source directory.

Finally, it compares the database catalog against the NAS file list based
on filenames and modification dates to determine:
1.  Files present on NAS but not in the DB catalog (new files).
2.  Files present in both but updated on NAS (updated files).
3.  Files present in the DB catalog but not on NAS (implicitly handled
    by not being in the 'new' or 'updated' lists).
4.  DB records corresponding to updated NAS files (to be deleted before re-insertion).

The results of the comparison (files to process and files to delete from DB)
are saved as JSON files to a specified output directory on the NAS.

Configuration for database, NAS, and processing parameters should be
set in the 'Configuration' section below or externalized to a config file.
"""

import psycopg2
import pandas as pd
import sys
import os
# --- Use pysmb instead of smbclient ---
from smb.SMBConnection import SMBConnection
from smb import smb_structs
import io # For writing strings to NAS
# --- End pysmb import ---
# --- Add SQLAlchemy import ---
import sqlalchemy
# --- End SQLAlchemy import ---
from datetime import datetime, timezone
import socket # For gethostname

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

# --- NAS Configuration ---
# Network attached storage connection parameters
NAS_PARAMS = {
    "ip": "your_nas_ip",
    "share": "your_share_name",
    "user": "your_nas_user",
    "password": "your_nas_password",
    "port": 445 # Default SMB port (can be 139)
}
# Base path on the NAS share containing the root folders for different document sources
NAS_BASE_INPUT_PATH = "path/to/your/base_input_folder"
# Base path on the NAS share where output JSON files will be stored
NAS_OUTPUT_FOLDER_PATH = "path/to/your/output_folder"

# --- Processing Configuration ---
# Define the specific document source to process in this run.
# This value is used for filtering the DB query and constructing NAS paths.
DOCUMENT_SOURCE = 'internal_esg'
# The name of the database table containing the file catalog.
DB_TABLE_NAME = 'apg_catalog'

# --- Full Refresh Mode ---
# Set to True to ignore DB/NAS comparison and process ALL NAS files,
# marking ALL existing DB records for this source for deletion.
FULL_REFRESH = False # Default is False (incremental update)

# --- pysmb Configuration ---
# Increase timeout for potentially slow NAS operations
smb_structs.SUPPORT_SMB2 = True # Enable SMB2/3 support if available
smb_structs.MAX_PAYLOAD_SIZE = 65536 # Can sometimes help with large directories
CLIENT_HOSTNAME = socket.gethostname() # Get local machine name for SMB connection

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
        print(f"   Successfully connected to NAS: {NAS_PARAMS['ip']}:{NAS_PARAMS['port']} on share '{NAS_PARAMS['share']}'")
        return conn
    except Exception as e:
        print(f"   [ERROR] Exception creating NAS connection: {e}")
        return None

def ensure_nas_dir_exists(conn, share_name, dir_path):
    """Ensures a directory exists on the NAS, creating it if necessary."""
    if not conn:
        print("   [ERROR] Cannot ensure NAS directory: No connection.")
        return False
    
    # pysmb needs paths relative to the share, using '/' as separator
    path_parts = dir_path.strip('/').split('/')
    current_path = ''
    try:
        for part in path_parts:
            if not part: continue
            current_path = os.path.join(current_path, part).replace('\\', '/')
            try:
                # Check if it exists by trying to list it
                conn.listPath(share_name, current_path)
                # print(f"      Directory exists: {current_path}")
            except Exception: # If listPath fails, assume it doesn't exist
                print(f"      Creating directory on NAS: {current_path}")
                conn.createDirectory(share_name, current_path)
        return True
    except Exception as e:
        print(f"   [ERROR] Failed to ensure/create NAS directory '{dir_path}': {e}")
        return False

def write_json_to_nas(share_name, nas_path_relative, data_string):
    """
    Writes a string (expected to be JSON) to a specified file path on the NAS using pysmb.

    Args:
        share_name (str): The name of the NAS share.
        nas_path_relative (str): The path relative to the share root (e.g., 'path/to/file.json').
        data_string (str): The string content to write to the file.

    Returns:
        bool: True if the write operation was successful, False otherwise.
    """
    conn = None
    print(f"   Attempting to write to NAS path: {share_name}/{nas_path_relative}")
    try:
        conn = create_nas_connection()
        if not conn:
            return False

        # Ensure the directory exists before writing the file
        dir_path = os.path.dirname(nas_path_relative).replace('\\', '/')
        if dir_path and not ensure_nas_dir_exists(conn, share_name, dir_path):
             print(f"   [ERROR] Failed to ensure output directory exists: {dir_path}")
             return False

        # Convert string to bytes and use BytesIO for pysmb storeFile
        data_bytes = data_string.encode('utf-8')
        file_obj = io.BytesIO(data_bytes)

        # Store the file
        bytes_written = conn.storeFile(share_name, nas_path_relative, file_obj)
        print(f"   Successfully wrote {bytes_written} bytes to: {share_name}/{nas_path_relative}")
        return True
    except Exception as e:
        print(f"   [ERROR] Unexpected error writing to NAS '{share_name}/{nas_path_relative}': {e}")
        return False
    finally:
        if conn:
            conn.close()
            # print("   NAS connection closed after write.")

def get_nas_files(share_name, base_folder_path):
    """
    Lists files recursively from a specific directory on an SMB share using pysmb.

    Retrieves file metadata including name, full path relative to the share root,
    size, and last modified timestamp (UTC). Skips common system/temporary files.

    Args:
        share_name (str): The name of the SMB share.
        base_folder_path (str): The path within the share to start listing from
                                (relative to the share root, e.g., 'data/source1').

    Returns:
        list[dict] | None: A list of dictionaries, where each dictionary represents
                           a file and contains 'file_name', 'file_path', 'file_size',
                           'date_last_modified', and 'date_created' keys (date_created might be same as modified).
                           Returns None if an error occurs during listing or connection.
    """
    files_list = []
    conn = None
    print(f" -> Attempting to list files from NAS path: {share_name}/{base_folder_path}")

    try:
        conn = create_nas_connection()
        if not conn:
            return None

        # Check if base path exists by trying to list it
        try:
             conn.listPath(share_name, base_folder_path)
        except Exception as e:
             print(f"   [ERROR] Base NAS path does not exist or is inaccessible: {share_name}/{base_folder_path} - {e}")
             return None # Critical error, cannot proceed

        print(f" -> Walking directory tree: {share_name}/{base_folder_path} ...")
        # --- Recursive function to walk directories ---
        def walk_nas_path(current_path_relative):
            try:
                items = conn.listPath(share_name, current_path_relative)
                for item in items:
                    # Skip '.' and '..' directory entries
                    if item.filename == '.' or item.filename == '..':
                        continue

                    # Skip common temporary or system files
                    if item.filename == '.DS_Store' or item.filename.startswith('~$') or item.filename.startswith('.'):
                        print(f"      Skipping system/temporary file: {item.filename}")
                        continue

                    # Construct the full path relative to the share root
                    full_path_relative = os.path.join(current_path_relative, item.filename).replace('\\', '/')

                    if item.isDirectory:
                        # Recurse into subdirectories
                        walk_nas_path(full_path_relative)
                    else:
                        # It's a file, process it
                        try:
                            # Convert modification time (epoch float) to timezone-aware UTC datetime
                            # pysmb's last_write_time is typically epoch timestamp
                            last_modified_dt = datetime.fromtimestamp(item.last_write_time, tz=timezone.utc)
                            # pysmb doesn't reliably provide creation time, use modification time as fallback
                            created_dt = datetime.fromtimestamp(item.create_time if item.create_time else item.last_write_time, tz=timezone.utc)

                            # Append file details to the list
                            files_list.append({
                                'file_name': item.filename,
                                'file_path': full_path_relative, # Store path relative to share root
                                'file_size': item.file_size,
                                'date_last_modified': last_modified_dt,
                                'date_created': created_dt # Add the determined creation date (might be same as modified)
                            })
                        except Exception as file_err:
                            print(f"      [WARNING] Could not process file '{full_path_relative}': {file_err}. Skipping file.")

            except Exception as list_err:
                 print(f"      [WARNING] Error listing path '{current_path_relative}': {list_err}. Skipping directory.")
        # --- End of recursive function ---

        # Start the walk from the base folder path
        walk_nas_path(base_folder_path)

        print(f" <- Successfully listed {len(files_list)} files from NAS.")
        return files_list

    except Exception as e:
        # Catch unexpected errors during the overall process
        print(f"   [ERROR] Unexpected error listing NAS files from '{share_name}/{base_folder_path}': {e}")
        return None # Indicate failure
    finally:
        if conn:
            conn.close()
            # print("   NAS connection closed after listing.")


# ==============================================================================
# --- Main Execution Logic ---
# ==============================================================================

if __name__ == "__main__":

    print("\n" + "="*60)
    print(f"--- Running Stage 1: Extract & Compare (using pysmb) ---")
    print(f"--- Document Source: {DOCUMENT_SOURCE} ---")
    print(f"--- DB Catalog Table: {DB_TABLE_NAME} ---")
    print("="*60 + "\n")

    # --- Construct NAS Paths (Relative to Share) ---
    print("[1] Constructing NAS Input and Output Paths (Relative)...")
    # Input path specific to the document source, relative to share
    nas_input_path_relative = os.path.join(NAS_BASE_INPUT_PATH, DOCUMENT_SOURCE).replace('\\', '/')
    print(f"   NAS Input Path (Relative): {NAS_PARAMS['share']}/{nas_input_path_relative}")

    # Output directory specific to the document source, relative to share
    nas_output_dir_relative = os.path.join(NAS_OUTPUT_FOLDER_PATH, DOCUMENT_SOURCE).replace('\\', '/')
    print(f"   NAS Output Directory (Relative): {NAS_PARAMS['share']}/{nas_output_dir_relative}")

    # Define specific output file paths (relative to share)
    db_output_relative_file = os.path.join(nas_output_dir_relative, '1A_catalog_in_postgres.json').replace('\\', '/')
    nas_output_relative_file = os.path.join(nas_output_dir_relative, '1B_files_in_nas.json').replace('\\', '/')
    process_output_relative_file = os.path.join(nas_output_dir_relative, '1C_nas_files_to_process.json').replace('\\', '/')
    delete_output_relative_file = os.path.join(nas_output_dir_relative, '1D_postgres_files_to_delete.json').replace('\\', '/')
    print(f"   DB Catalog Output File (Relative): {os.path.basename(db_output_relative_file)}")
    print(f"   NAS File List Output File (Relative): {os.path.basename(nas_output_relative_file)}")
    print(f"   Files to Process Output File (Relative): {os.path.basename(process_output_relative_file)}")
    print(f"   Files to Delete Output File (Relative): {os.path.basename(delete_output_relative_file)}")
    print("-" * 60)

    # --- Ensure NAS Output Directory Exists ---
    print("[2] Ensuring NAS Output Directory Exists...")
    conn_check = None
    try:
        conn_check = create_nas_connection()
        if not conn_check:
             print("   [CRITICAL ERROR] Failed to connect to NAS to check/create output directory.")
             sys.exit(1)
        if not ensure_nas_dir_exists(conn_check, NAS_PARAMS["share"], nas_output_dir_relative):
            print(f"   [CRITICAL ERROR] Failed to create/access NAS output directory '{nas_output_dir_relative}'.")
            sys.exit(1) # Cannot proceed without output directory
        else:
             print(f"   NAS output directory ensured: '{NAS_PARAMS['share']}/{nas_output_dir_relative}'")
    except Exception as e:
        print(f"   [CRITICAL ERROR] Unexpected error creating/accessing NAS directory '{nas_output_dir_relative}': {e}")
        sys.exit(1) # Cannot proceed
    finally:
        if conn_check:
            conn_check.close()
            # print("   NAS connection closed after directory check.")
    print("-" * 60)

    # Initialize variables
    db_engine = None # Changed from db_conn
    db_df = pd.DataFrame() # Initialize as empty DataFrame
    nas_df = pd.DataFrame() # Initialize as empty DataFrame

    # --- Get Data from PostgreSQL Catalog using SQLAlchemy ---
    print(f"[3] Fetching Data from PostgreSQL Table: '{DB_TABLE_NAME}'...")
    try:
        # Construct database URL for SQLAlchemy
        # Ensure password is handled correctly if it contains special characters (though create_engine usually handles this)
        db_url = f"postgresql+psycopg2://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}"
        print(f"   Creating SQLAlchemy engine for database '{DB_PARAMS['dbname']}' on {DB_PARAMS['host']}...")
        db_engine = sqlalchemy.create_engine(db_url)
        print("   Engine created.")

        # SQL query to select relevant columns for the specified document source
        # Use SQLAlchemy text for parameter binding with engines
        query = sqlalchemy.text(f"""
            SELECT id, file_name, file_path, date_last_modified, file_size,
                   document_source, document_type, document_name
            FROM {DB_TABLE_NAME}
            WHERE document_source = :source;
        """)
        print(f"   Executing query for document_source = '{DOCUMENT_SOURCE}'...")

        # Execute query and load results into a pandas DataFrame using the engine
        # Pass parameters in a dictionary for SQLAlchemy's text() object
        db_df = pd.read_sql_query(query, db_engine, params={"source": DOCUMENT_SOURCE})

        # --- Timestamp Handling (Database) ---
        if not db_df.empty and 'date_last_modified' in db_df.columns and not db_df['date_last_modified'].isnull().all():
            print("   Processing database timestamps...")
            # Convert to datetime objects, coercing errors to NaT
            db_df['date_last_modified'] = pd.to_datetime(db_df['date_last_modified'], errors='coerce')
            # Filter out NaT values before attempting timezone operations
            valid_db_dates = db_df['date_last_modified'].notna()
            # Check if timestamps are timezone-naive (only on valid dates)
            if not db_df.loc[valid_db_dates].empty and db_df.loc[valid_db_dates, 'date_last_modified'].dt.tz is None:
                print("   Localizing naive DB timestamps to UTC...")
                # Assume naive timestamps are UTC, make them timezone-aware (only on valid dates)
                db_df.loc[valid_db_dates, 'date_last_modified'] = db_df.loc[valid_db_dates, 'date_last_modified'].dt.tz_localize('UTC')
            elif not db_df.loc[valid_db_dates].empty:
                # Ensure all timestamps are in UTC for consistent comparison (only on valid dates)
                print("   Converting timezone-aware DB timestamps to UTC...")
                db_df.loc[valid_db_dates, 'date_last_modified'] = db_df.loc[valid_db_dates, 'date_last_modified'].dt.tz_convert('UTC')
            print("   Database timestamps processed (errors coerced to NaT).")
        elif db_df.empty:
             print("   Database catalog is empty for this source.")
        else:
             print("   No 'date_last_modified' data found or column missing in DB results.")


        print(f"   Query successful. Found {len(db_df)} records in DB catalog for '{DOCUMENT_SOURCE}'.")

        # --- Save DB Catalog to NAS ---
        print(f"\n   Saving DB catalog data to NAS file: '{os.path.basename(db_output_relative_file)}'...")
        # Convert DataFrame to JSON string with ISO date format
        db_json_string = db_df.to_json(orient='records', indent=4, date_format='iso')
        if not write_json_to_nas(NAS_PARAMS["share"], db_output_relative_file, db_json_string):
             print("   [CRITICAL ERROR] Failed to write DB catalog JSON to NAS. Exiting.")
             sys.exit(1) # Exit if saving fails

    # Removed specific psycopg2.Error catch
    except Exception as e:
        print(f"   [CRITICAL ERROR] An unexpected error occurred during DB operations: {e}")
        # Dispose the engine if it was created, releasing connection pool resources
        if db_engine is not None:
            db_engine.dispose()
        sys.exit(1) # Exit on other unexpected errors
    finally:
        # Dispose the engine to close connections in the pool
        if db_engine is not None:
            db_engine.dispose()
            print("\n   SQLAlchemy engine disposed.")
    print("-" * 60)


    # --- Get File List from NAS ---
    print(f"[4] Listing Files from NAS Source Path: '{NAS_PARAMS['share']}/{nas_input_path_relative}'...")
    nas_files_list = get_nas_files(
        NAS_PARAMS["share"],
        nas_input_path_relative # Use the relative path within the share
    )

    # Check if file listing was successful
    if nas_files_list is None:
        print("   [CRITICAL ERROR] Failed to retrieve file list from NAS. Exiting.")
        sys.exit(1) # Exit if NAS listing failed

    # Convert the list of file dictionaries to a pandas DataFrame
    nas_df = pd.DataFrame(nas_files_list)

    # --- Timestamp Handling (NAS) ---
    # NAS timestamps from get_nas_files should already be UTC-aware datetime objects
    if not nas_df.empty and 'date_last_modified' in nas_df.columns:
         print("   Processing NAS timestamps...")
         # Convert to datetime objects again just in case, coercing errors to NaT
         nas_df['date_last_modified'] = pd.to_datetime(nas_df['date_last_modified'], errors='coerce')
         # Filter out NaT values before attempting timezone conversion
         valid_nas_dates = nas_df['date_last_modified'].notna()
         # Ensure valid timestamps are timezone-aware UTC (get_nas_files should already do this, but be robust)
         if not nas_df.loc[valid_nas_dates].empty:
             if nas_df.loc[valid_nas_dates, 'date_last_modified'].dt.tz is None:
                 print("   Localizing naive NAS timestamps to UTC...")
                 nas_df.loc[valid_nas_dates, 'date_last_modified'] = nas_df.loc[valid_nas_dates, 'date_last_modified'].dt.tz_localize('UTC')
             else:
                 print("   Converting timezone-aware NAS timestamps to UTC...")
                 nas_df.loc[valid_nas_dates, 'date_last_modified'] = nas_df.loc[valid_nas_dates, 'date_last_modified'].dt.tz_convert('UTC')
         print("   NAS timestamps processed (errors coerced to NaT).")
    elif nas_df.empty:
        print("   NAS directory is empty or contains no listable files.")
    else:
        print("   No 'date_last_modified' data found in NAS file list.")


    # --- Save NAS File List to NAS ---
    print(f"\n   Saving NAS file list data to NAS file: '{os.path.basename(nas_output_relative_file)}'...")
    # Convert DataFrame to JSON string
    nas_json_string = nas_df.to_json(orient='records', indent=4, date_format='iso')
    if not write_json_to_nas(NAS_PARAMS["share"], nas_output_relative_file, nas_json_string):
        print("   [CRITICAL ERROR] Failed to write NAS file list JSON to NAS. Exiting.")
        sys.exit(1) # Exit if saving fails
    print("-" * 60)


    # --- Compare DB Catalog and NAS File List ---
    # (Comparison logic remains largely the same, using the generated DataFrames)
    print("[5] Comparing Database Catalog and NAS File List...")
    print(f"   Database records: {len(db_df)}")
    print(f"   NAS files found: {len(nas_df)}")

    # Initialize comparison result DataFrames
    files_to_process = pd.DataFrame(columns=['file_name', 'file_path', 'file_size', 'date_last_modified', 'date_created', 'reason'])
    files_to_delete = pd.DataFrame(columns=['id', 'file_name', 'file_path', 'document_source', 'document_type', 'document_name']) # Files to delete from DB
    both_files = pd.DataFrame() # Initialize to ensure it exists even in full refresh mode

    # --- Check for Full Refresh Mode ---
    if FULL_REFRESH:
        print("\n   *** FULL REFRESH MODE ENABLED ***")
        print("   Skipping DB/NAS comparison.")

        if nas_df.empty:
            print("   NAS directory is empty. No files to process even in full refresh mode.")
            files_to_process = pd.DataFrame(columns=['file_name', 'file_path', 'file_size', 'date_last_modified', 'date_created', 'reason'])
        else:
            print(f"   Marking all {len(nas_df)} NAS files for processing.")
            # Select necessary columns from nas_df
            process_cols = ['file_name', 'file_path', 'file_size', 'date_last_modified', 'date_created']
            existing_process_cols = [col for col in process_cols if col in nas_df.columns]
            files_to_process = nas_df[existing_process_cols].copy()
            files_to_process['reason'] = 'full_refresh'

        if db_df.empty:
            print("   DB catalog is empty for this source. No existing records to mark for deletion.")
            files_to_delete = pd.DataFrame(columns=['id', 'file_name', 'file_path', 'document_source', 'document_type', 'document_name'])
        else:
            print(f"   Marking all {len(db_df)} existing DB records for deletion.")
            # Select necessary columns from db_df
            delete_cols = ['id', 'file_name', 'file_path', 'document_source', 'document_type', 'document_name']
            existing_delete_cols = [col for col in delete_cols if col in db_df.columns]
            files_to_delete = db_df[existing_delete_cols].copy()

    else:
        # --- Incremental Update Logic (Original Comparison) ---
        print("\n   --- Incremental Update Mode ---")
        # Handle edge cases: both empty, NAS empty, DB empty
        if db_df.empty and nas_df.empty:
            print("   Result: Both DB catalog and NAS list are empty. No actions needed.")
        elif nas_df.empty:
            print("   Result: NAS list is empty. No files to process. All DB entries are potentially stale (but not deleted by this script).")
            # files_to_delete remains empty as we only delete based on NAS updates here
        elif db_df.empty:
            print("   Result: DB catalog is empty. All NAS files are considered 'new'.")
            # Include date_created when selecting columns
            new_cols = ['file_name', 'file_path', 'file_size', 'date_last_modified', 'date_created']
            existing_new_cols = [col for col in new_cols if col in nas_df.columns]
            files_to_process = nas_df[existing_new_cols].copy()
            files_to_process['reason'] = 'new'
            # files_to_delete remains empty
        else:
            # --- Perform the Comparison using Merge ---
            print("   Performing comparison based on 'file_name'...") # Using file_name as key assumes names are unique within the source folder

            # Ensure 'file_name' columns exist and are string type for reliable merging
            if 'file_name' not in nas_df.columns:
                print("   [CRITICAL ERROR] 'file_name' column missing in NAS DataFrame. Exiting.")
                sys.exit(1)
            if 'file_name' not in db_df.columns:
                print("   [CRITICAL ERROR] 'file_name' column missing in DB DataFrame. Exiting.")
                sys.exit(1)

            nas_df['file_name'] = nas_df['file_name'].astype(str)
            db_df['file_name'] = db_df['file_name'].astype(str)

            # Merge NAS file list (left) with DB catalog (right)
            # 'outer' merge keeps all rows from both DataFrames
            # 'indicator=True' adds a '_merge' column indicating the source of each row
            # NOTE: We merge on 'file_name' ONLY. If paths differ but names match, this treats them as the same file.
            # Consider merging on 'file_path' if paths should also match.
            comparison_df = pd.merge(
                nas_df,
                db_df,
                on='file_name', # Key to join on
                how='outer',
                suffixes=('_nas', '_db'), # Suffixes for overlapping columns (like file_path, date_last_modified)
                indicator=True # Adds '_merge' column ('left_only', 'right_only', 'both')
            )

            # --- Identify New Files ---
            # Files present only on NAS ('left_only' in the merge)
            new_files_mask = comparison_df['_merge'] == 'left_only'
            # Use columns from the NAS side (left)
            new_files_cols = ['file_name', 'file_path_nas', 'file_size_nas', 'date_last_modified_nas', 'date_created_nas']
            # Ensure all expected columns actually exist in comparison_df before selecting
            existing_new_cols = [col for col in new_files_cols if col in comparison_df.columns]
            if len(existing_new_cols) != len(new_files_cols):
                missing_cols = set(new_files_cols) - set(existing_new_cols)
                print(f"   [WARNING] Could not find expected columns {missing_cols} in merged data for new files list. Check merge logic.")
            new_files = comparison_df.loc[new_files_mask, existing_new_cols].copy()
            # Rename columns with suffixes
            new_files.rename(columns={
                'file_path_nas': 'file_path',
                'file_size_nas': 'file_size',
                'date_last_modified_nas': 'date_last_modified',
                'date_created_nas': 'date_created' # Rename NAS creation date
            }, inplace=True)
            new_files['reason'] = 'new'
            print(f"      Identified {len(new_files)} new files (present on NAS, not in DB).")

            # --- Identify Updated Files and Corresponding DB Entries to Delete ---
            # Files present in both NAS and DB ('both' in the merge)
            both_files_mask = comparison_df['_merge'] == 'both'
            both_files = comparison_df[both_files_mask].copy()
            updated_files_nas = pd.DataFrame(columns=['file_name', 'file_path', 'file_size', 'date_last_modified', 'date_created', 'reason']) # Initialize empty
            files_to_delete = pd.DataFrame(columns=['id', 'file_name', 'file_path', 'document_source', 'document_type', 'document_name']) # Initialize empty
            updated_mask = pd.Series(dtype=bool) # Initialize empty mask

            # Only perform update check if there are files present in both sources
            if not both_files.empty:
                # Check for updates based on modification time (NAS newer than DB)
                # Ensure both date columns are valid datetimes before comparison
                if 'date_last_modified_nas' in both_files.columns and 'date_last_modified_db' in both_files.columns:
                    # Handle potential NaT values before comparison
                    valid_dates_mask = both_files['date_last_modified_nas'].notna() & both_files['date_last_modified_db'].notna()
                    # Compare timestamps truncated to the minute (only on rows with valid dates)
                    # Initialize updated_mask for all 'both' rows as False
                    updated_mask = pd.Series(False, index=both_files.index)
                    # Apply comparison only where dates are valid
                    updated_mask[valid_dates_mask] = (
                        both_files.loc[valid_dates_mask, 'date_last_modified_nas'].dt.floor('min') >
                        both_files.loc[valid_dates_mask, 'date_last_modified_db'].dt.floor('min')
                    )

                    # Get NAS details for files identified as updated
                    # Use columns from the NAS side (left)
                    updated_files_cols = ['file_name', 'file_path_nas', 'file_size_nas', 'date_last_modified_nas', 'date_created_nas']
                    # Ensure all expected columns actually exist in both_files before selecting
                    existing_updated_cols = [col for col in updated_files_cols if col in both_files.columns]
                    if len(existing_updated_cols) != len(updated_files_cols):
                        missing_cols = set(updated_files_cols) - set(existing_updated_cols)
                        print(f"   [WARNING] Could not find expected columns {missing_cols} in merged data for updated files list. Check merge logic.")
                    # Select rows using the boolean mask 'updated_mask'
                    updated_files_nas = both_files.loc[updated_mask, existing_updated_cols].copy()
                    # Rename columns with suffixes
                    updated_files_nas.rename(columns={
                        'file_path_nas': 'file_path',
                        'file_size_nas': 'file_size',
                        'date_last_modified_nas': 'date_last_modified',
                        'date_created_nas': 'date_created' # Rename NAS creation date
                    }, inplace=True)
                    updated_files_nas['reason'] = 'updated'
                    print(f"      Identified {len(updated_files_nas)} updated files (newer on NAS than in DB).")

                    # Get DB details for files that need to be deleted because they were updated
                    # Select the original column names from the DB side of the merge
                    # (document_source, document_type, document_name don't get suffixes as they only exist in db_df)
                    db_cols_to_keep = ['id', 'file_name', 'file_path_db', 'document_source', 'document_type', 'document_name']
                    # Ensure all required columns exist before selecting
                    existing_db_cols = [col for col in db_cols_to_keep if col in both_files.columns]
                    if len(existing_db_cols) != len(db_cols_to_keep):
                        missing_cols = set(db_cols_to_keep) - set(existing_db_cols)
                        print(f"   [WARNING] Could not find expected columns {missing_cols} in merged data for deletion list. Check merge logic.")
                    # Select rows using the boolean mask 'updated_mask'
                    files_to_delete = both_files.loc[updated_mask, existing_db_cols].copy()
                    # Rename only the column that definitely has a suffix
                    files_to_delete.rename(columns={'file_path_db': 'file_path'}, inplace=True)
                    print(f"      Identified {len(files_to_delete)} DB records to delete (corresponding to updated files).")
                else:
                    print("      [WARNING] 'date_last_modified' columns missing in merged data for update check. Skipping update detection.")
                    # updated_files_nas and files_to_delete remain empty as initialized
            else:
                print("      No files found in both NAS and DB. Skipping update check.")
             # updated_files_nas and files_to_delete remain empty as initialized

        # --- Detailed File-by-File Comparison Logging ---
        # This logging should still work even if both_files is empty
        print("\n   Detailed Comparison Results:")
        # Use iterrows for more robust column access
        for index, row in comparison_df.iterrows():
            file_name = row.get('file_name', 'N/A') # Use .get for safety
            nas_time = row.get('date_last_modified_nas', pd.NaT)
            db_time = row.get('date_last_modified_db', pd.NaT)
            merge_status = row.get('_merge', None) # Get the merge status, default to None

            # Format timestamps for printing (handle NaT)
            nas_time_str = nas_time.isoformat() if pd.notna(nas_time) else "N/A (Not on NAS)"
            db_time_str = db_time.isoformat() if pd.notna(db_time) else "N/A (Not in DB)"

            # Determine status based on merge indicator and timestamp comparison
            status = "Error (Processing Failed)" # Default error status

            # Explicitly check if the merge status object is null/None first
            if pd.isna(merge_status):
                status = "Error (Missing Merge Status)"
            else:
                # Convert the non-null merge status object to string
                merge_status_val = str(merge_status)

                if merge_status_val == 'left_only':
                    status = "New"
                elif merge_status_val == 'right_only':
                    status = "Deleted/Moved (DB only)"
                elif merge_status_val == 'both':
                    # Re-check timestamps directly for 'both' rows, comparing down to the minute
                    if pd.notna(nas_time) and pd.notna(db_time):
                        nas_time_minute = nas_time.floor('min')
                        db_time_minute = db_time.floor('min')
                        if nas_time_minute > db_time_minute:
                            status = "Updated"
                        elif nas_time_minute == db_time_minute:
                            status = "Unchanged"
                        else: # nas_time_minute < db_time_minute (DB is newer? Should be rare)
                            status = "Unchanged (DB Newer/Same)" # Treat DB newer as unchanged for processing
                    else:
                        # Handle missing dates if merge status is 'both'
                        status = "Error (Missing Date in 'both')"
                else:
                    # Fallback for any unexpected merge status string
                    status = f"Error (Unexpected Merge Value: {merge_status_val})"
            # Note: The case where pd.isna(merge_status) is true is handled above


            print(f"      - File: {file_name}")
            print(f"        NAS Time: {nas_time_str}")
            print(f"        DB Time : {db_time_str}")
            print(f"        Status  : {status}")
        print("-" * 30) # Separator after detailed list

        # --- Combine New and Updated Files for Processing ---
        # Note: The DataFrames new_files and updated_files_nas are still created based on the masks above
        files_to_process = pd.concat([new_files, updated_files_nas], ignore_index=True)

        # --- Identify Files Only in DB (Potentially Deleted/Moved on NAS) ---
        # Files present only in the DB ('right_only' in the merge)
        # Note: This script currently *does not* automatically delete these from the DB.
        # It only identifies DB records to delete when a file is *updated* on the NAS.
        deleted_files_db = comparison_df[comparison_df['_merge'] == 'right_only']
        if not deleted_files_db.empty:
            print(f"      Note: Found {len(deleted_files_db)} files in DB but not on NAS (potentially deleted/moved). No action taken by this script.")
        # --- End of Incremental Update Comparison Logic ---

    # --- Final Summary of Comparison (Applies to both modes) ---
    # Calculate unchanged count (only relevant for incremental mode)
    unchanged_count = 0
    if not both_files.empty: # Check if 'both_files' DataFrame was populated
        if 'updated_mask' in locals() and not updated_mask.empty: # Check if 'updated_mask' exists and is not empty
            # Count rows in 'both_files' that are NOT in 'updated_files_nas' based on the updated_mask
            unchanged_count = len(both_files[~updated_mask])
        else: # If updated_mask wasn't created (e.g., missing date columns), assume all 'both' are unchanged
            unchanged_count = len(both_files)


    print(f"\n   Comparison Summary:")
    print(f"      - NAS Files to Process (New or Updated): {len(files_to_process)}")
    print(f"      - Existing DB Records to Delete (Updated Files): {len(files_to_delete)}")
    print(f"      - Files Found Unchanged (Timestamp Match): {unchanged_count}") # Added this line
    print("-" * 60)


    # --- Save Comparison Results to NAS ---
    print("[6] Saving Comparison Results to NAS...")

    # Save the list of files to be processed (new/updated)
    print(f"   Saving 'files to process' list to: '{os.path.basename(process_output_relative_file)}'...")
    process_json_string = files_to_process.to_json(orient='records', indent=4, date_format='iso')
    if not write_json_to_nas(NAS_PARAMS["share"], process_output_relative_file, process_json_string):
        print("   [CRITICAL ERROR] Failed to write 'files to process' JSON to NAS. Exiting.")
        sys.exit(1)

    # Save the list of DB records to be deleted (corresponding to updated files)
    print(f"   Saving 'files to delete' list to: '{os.path.basename(delete_output_relative_file)}'...")

    # Ensure 'id' column is integer type for JSON compatibility if it exists and has data
    if 'id' in files_to_delete.columns and not files_to_delete['id'].isnull().all():
         files_to_delete['id'] = files_to_delete['id'].astype('Int64') # Use nullable integer type

    # Convert the DataFrame to JSON
    delete_json_string = files_to_delete.to_json(orient='records', indent=4)
    if not write_json_to_nas(NAS_PARAMS["share"], delete_output_relative_file, delete_json_string):
        print("   [CRITICAL ERROR] Failed to write 'files to delete' JSON to NAS. Exiting.")
        sys.exit(1)

    print("-" * 60)

    # --- Create Skip Flag if No Files to Process ---
    print("[7] Managing Flag Files...")
    skip_flag_file_name = '_SKIP_SUBSEQUENT_STAGES.flag'
    refresh_flag_file_name = '_FULL_REFRESH.flag'
    skip_flag_relative_path = os.path.join(nas_output_dir_relative, skip_flag_file_name).replace('\\', '/')
    refresh_flag_relative_path = os.path.join(nas_output_dir_relative, refresh_flag_file_name).replace('\\', '/')
    conn_flag = None

    try:
        conn_flag = create_nas_connection()
        if not conn_flag:
            print("   [WARNING] Failed to connect to NAS to manage flag files. Skipping flag operations.")
        else:
            # --- Skip Flag Logic ---
            if files_to_process.empty:
                print(f"   No files to process found. Creating skip flag file: '{skip_flag_file_name}'")
                try:
                    # Create an empty file as a flag
                    conn_flag.storeFile(NAS_PARAMS["share"], skip_flag_relative_path, io.BytesIO(b''))
                    print(f"   Successfully created skip flag file: {skip_flag_relative_path}")
                except Exception as e:
                    print(f"   [WARNING] Error creating skip flag file '{skip_flag_relative_path}': {e}")
            else:
                print(f"   Files found for processing ({len(files_to_process)}). Ensuring skip flag does not exist.")
                try:
                    # Attempt to delete the flag file if it exists
                    conn_flag.deleteFiles(NAS_PARAMS["share"], skip_flag_relative_path)
                    print(f"   Removed potentially existing skip flag file: {skip_flag_relative_path}")
                except Exception as e:
                    # deleteFiles might fail if the file doesn't exist, which is fine.
                    # Check if the error message indicates "No such file" or similar
                    if "OBJECT_NAME_NOT_FOUND" not in str(e) and "STATUS_NO_SUCH_FILE" not in str(e):
                         print(f"   [INFO] Error removing skip flag file (may not exist or other issue): {e}")
                    else:
                         print(f"   Skip flag file did not exist.")


            # --- Full Refresh Flag Logic ---
            if FULL_REFRESH:
                print(f"   Full refresh mode enabled. Creating refresh flag file: '{refresh_flag_file_name}'")
                try:
                    conn_flag.storeFile(NAS_PARAMS["share"], refresh_flag_relative_path, io.BytesIO(b''))
                    print(f"   Successfully created refresh flag file: {refresh_flag_relative_path}")
                except Exception as e:
                    print(f"   [WARNING] Error creating refresh flag file '{refresh_flag_relative_path}': {e}")
            else:
                print(f"   Incremental mode. Ensuring refresh flag does not exist.")
                try:
                    conn_flag.deleteFiles(NAS_PARAMS["share"], refresh_flag_relative_path)
                    print(f"   Removed potentially existing refresh flag file: {refresh_flag_relative_path}")
                except Exception as e:
                    if "OBJECT_NAME_NOT_FOUND" not in str(e) and "STATUS_NO_SUCH_FILE" not in str(e):
                        print(f"   [INFO] Error removing refresh flag file (may not exist or other issue): {e}")
                    else:
                        print(f"   Refresh flag file did not exist.")

    except Exception as e:
        print(f"   [WARNING] Unexpected error during flag file management: {e}")
    finally:
        if conn_flag:
            conn_flag.close()
            # print("   NAS connection closed after flag management.")


    print("-" * 60)
    print("\n" + "="*60)
    print(f"--- Stage 1 Completed Successfully (using pysmb) ---")
    print("--- Output JSON files generated on NAS ---")
    print("="*60 + "\n")
