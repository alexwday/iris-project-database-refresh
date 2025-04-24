# -*- coding: utf-8 -*-
"""
Stage 1: Extract & Compare - PostgreSQL Catalog vs. NAS Filesystem

This script performs the first stage of a data synchronization process.
It connects to a PostgreSQL database to retrieve a catalog of files
(expected to be present) for a specific document source. It then connects
to a Network Attached Storage (NAS) device via SMB to list the actual files
present in the corresponding source directory.

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
import smbclient # For NAS connection
from datetime import datetime, timezone

# ==============================================================================
# --- Configuration ---
# ==============================================================================

# --- Database Configuration ---
# IMPORTANT: Replace placeholder values with your actual database credentials.
DB_PARAMS = {
    "host": "localhost",
    "port": "5432",
    "dbname": "maven-finance",
    "user": "your_username",  # Replace with your DB username
    "password": "your_password"   # Replace with your DB password
}

# --- NAS Configuration ---
# IMPORTANT: Replace placeholder values with your actual NAS details.
NAS_PARAMS = {
    "ip": "your_nas_ip",          # Replace with NAS IP address
    "share": "your_share_name",   # Replace with NAS share name
    "user": "your_nas_user",      # Replace with NAS username
    "password": "your_nas_password" # Replace with NAS password
}
# Base path on the NAS share containing the root folders for different document sources.
NAS_BASE_INPUT_PATH = "path/to/your/base_input_folder"
# Base path on the NAS share where output JSON files will be stored.
NAS_OUTPUT_FOLDER_PATH = "path/to/your/output_folder"

# --- Processing Configuration ---
# Define the specific document source to process in this run.
# This value is used for filtering the DB query and constructing NAS paths.
DOCUMENT_SOURCE = 'internal_esg'
# The name of the database table containing the file catalog.
DB_TABLE_NAME = 'apg_catalog'

# ==============================================================================
# --- Helper Functions ---
# ==============================================================================

def write_json_to_nas(smb_path, data_string):
    """
    Writes a string (expected to be JSON) to a specified file path on the NAS.

    Args:
        smb_path (str): The full SMB path (e.g., //nas_ip/share/path/to/file.json).
        data_string (str): The string content to write to the file.

    Returns:
        bool: True if the write operation was successful, False otherwise.
    """
    print(f"   Attempting to write to NAS path: {smb_path}")
    try:
        # Ensure the directory exists before writing the file
        dir_path = os.path.dirname(smb_path)
        if not smbclient.path.exists(dir_path):
             print(f"   Creating directory on NAS: {dir_path}")
             smbclient.makedirs(dir_path, exist_ok=True)

        with smbclient.open_file(smb_path, mode='w', encoding='utf-8') as f:
            f.write(data_string)
        print(f"   Successfully wrote to: {smb_path}")
        return True
    except smbclient.SambaClientError as e:
        print(f"   [ERROR] SMB Error writing to '{smb_path}': {e}")
        return False
    except Exception as e:
        print(f"   [ERROR] Unexpected error writing to NAS '{smb_path}': {e}")
        return False

def get_nas_files(nas_ip, share_name, base_folder_path, username, password):
    """
    Lists files recursively from a specific directory on an SMB share.

    Retrieves file metadata including name, full path relative to the share root,
    size, and last modified timestamp (UTC). Skips common system/temporary files.

    Args:
        nas_ip (str): The IP address of the NAS.
        share_name (str): The name of the SMB share.
        base_folder_path (str): The path within the share to start listing from
                                (relative to the share root, e.g., 'data/source1').
        username (str): The username for NAS authentication.
        password (str): The password for NAS authentication.

    Returns:
        list[dict] | None: A list of dictionaries, where each dictionary represents
                           a file and contains 'file_name', 'file_path', 'file_size',
                           'date_last_modified', and 'date_created' keys.
                           Returns None if an error occurs during listing or connection.
    """
    files_list = []
    smb_base_path = f"//{nas_ip}/{share_name}/{base_folder_path}"
    share_prefix = f"//{nas_ip}/{share_name}/" # Used to calculate relative path

    print(f" -> Configuring SMB client for user '{username}'...")
    try:
        # Set credentials globally for this smbclient instance/session
        smbclient.ClientConfig(username=username, password=password)
        print(f" -> Attempting to connect and list files from NAS path: {smb_base_path}")
    except Exception as e:
        # Catch potential config errors, though less common
        print(f"   [ERROR] Error setting SMB client config: {e}")
        # Allow execution to continue to the path check/walk, which will likely fail
        pass

    try:
        # Check if the base path exists before attempting to walk
        if not smbclient.path.exists(smb_base_path):
             print(f"   [ERROR] Base NAS path does not exist or is inaccessible: {smb_base_path}")
             return None # Critical error, cannot proceed

        print(f" -> Walking directory tree: {smb_base_path} ...")
        # Recursively walk the directory structure
        for dirpath, dirnames, filenames in smbclient.walk(smb_base_path):
            for filename in filenames:
                # Skip common temporary or system files
                if filename == '.DS_Store' or filename.startswith('~$') or filename.startswith('.'):
                    print(f"      Skipping system/temporary file: {filename}")
                    continue

                # Construct the full SMB path for the current file
                full_smb_path = os.path.join(dirpath, filename).replace('\\', '/')

                try:
                    # Get file metadata (stat)
                    stat_info = smbclient.stat(full_smb_path)
                    # Convert modification time to timezone-aware UTC datetime
                    last_modified_dt = datetime.fromtimestamp(stat_info.st_mtime, tz=timezone.utc)

                    # Attempt to get creation time (st_birthtime), fallback to modification time
                    creation_timestamp = None
                    if hasattr(stat_info, 'st_birthtime') and stat_info.st_birthtime:
                        try:
                            creation_timestamp = stat_info.st_birthtime
                            print(f"      Found birthtime: {creation_timestamp}") # Debug print
                        except Exception as e:
                             print(f"      [WARNING] Error accessing st_birthtime for {filename}: {e}. Falling back to mtime.")
                             creation_timestamp = stat_info.st_mtime # Fallback
                    else:
                        # print(f"      No st_birthtime found for {filename}. Falling back to mtime.") # Optional verbosity
                        creation_timestamp = stat_info.st_mtime # Fallback if attribute doesn't exist

                    created_dt = datetime.fromtimestamp(creation_timestamp, tz=timezone.utc)


                    # Calculate the file path relative to the share root
                    if full_smb_path.startswith(share_prefix):
                        full_path_from_share = full_smb_path[len(share_prefix):]
                    else:
                        # This case should ideally not happen if paths are constructed correctly
                        print(f"      [WARNING] Unexpected SMB path format encountered: {full_smb_path}. Using full path.")
                        full_path_from_share = full_smb_path # Keep full path as fallback

                    # Append file details to the list
                    files_list.append({
                        'file_name': filename,
                        'file_path': full_path_from_share, # Store path relative to share root
                        'file_size': stat_info.st_size,
                        'date_last_modified': last_modified_dt,
                        'date_created': created_dt # Add the determined creation date
                    })
                except smbclient.SambaClientError as stat_err:
                     print(f"      [WARNING] SMB Error getting stats for file '{full_smb_path}': {stat_err}. Skipping file.")
                except Exception as e:
                    # Catch other potential errors during stat or path processing
                    print(f"      [WARNING] Could not process file '{full_smb_path}': {e}. Skipping file.")

        print(f" <- Successfully listed {len(files_list)} files from NAS.")
        return files_list

    except smbclient.SambaClientError as e:
        print(f"   [ERROR] SMB Error listing files from '{smb_base_path}': {e}")
        return None # Indicate failure
    except Exception as e:
        # Catch unexpected errors during the walk process
        print(f"   [ERROR] Unexpected error listing NAS files from '{smb_base_path}': {e}")
        return None # Indicate failure


# ==============================================================================
# --- Main Execution Logic ---
# ==============================================================================

if __name__ == "__main__":

    print("\n" + "="*60)
    print(f"--- Running Stage 1: Extract & Compare ---")
    print(f"--- Document Source: {DOCUMENT_SOURCE} ---")
    print(f"--- DB Catalog Table: {DB_TABLE_NAME} ---")
    print("="*60 + "\n")

    # --- Construct NAS Paths ---
    print("[1] Constructing NAS Input and Output Paths...")
    # Input path specific to the document source
    nas_input_path_relative = os.path.join(NAS_BASE_INPUT_PATH, DOCUMENT_SOURCE).replace('\\', '/')
    nas_input_smb_path = f"//{NAS_PARAMS['ip']}/{NAS_PARAMS['share']}/{nas_input_path_relative}"
    print(f"   NAS Input Path (SMB): {nas_input_smb_path}")

    # Output directory specific to the document source
    nas_output_dir_relative = os.path.join(NAS_OUTPUT_FOLDER_PATH, DOCUMENT_SOURCE).replace('\\', '/')
    nas_output_dir_smb_path = f"//{NAS_PARAMS['ip']}/{NAS_PARAMS['share']}/{nas_output_dir_relative}"
    print(f"   NAS Output Directory (SMB): {nas_output_dir_smb_path}")

    # Define specific output file paths
    db_output_smb_file = os.path.join(nas_output_dir_smb_path, '1A_catalog_in_postgres.json').replace('\\', '/')
    nas_output_smb_file = os.path.join(nas_output_dir_smb_path, '1B_files_in_nas.json').replace('\\', '/')
    process_output_smb_file = os.path.join(nas_output_dir_smb_path, '1C_nas_files_to_process.json').replace('\\', '/')
    delete_output_smb_file = os.path.join(nas_output_dir_smb_path, '1D_postgres_files_to_delete.json').replace('\\', '/')
    print(f"   DB Catalog Output File: {os.path.basename(db_output_smb_file)}")
    print(f"   NAS File List Output File: {os.path.basename(nas_output_smb_file)}")
    print(f"   Files to Process Output File: {os.path.basename(process_output_smb_file)}")
    print(f"   Files to Delete Output File: {os.path.basename(delete_output_smb_file)}")
    print("-" * 60)

    # --- Ensure NAS Output Directory Exists ---
    print("[2] Ensuring NAS Output Directory Exists...")
    try:
        # Configure credentials (needed again if write_json_to_nas hasn't run yet or in case of separate process)
        smbclient.ClientConfig(username=NAS_PARAMS["user"], password=NAS_PARAMS["password"])
        if not smbclient.path.exists(nas_output_dir_smb_path):
            print(f"   Directory not found. Creating NAS output directory: '{nas_output_dir_smb_path}'")
            smbclient.makedirs(nas_output_dir_smb_path, exist_ok=True) # exist_ok=True prevents error if created between check and make
            print(f"   Successfully created directory.")
        else:
            print(f"   NAS output directory already exists: '{nas_output_dir_smb_path}'")
    except smbclient.SambaClientError as e:
        print(f"   [CRITICAL ERROR] SMB Error creating/accessing directory '{nas_output_dir_smb_path}': {e}")
        sys.exit(1) # Cannot proceed without output directory
    except Exception as e:
        print(f"   [CRITICAL ERROR] Unexpected error creating/accessing NAS directory '{nas_output_dir_smb_path}': {e}")
        sys.exit(1) # Cannot proceed
    print("-" * 60)

    # Initialize variables
    conn = None
    db_df = pd.DataFrame() # Initialize as empty DataFrame
    nas_df = pd.DataFrame() # Initialize as empty DataFrame

    # --- Get Data from PostgreSQL Catalog ---
    print(f"[3] Fetching Data from PostgreSQL Table: '{DB_TABLE_NAME}'...")
    try:
        print(f"   Connecting to database '{DB_PARAMS['dbname']}' on {DB_PARAMS['host']}...")
        conn = psycopg2.connect(**DB_PARAMS)
        print("   Connection successful.")

        # SQL query to select relevant columns for the specified document source
        query = f"""
            SELECT id, file_name, file_path, date_last_modified, file_size,
                   document_source, document_type, document_name
            FROM {DB_TABLE_NAME}
            WHERE document_source = %s;
        """
        print(f"   Executing query for document_source = '{DOCUMENT_SOURCE}'...")

        # Execute query and load results into a pandas DataFrame
        db_df = pd.read_sql_query(query, conn, params=(DOCUMENT_SOURCE,))

        # --- Timestamp Handling (Database) ---
        if not db_df.empty and 'date_last_modified' in db_df.columns and not db_df['date_last_modified'].isnull().all():
            print("   Processing database timestamps...")
            # Convert to datetime objects if not already
            db_df['date_last_modified'] = pd.to_datetime(db_df['date_last_modified'])
            # Check if timestamps are timezone-naive
            if db_df['date_last_modified'].dt.tz is None:
                print("   Localizing naive DB timestamps to UTC...")
                # Assume naive timestamps are UTC, make them timezone-aware
                db_df['date_last_modified'] = db_df['date_last_modified'].dt.tz_localize('UTC')
            else:
                print("   Converting timezone-aware DB timestamps to UTC...")
                # Ensure all timestamps are in UTC for consistent comparison
                db_df['date_last_modified'] = db_df['date_last_modified'].dt.tz_convert('UTC')
            print("   Database timestamps converted to UTC.")
        elif db_df.empty:
             print("   Database catalog is empty for this source.")
        else:
             print("   No 'date_last_modified' data found or column missing in DB results.")


        print(f"   Query successful. Found {len(db_df)} records in DB catalog for '{DOCUMENT_SOURCE}'.")

        # --- Save DB Catalog to NAS ---
        print(f"\n   Saving DB catalog data to NAS file: '{os.path.basename(db_output_smb_file)}'...")
        # Convert DataFrame to JSON string with ISO date format
        db_json_string = db_df.to_json(orient='records', indent=4, date_format='iso')
        if not write_json_to_nas(db_output_smb_file, db_json_string):
             print("   [CRITICAL ERROR] Failed to write DB catalog JSON to NAS. Exiting.")
             sys.exit(1) # Exit if saving fails

    except psycopg2.Error as db_err:
        print(f"   [CRITICAL ERROR] Database error: {db_err}")
        sys.exit(1) # Exit on database errors
    except Exception as e:
        print(f"   [CRITICAL ERROR] An unexpected error occurred during DB operations: {e}")
        sys.exit(1) # Exit on other unexpected errors
    finally:
        # Ensure the database connection is always closed
        if conn is not None:
            conn.close()
            print("\n   Database connection closed.")
    print("-" * 60)


    # --- Get File List from NAS ---
    print(f"[4] Listing Files from NAS Source Path: '{nas_input_smb_path}'...")
    nas_files_list = get_nas_files(
        NAS_PARAMS["ip"],
        NAS_PARAMS["share"],
        nas_input_path_relative, # Use the relative path within the share
        NAS_PARAMS["user"],
        NAS_PARAMS["password"]
    )

    # Check if file listing was successful
    if nas_files_list is None:
        print("   [CRITICAL ERROR] Failed to retrieve file list from NAS. Exiting.")
        sys.exit(1) # Exit if NAS listing failed

    # Convert the list of file dictionaries to a pandas DataFrame
    nas_df = pd.DataFrame(nas_files_list)

    # --- Timestamp Handling (NAS) ---
    # NAS timestamps from get_nas_files should already be UTC-aware
    if not nas_df.empty and 'date_last_modified' in nas_df.columns:
         print("   Ensuring NAS timestamps are timezone-aware UTC...")
         # Double-check conversion, though get_nas_files should handle this
         nas_df['date_last_modified'] = pd.to_datetime(nas_df['date_last_modified']).dt.tz_convert('UTC')
         print("   NAS timestamps confirmed as UTC.")
    elif nas_df.empty:
        print("   NAS directory is empty or contains no listable files.")
    else:
        print("   No 'date_last_modified' data found in NAS file list.")


    # --- Save NAS File List to NAS ---
    print(f"\n   Saving NAS file list data to NAS file: '{os.path.basename(nas_output_smb_file)}'...")
    # Convert DataFrame to JSON string
    nas_json_string = nas_df.to_json(orient='records', indent=4, date_format='iso')
    if not write_json_to_nas(nas_output_smb_file, nas_json_string):
        print("   [CRITICAL ERROR] Failed to write NAS file list JSON to NAS. Exiting.")
        sys.exit(1) # Exit if saving fails
    print("-" * 60)


    # --- Compare DB Catalog and NAS File List ---
    print("[5] Comparing Database Catalog and NAS File List...")
    print(f"   Database records: {len(db_df)}")
    print(f"   NAS files found: {len(nas_df)}")

    # Initialize comparison result DataFrames
    # Add date_created to files_to_process
    files_to_process = pd.DataFrame(columns=['file_name', 'file_path', 'file_size', 'date_last_modified', 'date_created', 'reason'])
    # Add document_source, document_type, document_name to files_to_delete
    files_to_delete = pd.DataFrame(columns=['id', 'file_name', 'file_path', 'document_source', 'document_type', 'document_name']) # Files to delete from DB

    # Handle edge cases: both empty, NAS empty, DB empty
    if db_df.empty and nas_df.empty:
        print("   Result: Both DB catalog and NAS list are empty. No actions needed.")
    elif nas_df.empty:
        print("   Result: NAS list is empty. No files to process. All DB entries are potentially stale (but not deleted by this script).")
        # files_to_delete remains empty as we only delete based on NAS updates here
    elif db_df.empty:
        print("   Result: DB catalog is empty. All NAS files are considered 'new'.")
        # Include date_created when selecting columns
        files_to_process = nas_df[['file_name', 'file_path', 'file_size', 'date_last_modified', 'date_created']].copy()
        files_to_process['reason'] = 'new'
        # files_to_delete remains empty
    else:
        # --- Perform the Comparison using Merge ---
        print("   Performing comparison based on 'file_name'...")

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
        # Include date_created_nas when selecting columns
        new_files_cols = ['file_name', 'file_path_nas', 'file_size_nas', 'date_last_modified_nas', 'date_created_nas']
        new_files = comparison_df.loc[new_files_mask, new_files_cols].copy()
        new_files.rename(columns={
            'file_path_nas': 'file_path',
            'file_size_nas': 'file_size',
            'date_last_modified_nas': 'date_last_modified',
            'date_created_nas': 'date_created' # Rename date_created
        }, inplace=True)
        new_files['reason'] = 'new'
        print(f"      Identified {len(new_files)} new files (present on NAS, not in DB).")

        # --- Identify Updated Files and Corresponding DB Entries to Delete ---
        # Files present in both NAS and DB ('both' in the merge)
        both_files_mask = comparison_df['_merge'] == 'both'
        both_files = comparison_df[both_files_mask].copy()

        # Check for updates based on modification time (NAS newer than DB)
        # Ensure both date columns are valid datetimes before comparison
        if 'date_last_modified_nas' in both_files.columns and 'date_last_modified_db' in both_files.columns:
             # Handle potential NaT values before comparison
             valid_dates_mask = both_files['date_last_modified_nas'].notna() & both_files['date_last_modified_db'].notna()
             updated_mask = valid_dates_mask & (both_files['date_last_modified_nas'] > both_files['date_last_modified_db'])

             # Get NAS details for files identified as updated
             # Include date_created_nas when selecting columns
             updated_files_cols = ['file_name', 'file_path_nas', 'file_size_nas', 'date_last_modified_nas', 'date_created_nas']
             updated_files_nas = both_files.loc[updated_mask, updated_files_cols].copy()
             updated_files_nas.rename(columns={
                 'file_path_nas': 'file_path',
                 'file_size_nas': 'file_size',
                 'date_last_modified_nas': 'date_last_modified',
                 'date_created_nas': 'date_created' # Rename date_created
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
             files_to_delete = both_files.loc[updated_mask, existing_db_cols].copy()
             # Rename only the column that definitely has a suffix
             files_to_delete.rename(columns={'file_path_db': 'file_path'}, inplace=True)
             print(f"      Identified {len(files_to_delete)} DB records to delete (corresponding to updated files).")
        else:
             print("      [WARNING] 'date_last_modified' columns missing in merged data for update check. Skipping update detection.")
             # Update the empty DataFrame definition here as well
             updated_files_nas = pd.DataFrame(columns=['file_name', 'file_path', 'file_size', 'date_last_modified', 'date_created', 'reason'])
             files_to_delete = pd.DataFrame(columns=['id', 'file_name', 'file_path', 'document_source', 'document_type', 'document_name'])


        # --- Combine New and Updated Files for Processing ---
        files_to_process = pd.concat([new_files, updated_files_nas], ignore_index=True)

        # --- Identify Files Only in DB (Potentially Deleted/Moved on NAS) ---
        # Files present only in the DB ('right_only' in the merge)
        # Note: This script currently *does not* automatically delete these from the DB.
        # It only identifies DB records to delete when a file is *updated* on the NAS.
        deleted_files_db = comparison_df[comparison_df['_merge'] == 'right_only']
        if not deleted_files_db.empty:
            print(f"      Note: Found {len(deleted_files_db)} files in DB but not on NAS (potentially deleted/moved). No action taken by this script.")


    # --- Final Summary of Comparison ---
    print(f"\n   Comparison Summary:")
    print(f"      - NAS Files to Process (New or Updated): {len(files_to_process)}")
    print(f"      - Existing DB Records to Delete (Updated Files): {len(files_to_delete)}")
    print("-" * 60)


    # --- Save Comparison Results to NAS ---
    print("[6] Saving Comparison Results to NAS...")

    # Save the list of files to be processed (new/updated)
    print(f"   Saving 'files to process' list to: '{os.path.basename(process_output_smb_file)}'...")
    process_json_string = files_to_process.to_json(orient='records', indent=4, date_format='iso')
    if not write_json_to_nas(process_output_smb_file, process_json_string):
        print("   [CRITICAL ERROR] Failed to write 'files to process' JSON to NAS. Exiting.")
        sys.exit(1)

    # Save the list of DB records to be deleted (corresponding to updated files)
    print(f"   Saving 'files to delete' list to: '{os.path.basename(delete_output_smb_file)}'...")

    # Ensure 'id' column is integer type for JSON compatibility if it exists and has data
    if 'id' in files_to_delete.columns and not files_to_delete['id'].isnull().all():
         files_to_delete['id'] = files_to_delete['id'].astype('Int64') # Use nullable integer type

    # Convert the DataFrame to JSON
    delete_json_string = files_to_delete.to_json(orient='records', indent=4)
    if not write_json_to_nas(delete_output_smb_file, delete_json_string):
        print("   [CRITICAL ERROR] Failed to write 'files to delete' JSON to NAS. Exiting.")
        sys.exit(1)

    print("-" * 60)
    print("\n" + "="*60)
    print(f"--- Stage 1 Completed Successfully ---")
    print("--- Output JSON files generated on NAS ---")
    print("="*60 + "\n")
