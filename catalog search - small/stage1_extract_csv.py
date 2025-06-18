# -*- coding: utf-8 -*-
"""
Stage 1: Extract & Compare - CSV Catalog vs. NAS Filesystem (using pysmb)

This script performs the first stage of a data synchronization process.
It reads from CSV files (master_catalog.csv and master_content.csv) to retrieve 
a catalog of files for a specific document source. It then connects to a 
Network Attached Storage (NAS) device via SMB (using pysmb) to list the actual 
files present in the corresponding source directory.

Auto-initialization: If the master CSV files don't exist, they are created with 
proper schemas and headers.

Finally, it compares the CSV catalog against the NAS file list based on filenames 
and modification dates to determine:
1. Files present on NAS but not in the CSV catalog (new files).
2. Files present in both but updated on NAS (updated files).
3. Files present in the CSV catalog but not on NAS (to be marked for deletion).
4. CSV records corresponding to updated NAS files (to be deleted before re-insertion).

The results of the comparison (files to process and files to delete from CSV)
are saved as JSON files to a specified output directory on the NAS.

Configuration for NAS and processing parameters should be set in the 
'Configuration' section below or externalized to a config file.
"""

import pandas as pd
import sys
import os
from smb.SMBConnection import SMBConnection
from smb import smb_structs
import io
from datetime import datetime, timezone
import socket
import json

# ==============================================================================
# --- Configuration ---
# ==============================================================================

# --- CSV Configuration ---
# Paths to master CSV files on NAS (relative to share)
MASTER_CSV_FOLDER_PATH = "path/to/master_csv_folder"
MASTER_CATALOG_CSV = "master_catalog.csv"
MASTER_CONTENT_CSV = "master_content.csv"

# --- NAS Configuration ---
# Network attached storage connection parameters
NAS_PARAMS = {
        "ip": "your_nas_ip",
        "share": "your_share_name",
        "user": "your_nas_user",
        "password": "your_nas_password",
        "port": 445
}
# Base path on the NAS share containing the root folders for different document sources
NAS_BASE_INPUT_PATH = "path/to/your/base_input_folder"
# Base path on the NAS share where output JSON files will be stored
NAS_OUTPUT_FOLDER_PATH = "path/to/your/output_folder"

# --- Processing Configuration ---
# Document sources configuration - each line contains source name and detail level
DOCUMENT_SOURCES = """
internal_cheatsheets,detailed
internal_esg,standard
# internal_policies,concise
financial_reports,detailed
marketing_materials,concise
# technical_docs,detailed
"""

def load_document_sources():
        """Parse document sources configuration - works for all stages"""
        sources = []
        for line in DOCUMENT_SOURCES.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(',')
                if len(parts) == 2:
                    source_name = parts[0].strip()
                    detail_level = parts[1].strip()
                    sources.append({
                        'name': source_name,
                        'detail_level': detail_level
                    })
                else:
                    print(f"Warning: Invalid config line ignored: {line}")
        return sources

# --- Full Refresh Mode ---
# Set to True to ignore CSV/NAS comparison and process ALL NAS files,
# marking ALL existing CSV records for this source for deletion.
FULL_REFRESH = False

# --- pysmb Configuration ---
smb_structs.SUPPORT_SMB2 = True
smb_structs.MAX_PAYLOAD_SIZE = 65536
CLIENT_HOSTNAME = socket.gethostname()

# ==============================================================================
# --- CSV Schema Definitions ---
# ==============================================================================

# Master catalog CSV schema - based on apg_catalog table
CATALOG_SCHEMA = {
        'id': 'int64',
        'created_at': 'datetime64[ns, UTC]',
        'document_source': 'object',
        'document_type': 'object', 
        'document_name': 'object',
        'document_description': 'object',
        'document_usage': 'object',
        'document_usage_embedding': 'object',  # Will store JSON string representation
        'document_description_embedding': 'object',  # Will store JSON string representation
        'date_created': 'datetime64[ns, UTC]',
        'date_last_modified': 'datetime64[ns, UTC]',
        'file_name': 'object',
        'file_type': 'object',
        'file_size': 'int64',
        'file_path': 'object',
        'file_link': 'object'
}

# Master content CSV schema - based on apg_content table
CONTENT_SCHEMA = {
        'id': 'int64',
        'created_at': 'datetime64[ns, UTC]',
        'document_source': 'object',
        'document_type': 'object',
        'document_name': 'object',
        'section_id': 'int64',
        'section_name': 'object',
        'section_summary': 'object',
        'section_content': 'object',
        'page_number': 'int64'
}

# ==============================================================================
# --- Helper Functions ---
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
        
        path_parts = dir_path.strip('/').split('/')
        current_path = ''
        try:
            for part in path_parts:
                if not part: continue
                current_path = os.path.join(current_path, part).replace('\\', '/')
                try:
                    conn.listPath(share_name, current_path)
                except Exception:
                    print(f"      Creating directory on NAS: {current_path}")
                    conn.createDirectory(share_name, current_path)
            return True
        except Exception as e:
            print(f"   [ERROR] Failed to ensure/create NAS directory '{dir_path}': {e}")
            return False

def write_json_to_nas(share_name, nas_path_relative, data_string):
        """Writes a string (expected to be JSON) to a specified file path on the NAS using pysmb."""
        conn = None
        print(f"   Attempting to write to NAS path: {share_name}/{nas_path_relative}")
        try:
            conn = create_nas_connection()
            if not conn:
                return False

            dir_path = os.path.dirname(nas_path_relative).replace('\\', '/')
            if dir_path and not ensure_nas_dir_exists(conn, share_name, dir_path):
                 print(f"   [ERROR] Failed to ensure output directory exists: {dir_path}")
                 return False

            data_bytes = data_string.encode('utf-8')
            file_obj = io.BytesIO(data_bytes)

            bytes_written = conn.storeFile(share_name, nas_path_relative, file_obj)
            print(f"   Successfully wrote {bytes_written} bytes to: {share_name}/{nas_path_relative}")
            return True
        except Exception as e:
            print(f"   [ERROR] Unexpected error writing to NAS '{share_name}/{nas_path_relative}': {e}")
            return False
        finally:
            if conn:
                conn.close()

def get_nas_files(share_name, base_folder_path):
        """Lists files recursively from a specific directory on an SMB share using pysmb."""
        files_list = []
        conn = None
        print(f" -> Attempting to list files from NAS path: {share_name}/{base_folder_path}")

        try:
            conn = create_nas_connection()
            if not conn:
                return None

            try:
                 conn.listPath(share_name, base_folder_path)
            except Exception as e:
                 print(f"   [ERROR] Base NAS path does not exist or is inaccessible: {share_name}/{base_folder_path} - {e}")
                 return None

            print(f" -> Walking directory tree: {share_name}/{base_folder_path} ...")
            
            def walk_nas_path(current_path_relative):
                try:
                    items = conn.listPath(share_name, current_path_relative)
                    for item in items:
                        if item.filename == '.' or item.filename == '..':
                            continue

                        if item.filename == '.DS_Store' or item.filename.startswith('~$') or item.filename.startswith('.'):
                            print(f"      Skipping system/temporary file: {item.filename}")
                            continue

                        full_path_relative = os.path.join(current_path_relative, item.filename).replace('\\', '/')

                        if item.isDirectory:
                            walk_nas_path(full_path_relative)
                        else:
                            try:
                                last_modified_dt = datetime.fromtimestamp(item.last_write_time, tz=timezone.utc)
                                created_dt = datetime.fromtimestamp(item.create_time if item.create_time else item.last_write_time, tz=timezone.utc)

                                files_list.append({
                                    'file_name': item.filename,
                                    'file_path': full_path_relative,
                                    'file_size': item.file_size,
                                    'date_last_modified': last_modified_dt,
                                    'date_created': created_dt
                                })
                            except Exception as file_err:
                                print(f"      [WARNING] Could not process file '{full_path_relative}': {file_err}. Skipping file.")

                except Exception as list_err:
                     print(f"      [WARNING] Error listing path '{current_path_relative}': {list_err}. Skipping directory.")

            walk_nas_path(base_folder_path)

            print(f" <- Successfully listed {len(files_list)} files from NAS.")
            return files_list

        except Exception as e:
            print(f"   [ERROR] Unexpected error listing NAS files from '{share_name}/{base_folder_path}': {e}")
            return None
        finally:
            if conn:
                conn.close()

def check_nas_file_exists(share_name, file_path):
        """Check if a file exists on NAS."""
        conn = None
        try:
            conn = create_nas_connection()
            if not conn:
                return False
            conn.getAttributes(share_name, file_path)
            return True
        except:
            return False
        finally:
            if conn:
                conn.close()

def write_csv_to_nas(share_name, nas_path_relative, df):
        """Write a DataFrame as CSV to NAS."""
        conn = None
        try:
            conn = create_nas_connection()
            if not conn:
                return False

            # Ensure directory exists
            dir_path = os.path.dirname(nas_path_relative).replace('\\', '/')
            if dir_path and not ensure_nas_dir_exists(conn, share_name, dir_path):
                print(f"   [ERROR] Failed to ensure CSV directory exists: {dir_path}")
                return False

            # Convert DataFrame to CSV string
            csv_content = df.to_csv(index=False)
            csv_bytes = csv_content.encode('utf-8')
            file_obj = io.BytesIO(csv_bytes)

            # Write to NAS
            bytes_written = conn.storeFile(share_name, nas_path_relative, file_obj)
            print(f"   Successfully wrote {bytes_written} bytes to: {share_name}/{nas_path_relative}")
            return True
        except Exception as e:
            print(f"   [ERROR] Failed to write CSV to NAS '{share_name}/{nas_path_relative}': {e}")
            return False
        finally:
            if conn:
                conn.close()

def read_csv_from_nas(share_name, nas_path_relative):
        """Read a CSV file from NAS into a DataFrame."""
        conn = None
        try:
            if not check_nas_file_exists(share_name, nas_path_relative):
                return pd.DataFrame()

            conn = create_nas_connection()
            if not conn:
                return None

            file_obj = io.BytesIO()
            file_attributes, filesize = conn.retrieveFile(share_name, nas_path_relative, file_obj)
            file_obj.seek(0)
            csv_content = file_obj.read().decode('utf-8')
            
            # Read CSV from string
            from io import StringIO
            df = pd.read_csv(StringIO(csv_content))
            print(f"   Successfully read CSV from: {share_name}/{nas_path_relative} ({len(df)} records)")
            return df
        except Exception as e:
            print(f"   [ERROR] Failed to read CSV from NAS '{share_name}/{nas_path_relative}': {e}")
            return None
        finally:
            if conn:
                conn.close()

def initialize_master_csvs():
        """Initialize master CSV files on NAS with proper schemas if they don't exist."""
        catalog_path = os.path.join(MASTER_CSV_FOLDER_PATH, MASTER_CATALOG_CSV).replace('\\', '/')
        content_path = os.path.join(MASTER_CSV_FOLDER_PATH, MASTER_CONTENT_CSV).replace('\\', '/')
        
        # Initialize master_catalog.csv on NAS
        if not check_nas_file_exists(NAS_PARAMS["share"], catalog_path):
            print(f"   Creating master catalog CSV on NAS: {NAS_PARAMS['share']}/{catalog_path}")
            catalog_df = pd.DataFrame(columns=list(CATALOG_SCHEMA.keys()))
            catalog_df = catalog_df.astype(CATALOG_SCHEMA)
            if write_csv_to_nas(NAS_PARAMS["share"], catalog_path, catalog_df):
                print(f"   Created empty master_catalog.csv with {len(CATALOG_SCHEMA)} columns")
            else:
                print(f"   [ERROR] Failed to create master_catalog.csv on NAS")
                return False
        else:
            print(f"   Master catalog CSV already exists on NAS: {NAS_PARAMS['share']}/{catalog_path}")
        
        # Initialize master_content.csv on NAS
        if not check_nas_file_exists(NAS_PARAMS["share"], content_path):
            print(f"   Creating master content CSV on NAS: {NAS_PARAMS['share']}/{content_path}")
            content_df = pd.DataFrame(columns=list(CONTENT_SCHEMA.keys()))
            content_df = content_df.astype(CONTENT_SCHEMA)
            if write_csv_to_nas(NAS_PARAMS["share"], content_path, content_df):
                print(f"   Created empty master_content.csv with {len(CONTENT_SCHEMA)} columns")
            else:
                print(f"   [ERROR] Failed to create master_content.csv on NAS")
                return False
        else:
            print(f"   Master content CSV already exists on NAS: {NAS_PARAMS['share']}/{content_path}")
        
        return True

def load_master_catalog(document_source):
        """Load the master catalog CSV from NAS and filter by document source."""
        catalog_path = os.path.join(MASTER_CSV_FOLDER_PATH, MASTER_CATALOG_CSV).replace('\\', '/')
        
        try:
            catalog_df = read_csv_from_nas(NAS_PARAMS["share"], catalog_path)
            
            if catalog_df is None:
                print(f"   [ERROR] Failed to read master catalog from NAS")
                return pd.DataFrame(columns=list(CATALOG_SCHEMA.keys()))
            
            if catalog_df.empty:
                print(f"   Master catalog CSV is empty")
                return catalog_df
            
            # Filter by document source
            if 'document_source' in catalog_df.columns:
                catalog_df = catalog_df[catalog_df['document_source'] == document_source]
            
            # Handle timestamp columns
            if not catalog_df.empty:
                for col in ['created_at', 'date_created', 'date_last_modified']:
                    if col in catalog_df.columns:
                        catalog_df[col] = pd.to_datetime(catalog_df[col], errors='coerce', utc=True)
            
            print(f"   Loaded {len(catalog_df)} records from master catalog for source '{document_source}'")
            return catalog_df
                
        except Exception as e:
            print(f"   [ERROR] Failed to load master catalog from NAS: {e}")
            return pd.DataFrame(columns=list(CATALOG_SCHEMA.keys()))

# ==============================================================================
# --- Main Execution Logic ---
# ==============================================================================

if __name__ == "__main__":

    print("\n" + "="*60)
    print(f"--- Running Stage 1: Extract & Compare (CSV-based) ---")
    print("="*60 + "\n")
    
    # Get document sources
    sources = load_document_sources()
    print(f"[0] Processing {len(sources)} document sources:")
    for source in sources:
        print(f"   - {source['name']} (detail level: {source['detail_level']})")
    print("-" * 60)

    # --- Initialize Master CSVs on NAS ---
    print("\n[1] Initializing Master CSV Files on NAS...")
    if not initialize_master_csvs():
        print("[CRITICAL ERROR] Failed to initialize master CSV files on NAS. Exiting.")
        sys.exit(1)
    print("-" * 60)
    
    # Track overall processing results
    all_sources_processed = []
    sources_with_files = []
    
    # Process each document source
    for source_config in sources:
            DOCUMENT_SOURCE = source_config['name']
            
            print(f"\n{'='*60}")
            print(f"Processing Document Source: {DOCUMENT_SOURCE}")
            print(f"{'='*60}\n")

            # --- Construct NAS Paths ---
            print("[2] Constructing NAS Input and Output Paths...")
            nas_input_path_relative = os.path.join(NAS_BASE_INPUT_PATH, DOCUMENT_SOURCE).replace('\\', '/')
            print(f"   NAS Input Path: {NAS_PARAMS['share']}/{nas_input_path_relative}")

            nas_output_dir_relative = os.path.join(NAS_OUTPUT_FOLDER_PATH, DOCUMENT_SOURCE).replace('\\', '/')
            print(f"   NAS Output Directory: {NAS_PARAMS['share']}/{nas_output_dir_relative}")

            # Define output file paths
            csv_output_relative_file = os.path.join(nas_output_dir_relative, '1A_catalog_in_csv.json').replace('\\', '/')
            nas_output_relative_file = os.path.join(nas_output_dir_relative, '1B_files_in_nas.json').replace('\\', '/')
            process_output_relative_file = os.path.join(nas_output_dir_relative, '1C_nas_files_to_process.json').replace('\\', '/')
            delete_output_relative_file = os.path.join(nas_output_dir_relative, '1D_csv_files_to_delete.json').replace('\\', '/')
            print("-" * 60)

            # --- Ensure NAS Output Directory Exists ---
            print("[3] Ensuring NAS Output Directory Exists...")
            conn_check = None
            try:
                conn_check = create_nas_connection()
                if not conn_check:
                     print("   [ERROR] Failed to connect to NAS to check/create output directory.")
                     continue  # Skip this source
                if not ensure_nas_dir_exists(conn_check, NAS_PARAMS["share"], nas_output_dir_relative):
                    print(f"   [ERROR] Failed to create/access NAS output directory '{nas_output_dir_relative}'.")
                    continue  # Skip this source
                else:
                     print(f"   NAS output directory ensured: '{NAS_PARAMS['share']}/{nas_output_dir_relative}'")
            except Exception as e:
                print(f"   [ERROR] Unexpected error creating/accessing NAS directory '{nas_output_dir_relative}': {e}")
                continue  # Skip this source
            finally:
                if conn_check:
                    conn_check.close()
            print("-" * 60)

            # --- Load Data from Master CSV ---
            print(f"[4] Loading Data from Master CSV Catalog...")
            csv_df = load_master_catalog(DOCUMENT_SOURCE)

            # Save CSV catalog to NAS
            print(f"\n   Saving CSV catalog data to NAS file: '{os.path.basename(csv_output_relative_file)}'...")
            csv_json_string = csv_df.to_json(orient='records', indent=4, date_format='iso')
            if not write_json_to_nas(NAS_PARAMS["share"], csv_output_relative_file, csv_json_string):
                 print("   [ERROR] Failed to write CSV catalog JSON to NAS. Skipping this source.")
                 continue
            print("-" * 60)

            # --- Get File List from NAS ---
            print(f"[5] Listing Files from NAS Source Path: '{NAS_PARAMS['share']}/{nas_input_path_relative}'...")
            nas_files_list = get_nas_files(NAS_PARAMS["share"], nas_input_path_relative)

            if nas_files_list is None:
                print("   [ERROR] Failed to retrieve file list from NAS. Skipping this source.")
                continue

            nas_df = pd.DataFrame(nas_files_list)

            # Handle NAS timestamps
            if not nas_df.empty and 'date_last_modified' in nas_df.columns:
                 print("   Processing NAS timestamps...")
                 nas_df['date_last_modified'] = pd.to_datetime(nas_df['date_last_modified'], errors='coerce')
                 valid_nas_dates = nas_df['date_last_modified'].notna()
                 if not nas_df.loc[valid_nas_dates].empty:
                     if nas_df.loc[valid_nas_dates, 'date_last_modified'].dt.tz is None:
                         print("   Localizing naive NAS timestamps to UTC...")
                         nas_df.loc[valid_nas_dates, 'date_last_modified'] = nas_df.loc[valid_nas_dates, 'date_last_modified'].dt.tz_localize('UTC')
                     else:
                         print("   Converting timezone-aware NAS timestamps to UTC...")
                         nas_df.loc[valid_nas_dates, 'date_last_modified'] = nas_df.loc[valid_nas_dates, 'date_last_modified'].dt.tz_convert('UTC')
                 print("   NAS timestamps processed.")

            # Save NAS file list to NAS
            print(f"\n   Saving NAS file list data to NAS file: '{os.path.basename(nas_output_relative_file)}'...")
            nas_json_string = nas_df.to_json(orient='records', indent=4, date_format='iso')
            if not write_json_to_nas(NAS_PARAMS["share"], nas_output_relative_file, nas_json_string):
                print("   [ERROR] Failed to write NAS file list JSON to NAS. Skipping this source.")
                continue
            print("-" * 60)

            # --- Compare CSV Catalog and NAS File List ---
            print("[6] Comparing CSV Catalog and NAS File List...")
            print(f"   CSV records: {len(csv_df)}")
            print(f"   NAS files found: {len(nas_df)}")

            # Initialize comparison result DataFrames
            files_to_process = pd.DataFrame(columns=['file_name', 'file_path', 'file_size', 'date_last_modified', 'date_created', 'reason'])
            files_to_delete = pd.DataFrame(columns=['id', 'file_name', 'file_path', 'document_source', 'document_type', 'document_name'])
            both_files = pd.DataFrame()

            # --- Check for Full Refresh Mode ---
            if FULL_REFRESH:
                print("\n   *** FULL REFRESH MODE ENABLED ***")
                print("   Skipping CSV/NAS comparison.")

                if nas_df.empty:
                    print("   NAS directory is empty. No files to process even in full refresh mode.")
                    files_to_process = pd.DataFrame(columns=['file_name', 'file_path', 'file_size', 'date_last_modified', 'date_created', 'reason'])
                else:
                    print(f"   Marking all {len(nas_df)} NAS files for processing.")
                    process_cols = ['file_name', 'file_path', 'file_size', 'date_last_modified', 'date_created']
                    existing_process_cols = [col for col in process_cols if col in nas_df.columns]
                    files_to_process = nas_df[existing_process_cols].copy()
                    files_to_process['reason'] = 'full_refresh'

                if csv_df.empty:
                    print("   CSV catalog is empty for this source. No existing records to mark for deletion.")
                    files_to_delete = pd.DataFrame(columns=['id', 'file_name', 'file_path', 'document_source', 'document_type', 'document_name'])
                else:
                    print(f"   Marking all {len(csv_df)} existing CSV records for deletion.")
                    delete_cols = ['id', 'file_name', 'file_path', 'document_source', 'document_type', 'document_name']
                    existing_delete_cols = [col for col in delete_cols if col in csv_df.columns]
                    files_to_delete = csv_df[existing_delete_cols].copy()

            else:
                # --- Incremental Update Logic ---
                print("\n   --- Incremental Update Mode ---")
                if csv_df.empty and nas_df.empty:
                    print("   Result: Both CSV catalog and NAS list are empty. No actions needed.")
                    
                    # Initialize variables needed later in the flow
                    files_to_process = pd.DataFrame(columns=['file_name', 'file_path', 'file_size', 'date_last_modified', 'date_created', 'reason'])
                    files_to_delete = pd.DataFrame(columns=['id', 'file_name', 'file_path', 'document_source', 'document_type', 'document_name'])
                    comparison_df = pd.DataFrame(columns=['_merge'])  # Empty comparison_df with _merge column
                elif nas_df.empty:
                    print("   Result: NAS list is empty. No files to process.")
                    
                    # Initialize variables needed later in the flow
                    files_to_process = pd.DataFrame(columns=['file_name', 'file_path', 'file_size', 'date_last_modified', 'date_created', 'reason'])
                    files_to_delete = pd.DataFrame(columns=['id', 'file_name', 'file_path', 'document_source', 'document_type', 'document_name'])
                    comparison_df = pd.DataFrame(columns=['_merge'])  # Empty comparison_df with _merge column
                elif csv_df.empty:
                    print("   Result: CSV catalog is empty. All NAS files are considered 'new'.")
                    new_cols = ['file_name', 'file_path', 'file_size', 'date_last_modified', 'date_created']
                    existing_new_cols = [col for col in new_cols if col in nas_df.columns]
                    files_to_process = nas_df[existing_new_cols].copy()
                    files_to_process['reason'] = 'new'
                    
                    # Initialize variables needed later in the flow
                    files_to_delete = pd.DataFrame(columns=['id', 'file_name', 'file_path', 'document_source', 'document_type', 'document_name'])
                    comparison_df = pd.DataFrame(columns=['_merge'])  # Empty comparison_df with _merge column
                else:
                    # --- Perform the Comparison using Merge ---
                    print("   Performing comparison based on 'file_name'...")

                    if 'file_name' not in nas_df.columns:
                        print("   [ERROR] 'file_name' column missing in NAS DataFrame. Skipping this source.")
                        continue
                    if 'file_name' not in csv_df.columns:
                        print("   [ERROR] 'file_name' column missing in CSV DataFrame. Skipping this source.")
                        continue

                    nas_df['file_name'] = nas_df['file_name'].astype(str)
                    csv_df['file_name'] = csv_df['file_name'].astype(str)

                    # Merge NAS file list with CSV catalog
                    comparison_df = pd.merge(
                        nas_df,
                        csv_df,
                        on='file_name',
                        how='outer',
                        suffixes=('_nas', '_csv'),
                        indicator=True
                    )

                    # --- Identify New Files ---
                    new_files_mask = comparison_df['_merge'] == 'left_only'
                    new_files_cols = ['file_name', 'file_path_nas', 'file_size_nas', 'date_last_modified_nas', 'date_created_nas']
                    existing_new_cols = [col for col in new_files_cols if col in comparison_df.columns]
                    if len(existing_new_cols) != len(new_files_cols):
                        missing_cols = set(new_files_cols) - set(existing_new_cols)
                        print(f"   [WARNING] Could not find expected columns {missing_cols} in merged data for new files list.")
                    new_files = comparison_df.loc[new_files_mask, existing_new_cols].copy()
                    new_files.rename(columns={
                        'file_path_nas': 'file_path',
                        'file_size_nas': 'file_size',
                        'date_last_modified_nas': 'date_last_modified',
                        'date_created_nas': 'date_created'
                    }, inplace=True)
                    new_files['reason'] = 'new'
                    print(f"      Identified {len(new_files)} new files (present on NAS, not in CSV).")

                    # --- Identify Updated Files and Corresponding CSV Entries to Delete ---
                    both_files_mask = comparison_df['_merge'] == 'both'
                    both_files = comparison_df[both_files_mask].copy()
                    updated_files_nas = pd.DataFrame(columns=['file_name', 'file_path', 'file_size', 'date_last_modified', 'date_created', 'reason'])
                    files_to_delete = pd.DataFrame(columns=['id', 'file_name', 'file_path', 'document_source', 'document_type', 'document_name'])
                    updated_mask = pd.Series(dtype=bool)

                    if not both_files.empty:
                        if 'date_last_modified_nas' in both_files.columns and 'date_last_modified_csv' in both_files.columns:
                            valid_dates_mask = both_files['date_last_modified_nas'].notna() & both_files['date_last_modified_csv'].notna()
                            updated_mask = pd.Series(False, index=both_files.index)
                            updated_mask[valid_dates_mask] = (
                                both_files.loc[valid_dates_mask, 'date_last_modified_nas'].dt.floor('min') >
                                both_files.loc[valid_dates_mask, 'date_last_modified_csv'].dt.floor('min')
                            )

                            # Get NAS details for updated files
                            updated_files_cols = ['file_name', 'file_path_nas', 'file_size_nas', 'date_last_modified_nas', 'date_created_nas']
                            existing_updated_cols = [col for col in updated_files_cols if col in both_files.columns]
                            updated_files_nas = both_files.loc[updated_mask, existing_updated_cols].copy()
                            updated_files_nas.rename(columns={
                                'file_path_nas': 'file_path',
                                'file_size_nas': 'file_size',
                                'date_last_modified_nas': 'date_last_modified',
                                'date_created_nas': 'date_created'
                            }, inplace=True)
                            updated_files_nas['reason'] = 'updated'
                            print(f"      Identified {len(updated_files_nas)} updated files (newer on NAS than in CSV).")

                            # Get CSV details for files that need to be deleted
                            csv_cols_to_keep = ['id', 'file_name', 'file_path_csv', 'document_source', 'document_type', 'document_name']
                            existing_csv_cols = [col for col in csv_cols_to_keep if col in both_files.columns]
                            files_to_delete = both_files.loc[updated_mask, existing_csv_cols].copy()
                            files_to_delete.rename(columns={'file_path_csv': 'file_path'}, inplace=True)
                            print(f"      Identified {len(files_to_delete)} CSV records to delete (corresponding to updated files).")

                # --- Identify Files Only in CSV (Deleted/Moved on NAS) ---
                deleted_files_csv = comparison_df[comparison_df['_merge'] == 'right_only']
                if not deleted_files_csv.empty:
                    print(f"      Found {len(deleted_files_csv)} files in CSV but not on NAS. Adding to deletion list.")
                    csv_cols_to_keep = ['id', 'file_name', 'file_path_csv', 'document_source', 'document_type', 'document_name']
                    existing_csv_cols = [col for col in csv_cols_to_keep if col in deleted_files_csv.columns]
                    deleted_files_to_remove = deleted_files_csv[existing_csv_cols].copy()
                    deleted_files_to_remove.rename(columns={'file_path_csv': 'file_path'}, inplace=True)
                    files_to_delete = pd.concat([files_to_delete, deleted_files_to_remove], ignore_index=True)
                    print(f"      Added {len(deleted_files_to_remove)} CSV records to deletion list (files no longer on NAS).")

                    # --- Combine New and Updated Files for Processing ---
                    files_to_process = pd.concat([new_files, updated_files_nas], ignore_index=True)

            # --- Final Summary ---
            unchanged_count = 0
            if not both_files.empty:
                if 'updated_mask' in locals() and not updated_mask.empty:
                    unchanged_count = len(both_files[~updated_mask])
                else:
                    unchanged_count = len(both_files)

            print(f"\n   Comparison Summary:")
            print(f"      - NAS Files to Process (New or Updated): {len(files_to_process)}")
            print(f"      - Existing CSV Records to Delete (Updated Files + Missing Files): {len(files_to_delete)}")
            print(f"      - Files Found Unchanged (Timestamp Match): {unchanged_count}")
            print("-" * 60)

            # --- Save Comparison Results to NAS ---
            print("[7] Saving Comparison Results to NAS...")

            # Save files to process
            print(f"   Saving 'files to process' list to: '{os.path.basename(process_output_relative_file)}'...")
            process_json_string = files_to_process.to_json(orient='records', indent=4, date_format='iso')
            if not write_json_to_nas(NAS_PARAMS["share"], process_output_relative_file, process_json_string):
                print("   [CRITICAL ERROR] Failed to write 'files to process' JSON to NAS. Exiting.")
                sys.exit(1)

            # Save files to delete
            print(f"   Saving 'files to delete' list to: '{os.path.basename(delete_output_relative_file)}'...")
            if 'id' in files_to_delete.columns and not files_to_delete['id'].isnull().all():
                 files_to_delete['id'] = files_to_delete['id'].astype('Int64')
            delete_json_string = files_to_delete.to_json(orient='records', indent=4)
            if not write_json_to_nas(NAS_PARAMS["share"], delete_output_relative_file, delete_json_string):
                print("   [CRITICAL ERROR] Failed to write 'files to delete' JSON to NAS. Exiting.")
                sys.exit(1)
            print("-" * 60)

            # --- Create Skip Flag if No Files to Process ---
            print("[8] Managing Flag Files...")
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
                    # Skip flag logic - only skip if no files to process AND no files to delete
                    if files_to_process.empty and files_to_delete.empty:
                        print(f"   No files to process and no files to delete found. Creating skip flag file: '{skip_flag_file_name}'")
                        try:
                            conn_flag.storeFile(NAS_PARAMS["share"], skip_flag_relative_path, io.BytesIO(b''))
                            print(f"   Successfully created skip flag file: {skip_flag_relative_path}")
                        except Exception as e:
                            print(f"   [WARNING] Error creating skip flag file '{skip_flag_relative_path}': {e}")
                    else:
                        action_reason = []
                        if not files_to_process.empty:
                            action_reason.append(f"{len(files_to_process)} files to process")
                        if not files_to_delete.empty:
                            action_reason.append(f"{len(files_to_delete)} files to delete")
                        
                        print(f"   Work found: {', '.join(action_reason)}. Ensuring skip flag does not exist.")
                        try:
                            conn_flag.deleteFiles(NAS_PARAMS["share"], skip_flag_relative_path)
                            print(f"   Removed potentially existing skip flag file: {skip_flag_relative_path}")
                        except Exception as e:
                            if "OBJECT_NAME_NOT_FOUND" not in str(e) and "STATUS_NO_SUCH_FILE" not in str(e):
                                 print(f"   [INFO] Error removing skip flag file (may not exist): {e}")
                            else:
                                 print(f"   Skip flag file did not exist.")

                    # Full refresh flag logic
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
                                print(f"   [INFO] Error removing refresh flag file (may not exist): {e}")
                            else:
                                print(f"   Refresh flag file did not exist.")

            except Exception as e:
                print(f"   [WARNING] Unexpected error during flag file management: {e}")
            finally:
                if conn_flag:
                    conn_flag.close()

                # Track this source as processed
                all_sources_processed.append(DOCUMENT_SOURCE)
                if not files_to_process.empty:
                    sources_with_files.append(DOCUMENT_SOURCE)
                
                print(f"   Source '{DOCUMENT_SOURCE}' processing completed.")
                print("-" * 60)

    # Final summary
    print("\n" + "="*60)
    print(f"--- Stage 1 Completed Successfully (CSV-based) ---")
    print(f"--- Processed {len(all_sources_processed)} sources ---")
    print(f"--- Sources with files to process: {len(sources_with_files)} ---")
    if sources_with_files:
        print(f"--- Sources with files: {', '.join(sources_with_files)} ---")
    print("="*60 + "\n")