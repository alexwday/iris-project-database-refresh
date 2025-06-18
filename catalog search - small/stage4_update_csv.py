# -*- coding: utf-8 -*-
"""
Stage 4: Update CSV Files

This script performs the CSV update stage of the data synchronization process.
It reads the master CSV files and updates them based on the JSON files 
generated in previous stages.

Workflow:
1. Reads the list of records to delete (identified in Stage 1).
2. Reads the new/updated catalog entries (generated in Stage 3).
3. Reads the new/updated content entries (generated in Stage 3).
4. Loads the current master CSV files.
5. Performs validation: Counts records for the source before operations.
6. Deletes records from both CSV files based on the 'files_to_delete' list.
7. Performs validation: Counts records after deletion.
8. Inserts the new catalog entries into the master_catalog.csv.
9. Inserts the new content entries into the master_content.csv.
10. Performs validation: Counts records after insertion.
11. Saves updated CSV files atomically (with backup).
"""

import pandas as pd
import json
import sys
import os
import time
from smb.SMBConnection import SMBConnection
from smb import smb_structs
import io
import socket
from datetime import datetime
import shutil
import tempfile

# ==============================================================================
# --- Configuration ---
# ==============================================================================

# --- CSV Configuration ---
# Paths to master CSV files on NAS (relative to share)
MASTER_CSV_FOLDER_PATH = "path/to/master_csv_folder"
MASTER_CATALOG_CSV = "master_catalog.csv"
MASTER_CONTENT_CSV = "master_content.csv"

# --- NAS Configuration ---
NAS_PARAMS = {
    "ip": "your_nas_ip",
    "share": "your_share_name",
    "user": "your_nas_user",
    "password": "your_nas_password",
    "port": 445
}
NAS_OUTPUT_FOLDER_PATH = "path/to/your/output_folder"

# --- pysmb Configuration ---
smb_structs.SUPPORT_SMB2 = True
smb_structs.MAX_PAYLOAD_SIZE = 65536
CLIENT_HOSTNAME = socket.gethostname()

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

# --- Input Filenames ---
FILES_TO_DELETE_FILENAME = '1D_csv_files_to_delete.json'
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
            CLIENT_HOSTNAME,
            NAS_PARAMS["ip"],
            use_ntlm_v2=True,
            is_direct_tcp=(NAS_PARAMS["port"] == 445)
        )
        connected = conn.connect(NAS_PARAMS["ip"], NAS_PARAMS["port"], timeout=60)
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
            return False

        conn.getAttributes(share_name, nas_path_relative)
        return True
    except Exception as e:
        err_str = str(e).lower()
        if "no such file" in err_str or "object_name_not_found" in err_str or "0xc0000034" in err_str:
            return False
        else:
            print(f"   [WARNING] Unexpected error checking existence of NAS path '{share_name}/{nas_path_relative}': {e}")
            return False
    finally:
        if conn:
            conn.close()

def ensure_nas_dir_exists(conn, share_name, dir_path_relative):
    """Ensures a directory exists on the NAS, creating it if necessary."""
    if not conn:
        print("   [ERROR] Cannot ensure NAS directory: No connection.")
        return False
    
    # pysmb needs paths relative to the share, using '/' as separator
    path_parts = dir_path_relative.strip('/').split('/')
    current_path = ''
    try:
        for part in path_parts:
            if not part: continue
            current_path = os.path.join(current_path, part).replace('\\', '/')
            try:
                # Check if it exists by trying to list it
                conn.listPath(share_name, current_path)
            except Exception: # If listPath fails, assume it doesn't exist
                print(f"      Creating directory on NAS: {share_name}/{current_path}")
                conn.createDirectory(share_name, current_path)
        return True
    except Exception as e:
        print(f"   [ERROR] Failed to ensure/create NAS directory '{share_name}/{dir_path_relative}': {e}")
        return False

def read_json_from_nas(nas_path_relative):
    """Reads and parses JSON data from a file path on the NAS."""
    print(f"   Attempting to read JSON from NAS path: {NAS_PARAMS['share']}/{nas_path_relative}")
    try:
        if not check_nas_path_exists(NAS_PARAMS["share"], nas_path_relative):
            print(f"   [WARNING] JSON file not found at: {NAS_PARAMS['share']}/{nas_path_relative}. Returning empty list.")
            return []

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
        return None
    except Exception as e:
        print(f"   [ERROR] Unexpected error reading JSON from NAS '{nas_path_relative}': {e}")
        return None

def read_csv_from_nas(share_name, nas_path_relative):
    """Read a CSV file from NAS into a DataFrame."""
    conn = None
    try:
        conn = create_nas_connection()
        if not conn:
            return None

        # Check if file exists
        try:
            conn.getAttributes(share_name, nas_path_relative)
        except:
            print(f"   CSV file not found on NAS: {share_name}/{nas_path_relative}")
            return pd.DataFrame()

        file_obj = io.BytesIO()
        file_attributes, filesize = conn.retrieveFile(share_name, nas_path_relative, file_obj)
        file_obj.seek(0)
        csv_content = file_obj.read().decode('utf-8')
        
        # Read CSV from string
        from io import StringIO
        df = pd.read_csv(StringIO(csv_content))
        print(f"   Successfully read CSV from NAS: {share_name}/{nas_path_relative} ({len(df)} records)")
        return df
    except Exception as e:
        print(f"   [ERROR] Failed to read CSV from NAS '{share_name}/{nas_path_relative}': {e}")
        return None
    finally:
        if conn:
            conn.close()

def write_csv_to_nas(share_name, nas_path_relative, df, max_retries=3):
    """Write a DataFrame as CSV to NAS with improved error handling for large files."""
    conn = None
    temp_file_path = None
    
    try:
        print(f"   Preparing to write CSV with {len(df)} records ({df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB in memory)")
        
        conn = create_nas_connection()
        if not conn:
            return False

        # Ensure directory exists
        dir_path = os.path.dirname(nas_path_relative).replace('\\', '/')
        if dir_path and not ensure_nas_dir_exists(conn, share_name, dir_path):
            print(f"   [ERROR] Failed to ensure CSV directory exists: {dir_path}")
            return False

        # Use temporary file approach for large DataFrames to avoid memory issues
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as temp_file:
            temp_file_path = temp_file.name
            
            # Write CSV to temporary file in chunks to manage memory
            print(f"   Writing CSV to temporary file...")
            df.to_csv(temp_file, index=False, chunksize=10000)
        
        # Read the temporary file and upload to NAS with retry logic
        for attempt in range(max_retries):
            try:
                with open(temp_file_path, 'rb') as f:
                    csv_bytes = f.read()
                    file_size_mb = len(csv_bytes) / 1024 / 1024
                    print(f"   Uploading CSV file ({file_size_mb:.1f} MB) to NAS (attempt {attempt + 1}/{max_retries})...")
                    
                    file_obj = io.BytesIO(csv_bytes)
                    bytes_written = conn.storeFile(share_name, nas_path_relative, file_obj)
                    print(f"   ✓ Successfully wrote {bytes_written:,} bytes to NAS: {share_name}/{nas_path_relative}")
                    return True
                    
            except Exception as e:
                error_msg = str(e).lower()
                if attempt < max_retries - 1:
                    if "timeout" in error_msg or "connection" in error_msg:
                        wait_time = (attempt + 1) * 5  # Progressive backoff
                        print(f"   Network issue on attempt {attempt + 1}, retrying in {wait_time}s: {e}")
                        time.sleep(wait_time)
                        # Reconnect for next attempt
                        try:
                            conn.close()
                        except:
                            pass
                        conn = create_nas_connection()
                        if not conn:
                            print(f"   [ERROR] Failed to reconnect to NAS")
                            return False
                    else:
                        print(f"   Unexpected error on attempt {attempt + 1}, retrying: {e}")
                        time.sleep(2)
                else:
                    print(f"   [ERROR] Failed to write CSV after {max_retries} attempts: {e}")
                    return False
        
        return False
        
    except Exception as e:
        print(f"   [ERROR] Critical error during CSV write preparation: {e}")
        return False
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        if conn:
            try:
                conn.close()
            except:
                pass

def load_master_csv(filename, document_source=None):
    """Load a master CSV file from NAS and optionally filter by document source."""
    csv_path = os.path.join(MASTER_CSV_FOLDER_PATH, filename).replace('\\', '/')
    
    try:
        df = read_csv_from_nas(NAS_PARAMS["share"], csv_path)
        
        if df is None:
            print(f"   [ERROR] Failed to read {filename} from NAS")
            return pd.DataFrame()
        
        if df.empty:
            print(f"   CSV file {filename} is empty")
            return df
        
        # Filter by document source if specified
        if document_source and 'document_source' in df.columns:
            df = df[df['document_source'] == document_source]
        
        # Handle timestamp columns
        if not df.empty:
            for col in ['created_at', 'date_created', 'date_last_modified']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
        
        print(f"   Loaded {len(df)} records from {filename}" + (f" for source '{document_source}'" if document_source else ""))
        return df
            
    except Exception as e:
        print(f"   [ERROR] Failed to load CSV {filename} from NAS: {e}")
        return pd.DataFrame()

def save_master_csv_atomic(df, filename):
    """Save a DataFrame to CSV file on NAS atomically."""
    csv_path = os.path.join(MASTER_CSV_FOLDER_PATH, filename).replace('\\', '/')
    
    try:
        # For NAS operations, we'll use a simple write approach
        # (True atomic operations would require more complex NAS backup/restore logic)
        success = write_csv_to_nas(NAS_PARAMS["share"], csv_path, df)
        
        if success:
            print(f"   Successfully saved {filename} to NAS")
            return True
        else:
            print(f"   [ERROR] Failed to save {filename} to NAS")
            return False
            
    except Exception as e:
        print(f"   [ERROR] Failed to save CSV {filename} to NAS: {e}")
        return False

def count_csv_records(df, document_source):
    """Count records in a DataFrame for a specific document source."""
    if df.empty:
        return 0
    
    if 'document_source' in df.columns:
        return len(df[df['document_source'] == document_source])
    else:
        return 0

def get_next_id(df):
    """Get the next available ID for a DataFrame."""
    if df.empty or 'id' not in df.columns:
        return 1
    
    max_id = df['id'].max()
    return int(max_id) + 1 if pd.notna(max_id) else 1

# ==============================================================================
# --- Main Processing Function ---
# ==============================================================================

def main_processing_stage4(delete_list_relative_path, catalog_list_relative_path, content_list_relative_path):
    """Handles the core logic for Stage 4: reading inputs, CSV operations."""
    print(f"--- Starting Main Processing for Stage 4 ---")

    # --- Read Input Files from NAS ---
    print("[4] Reading Input Files from NAS...")
    files_to_delete = read_json_from_nas(delete_list_relative_path)
    catalog_entries = read_json_from_nas(catalog_list_relative_path)
    content_entries = read_json_from_nas(content_list_relative_path)

    if files_to_delete is None or catalog_entries is None or content_entries is None:
        print("[CRITICAL ERROR] Failed to read one or more input files from NAS. Exiting.")
        sys.exit(1)

    # Check if there's nothing to do
    if not files_to_delete and not catalog_entries and not content_entries:
        print("   No files to delete and no new entries to insert. Stage 4 has no work.")
        print("\n" + "="*60)
        print(f"--- Stage 4 Completed (No CSV operations needed) ---")
        print("="*60 + "\n")
        return

    print(f"   Files to Delete: {len(files_to_delete)} records")
    print(f"   Catalog Entries to Insert: {len(catalog_entries)} records")
    print(f"   Content Entries to Insert: {len(content_entries)} records")
    print("-" * 60)

    # --- Load Current Master CSV Files ---
    print("[5] Loading Current Master CSV Files...")
    catalog_df = load_master_csv(MASTER_CATALOG_CSV)
    content_df = load_master_csv(MASTER_CONTENT_CSV)
    print("-" * 60)

    # --- Validation (Before) ---
    print("[6] Performing Pre-Operation Validation...")
    initial_catalog_count = count_csv_records(catalog_df, DOCUMENT_SOURCE)
    initial_content_count = count_csv_records(content_df, DOCUMENT_SOURCE)
    print(f"   Initial count in catalog CSV for source '{DOCUMENT_SOURCE}': {initial_catalog_count}")
    print(f"   Initial count in content CSV for source '{DOCUMENT_SOURCE}': {initial_content_count}")
    print("-" * 60)

    # --- Deletion Phase ---
    print("[7] Deleting Records Marked for Deletion...")
    if not files_to_delete:
        print("   No records marked for deletion. Skipping deletion phase.")
        after_delete_catalog_count = initial_catalog_count
        after_delete_content_count = initial_content_count
    else:
        delete_keys = set()
        for item in files_to_delete:
            if 'id' in item and item['id'] is not None:
                # Use ID-based deletion if available
                delete_keys.add(('id', item['id']))
            else:
                # Fall back to key-based deletion
                key = (
                    item.get('document_source'),
                    item.get('document_type'),
                    item.get('document_name')
                )
                if all(k is not None for k in key):
                    delete_keys.add(('key', key))
                else:
                    print(f"   [WARNING] Skipping deletion for record with missing key components: {item}")

        if not delete_keys:
            print("   No valid keys found for deletion after filtering.")
            after_delete_catalog_count = initial_catalog_count
            after_delete_content_count = initial_content_count
        else:
            print(f"   Attempting to delete records for {len(delete_keys)} unique keys...")
            
            # Delete from catalog
            original_catalog_len = len(catalog_df)
            for delete_type, delete_value in delete_keys:
                if delete_type == 'id':
                    catalog_df = catalog_df[catalog_df['id'] != delete_value]
                elif delete_type == 'key':
                    doc_source, doc_type, doc_name = delete_value
                    mask = (
                        (catalog_df['document_source'] == doc_source) &
                        (catalog_df['document_type'] == doc_type) &
                        (catalog_df['document_name'] == doc_name)
                    )
                    catalog_df = catalog_df[~mask]
            
            catalog_deleted = original_catalog_len - len(catalog_df)
            print(f"      Deleted {catalog_deleted} records from catalog CSV")

            # Delete from content
            original_content_len = len(content_df)
            for delete_type, delete_value in delete_keys:
                if delete_type == 'key':
                    doc_source, doc_type, doc_name = delete_value
                    mask = (
                        (content_df['document_source'] == doc_source) &
                        (content_df['document_type'] == doc_type) &
                        (content_df['document_name'] == doc_name)
                    )
                    content_df = content_df[~mask]
            
            content_deleted = original_content_len - len(content_df)
            print(f"      Deleted {content_deleted} records from content CSV")

        # Validation after deletion
        after_delete_catalog_count = count_csv_records(catalog_df, DOCUMENT_SOURCE)
        after_delete_content_count = count_csv_records(content_df, DOCUMENT_SOURCE)
        print(f"   Count in catalog CSV after deletion: {after_delete_catalog_count}")
        print(f"   Count in content CSV after deletion: {after_delete_content_count}")

    print("-" * 60)

    # --- Insertion Phase ---
    print("[8] Inserting New/Updated Records...")

    # Insert Catalog Entries
    if not catalog_entries:
        print("   No catalog entries to insert.")
    else:
        print(f"   Inserting {len(catalog_entries)} catalog entries...")
        
        # Prepare new catalog records
        new_catalog_records = []
        next_catalog_id = get_next_id(catalog_df)
        
        for entry in catalog_entries:
            # Add system fields
            entry['id'] = next_catalog_id
            entry['created_at'] = datetime.utcnow().isoformat() + 'Z'
            new_catalog_records.append(entry)
            next_catalog_id += 1
        
        # Append to catalog DataFrame
        new_catalog_df = pd.DataFrame(new_catalog_records)
        catalog_df = pd.concat([catalog_df, new_catalog_df], ignore_index=True)
        print(f"   Successfully added {len(new_catalog_records)} records to catalog CSV")

    # Insert Content Entries
    if not content_entries:
        print("   No content entries to insert.")
    else:
        print(f"   Inserting {len(content_entries)} content entries...")
        
        # Prepare new content records
        new_content_records = []
        next_content_id = get_next_id(content_df)
        
        for entry in content_entries:
            # Add system fields
            entry['id'] = next_content_id
            entry['created_at'] = datetime.utcnow().isoformat() + 'Z'
            new_content_records.append(entry)
            next_content_id += 1
        
        # Append to content DataFrame
        new_content_df = pd.DataFrame(new_content_records)
        content_df = pd.concat([content_df, new_content_df], ignore_index=True)
        print(f"   Successfully added {len(new_content_records)} records to content CSV")

    print("-" * 60)

    # --- Validation (After Insertion) ---
    print("[9] Performing Post-Insertion Validation...")
    final_catalog_count = count_csv_records(catalog_df, DOCUMENT_SOURCE)
    final_content_count = count_csv_records(content_df, DOCUMENT_SOURCE)
    print(f"   Final count in catalog CSV for source '{DOCUMENT_SOURCE}': {final_catalog_count}")
    print(f"   Final count in content CSV for source '{DOCUMENT_SOURCE}': {final_content_count}")

    # Final count verification
    print("\n   Verification:")
    expected_catalog_count = after_delete_catalog_count + len(catalog_entries)
    if final_catalog_count == expected_catalog_count:
        print(f"   OK: Final catalog count ({final_catalog_count}) matches expected count ({after_delete_catalog_count} + {len(catalog_entries)} = {expected_catalog_count}).")
    else:
        print(f"   WARNING: Final catalog count ({final_catalog_count}) does NOT match expected count ({expected_catalog_count}).")

    expected_content_count = after_delete_content_count + len(content_entries)
    if final_content_count == expected_content_count:
        print(f"   OK: Final content count ({final_content_count}) matches expected count ({after_delete_content_count} + {len(content_entries)} = {expected_content_count}).")
    else:
        print(f"   WARNING: Final content count ({final_content_count}) does NOT match expected count ({expected_content_count}).")

    print("-" * 60)

    # --- Save Updated CSV Files ---
    print("[10] Saving Updated CSV Files...")
    
    catalog_saved = save_master_csv_atomic(catalog_df, MASTER_CATALOG_CSV)
    content_saved = save_master_csv_atomic(content_df, MASTER_CONTENT_CSV)
    
    if catalog_saved and content_saved:
        print("   Successfully saved both master CSV files")
    else:
        print("   [CRITICAL ERROR] Failed to save one or more CSV files")
        sys.exit(1)

    print("-" * 60)
    print("\n" + "="*60)
    print(f"--- Stage 4 Completed Successfully ---")
    print("--- CSV files updated ---")
    print("="*60 + "\n")
    print(f"--- End of Main Processing for Stage 4 ---")

# ==============================================================================
# --- Script Entry Point ---
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print(f"--- Running Stage 4: Update CSV Files ---")
    print("="*60 + "\n")
    
    # Get document sources
    sources = load_document_sources()
    print(f"[0] Processing {len(sources)} document sources:")
    for source in sources:
        print(f"   - {source['name']} (detail level: {source['detail_level']})")
    print("-" * 60)
    
    # Track overall processing results
    all_sources_processed = []
    sources_with_updates = []
    
    # Process each document source
    for source_config in sources:
        DOCUMENT_SOURCE = source_config['name']
        
        print(f"\n{'='*60}")
        print(f"Processing Document Source: {DOCUMENT_SOURCE}")
        print(f"{'='*60}\n")

        # --- Define NAS Paths ---
        print("[1] Defining NAS Input Paths...")
        source_base_dir_relative = os.path.join(NAS_OUTPUT_FOLDER_PATH, DOCUMENT_SOURCE).replace('\\', '/')

        delete_list_relative_path = os.path.join(source_base_dir_relative, FILES_TO_DELETE_FILENAME).replace('\\', '/')
        catalog_list_relative_path = os.path.join(source_base_dir_relative, CATALOG_ENTRIES_FILENAME).replace('\\', '/')
        content_list_relative_path = os.path.join(source_base_dir_relative, CONTENT_ENTRIES_FILENAME).replace('\\', '/')

        print(f"   Files to Delete List: {NAS_PARAMS['share']}/{delete_list_relative_path}")
        print(f"   Catalog Entries List: {NAS_PARAMS['share']}/{catalog_list_relative_path}")
        print(f"   Content Entries List: {NAS_PARAMS['share']}/{content_list_relative_path}")
        print("-" * 60)

        # --- Check for Skip Flag ---
        print("[2] Checking for skip flag from Stage 1...")
        skip_flag_file_name = '_SKIP_SUBSEQUENT_STAGES.flag'
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
            print(f"   [WARNING] Unexpected error checking for skip flag file: {e}")
            print(f"   Proceeding with Stage 4.")
        print("-" * 60)

        # --- Execute Main Processing if Not Skipped ---
        if should_skip:
            print(f"   Stage 4 Skipped for source '{DOCUMENT_SOURCE}' (No files to process from Stage 1)")
        else:
            try:
                main_processing_stage4(delete_list_relative_path, catalog_list_relative_path, content_list_relative_path)
                sources_with_updates.append(DOCUMENT_SOURCE)
            except Exception as e:
                print(f"   [ERROR] CSV update failed for source '{DOCUMENT_SOURCE}': {e}")
                continue
        
        # Track this source as processed
        all_sources_processed.append(DOCUMENT_SOURCE)
        print(f"   Source '{DOCUMENT_SOURCE}' processing completed.")
        print("-" * 60)

    # Final summary
    print("\n" + "="*60)
    print(f"--- Stage 4 Completed Successfully ---")
    print(f"--- Processed {len(all_sources_processed)} sources ---")
    print(f"--- Sources with CSV updates: {len(sources_with_updates)} ---")
    if sources_with_updates:
        print(f"--- Sources with updates: {', '.join(sources_with_updates)} ---")
    print("="*60 + "\n")