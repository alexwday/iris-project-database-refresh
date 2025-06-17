# -*- coding: utf-8 -*-
"""
Stage 5: Create Timestamped Deployment Files and Archive Processing Run

This script creates timestamped, deployment-ready CSV files for IT pickup and 
archives the entire processing run for audit and rollback purposes.

Workflow:
1. Loads the updated master CSV files from Stage 4
2. Validates data integrity and format consistency
3. Applies final transformations needed for PostgreSQL import
4. Creates timestamped deployment copies for IT pickup
5. Generates deployment metadata and summary report
6. Archives the entire processing run (zip + move to archive folder)
7. Cleans up working files
"""

import pandas as pd
import json
import sys
import os
from smb.SMBConnection import SMBConnection
from smb import smb_structs
import io
import socket
from datetime import datetime, timezone
import hashlib
import zipfile
import tempfile

# ==============================================================================
# --- Configuration ---
# ==============================================================================

# --- CSV Configuration ---
# Paths to master CSV files on NAS (relative to share) - these are outputs from Stage 4
MASTER_CSV_FOLDER_PATH = "path/to/master_csv_folder"
MASTER_CATALOG_CSV = "master_catalog.csv"
MASTER_CONTENT_CSV = "master_content.csv"

# --- Deployment Configuration ---
# Timestamped files for IT pickup (format: catalog_YYYY-MM-DD_HH-MM-SS.csv)
DEPLOYMENT_CATALOG_PREFIX = "catalog"
DEPLOYMENT_CONTENT_PREFIX = "content"
DEPLOYMENT_METADATA_JSON = "deployment_metadata.json"
SUMMARY_REPORT_JSON = "summary_report.json"

# --- Archive Configuration ---
ARCHIVE_SUBFOLDER_NAME = "_archive"

# --- NAS Configuration ---
NAS_PARAMS = {
    "ip": "your_nas_ip",
    "share": "your_share_name",
    "user": "your_nas_user",
    "password": "your_nas_password",
    "port": 445
}
NAS_OUTPUT_FOLDER_PATH = "path/to/your/output_folder"
NAS_DEPLOYMENT_FOLDER_PATH = "path/to/deployment_folder"

# --- pysmb Configuration ---
smb_structs.SUPPORT_SMB2 = True
smb_structs.MAX_PAYLOAD_SIZE = 65536
CLIENT_HOSTNAME = socket.gethostname()

# --- Processing Configuration ---
DOCUMENT_SOURCE = 'internal_esg'

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

def write_file_to_nas(share_name, nas_path_relative, content, is_binary=False):
    """Writes content to a specified file path on the NAS."""
    conn = None
    print(f"   Writing to NAS path: {share_name}/{nas_path_relative}")
    try:
        conn = create_nas_connection()
        if not conn:
            return False

        dir_path = os.path.dirname(nas_path_relative).replace('\\', '/')
        if dir_path and not ensure_nas_dir_exists(conn, share_name, dir_path):
             print(f"   [ERROR] Failed to ensure output directory exists: {dir_path}")
             return False

        if is_binary:
            file_obj = io.BytesIO(content)
        else:
            content_bytes = content.encode('utf-8') if isinstance(content, str) else content
            file_obj = io.BytesIO(content_bytes)

        bytes_written = conn.storeFile(share_name, nas_path_relative, file_obj)
        print(f"   Successfully wrote {bytes_written} bytes to: {share_name}/{nas_path_relative}")
        return True
    except Exception as e:
        print(f"   [ERROR] Unexpected error writing to NAS '{share_name}/{nas_path_relative}': {e}")
        return False
    finally:
        if conn:
            conn.close()

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

def load_master_csv(filename):
    """Load a master CSV file from NAS."""
    csv_path = os.path.join(MASTER_CSV_FOLDER_PATH, filename).replace('\\', '/')
    
    try:
        df = read_csv_from_nas(NAS_PARAMS["share"], csv_path)
        
        if df is None:
            print(f"   [ERROR] Failed to read {filename} from NAS")
            return pd.DataFrame()
        
        if df.empty:
            print(f"   CSV file {filename} is empty")
            return df
        
        # Handle timestamp columns
        if not df.empty:
            for col in ['created_at', 'date_created', 'date_last_modified']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
        
        print(f"   Loaded {len(df)} records from {filename}")
        return df
            
    except Exception as e:
        print(f"   [ERROR] Failed to load CSV {filename} from NAS: {e}")
        return pd.DataFrame()

def calculate_file_hash(df):
    """Calculate a hash of the DataFrame content for integrity checking."""
    # Convert DataFrame to JSON string and calculate MD5 hash
    json_str = df.to_json(orient='records')
    return hashlib.md5(json_str.encode('utf-8')).hexdigest()

def validate_csv_data(df, csv_type):
    """Validate CSV data integrity and format."""
    issues = []
    
    if df.empty:
        issues.append(f"{csv_type}: DataFrame is empty")
        return issues
    
    # Check for required columns based on CSV type
    if csv_type == "catalog":
        required_cols = ['id', 'document_source', 'document_type', 'document_name', 'file_name']
    elif csv_type == "content":
        required_cols = ['id', 'document_source', 'document_type', 'document_name', 'section_content']
    else:
        required_cols = ['id']
    
    for col in required_cols:
        if col not in df.columns:
            issues.append(f"{csv_type}: Missing required column '{col}'")
    
    # Check for null values in critical columns
    if 'id' in df.columns:
        null_ids = df['id'].isnull().sum()
        if null_ids > 0:
            issues.append(f"{csv_type}: {null_ids} records have null IDs")
    
    # Check for duplicate IDs
    if 'id' in df.columns:
        duplicate_ids = df['id'].duplicated().sum()
        if duplicate_ids > 0:
            issues.append(f"{csv_type}: {duplicate_ids} duplicate IDs found")
    
    # Check document source consistency
    if 'document_source' in df.columns:
        unique_sources = df['document_source'].unique()
        if len(unique_sources) > 1:
            issues.append(f"{csv_type}: Multiple document sources found: {list(unique_sources)}")
    
    return issues

def prepare_final_csv(df, csv_type):
    """Prepare CSV data for final output with PostgreSQL-compatible formatting."""
    if df.empty:
        return df
    
    df_final = df.copy()
    
    # Ensure proper data types
    if 'id' in df_final.columns:
        df_final['id'] = pd.to_numeric(df_final['id'], errors='coerce').astype('Int64')
    
    if 'file_size' in df_final.columns:
        df_final['file_size'] = pd.to_numeric(df_final['file_size'], errors='coerce').astype('Int64')
    
    if 'section_id' in df_final.columns:
        df_final['section_id'] = pd.to_numeric(df_final['section_id'], errors='coerce').astype('Int64')
    
    if 'page_number' in df_final.columns:
        df_final['page_number'] = pd.to_numeric(df_final['page_number'], errors='coerce').astype('Int64')
    
    # Format timestamp columns for PostgreSQL
    timestamp_cols = ['created_at', 'date_created', 'date_last_modified']
    for col in timestamp_cols:
        if col in df_final.columns:
            df_final[col] = pd.to_datetime(df_final[col], errors='coerce', utc=True)
            # Format as ISO string with timezone
            df_final[col] = df_final[col].dt.strftime('%Y-%m-%d %H:%M:%S+00')
    
    # Handle embedding columns (convert back to JSON string if needed)
    embedding_cols = ['document_usage_embedding', 'document_description_embedding']
    for col in embedding_cols:
        if col in df_final.columns:
            # Ensure embedding data is properly formatted as JSON string or NULL
            df_final[col] = df_final[col].apply(lambda x: json.dumps(x) if pd.notna(x) and x != '' else None)
    
    # Clean text fields
    text_cols = ['document_description', 'document_usage', 'section_summary', 'section_content']
    for col in text_cols:
        if col in df_final.columns:
            # Remove null bytes and control characters that might cause PostgreSQL issues
            df_final[col] = df_final[col].astype(str).str.replace('\x00', '', regex=False)
            df_final[col] = df_final[col].str.replace('\r\n', '\n', regex=False)
            df_final[col] = df_final[col].replace('nan', None)
    
    # Sort by ID for consistent output
    if 'id' in df_final.columns:
        df_final = df_final.sort_values('id').reset_index(drop=True)
    
    return df_final

def generate_deployment_metadata(catalog_df, content_df, timestamp):
    """Generate deployment metadata for IT."""
    metadata = {
        "deployment_info": {
            "timestamp": timestamp,
            "document_source": DOCUMENT_SOURCE,
            "pipeline_version": "CSV_Migration_v1.0",
            "stage5_version": "2.0"
        },
        "file_info": {
            "catalog_file": f"{DEPLOYMENT_CATALOG_PREFIX}_{timestamp}.csv",
            "content_file": f"{DEPLOYMENT_CONTENT_PREFIX}_{timestamp}.csv",
            "catalog_records": len(catalog_df),
            "content_records": len(content_df),
            "catalog_hash": calculate_file_hash(catalog_df),
            "content_hash": calculate_file_hash(content_df)
        },
        "schema_info": {
            "catalog_columns": list(catalog_df.columns) if not catalog_df.empty else [],
            "content_columns": list(content_df.columns) if not content_df.empty else [],
            "target_tables": {
                "catalog": "apg_catalog",
                "content": "apg_content"
            }
        },
        "instructions": {
            "import_order": ["Delete existing records for document_source", "Import catalog CSV", "Import content CSV"],
            "delete_query": f"DELETE FROM apg_catalog WHERE document_source = '{DOCUMENT_SOURCE}'; DELETE FROM apg_content WHERE document_source = '{DOCUMENT_SOURCE}';",
            "notes": [
                "CSV files are formatted for PostgreSQL COPY command",
                "Timestamp columns are in UTC format",
                "Embedding columns contain JSON strings or NULL",
                "Files have been validated for data integrity"
            ]
        }
    }
    
    return metadata

def safe_datetime_format(dt_value):
    """Safely format a datetime value to ISO string, handling various input types."""
    if dt_value is None or pd.isna(dt_value):
        return None
    
    # If it's already a string, return as-is
    if isinstance(dt_value, str):
        return dt_value
    
    # If it's a datetime object, format it
    try:
        if hasattr(dt_value, 'isoformat'):
            return dt_value.isoformat()
        elif hasattr(dt_value, 'strftime'):
            return dt_value.strftime('%Y-%m-%d %H:%M:%S')
        else:
            # Fallback: convert to string
            return str(dt_value)
    except Exception:
        # Last resort: convert to string
        return str(dt_value)

def generate_summary_report(catalog_df, content_df, validation_issues, timestamp):
    """Generate a summary report of the deployment."""
    report = {
        "summary": {
            "timestamp": timestamp,
            "document_source": DOCUMENT_SOURCE,
            "status": "SUCCESS" if not validation_issues else "WARNING",
            "total_records": len(catalog_df) + len(content_df)
        },
        "catalog_stats": {
            "total_records": len(catalog_df),
            "unique_documents": catalog_df['document_name'].nunique() if 'document_name' in catalog_df.columns and not catalog_df.empty else 0,
            "file_types": catalog_df['file_type'].value_counts().to_dict() if 'file_type' in catalog_df.columns and not catalog_df.empty else {},
            "date_range": {
                "earliest": safe_datetime_format(catalog_df['date_created'].min()) if 'date_created' in catalog_df.columns and not catalog_df.empty and catalog_df['date_created'].notna().any() else None,
                "latest": safe_datetime_format(catalog_df['date_created'].max()) if 'date_created' in catalog_df.columns and not catalog_df.empty and catalog_df['date_created'].notna().any() else None
            }
        },
        "content_stats": {
            "total_records": len(content_df),
            "unique_documents": content_df['document_name'].nunique() if 'document_name' in content_df.columns and not content_df.empty else 0,
            "avg_sections_per_doc": round(len(content_df) / content_df['document_name'].nunique(), 2) if 'document_name' in content_df.columns and not content_df.empty and content_df['document_name'].nunique() > 0 else 0,
            "page_range": {
                "min_page": int(content_df['page_number'].min()) if 'page_number' in content_df.columns and not content_df.empty and content_df['page_number'].notna().any() else None,
                "max_page": int(content_df['page_number'].max()) if 'page_number' in content_df.columns and not content_df.empty and content_df['page_number'].notna().any() else None
            }
        },
        "validation": {
            "issues_found": len(validation_issues),
            "issues": validation_issues
        }
    }
    
    return report

def archive_processing_run(timestamp):
    """Archive the entire processing run by zipping and moving to archive folder."""
    try:
        # Define source and destination paths
        source_dir_relative = os.path.join(NAS_OUTPUT_FOLDER_PATH, DOCUMENT_SOURCE).replace('\\', '/')
        archive_dir_relative = os.path.join(NAS_OUTPUT_FOLDER_PATH, ARCHIVE_SUBFOLDER_NAME).replace('\\', '/')
        archive_filename = f"{DOCUMENT_SOURCE}_{timestamp}.zip"
        archive_path_relative = os.path.join(archive_dir_relative, archive_filename).replace('\\', '/')
        
        print(f"   Creating archive: {archive_filename}")
        print(f"   Source: {NAS_PARAMS['share']}/{source_dir_relative}")
        print(f"   Archive: {NAS_PARAMS['share']}/{archive_path_relative}")
        
        # Create temporary zip file locally
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
            temp_zip_path = temp_zip.name
        
        try:
            # Download all files from processing directory and zip them
            conn = create_nas_connection()
            if not conn:
                print("   [ERROR] Failed to connect to NAS for archiving")
                return False
            
            # Create zip file
            with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Walk through the source directory and add all files to zip
                files_added = 0
                try:
                    files = conn.listPath(NAS_PARAMS["share"], source_dir_relative)
                    for file_info in files:
                        if file_info.filename in ['.', '..']:
                            continue
                        
                        file_path = os.path.join(source_dir_relative, file_info.filename).replace('\\', '/')
                        if not file_info.isDirectory:
                            # Download file content
                            file_obj = io.BytesIO()
                            conn.retrieveFile(NAS_PARAMS["share"], file_path, file_obj)
                            file_obj.seek(0)
                            file_content = file_obj.read()
                            
                            # Add to zip
                            zipf.writestr(file_info.filename, file_content)
                            files_added += 1
                        
                except Exception as e:
                    print(f"   [ERROR] Error reading source directory for archiving: {e}")
                    return False
                
                print(f"   Added {files_added} files to archive")
            
            # Ensure archive directory exists
            if not ensure_nas_dir_exists(conn, NAS_PARAMS["share"], archive_dir_relative):
                print("   [ERROR] Failed to create archive directory")
                return False
            
            # Upload zip file to NAS
            with open(temp_zip_path, 'rb') as zip_file:
                zip_content = zip_file.read()
            
            file_obj = io.BytesIO(zip_content)
            bytes_written = conn.storeFile(NAS_PARAMS["share"], archive_path_relative, file_obj)
            print(f"   Successfully archived {bytes_written} bytes to: {archive_filename}")
            
            conn.close()
            return True
            
        finally:
            # Clean up temporary zip file
            try:
                os.unlink(temp_zip_path)
            except Exception:
                pass
                
    except Exception as e:
        print(f"   [ERROR] Failed to archive processing run: {e}")
        return False

# ==============================================================================
# --- Main Processing Function ---
# ==============================================================================

def main_processing_stage5():
    """Handles the core logic for Stage 5: create timestamped deployment files and archive processing run."""
    print(f"--- Starting Main Processing for Stage 5 ---")
    
    # Generate timestamp for this deployment
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    print(f"   Deployment timestamp: {timestamp}")

    # --- Load Updated Master CSV Files ---
    print("[2] Loading Updated Master CSV Files from Stage 4...")
    catalog_df = load_master_csv(MASTER_CATALOG_CSV)
    content_df = load_master_csv(MASTER_CONTENT_CSV)
    
    if catalog_df is None or content_df is None:
        print("[CRITICAL ERROR] Failed to load master CSV files. Exiting.")
        sys.exit(1)
    
    print(f"   Loaded catalog: {len(catalog_df)} records")
    print(f"   Loaded content: {len(content_df)} records")
    print("-" * 60)

    # --- Validate Data Integrity ---
    print("[3] Validating Data Integrity...")
    catalog_issues = validate_csv_data(catalog_df, "catalog")
    content_issues = validate_csv_data(content_df, "content")
    all_issues = catalog_issues + content_issues
    
    if all_issues:
        print(f"   Found {len(all_issues)} validation issues:")
        for issue in all_issues:
            print(f"     - {issue}")
        if any("Missing required column" in issue for issue in all_issues):
            print("[CRITICAL ERROR] Critical validation failures found. Exiting.")
            sys.exit(1)
    else:
        print("   All validation checks passed")
    print("-" * 60)

    # --- Prepare Final CSV Data ---
    print("[4] Preparing CSV Data for Deployment...")
    final_catalog_df = prepare_final_csv(catalog_df, "catalog")
    final_content_df = prepare_final_csv(content_df, "content")
    
    print(f"   Prepared catalog: {len(final_catalog_df)} records")
    print(f"   Prepared content: {len(final_content_df)} records")
    print("-" * 60)

    # --- Create Timestamped Deployment Files ---
    print("[5] Creating Timestamped Deployment Files for IT...")
    
    # Define deployment paths - directly in deployment folder (no subfolder)
    
    # Timestamped filenames
    catalog_filename = f"{DEPLOYMENT_CATALOG_PREFIX}_{timestamp}.csv"
    content_filename = f"{DEPLOYMENT_CONTENT_PREFIX}_{timestamp}.csv"
    
    catalog_output_path = os.path.join(NAS_DEPLOYMENT_FOLDER_PATH, catalog_filename).replace('\\', '/')
    content_output_path = os.path.join(NAS_DEPLOYMENT_FOLDER_PATH, content_filename).replace('\\', '/')
    
    # Convert DataFrames to CSV strings
    catalog_csv_content = final_catalog_df.to_csv(index=False, na_rep='NULL')
    content_csv_content = final_content_df.to_csv(index=False, na_rep='NULL')
    
    print(f"   Generated timestamped deployment files:")
    print(f"     - {catalog_filename}")
    print(f"     - {content_filename}")
    print("-" * 60)

    # --- Write Deployment Files to NAS ---
    print("[6] Writing Timestamped Files to IT Pickup Folder...")
    
    files_to_write = [
        (catalog_output_path, catalog_csv_content, f"Timestamped catalog CSV ({catalog_filename})"),
        (content_output_path, content_csv_content, f"Timestamped content CSV ({content_filename})")
    ]
    
    success_count = 0
    for file_path, content, description in files_to_write:
        print(f"   Writing {description}...")
        if write_file_to_nas(NAS_PARAMS["share"], file_path, content):
            success_count += 1
        else:
            print(f"   [ERROR] Failed to write {description}")
    
    if success_count != len(files_to_write):
        print(f"[CRITICAL ERROR] Failed to write {len(files_to_write) - success_count} deployment files. Exiting.")
        sys.exit(1)
    
    print(f"   Successfully wrote all {len(files_to_write)} deployment files")
    print("-" * 60)

    # --- Archive Processing Run ---
    print("[7] Archiving Processing Run...")
    archive_success = archive_processing_run(timestamp)
    if not archive_success:
        print("   [WARNING] Failed to archive processing run, but deployment files are ready.")
    print("-" * 60)

    # --- Final Summary ---
    print("[8] Deployment and Archive Summary...")
    print(f"   Document Source: {DOCUMENT_SOURCE}")
    print(f"   Deployment Timestamp: {timestamp}")
    print(f"   Catalog Records: {len(final_catalog_df)}")
    print(f"   Content Records: {len(final_content_df)}")
    print(f"   Deployment Location: {NAS_PARAMS['share']}/{NAS_DEPLOYMENT_FOLDER_PATH}")
    print(f"   Validation Issues: {len(all_issues)}")
    print(f"   Files Ready for IT Pickup:")
    print(f"     - {catalog_filename}")
    print(f"     - {content_filename}")
    if archive_success:
        print(f"   Processing run archived successfully")
    
    print("-" * 60)
    print("\n" + "="*60)
    print(f"--- Stage 5 Completed Successfully ---")
    print("--- Deployment files ready for IT pickup ---")
    print("="*60 + "\n")

# ==============================================================================
# --- Script Entry Point ---
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print(f"--- Running Stage 5: Output Final CSV Files ---")
    print(f"--- Document Source: {DOCUMENT_SOURCE} ---")
    print("="*60 + "\n")

    # --- Check for Skip Flag ---
    print("[1] Checking for skip flag from Stage 1...")
    source_base_dir_relative = os.path.join(NAS_OUTPUT_FOLDER_PATH, DOCUMENT_SOURCE).replace('\\', '/')
    skip_flag_file_name = '_SKIP_SUBSEQUENT_STAGES.flag'
    skip_flag_relative_path = os.path.join(source_base_dir_relative, skip_flag_file_name).replace('\\', '/')
    
    print(f"   Checking for flag file: {NAS_PARAMS['share']}/{skip_flag_relative_path}")
    should_skip = False
    
    try:
        conn = create_nas_connection()
        if conn:
            try:
                conn.getAttributes(NAS_PARAMS["share"], skip_flag_relative_path)
                should_skip = True
                print(f"   Skip flag file found. Stage 1 indicated no files to process.")
            except:
                print(f"   Skip flag file not found. Proceeding with Stage 5.")
            conn.close()
        else:
            print(f"   Could not connect to NAS. Proceeding with Stage 5.")
    except Exception as e:
        print(f"   [WARNING] Error checking for skip flag: {e}")
        print(f"   Proceeding with Stage 5.")
    
    print("-" * 60)

    # --- Execute Main Processing if Not Skipped ---
    if should_skip:
        print("\n" + "="*60)
        print(f"--- Stage 5 Skipped (No files to process from Stage 1) ---")
        print("="*60 + "\n")
    else:
        main_processing_stage5()