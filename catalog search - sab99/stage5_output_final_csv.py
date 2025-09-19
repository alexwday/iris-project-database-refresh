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
import time
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
    "port": 445,
}
NAS_OUTPUT_FOLDER_PATH = "path/to/your/output_folder"
NAS_DEPLOYMENT_FOLDER_PATH = "path/to/deployment_folder"

# --- pysmb Configuration ---
smb_structs.SUPPORT_SMB2 = True
smb_structs.SUPPORT_SMB2x = True  # Enable SMB 2.x support
smb_structs.MAX_PAYLOAD_SIZE = 65536
CLIENT_HOSTNAME = socket.gethostname()

# --- Processing Configuration ---
# Document sources configuration - each line contains source name and detail level
DOCUMENT_SOURCES = """
internal_sab_99,detailed
"""


def load_document_sources():
    """Parse document sources configuration - works for all stages"""
    sources = []
    for line in DOCUMENT_SOURCES.strip().split("\n"):
        line = line.strip()
        if line and not line.startswith("#"):
            parts = line.split(",")
            if len(parts) == 2:
                source_name = parts[0].strip()
                detail_level = parts[1].strip()
                sources.append({"name": source_name, "detail_level": detail_level})
            else:
                print(f"Warning: Invalid config line ignored: {line}")
    return sources


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
            is_direct_tcp=(NAS_PARAMS["port"] == 445),
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

    path_parts = dir_path.strip("/").split("/")
    current_path = ""
    try:
        for part in path_parts:
            if not part:
                continue
            current_path = os.path.join(current_path, part).replace("\\", "/")
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

        dir_path = os.path.dirname(nas_path_relative).replace("\\", "/")
        if dir_path and not ensure_nas_dir_exists(conn, share_name, dir_path):
            print(f"   [ERROR] Failed to ensure output directory exists: {dir_path}")
            return False

        if is_binary:
            file_obj = io.BytesIO(content)
        else:
            content_bytes = (
                content.encode("utf-8") if isinstance(content, str) else content
            )
            file_obj = io.BytesIO(content_bytes)

        bytes_written = conn.storeFile(share_name, nas_path_relative, file_obj)
        print(
            f"   Successfully wrote {bytes_written} bytes to: {share_name}/{nas_path_relative}"
        )
        return True
    except Exception as e:
        print(
            f"   [ERROR] Unexpected error writing to NAS '{share_name}/{nas_path_relative}': {e}"
        )
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
        file_attributes, filesize = conn.retrieveFile(
            share_name, nas_path_relative, file_obj
        )
        file_obj.seek(0)
        csv_content = file_obj.read().decode("utf-8")

        # Read CSV from string
        from io import StringIO

        df = pd.read_csv(StringIO(csv_content))
        print(
            f"   Successfully read CSV from NAS: {share_name}/{nas_path_relative} ({len(df)} records)"
        )
        return df
    except Exception as e:
        print(
            f"   [ERROR] Failed to read CSV from NAS '{share_name}/{nas_path_relative}': {e}"
        )
        return None
    finally:
        if conn:
            conn.close()


def load_master_csv(filename):
    """Load a master CSV file from NAS."""
    csv_path = os.path.join(MASTER_CSV_FOLDER_PATH, filename).replace("\\", "/")

    try:
        df = read_csv_from_nas(NAS_PARAMS["share"], csv_path)

        if df is None:
            print(f"   [ERROR] Failed to read {filename} from NAS")
            return pd.DataFrame()

        if df.empty:
            print(f"   CSV file {filename} is empty")
            return df

        # Handle timestamp columns (only process columns that actually exist)
        if not df.empty:
            timestamp_cols = ["created_at", "date_created", "date_last_modified"]
            for col in timestamp_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

        print(f"   Loaded {len(df)} records from {filename}")
        return df

    except Exception as e:
        print(f"   [ERROR] Failed to load CSV {filename} from NAS: {e}")
        return pd.DataFrame()


def calculate_file_hash(df):
    """Calculate a hash of the DataFrame content for integrity checking."""
    # Convert DataFrame to JSON string and calculate MD5 hash
    json_str = df.to_json(orient="records")
    return hashlib.md5(json_str.encode("utf-8")).hexdigest()


def validate_csv_data(df, csv_type):
    """Validate CSV data integrity and format."""
    issues = []

    if df.empty:
        issues.append(f"{csv_type}: DataFrame is empty")
        return issues

    # Check for required columns based on CSV type
    if csv_type == "catalog":
        required_cols = [
            "id",
            "document_source",
            "document_type",
            "document_name",
            "file_name",
        ]
    elif csv_type == "content":
        required_cols = [
            "id",
            "document_source",
            "document_type",
            "document_name",
            "section_content",
        ]
    else:
        required_cols = ["id"]

    for col in required_cols:
        if col not in df.columns:
            issues.append(f"{csv_type}: Missing required column '{col}'")

    # Check for null values in critical columns
    if "id" in df.columns:
        null_ids = df["id"].isnull().sum()
        if null_ids > 0:
            issues.append(f"{csv_type}: {null_ids} records have null IDs")

    # Check for duplicate IDs
    if "id" in df.columns:
        duplicate_ids = df["id"].duplicated().sum()
        if duplicate_ids > 0:
            issues.append(f"{csv_type}: {duplicate_ids} duplicate IDs found")

    # Check document source consistency
    if "document_source" in df.columns:
        unique_sources = df["document_source"].unique()
        if len(unique_sources) > 1:
            issues.append(
                f"{csv_type}: Multiple document sources found: {list(unique_sources)}"
            )

    return issues


def prepare_final_csv(df, csv_type):
    """Prepare CSV data for final output with PostgreSQL-compatible formatting."""
    if df.empty:
        return df

    df_final = df.copy()

    # Remove PostgreSQL auto-generated fields and processing artifacts for clean import
    # These will be auto-generated by PostgreSQL on import or are not part of the schema
    postgres_auto_fields = ["id", "created_at", "processed_json_path"]
    for field in postgres_auto_fields:
        if field in df_final.columns:
            df_final = df_final.drop(field, axis=1)
            print(f"   Removed auto-generated field: {field}")
    
    # Remove table-specific columns that don't belong in each table
    if csv_type == "content":
        # Content table should NOT have catalog-only columns
        catalog_only_fields = [
            "date_created", "date_last_modified", "file_name", "file_type", 
            "file_size", "file_path", "file_link", "document_description", 
            "document_usage", "document_usage_embedding", "document_description_embedding"
        ]
        for field in catalog_only_fields:
            if field in df_final.columns:
                df_final = df_final.drop(field, axis=1)
                print(f"   Removed catalog-only field from content table: {field}")
    elif csv_type == "catalog":
        # Catalog table should NOT have content-only columns  
        content_only_fields = ["section_id", "section_name", "section_summary", "section_content", "page_number"]
        for field in content_only_fields:
            if field in df_final.columns:
                df_final = df_final.drop(field, axis=1)
                print(f"   Removed content-only field from catalog table: {field}")

    # Ensure proper data types for remaining fields

    if "file_size" in df_final.columns:
        df_final["file_size"] = pd.to_numeric(
            df_final["file_size"], errors="coerce"
        ).astype("Int64")

    if "section_id" in df_final.columns:
        df_final["section_id"] = pd.to_numeric(
            df_final["section_id"], errors="coerce"
        ).astype("Int64")

    if "page_number" in df_final.columns:
        df_final["page_number"] = pd.to_numeric(
            df_final["page_number"], errors="coerce"
        ).astype("Int64")

    # Format timestamp columns for PostgreSQL (excluding created_at which was removed)
    # Note: Only catalog table has date_created/date_last_modified columns
    if csv_type == "catalog":
        timestamp_cols = ["date_created", "date_last_modified"]
        for col in timestamp_cols:
            if col in df_final.columns:
                df_final[col] = pd.to_datetime(df_final[col], errors="coerce", utc=True)
                # Format as ISO string with timezone
                df_final[col] = df_final[col].dt.strftime("%Y-%m-%d %H:%M:%S+00")

    # Handle embedding columns (format for PostgreSQL vector type)
    # Note: Only catalog table has embedding columns
    if csv_type == "catalog":
        embedding_cols = ["document_usage_embedding", "document_description_embedding"]
        for col in embedding_cols:
            if col in df_final.columns:
                # Format for PostgreSQL vector type: [1,2,3] not "[1,2,3]"
                def format_vector(x):
                    if pd.isna(x) or x == "" or x is None:
                        return None
                    if isinstance(x, str):
                        # If already a string, remove any extra quotes and ensure proper format
                        x = x.strip().strip('"')
                        if x.startswith("[") and x.endswith("]"):
                            return x
                        else:
                            return None
                    elif isinstance(x, (list, tuple)):
                        # Convert list/tuple to vector format
                        return "[" + ",".join(map(str, x)) + "]"
                    else:
                        return None

                df_final[col] = df_final[col].apply(format_vector)

    # Clean text fields
    text_cols = [
        "document_description",
        "document_usage",
        "section_summary",
        "section_content",
    ]
    for col in text_cols:
        if col in df_final.columns:
            # Remove null bytes and control characters that might cause PostgreSQL issues
            df_final[col] = (
                df_final[col].astype(str).str.replace("\x00", "", regex=False)
            )
            df_final[col] = df_final[col].str.replace("\r\n", "\n", regex=False)
            df_final[col] = df_final[col].replace("nan", None)

    # Sort for consistent output (using document identifiers since id was removed)
    sort_cols = []
    for col in ["document_source", "document_type", "document_name", "section_id"]:
        if col in df_final.columns:
            sort_cols.append(col)

    if sort_cols:
        df_final = df_final.sort_values(sort_cols).reset_index(drop=True)

    return df_final


def generate_deployment_metadata(catalog_df, content_df, timestamp, sources_included):
    """Generate deployment metadata for IT."""
    # Build the delete query without nested f-strings
    source_list = ", ".join([f"'{src}'" for src in sources_included])
    delete_query = f"DELETE FROM apg_catalog WHERE document_source IN ({source_list}); DELETE FROM apg_content WHERE document_source IN ({source_list});"

    metadata = {
        "deployment_info": {
            "timestamp": timestamp,
            "document_sources": sources_included,  # List of all sources included
            "pipeline_version": "CSV_Migration_v1.0",
            "stage5_version": "2.0",
        },
        "file_info": {
            "catalog_file": f"{DEPLOYMENT_CATALOG_PREFIX}_{timestamp}.csv",
            "content_file": f"{DEPLOYMENT_CONTENT_PREFIX}_{timestamp}.csv",
            "catalog_records": len(catalog_df),
            "content_records": len(content_df),
            "catalog_hash": calculate_file_hash(catalog_df),
            "content_hash": calculate_file_hash(content_df),
        },
        "schema_info": {
            "catalog_columns": list(catalog_df.columns) if not catalog_df.empty else [],
            "content_columns": list(content_df.columns) if not content_df.empty else [],
            "target_tables": {"catalog": "apg_catalog", "content": "apg_content"},
        },
        "instructions": {
            "import_order": [
                "Delete existing records for document_source",
                "Import catalog CSV",
                "Import content CSV",
            ],
            "delete_query": delete_query,
            "notes": [
                "CSV files are formatted for PostgreSQL COPY command",
                "Timestamp columns are in UTC format",
                "Embedding columns contain JSON strings or NULL",
                "Files have been validated for data integrity",
            ],
        },
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
        if hasattr(dt_value, "isoformat"):
            return dt_value.isoformat()
        elif hasattr(dt_value, "strftime"):
            return dt_value.strftime("%Y-%m-%d %H:%M:%S")
        else:
            # Fallback: convert to string
            return str(dt_value)
    except Exception:
        # Last resort: convert to string
        return str(dt_value)


def generate_summary_report(
    catalog_df, content_df, validation_issues, timestamp, sources_included
):
    """Generate a summary report of the deployment."""
    report = {
        "summary": {
            "timestamp": timestamp,
            "document_sources": sources_included,  # List of all sources included
            "status": "SUCCESS" if not validation_issues else "WARNING",
            "total_records": len(catalog_df) + len(content_df),
        },
        "catalog_stats": {
            "total_records": len(catalog_df),
            "unique_documents": (
                catalog_df["document_name"].nunique()
                if "document_name" in catalog_df.columns and not catalog_df.empty
                else 0
            ),
            "file_types": (
                catalog_df["file_type"].value_counts().to_dict()
                if "file_type" in catalog_df.columns and not catalog_df.empty
                else {}
            ),
            "date_range": {
                "earliest": (
                    safe_datetime_format(catalog_df["date_created"].min())
                    if "date_created" in catalog_df.columns
                    and not catalog_df.empty
                    and catalog_df["date_created"].notna().any()
                    else None
                ),
                "latest": (
                    safe_datetime_format(catalog_df["date_created"].max())
                    if "date_created" in catalog_df.columns
                    and not catalog_df.empty
                    and catalog_df["date_created"].notna().any()
                    else None
                ),
            },
        },
        "content_stats": {
            "total_records": len(content_df),
            "unique_documents": (
                content_df["document_name"].nunique()
                if "document_name" in content_df.columns and not content_df.empty
                else 0
            ),
            "avg_sections_per_doc": (
                round(len(content_df) / content_df["document_name"].nunique(), 2)
                if "document_name" in content_df.columns
                and not content_df.empty
                and content_df["document_name"].nunique() > 0
                else 0
            ),
            "page_range": {
                "min_page": (
                    int(content_df["page_number"].min())
                    if "page_number" in content_df.columns
                    and not content_df.empty
                    and content_df["page_number"].notna().any()
                    else None
                ),
                "max_page": (
                    int(content_df["page_number"].max())
                    if "page_number" in content_df.columns
                    and not content_df.empty
                    and content_df["page_number"].notna().any()
                    else None
                ),
            },
        },
        "validation": {
            "issues_found": len(validation_issues),
            "issues": validation_issues,
        },
    }

    return report


def archive_directory_recursive(conn, share_name, dir_path, zipf, base_path=""):
    """
    Recursively archive all files and subdirectories.

    Args:
        conn: SMB connection object
        share_name: Name of the SMB share
        dir_path: Path to directory to archive (relative to share)
        zipf: ZipFile object to add files to
        base_path: Base path within the zip file for maintaining directory structure

    Returns:
        int: Number of files archived
    """
    files_archived = 0

    try:
        files = conn.listPath(share_name, dir_path)
        for file_info in files:
            if file_info.filename in [".", ".."]:
                continue

            file_path = os.path.join(dir_path, file_info.filename).replace("\\", "/")
            archive_path = os.path.join(base_path, file_info.filename).replace(
                "\\", "/"
            )

            if file_info.isDirectory:
                # Recursively archive subdirectory
                files_archived += archive_directory_recursive(
                    conn, share_name, file_path, zipf, archive_path
                )
            else:
                # Archive the file
                try:
                    file_obj = io.BytesIO()
                    conn.retrieveFile(share_name, file_path, file_obj)
                    file_obj.seek(0)
                    zipf.writestr(archive_path, file_obj.read())
                    files_archived += 1
                except Exception as e:
                    print(f"      [WARNING] Failed to archive file {archive_path}: {e}")

    except Exception as e:
        print(f"      [WARNING] Failed to list directory {dir_path}: {e}")

    return files_archived


def delete_directory_simple(conn, share_name, dir_path, max_retries=3):
    """
    Simple recursive directory deletion with minimal retries.
    Focus on speed over comprehensive error handling.

    Args:
        conn: SMB connection object
        share_name: Name of the SMB share
        dir_path: Path to directory to delete (relative to share)
        max_retries: Maximum retry attempts for the entire operation

    Returns:
        bool: True if deletion completed (even partially), False on critical failure
    """
    for attempt in range(max_retries):
        try:
            # Try to delete everything recursively
            _delete_dir_contents(conn, share_name, dir_path)

            # Try to delete the main directory
            try:
                conn.deleteDirectory(share_name, dir_path)
                print(
                    f"      ✓ Successfully deleted directory: {os.path.basename(dir_path)}"
                )
                return True
            except Exception as e:
                error_msg = str(e).lower()
                if "not found" in error_msg or "object_name_not_found" in error_msg:
                    # Already deleted
                    return True
                elif attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    print(
                        f"      [WARNING] Could not delete main directory {dir_path}, but contents were cleared"
                    )
                    return True  # Consider it successful if contents are gone

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"      Retry {attempt + 1}/{max_retries} for directory deletion")
                time.sleep(2)
            else:
                print(
                    f"      [WARNING] Directory deletion incomplete after {max_retries} attempts: {e}"
                )
                return True  # Still return True to continue processing

    return True


def _delete_dir_contents(conn, share_name, dir_path):
    """
    Helper function to delete directory contents without excessive logging.
    """
    try:
        files = conn.listPath(share_name, dir_path)

        # First delete all files
        for file_info in files:
            if file_info.filename in [".", ".."]:
                continue

            file_path = os.path.join(dir_path, file_info.filename).replace("\\", "/")

            if not file_info.isDirectory:
                try:
                    conn.deleteFiles(share_name, file_path)
                except Exception:
                    # Ignore individual file errors
                    pass

        # Then delete all subdirectories
        for file_info in files:
            if file_info.filename in [".", ".."]:
                continue

            if file_info.isDirectory:
                subdir_path = os.path.join(dir_path, file_info.filename).replace(
                    "\\", "/"
                )
                try:
                    # Recursively delete subdirectory
                    _delete_dir_contents(conn, share_name, subdir_path)
                    conn.deleteDirectory(share_name, subdir_path)
                except Exception:
                    # Ignore individual directory errors
                    pass

    except Exception:
        # Ignore listing errors
        pass


def archive_processing_run(document_source, timestamp):
    """Archive the entire processing run by zipping and moving to archive folder."""
    try:
        # Define source and destination paths
        source_dir_relative = os.path.join(
            NAS_OUTPUT_FOLDER_PATH, document_source
        ).replace("\\", "/")
        archive_dir_relative = os.path.join(
            NAS_OUTPUT_FOLDER_PATH, ARCHIVE_SUBFOLDER_NAME
        ).replace("\\", "/")
        archive_filename = f"{document_source}_{timestamp}.zip"
        archive_path_relative = os.path.join(
            archive_dir_relative, archive_filename
        ).replace("\\", "/")

        print(f"   Creating archive: {archive_filename}")
        print(f"   Source: {NAS_PARAMS['share']}/{source_dir_relative}")
        print(f"   Archive: {NAS_PARAMS['share']}/{archive_path_relative}")

        # Create temporary zip file locally
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_zip:
            temp_zip_path = temp_zip.name

        try:
            # Download all files from processing directory and zip them
            conn = create_nas_connection()
            if not conn:
                print("   [ERROR] Failed to connect to NAS for archiving")
                return False

            # Create zip file with recursive archiving
            with zipfile.ZipFile(temp_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                print("   Archiving all files and subdirectories...")
                files_added = archive_directory_recursive(
                    conn, NAS_PARAMS["share"], source_dir_relative, zipf
                )
                print(f"   Added {files_added} files to archive")

                if files_added == 0:
                    print("   [WARNING] No files found to archive")

            # Ensure archive directory exists
            if not ensure_nas_dir_exists(
                conn, NAS_PARAMS["share"], archive_dir_relative
            ):
                print("   [ERROR] Failed to create archive directory")
                return False

            # Upload zip file to NAS
            with open(temp_zip_path, "rb") as zip_file:
                zip_content = zip_file.read()

            file_obj = io.BytesIO(zip_content)
            bytes_written = conn.storeFile(
                NAS_PARAMS["share"], archive_path_relative, file_obj
            )
            print(
                f"   Successfully archived {bytes_written} bytes to: {archive_filename}"
            )

            # Remove the source directory after successful archiving
            print(f"   Cleaning up source directory: {source_dir_relative}")
            cleanup_success = delete_directory_simple(
                conn, NAS_PARAMS["share"], source_dir_relative
            )
            if cleanup_success:
                print(f"   ✓ Source directory cleaned up: {source_dir_relative}")
            else:
                print(
                    f"   [WARNING] Could not fully clean up source directory: {source_dir_relative}"
                )

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


def main_processing_stage5_deployment():
    """Handles the core logic for Stage 5: create timestamped deployment files (runs once for all sources)."""
    print(f"--- Starting Main Processing for Stage 5 (Deployment Files) ---")

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

    catalog_output_path = os.path.join(
        NAS_DEPLOYMENT_FOLDER_PATH, catalog_filename
    ).replace("\\", "/")
    content_output_path = os.path.join(
        NAS_DEPLOYMENT_FOLDER_PATH, content_filename
    ).replace("\\", "/")

    # Convert DataFrames to CSV strings
    catalog_csv_content = final_catalog_df.to_csv(index=False, na_rep="NULL")
    content_csv_content = final_content_df.to_csv(index=False, na_rep="NULL")

    print(f"   Generated timestamped deployment files:")
    print(f"     - {catalog_filename}")
    print(f"     - {content_filename}")
    print("-" * 60)

    # --- Write Deployment Files to NAS ---
    print("[6] Writing Timestamped Files to IT Pickup Folder...")

    files_to_write = [
        (
            catalog_output_path,
            catalog_csv_content,
            f"Timestamped catalog CSV ({catalog_filename})",
        ),
        (
            content_output_path,
            content_csv_content,
            f"Timestamped content CSV ({content_filename})",
        ),
    ]

    success_count = 0
    for file_path, content, description in files_to_write:
        print(f"   Writing {description}...")
        if write_file_to_nas(NAS_PARAMS["share"], file_path, content):
            success_count += 1
        else:
            print(f"   [ERROR] Failed to write {description}")

    if success_count != len(files_to_write):
        print(
            f"[CRITICAL ERROR] Failed to write {len(files_to_write) - success_count} deployment files. Exiting."
        )
        sys.exit(1)

    print(f"   Successfully wrote all {len(files_to_write)} deployment files")
    print("-" * 60)

    # Note: Archiving moved to per-source processing
    print(
        "[7] Deployment files created successfully. Archiving will be done per-source."
    )
    print("-" * 60)

    # --- Final Summary ---
    print("[8] Deployment Summary...")
    print(f"   Document Sources: ALL_SOURCES_COMBINED")
    print(f"   Deployment Timestamp: {timestamp}")
    print(f"   Catalog Records: {len(final_catalog_df)}")
    print(f"   Content Records: {len(final_content_df)}")
    print(f"   Deployment Location: {NAS_PARAMS['share']}/{NAS_DEPLOYMENT_FOLDER_PATH}")
    print(f"   Validation Issues: {len(all_issues)}")
    print(f"   Files Ready for IT Pickup:")
    print(f"     - {catalog_filename}")
    print(f"     - {content_filename}")
    print(f"   Note: Individual source archiving will be handled separately")

    print("-" * 60)

    return timestamp  # Return timestamp for use in archiving


# ==============================================================================
# --- Script Entry Point ---
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(f"--- Running Stage 5: Output Final CSV Files (All Sources) ---")
    print("=" * 60 + "\n")

    # Get document sources
    sources = load_document_sources()
    print(f"[0] Processing {len(sources)} document sources:")
    for source in sources:
        print(f"   - {source['name']} (detail level: {source['detail_level']})")
    print("-" * 60)

    # Track overall processing results
    all_sources_processed = []
    sources_archived = []

    # --- Step 1: Create Deployment Files (Once for All Sources) ---
    print("[1] Creating final deployment files from all processed sources...")

    # Check if any sources have content to process (for deployment files only)
    sources_to_process = []
    for source_config in sources:
        DOCUMENT_SOURCE = source_config["name"]
        source_base_dir_relative = os.path.join(
            NAS_OUTPUT_FOLDER_PATH, DOCUMENT_SOURCE
        ).replace("\\", "/")
        skip_flag_relative_path = os.path.join(
            source_base_dir_relative, "_SKIP_SUBSEQUENT_STAGES.flag"
        ).replace("\\", "/")

        try:
            if not check_nas_path_exists(NAS_PARAMS["share"], skip_flag_relative_path):
                sources_to_process.append(source_config)
                print(
                    f"   Source '{DOCUMENT_SOURCE}' has content to include in deployment"
                )
            else:
                print(
                    f"   Source '{DOCUMENT_SOURCE}' was skipped (no files to process, but will still be archived)"
                )
        except:
            # Assume source has content if we can't check
            sources_to_process.append(source_config)
            print(f"   Source '{DOCUMENT_SOURCE}' included (skip flag check failed)")

    print(f"   Found {len(sources_to_process)} sources with content for deployment")

    # Always proceed with deployment and archiving - even if no sources have new content,
    # we still need to archive and clean up any folders created during Stage 1
    print("-" * 60)

    # --- Execute Deployment File Creation (Once) ---
    try:
        if sources_to_process:
            deployment_timestamp = main_processing_stage5_deployment()
            print(
                f"   Deployment files created successfully with timestamp: {deployment_timestamp}"
            )
        else:
            print(f"   No sources have new content - skipping deployment file creation")
            deployment_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            print(f"   Using timestamp for archiving: {deployment_timestamp}")
    except Exception as e:
        print(f"\n[ERROR] Stage 5 deployment file creation failed: {e}")
        sys.exit(1)

    print("-" * 60)

    # --- Step 2: Archive Each Source Individually ---
    print("[2] Archiving each source's processing run...")

    for source_config in sources:
        DOCUMENT_SOURCE = source_config["name"]

        print(f"\n{'='*60}")
        print(f"Archiving Document Source: {DOCUMENT_SOURCE}")
        print(f"{'='*60}\n")

        # Always perform archiving and cleanup to ensure folders are cleaned up,
        # regardless of whether files were processed in this run
        source_base_dir_relative = os.path.join(
            NAS_OUTPUT_FOLDER_PATH, DOCUMENT_SOURCE
        ).replace("\\", "/")
        skip_flag_relative_path = os.path.join(
            source_base_dir_relative, "_SKIP_SUBSEQUENT_STAGES.flag"
        ).replace("\\", "/")

        skip_flag_exists = False
        try:
            if check_nas_path_exists(NAS_PARAMS["share"], skip_flag_relative_path):
                print(
                    f"   Skip flag found for '{DOCUMENT_SOURCE}', but still proceeding with archiving for cleanup."
                )
                skip_flag_exists = True
            else:
                print(
                    f"   No skip flag found. Proceeding with archiving for '{DOCUMENT_SOURCE}'."
                )
        except Exception as e:
            print(f"   [WARNING] Error checking skip flag for '{DOCUMENT_SOURCE}': {e}")
            print(f"   Proceeding with archiving attempt.")

        # Always archive and cleanup - prior stages may have created subfolders that need cleanup
        try:
            print(f"   Archiving processing run for source '{DOCUMENT_SOURCE}'...")
            archive_success = archive_processing_run(
                DOCUMENT_SOURCE, deployment_timestamp
            )
            if archive_success:
                sources_archived.append(DOCUMENT_SOURCE)
                print(f"   Successfully archived source '{DOCUMENT_SOURCE}'")
            else:
                print(
                    f"   [WARNING] Failed to archive source '{DOCUMENT_SOURCE}', but continuing with other sources."
                )
        except Exception as e:
            print(f"   [ERROR] Archiving failed for source '{DOCUMENT_SOURCE}': {e}")
            print(f"   Continuing with other sources.")

        # Track this source as processed
        all_sources_processed.append(DOCUMENT_SOURCE)
        print(f"   Source '{DOCUMENT_SOURCE}' archiving completed.")
        print("-" * 60)

    # Final summary
    print("\n" + "=" * 60)
    print(f"--- Stage 5 Completed Successfully ---")
    print(f"--- Processed {len(all_sources_processed)} sources ---")
    print(f"--- Created deployment files from {len(sources_to_process)} sources ---")
    if sources_to_process:
        processed_names = [s["name"] for s in sources_to_process]
        print(f"--- Sources with deployment data: {', '.join(processed_names)} ---")
    print(f"--- Successfully archived {len(sources_archived)} sources ---")
    if sources_archived:
        print(f"--- Sources archived: {', '.join(sources_archived)} ---")
    print("=" * 60 + "\n")
