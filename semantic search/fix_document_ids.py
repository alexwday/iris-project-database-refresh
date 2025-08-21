#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix Document IDs Script
Fixes document_id fields in master CSV by replacing hyphens with underscores
for IASB document types (iasb-ias, iasb-ifrs, iasb-ifric, iasb-sic).

This script:
1. Loads the master CSV from NAS
2. Fixes all document_id fields (iasb-* -> iasb_*)
3. Saves the corrected master CSV
4. Creates a timestamped deployment copy
5. Reports statistics on changes made
"""

import os
import csv
import io
import socket
from datetime import datetime
from smb.SMBConnection import SMBConnection
from smb import smb_structs

# ==============================================================================
# Configuration - Copy from Stage 5
# ==============================================================================

# --- NAS Configuration ---
NAS_PARAMS = {
    "ip": "your_nas_ip",  # TODO: Replace with actual NAS IP
    "share": "your_share_name",  # TODO: Replace with actual share name
    "user": "your_nas_user",  # TODO: Replace with actual NAS username
    "password": "your_nas_password",  # TODO: Replace with actual NAS password
    "port": 445,  # Default SMB port (can be 139)
}

# --- File Paths ---
MASTER_CSV_PATH = "semantic_search/pipeline_output/stage5/master_database.csv"
DEPLOYMENT_FOLDER_PATH = "semantic_search/deployment"
DEPLOYMENT_PREFIX = "iris_semantic_search"

# --- pysmb Configuration ---
smb_structs.SUPPORT_SMB2 = True
smb_structs.MAX_PAYLOAD_SIZE = 65536
CLIENT_HOSTNAME = socket.gethostname()

# Document ID patterns to fix
DOCUMENT_ID_FIXES = {
    "iasb-ias": "iasb_ias",
    "iasb-ifrs": "iasb_ifrs",
    "iasb-ifric": "iasb_ifric",
    "iasb-sic": "iasb_sic"
}

# ==============================================================================
# NAS Helper Functions
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
            print("❌ Failed to connect to NAS")
            return None
        return conn
    except Exception as e:
        print(f"❌ Exception creating NAS connection: {e}")
        return None

def read_from_nas(share_name, nas_path_relative):
    """Reads content (as bytes) from a file path on the NAS."""
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
        print(f"❌ Error reading from NAS: {e}")
        return None
    finally:
        if file_obj:
            try:
                file_obj.close()
            except:
                pass
        if conn:
            conn.close()

def write_to_nas(share_name, nas_path_relative, content_bytes):
    """Writes bytes to a file path on the NAS."""
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
                    try:
                        conn.createDirectory(share_name, current_path)
                    except:
                        pass  # Directory might already exist
        
        file_obj = io.BytesIO(content_bytes)
        bytes_written = conn.storeFile(share_name, nas_path_relative, file_obj)
        
        if bytes_written == 0 and len(content_bytes) > 0:
            print(f"❌ No bytes written to {nas_path_relative}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Error writing to NAS: {e}")
        return False
    finally:
        if conn:
            conn.close()

# ==============================================================================
# Main Processing
# ==============================================================================

def fix_document_ids():
    """Main function to fix document IDs in master CSV."""
    
    print("="*60)
    print("🔧 Document ID Fix Script")
    print("="*60)
    
    # Step 1: Load master CSV from NAS
    print(f"\n📥 Loading master CSV from NAS: {MASTER_CSV_PATH}")
    csv_bytes = read_from_nas(NAS_PARAMS["share"], MASTER_CSV_PATH)
    
    if csv_bytes is None:
        print("❌ Failed to read master CSV from NAS")
        return False
    
    csv_content = csv_bytes.decode('utf-8')
    print(f"✅ Loaded master CSV ({len(csv_bytes):,} bytes)")
    
    # Step 2: Parse CSV and fix document_ids
    print("\n🔍 Processing CSV to fix document IDs...")
    
    # Read CSV into memory
    csv_reader = csv.DictReader(io.StringIO(csv_content))
    rows = list(csv_reader)
    fieldnames = csv_reader.fieldnames
    
    print(f"   Total rows: {len(rows)}")
    
    # Track changes
    changes_by_pattern = {pattern: 0 for pattern in DOCUMENT_ID_FIXES.keys()}
    
    # Fix document_ids
    for row in rows:
        if 'document_id' in row:
            original_id = row['document_id']
            # Check each pattern
            for old_pattern, new_pattern in DOCUMENT_ID_FIXES.items():
                if original_id and original_id.startswith(old_pattern):
                    # Replace the pattern
                    new_id = original_id.replace(old_pattern, new_pattern, 1)
                    row['document_id'] = new_id
                    changes_by_pattern[old_pattern] += 1
                    break
    
    # Report changes
    print("\n📊 Changes made:")
    total_changes = 0
    for pattern, count in changes_by_pattern.items():
        if count > 0:
            print(f"   {pattern} -> {DOCUMENT_ID_FIXES[pattern]}: {count} rows")
            total_changes += count
    
    if total_changes == 0:
        print("   No changes needed - all document IDs are already correct")
    else:
        print(f"   Total rows modified: {total_changes}")
    
    # Step 3: Write updated CSV back to memory
    print("\n💾 Preparing updated CSV...")
    output_buffer = io.StringIO()
    csv_writer = csv.DictWriter(output_buffer, fieldnames=fieldnames)
    csv_writer.writeheader()
    csv_writer.writerows(rows)
    
    updated_csv_content = output_buffer.getvalue()
    updated_csv_bytes = updated_csv_content.encode('utf-8')
    
    # Step 4: Save updated master CSV
    print(f"\n💾 Saving updated master CSV to NAS: {MASTER_CSV_PATH}")
    if write_to_nas(NAS_PARAMS["share"], MASTER_CSV_PATH, updated_csv_bytes):
        print(f"✅ Successfully saved updated master CSV")
    else:
        print("❌ Failed to save updated master CSV")
        return False
    
    # Step 5: Create timestamped deployment copy
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    deployment_filename = f"{DEPLOYMENT_PREFIX}_{timestamp}.csv"
    deployment_path = os.path.join(DEPLOYMENT_FOLDER_PATH, deployment_filename).replace("\\", "/")
    
    print(f"\n🚀 Creating deployment copy: {deployment_filename}")
    if write_to_nas(NAS_PARAMS["share"], deployment_path, updated_csv_bytes):
        print(f"✅ Deployment file created successfully")
        print(f"   Location: {deployment_path}")
    else:
        print("❌ Failed to create deployment copy")
        return False
    
    # Step 6: Create backup of original (just in case)
    backup_filename = f"master_database_before_fix_{timestamp}.csv"
    backup_path = os.path.join("semantic_search/pipeline_output/stage5/backups", backup_filename).replace("\\", "/")
    
    print(f"\n📦 Creating backup of original: {backup_filename}")
    if write_to_nas(NAS_PARAMS["share"], backup_path, csv_bytes):  # Original content
        print(f"✅ Backup saved successfully")
    else:
        print("⚠️ Failed to save backup (continuing anyway)")
    
    # Final summary
    print("\n" + "="*60)
    print("✅ Document ID Fix Complete!")
    print(f"   Total rows processed: {len(rows)}")
    print(f"   Total rows fixed: {total_changes}")
    print(f"   Master CSV updated: {MASTER_CSV_PATH}")
    print(f"   Deployment copy: {deployment_filename}")
    print("\n📌 Next steps:")
    print("   1. Run Stage 6 to upload the corrected data to PostgreSQL")
    print("   2. Stage 6 will clear the table and upload all corrected records")
    print("="*60)
    
    return True

if __name__ == "__main__":
    # Check if NAS configuration is set
    if "your_nas_ip" in NAS_PARAMS["ip"]:
        print("❌ Please update NAS configuration in this script before running")
        print("   Edit the NAS_PARAMS dictionary with your actual NAS credentials")
    else:
        success = fix_document_ids()
        exit(0 if success else 1)