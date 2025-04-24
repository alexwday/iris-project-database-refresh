import psycopg2
import pandas as pd
import sys
import os
import smbclient # For NAS connection
from datetime import datetime, timezone

# --- Configuration ---
# Database Configuration
DB_PARAMS = {
    "host": "localhost",
    "port": "5432",
    "dbname": "maven-finance",
    "user": "your_username",  # Replace with your DB username
    "password": "your_password"   # Replace with your DB password
}
# NAS Configuration
NAS_PARAMS = {
    "ip": "your_nas_ip",          # Replace with NAS IP address
    "share": "your_share_name",   # Replace with NAS share name
    # "path" is removed, will be constructed dynamically
    "user": "your_nas_user",      # Replace with NAS username
    "password": "your_nas_password" # Replace with NAS password
}
NAS_BASE_INPUT_PATH = "path/to/your/base_input_folder" # Base path on the share containing document source folders
NAS_OUTPUT_FOLDER_PATH = "path/to/your/output_folder" # Base path on the share for outputs

# Processing Configuration
DOCUMENT_SOURCE = 'internal_esg' # Define the document source to process (used for DB query and path construction)
DB_TABLE_NAME = 'apg_catalog'    # Query the catalog table

# --- Helper Functions ---
def write_json_to_nas(smb_path, data_string):
    """Writes a string (JSON) to a file path on the NAS using smbclient."""
    try:
        with smbclient.open_file(smb_path, mode='w', encoding='utf-8') as f:
            f.write(data_string)
        print(f"Successfully wrote to NAS path: {smb_path}")
        return True
    except smbclient.SambaClientError as e:
        print(f"SMB Error writing to '{smb_path}': {e}")
        return False
    except Exception as e:
        print(f"Unexpected error writing to NAS '{smb_path}': {e}")
        return False

def get_nas_files(nas_ip, share_name, base_folder_path, username, password):
    """Lists files recursively from an SMB share."""
    files_list = []
    smb_base_path = f"//{nas_ip}/{share_name}/{base_folder_path}"
    
    try:
        smbclient.ClientConfig(username=username, password=password)
        print(f"Attempting to connect to NAS: {smb_base_path}")
    except Exception as e:
        print(f"Error setting SMB client config: {e}")
        pass 

    try:
        if not smbclient.path.exists(smb_base_path):
             print(f"Error: Base NAS path does not exist: {smb_base_path}")
             return None 

        for dirpath, dirnames, filenames in smbclient.walk(smb_base_path):
            relative_dirpath = os.path.relpath(dirpath, smb_base_path)
            relative_dirpath = relative_dirpath.replace('\\', '/').strip('/')
            if relative_dirpath == '.':
                 relative_dirpath = '' 

            for filename in filenames:
                if filename == '.DS_Store' or filename.startswith('~$'):
                    print(f"Skipping temporary/system file: {filename}")
                    continue 

                full_smb_path = os.path.join(dirpath, filename).replace('\\', '/')
                try:
                    stat_info = smbclient.stat(full_smb_path)
                    last_modified_dt = datetime.fromtimestamp(stat_info.st_mtime, tz=timezone.utc)
                    relative_file_path = os.path.join(relative_dirpath, filename).replace('\\', '/')
                    
                    files_list.append({
                        'file_name': filename,
                        'file_path': relative_file_path.strip('/'), 
                        'file_size': stat_info.st_size,
                        'date_last_modified': last_modified_dt 
                    })
                except Exception as e:
                    print(f"Warning: Could not stat file '{full_smb_path}': {e}")
        print(f"Successfully listed {len(files_list)} files from NAS.")
        return files_list
    except smbclient.SambaClientError as e:
        print(f"SMB Error listing files from '{smb_base_path}': {e}")
        return None 
    except Exception as e:
        print(f"Unexpected error listing NAS files: {e}")
        return None 


# --- Main Execution Logic ---
if __name__ == "__main__":

    # Construct NAS output paths using the base path and document source
    nas_base_output_smb_path = f"//{NAS_PARAMS['ip']}/{NAS_PARAMS['share']}/{NAS_OUTPUT_FOLDER_PATH}" # Reverted variable name
    nas_output_dir_smb_path = os.path.join(nas_base_output_smb_path, DOCUMENT_SOURCE).replace('\\', '/') # Final output dir uses DOCUMENT_SOURCE
    
    db_output_smb_file = os.path.join(nas_output_dir_smb_path, '1A_catalog_in_postgres.json').replace('\\', '/')
    nas_output_smb_file = os.path.join(nas_output_dir_smb_path, '1B_files_in_nas.json').replace('\\', '/')
    process_output_smb_file = os.path.join(nas_output_dir_smb_path, '1C_nas_files_to_process.json').replace('\\', '/')
    delete_output_smb_file = os.path.join(nas_output_dir_smb_path, '1D_postgres_files_to_delete.json').replace('\\', '/')

    print(f"--- Running Stage 1: Extract & Compare ---")
    print(f"Source: {DOCUMENT_SOURCE}, DB Table: {DB_TABLE_NAME}")
    print(f"NAS Output Directory: {nas_output_dir_smb_path}")

    # Ensure NAS output directory exists using smbclient
    try:
        smbclient.ClientConfig(username=NAS_PARAMS["user"], password=NAS_PARAMS["password"])
        if not smbclient.path.exists(nas_output_dir_smb_path):
            smbclient.makedirs(nas_output_dir_smb_path, exist_ok=True) 
            print(f"Created NAS output directory: '{nas_output_dir_smb_path}'")
        else:
            print(f"NAS output directory already exists: '{nas_output_dir_smb_path}'")
    except smbclient.SambaClientError as e:
        print(f"SMB Error creating directory '{nas_output_dir_smb_path}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error creating NAS directory '{nas_output_dir_smb_path}': {e}")
        sys.exit(1)

    conn = None
    db_df = pd.DataFrame()
    nas_df = pd.DataFrame()

    # 1. Get Data from PostgreSQL Catalog
    try:
        print("Connecting to database...")
        conn = psycopg2.connect(**DB_PARAMS)
        print("Connection successful.")

        query = f"""
            SELECT id, file_name, file_path, date_last_modified, file_size 
            FROM {DB_TABLE_NAME} 
            WHERE document_source = %s;
        """
        print(f"Executing query on {DB_TABLE_NAME}...")
        
        db_df = pd.read_sql_query(query, conn, params=(DOCUMENT_SOURCE,))
        if 'date_last_modified' in db_df.columns and not db_df['date_last_modified'].isnull().all():
            db_df['date_last_modified'] = pd.to_datetime(db_df['date_last_modified'])
            if db_df['date_last_modified'].dt.tz is None:
                print("Localizing naive DB timestamps to UTC...")
                db_df['date_last_modified'] = db_df['date_last_modified'].dt.tz_localize('UTC')
            else:
                print("Converting timezone-aware DB timestamps to UTC...")
                db_df['date_last_modified'] = db_df['date_last_modified'].dt.tz_convert('UTC')
        print(f"Query successful. Found {len(db_df)} records in DB catalog.")

        print(f"Saving DB catalog data to NAS: '{db_output_smb_file}'...")
        db_json_string = db_df.to_json(orient='records', indent=4, date_format='iso')
        if not write_json_to_nas(db_output_smb_file, db_json_string):
             sys.exit(1) 

    except psycopg2.Error as db_err:
        print(f"Database error: {db_err}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during DB operations: {e}")
        sys.exit(1)
    finally:
        if conn is not None:
            conn.close()
            print("Database connection closed.")

    # 2. Get File List from NAS
    # Construct the dynamic input path
    nas_input_path = os.path.join(NAS_BASE_INPUT_PATH, DOCUMENT_SOURCE).replace('\\', '/')
    print(f"Listing files from NAS source path: //{NAS_PARAMS['ip']}/{NAS_PARAMS['share']}/{nas_input_path}")
    nas_files_list = get_nas_files(
        NAS_PARAMS["ip"],
        NAS_PARAMS["share"],
        nas_input_path, # Use dynamically constructed path
        NAS_PARAMS["user"],
        NAS_PARAMS["password"]
    )

    if nas_files_list is None:
        print("Failed to retrieve file list from NAS. Exiting.")
        sys.exit(1)
        
    nas_df = pd.DataFrame(nas_files_list)
    if 'date_last_modified' in nas_df.columns:
        nas_df['date_last_modified'] = pd.to_datetime(nas_df['date_last_modified']).dt.tz_convert('UTC')

    print(f"Saving NAS file list data to NAS: '{nas_output_smb_file}'...")
    nas_json_string = nas_df.to_json(orient='records', indent=4, date_format='iso')
    if not write_json_to_nas(nas_output_smb_file, nas_json_string):
        sys.exit(1) 

    # 3. Compare DB Catalog and NAS File List
    print("Comparing database catalog and NAS file list using 'file_name'...")
    if db_df.empty and nas_df.empty:
        print("Both DB catalog and NAS list are empty. No comparison needed.")
        files_to_process = pd.DataFrame(columns=['file_name', 'file_path', 'file_size', 'date_last_modified', 'reason'])
        files_to_delete = pd.DataFrame(columns=['id', 'file_name', 'file_path']) 
    elif nas_df.empty:
        print("NAS list is empty. No files to process.")
        files_to_process = pd.DataFrame(columns=['file_name', 'file_path', 'file_size', 'date_last_modified', 'reason'])
        files_to_delete = pd.DataFrame(columns=['id', 'file_name', 'file_path'])
    elif db_df.empty:
        print("DB catalog is empty. All NAS files are new.")
        files_to_process = nas_df.copy()
        files_to_process['reason'] = 'new'
        files_to_delete = pd.DataFrame(columns=['id', 'file_name', 'file_path'])
    else:
        # Ensure file_name columns exist and are strings for merging
        if 'file_name' not in nas_df.columns:
            print("Error: 'file_name' column missing in NAS DataFrame.")
            sys.exit(1)
        if 'file_name' not in db_df.columns:
            print("Error: 'file_name' column missing in DB DataFrame.")
            sys.exit(1)
            
        nas_df['file_name'] = nas_df['file_name'].astype(str)
        db_df['file_name'] = db_df['file_name'].astype(str)
        
        # Using exact filename for merging
        merge_on_key = 'file_name' 

        print(f"Performing merge operation on key: '{merge_on_key}'...")
        # Merge the two dataframes based on the filename
        comparison_df = pd.merge(
            nas_df,
            db_df,
            on=merge_on_key, 
            how='outer',
            suffixes=('_nas', '_db'),
            indicator=True 
        )

        # Identify new files (only on NAS)
        new_files = comparison_df[comparison_df['_merge'] == 'left_only'][['file_name', 'file_path_nas', 'file_size_nas', 'date_last_modified_nas']].copy()
        new_files.rename(columns={'file_path_nas': 'file_path', 'file_size_nas': 'file_size', 'date_last_modified_nas': 'date_last_modified'}, inplace=True)
        new_files['reason'] = 'new'

        # Identify files present in both, check for updates
        both_files = comparison_df[comparison_df['_merge'] == 'both'].copy()
        updated_mask = both_files['date_last_modified_nas'] > both_files['date_last_modified_db']
        
        # Get NAS details for updated files
        updated_files_nas = both_files.loc[updated_mask, ['file_name', 'file_path_nas', 'file_size_nas', 'date_last_modified_nas']].copy()
        updated_files_nas.rename(columns={'file_path_nas': 'file_path', 'file_size_nas': 'file_size', 'date_last_modified_nas': 'date_last_modified'}, inplace=True)
        updated_files_nas['reason'] = 'updated'
        
        # Identify files to delete from DB (those that were updated)
        files_to_delete = both_files.loc[updated_mask, ['id', 'file_name', 'file_path_db']].copy()
        files_to_delete.rename(columns={'file_path_db': 'file_path'}, inplace=True)

        # Combine new and updated files for processing
        files_to_process = pd.concat([new_files, updated_files_nas], ignore_index=True)

    print(f"Comparison complete: {len(files_to_process)} NAS files to process, {len(files_to_delete)} existing DB records to delete.")

    # 4. Save Comparison Results to NAS
    print(f"Saving files to process list to NAS: '{process_output_smb_file}'...")
    process_json_string = files_to_process.to_json(orient='records', indent=4, date_format='iso')
    if not write_json_to_nas(process_output_smb_file, process_json_string):
        sys.exit(1)

    print(f"Saving files to delete list to NAS: '{delete_output_smb_file}'...")
    if 'id' in files_to_delete.columns:
         files_to_delete['id'] = files_to_delete['id'].astype('Int64') 
    delete_json_string = files_to_delete.to_json(orient='records', indent=4)
    if not write_json_to_nas(delete_output_smb_file, delete_json_string):
        sys.exit(1)
            
    print(f"--- Stage 1 Completed ---")
