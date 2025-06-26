# -*- coding: utf-8 -*-
"""
Stage 7: Export Textbook Database to CSV

Purpose:
Standalone script to download all records from iris_textbook_chunks table,
ensure they match the iris_textbook_database schema, and export to CSV.
Postgres-generated fields (id, created_at) are left blank for the export.

Input:
- Database connection to iris_textbook_chunks table
- Target schema for iris_textbook_database

Output:
- CSV file named: iris_textbook_database_YYYYMMDD_HHMMSS.csv
- Console output showing first few rows for verification
"""

import os
import json
import logging
import pandas as pd
import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime
from pathlib import Path
import numpy as np

# Try to import pgvector if available
try:
    from pgvector.psycopg2 import register_vector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False
    logging.warning("pgvector not installed. Vector handling may be limited.")

# ==============================================================================
# Configuration
# ==============================================================================

# --- Database Configuration ---
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "ey_guidance")
DB_USER = os.environ.get("DB_USER", "user")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "password")

# --- Source and Target ---
SOURCE_TABLE = "iris_textbook_chunks"
TARGET_TABLE = "iris_textbook_database"  # For naming the output file
OUTPUT_DIR = "pipeline_output/stage7"
LOG_DIR = "pipeline_output/logs"

# --- Display Configuration ---
PREVIEW_ROWS = 5  # Number of rows to display after export
PANDAS_DISPLAY_MAX_COLUMNS = None  # Show all columns
PANDAS_DISPLAY_WIDTH = None  # Don't wrap DataFrame display

# --- Logging Setup ---
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
log_file = Path(LOG_DIR) / 'stage7_export_database.log'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Configure pandas display options
pd.set_option('display.max_columns', PANDAS_DISPLAY_MAX_COLUMNS)
pd.set_option('display.width', PANDAS_DISPLAY_WIDTH)
pd.set_option('display.max_colwidth', 50)  # Truncate long text fields

# ==============================================================================
# Database Functions
# ==============================================================================

def get_db_connection():
    """Establishes and returns a PostgreSQL database connection."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            connect_timeout=30,  # 30 second connection timeout
            options='-c statement_timeout=600000'  # 10 minute statement timeout
        )
        logging.info(f"Database connection established to {DB_NAME} on {DB_HOST}:{DB_PORT}")
        
        # Set connection to autocommit mode to avoid transaction issues
        conn.set_session(autocommit=True)
        
        # Register pgvector type if available
        if PGVECTOR_AVAILABLE:
            register_vector(conn)
            logging.info("pgvector type registered")
        
        return conn
    except psycopg2.Error as e:
        logging.error(f"Database connection error: {e}")
        return None

def fetch_all_records(conn, table_name):
    """Fetches all records from the specified table."""
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            # Get column information first
            cur.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = %s 
                ORDER BY ordinal_position;
            """, (table_name,))
            columns_info = cur.fetchall()
            logging.info(f"Found {len(columns_info)} columns in {table_name}")
            
            # Get record count first
            cur.execute(f"SELECT COUNT(*) FROM {table_name};")
            record_count = cur.fetchone()[0]
            logging.info(f"Table contains {record_count} records")
            
            # Fetch all data with server-side cursor for large datasets
            logging.info("Starting data fetch... This may take a while for large datasets.")
            
            # Use a named cursor for server-side processing
            with conn.cursor(name='fetch_all_cursor', cursor_factory=DictCursor) as named_cur:
                named_cur.itersize = 10000  # Fetch 10k rows at a time
                named_cur.execute(f"SELECT * FROM {table_name} ORDER BY sequence_number;")
                
                records = []
                batch_num = 0
                while True:
                    batch = named_cur.fetchmany(10000)
                    if not batch:
                        break
                    records.extend(batch)
                    batch_num += 1
                    logging.info(f"Fetched batch {batch_num} ({len(records)} records so far)")
                
            logging.info(f"Successfully fetched all {len(records)} records from {table_name}")
            
            return records, columns_info
    except psycopg2.Error as e:
        logging.error(f"Database error fetching data from {table_name}: {e}")
        return None, None
    except Exception as e:
        logging.error(f"Unexpected error fetching data: {e}")
        return None, None

# ==============================================================================
# Schema Mapping Functions
# ==============================================================================

def map_to_target_schema(records):
    """
    Maps records from iris_textbook_chunks to iris_textbook_database schema.
    Returns a list of dictionaries with the target schema fields.
    """
    mapped_records = []
    
    for record in records:
        # Convert DictRow to regular dict for easier handling
        record_dict = dict(record)
        
        # Create mapped record with target schema
        mapped_record = {
            # System fields (leave blank for Postgres to generate)
            'id': None,  # Will be generated by SERIAL PRIMARY KEY
            'created_at': None,  # Will be generated by DEFAULT NOW()
            
            # Structural positioning fields
            'document_id': record_dict.get('document_id'),
            'chapter_number': record_dict.get('chapter_number'),
            'section_number': record_dict.get('section_number'),
            'part_number': record_dict.get('part_number'),
            'sequence_number': record_dict.get('sequence_number'),
            
            # Chapter-level metadata
            'chapter_name': record_dict.get('chapter_name'),
            'chapter_tags': record_dict.get('chapter_tags', []),
            'chapter_summary': record_dict.get('chapter_summary'),
            'chapter_token_count': record_dict.get('chapter_token_count'),
            
            # Section-level pagination & importance
            'section_start_page': record_dict.get('section_start_page'),
            'section_end_page': record_dict.get('section_end_page'),
            'section_importance_score': record_dict.get('section_importance_score'),
            'section_token_count': record_dict.get('section_token_count'),
            
            # Section-level metadata
            'section_hierarchy': record_dict.get('section_hierarchy'),
            'section_title': record_dict.get('section_title'),
            'section_standard': record_dict.get('section_standard'),
            'section_standard_codes': record_dict.get('section_standard_codes', []),
            'section_references': record_dict.get('section_references', []),
            
            # Content & embedding
            'content': record_dict.get('content'),
            'embedding': record_dict.get('embedding'),
            'text_search_vector': None  # Will be generated by Postgres
        }
        
        mapped_records.append(mapped_record)
    
    return mapped_records

def convert_to_dataframe(mapped_records):
    """
    Converts mapped records to a pandas DataFrame.
    Handles special data types like arrays and vectors.
    """
    # Convert to DataFrame
    df = pd.DataFrame(mapped_records)
    
    # Handle PostgreSQL array fields - convert Python lists to PostgreSQL array format
    array_fields = ['chapter_tags', 'section_standard_codes', 'section_references']
    for field in array_fields:
        if field in df.columns:
            # Convert lists to PostgreSQL array format: {item1,item2,item3}
            df[field] = df[field].apply(
                lambda x: '{' + ','.join(f'"{item}"' for item in x) + '}' 
                if isinstance(x, list) and x else '{}'
            )
    
    # Handle embedding field - convert to string representation if needed
    if 'embedding' in df.columns:
        # Embeddings are typically stored as lists of floats
        # For CSV export, we'll convert to string representation
        df['embedding'] = df['embedding'].apply(
            lambda x: '[' + ','.join(map(str, x)) + ']' 
            if isinstance(x, (list, np.ndarray)) and len(x) > 0 else None
        )
    
    # Replace None/NaN with empty strings for CSV export
    # Keep None for id and created_at as they should be NULL
    for col in df.columns:
        if col not in ['id', 'created_at', 'text_search_vector']:
            df[col] = df[col].fillna('')
    
    return df

# ==============================================================================
# Main Function
# ==============================================================================

def run_stage7():
    """Main function to execute Stage 7 export process."""
    logging.info("--- Starting Stage 7: Export Textbook Database to CSV ---")
    
    # Connect to database
    conn = get_db_connection()
    if not conn:
        logging.error("Failed to connect to database. Exiting.")
        return False
    
    try:
        # Fetch all records
        records, columns_info = fetch_all_records(conn, SOURCE_TABLE)
        if records is None:
            logging.error("Failed to fetch records. Exiting.")
            return False
        
        # Log source table schema
        logging.info(f"\n--- Source Table Schema ({SOURCE_TABLE}) ---")
        for col in columns_info:
            logging.info(f"  {col['column_name']}: {col['data_type']}")
        
        # Map to target schema
        logging.info("\n--- Mapping to Target Schema ---")
        mapped_records = map_to_target_schema(records)
        logging.info(f"Mapped {len(mapped_records)} records to target schema")
        
        # Convert to DataFrame
        df = convert_to_dataframe(mapped_records)
        logging.info(f"Created DataFrame with shape: {df.shape}")
        
        # Generate filename with timestamp in format: YYYY-MM-DD_HH-MM-SS
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = f"{TARGET_TABLE}_{timestamp}.csv"
        output_path = Path(OUTPUT_DIR) / output_filename
        
        # Export to CSV
        logging.info(f"\n--- Exporting to CSV ---")
        df.to_csv(output_path, index=False, encoding='utf-8')
        logging.info(f"Successfully exported to: {output_path}")
        
        # Display file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logging.info(f"File size: {file_size_mb:.2f} MB")
        
        # Preview first few rows
        logging.info(f"\n--- Preview of First {PREVIEW_ROWS} Rows ---")
        print(f"\nFirst {PREVIEW_ROWS} rows of exported data:")
        print("=" * 80)
        
        # Display basic info
        preview_df = df.head(PREVIEW_ROWS)
        
        # Show selected columns first for readability
        key_columns = ['document_id', 'chapter_number', 'section_number', 
                      'part_number', 'sequence_number', 'chapter_name']
        print("\n1. Key Identifiers:")
        print(preview_df[key_columns].to_string())
        
        print("\n2. Section Info:")
        section_columns = ['section_title', 'section_hierarchy', 'section_importance_score']
        print(preview_df[section_columns].to_string())
        
        print("\n3. Content Preview (first 100 chars):")
        for idx, row in preview_df.iterrows():
            content_preview = str(row['content'])[:100] + "..." if len(str(row['content'])) > 100 else str(row['content'])
            print(f"Row {idx}: {content_preview}")
        
        print("\n4. Array Fields:")
        array_cols = ['chapter_tags', 'section_standard_codes', 'section_references']
        print(preview_df[array_cols].to_string())
        
        print("\n5. System Fields (should be None/empty):")
        system_cols = ['id', 'created_at', 'text_search_vector']
        print(preview_df[system_cols].to_string())
        
        # Summary statistics
        print(f"\n--- Export Summary ---")
        print(f"Total records exported: {len(df)}")
        print(f"Output file: {output_path}")
        print(f"File size: {file_size_mb:.2f} MB")
        
        # Check for any potential issues
        print(f"\n--- Data Validation ---")
        null_content = df['content'].isnull().sum()
        print(f"Records with null content: {null_content}")
        
        null_embedding = df['embedding'].isnull().sum() + (df['embedding'] == '').sum()
        print(f"Records with null/empty embedding: {null_embedding}")
        
        # Check unique documents
        unique_docs = df['document_id'].nunique()
        print(f"Unique document IDs: {unique_docs}")
        print(f"Document IDs: {df['document_id'].unique()}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error during export process: {e}", exc_info=True)
        return False
    
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")

# ==============================================================================
# Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    success = run_stage7()
    if success:
        logging.info("--- Stage 7 Completed Successfully ---")
    else:
        logging.error("--- Stage 7 Failed ---")