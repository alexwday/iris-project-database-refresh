# -*- coding: utf-8 -*-
"""
Stage 7: Export Textbook Database to CSV (Fixed Version)

Purpose:
Export all records from iris_textbook_chunks table exactly as stored in PostgreSQL,
only blanking out auto-generated fields (id, created_at).
The output CSV should be directly importable back into PostgreSQL.

Input:
- Database connection to iris_textbook_chunks table

Output:
- CSV file named: iris_textbook_database_YYYYMMDD_HHMMSS.csv
- Data preserved exactly as stored in PostgreSQL
"""

import os
import csv
import logging
import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime
from pathlib import Path
import io

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

# --- Logging Setup ---
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
log_file = Path(LOG_DIR) / 'stage7_export_database_fixed.log'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

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
            password=DB_PASSWORD
        )
        logging.info(f"Database connection established to {DB_NAME} on {DB_HOST}:{DB_PORT}")
        return conn
    except psycopg2.Error as e:
        logging.error(f"Database connection error: {e}")
        return None

def export_to_csv_with_copy(conn, table_name, output_path):
    """
    Uses PostgreSQL COPY command to export data to CSV.
    This preserves data types exactly as stored in PostgreSQL.
    """
    try:
        with conn.cursor() as cur:
            # First, get the column names in order
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = %s 
                ORDER BY ordinal_position;
            """, (table_name,))
            columns = [row[0] for row in cur.fetchall()]
            logging.info(f"Found {len(columns)} columns in {table_name}")
            
            # Create a query that selects all columns but replaces id and created_at with NULL
            select_parts = []
            for col in columns:
                if col in ['id', 'created_at']:
                    select_parts.append(f"NULL as {col}")
                else:
                    select_parts.append(col)
            
            select_query = f"SELECT {', '.join(select_parts)} FROM {table_name} ORDER BY sequence_number"
            
            # Use COPY TO with CSV format
            copy_query = f"COPY ({select_query}) TO STDOUT WITH (FORMAT CSV, HEADER TRUE, NULL '')"
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                cur.copy_expert(copy_query, f)
            
            # Get row count
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cur.fetchone()[0]
            
            return row_count
            
    except psycopg2.Error as e:
        logging.error(f"Error exporting data: {e}")
        return None

def verify_export(output_path, expected_count):
    """Verify the exported CSV file."""
    try:
        # Count rows properly using CSV reader (handles multi-line fields)
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            row_count = sum(1 for row in reader)
        
        logging.info(f"CSV contains {row_count} data rows (expected: {expected_count})")
        
        # Check file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logging.info(f"File size: {file_size_mb:.2f} MB")
        
        # Preview first few rows
        preview_rows = 5
        print(f"\n--- Preview of First {preview_rows} Rows ---")
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for i, row in enumerate(reader):
                if i >= preview_rows:
                    break
                
                print(f"\n{'='*80}")
                print(f"Row {i+1}:")
                print(f"{'='*80}")
                
                # Display fields in organized groups
                print("\n1. Document & Position Info:")
                print(f"   document_id: {row.get('document_id', 'N/A')}")
                print(f"   chapter_number: {row.get('chapter_number', 'N/A')}")
                print(f"   section_number: {row.get('section_number', 'N/A')}")
                print(f"   part_number: {row.get('part_number', 'N/A')}")
                print(f"   sequence_number: {row.get('sequence_number', 'N/A')}")
                
                print("\n2. Chapter Info:")
                print(f"   chapter_name: {row.get('chapter_name', 'N/A')[:80]}{'...' if len(row.get('chapter_name', '')) > 80 else ''}")
                print(f"   chapter_tags: {row.get('chapter_tags', 'N/A')}")
                print(f"   chapter_summary: {row.get('chapter_summary', 'N/A')[:100]}{'...' if len(row.get('chapter_summary', '')) > 100 else ''}")
                print(f"   chapter_token_count: {row.get('chapter_token_count', 'N/A')}")
                
                print("\n3. Section Info:")
                print(f"   section_title: {row.get('section_title', 'N/A')[:80]}{'...' if len(row.get('section_title', '')) > 80 else ''}")
                print(f"   section_hierarchy: {row.get('section_hierarchy', 'N/A')}")
                print(f"   section_start_page: {row.get('section_start_page', 'N/A')}")
                print(f"   section_end_page: {row.get('section_end_page', 'N/A')}")
                print(f"   section_importance_score: {row.get('section_importance_score', 'N/A')}")
                print(f"   section_token_count: {row.get('section_token_count', 'N/A')}")
                
                print("\n4. Arrays & References:")
                print(f"   section_standard: {row.get('section_standard', 'N/A')}")
                print(f"   section_standard_codes: {row.get('section_standard_codes', 'N/A')}")
                print(f"   section_references: {row.get('section_references', 'N/A')}")
                
                print("\n5. Content & Embedding:")
                content = row.get('content', 'N/A')
                print(f"   content: {content[:150]}{'...' if len(content) > 150 else ''}")
                print(f"   content_length: {len(content)} chars")
                
                embedding = row.get('embedding', '')
                if embedding:
                    # Show just the first few values and total count
                    if embedding.startswith('[') and embedding.endswith(']'):
                        try:
                            # Extract first few numbers
                            values = embedding[1:-1].split(',')[:5]
                            preview = '[' + ','.join(values) + f',... ({len(embedding[1:-1].split(","))} total values)]'
                            print(f"   embedding: {preview}")
                        except:
                            print(f"   embedding: <present, {len(embedding)} chars>")
                    else:
                        print(f"   embedding: <present, {len(embedding)} chars>")
                else:
                    print(f"   embedding: <empty>")
                
                print("\n6. Auto-generated Fields (should be empty):")
                print(f"   id: '{row.get('id', '')}' {'✓ empty' if not row.get('id') else '✗ NOT EMPTY'}")
                print(f"   created_at: '{row.get('created_at', '')}' {'✓ empty' if not row.get('created_at') else '✗ NOT EMPTY'}")
                print(f"   text_search_vector: '{row.get('text_search_vector', '')}' {'✓ empty' if not row.get('text_search_vector') else '✗ NOT EMPTY'}")
        
        print(f"\n{'='*80}")
        print(f"Total rows in CSV: {row_count}")
        print(f"{'='*80}")
        
        return row_count == expected_count
        
    except Exception as e:
        logging.error(f"Error verifying export: {e}")
        return False

# ==============================================================================
# Main Function
# ==============================================================================

def run_stage7_fixed():
    """Main function to execute Stage 7 export process (fixed version)."""
    logging.info("--- Starting Stage 7: Export Textbook Database to CSV (Fixed Version) ---")
    
    # Connect to database
    conn = get_db_connection()
    if not conn:
        logging.error("Failed to connect to database. Exiting.")
        return False
    
    try:
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = f"{TARGET_TABLE}_{timestamp}.csv"
        output_path = Path(OUTPUT_DIR) / output_filename
        
        # Export using COPY command
        logging.info(f"Exporting data from {SOURCE_TABLE} to {output_path}")
        row_count = export_to_csv_with_copy(conn, SOURCE_TABLE, output_path)
        
        if row_count is None:
            logging.error("Export failed.")
            return False
        
        logging.info(f"Successfully exported {row_count} records to {output_path}")
        
        # Verify the export
        if verify_export(output_path, row_count):
            logging.info("Export verification passed.")
        else:
            logging.warning("Export verification detected issues.")
        
        # Summary
        print(f"\n--- Export Summary ---")
        print(f"Total records exported: {row_count}")
        print(f"Output file: {output_path}")
        print(f"File size: {output_path.stat().st_size / (1024 * 1024):.2f} MB")
        print(f"\nThis CSV preserves all data exactly as stored in PostgreSQL:")
        print(f"- Arrays remain in PostgreSQL format")
        print(f"- Embeddings remain as PostgreSQL vectors")
        print(f"- NULL values are preserved")
        print(f"- Only 'id' and 'created_at' are blanked out")
        
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
    success = run_stage7_fixed()
    if success:
        logging.info("--- Stage 7 (Fixed) Completed Successfully ---")
    else:
        logging.error("--- Stage 7 (Fixed) Failed ---")