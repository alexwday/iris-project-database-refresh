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
        # Count rows (excluding header)
        with open(output_path, 'r', encoding='utf-8') as f:
            row_count = sum(1 for line in f) - 1  # Subtract header
        
        logging.info(f"CSV contains {row_count} data rows (expected: {expected_count})")
        
        # Check file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logging.info(f"File size: {file_size_mb:.2f} MB")
        
        # Preview first few rows
        logging.info("\n--- Preview of First 5 Rows ---")
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 5:
                    break
                logging.info(f"\nRow {i+1}:")
                logging.info(f"  document_id: {row.get('document_id', 'N/A')}")
                logging.info(f"  sequence_number: {row.get('sequence_number', 'N/A')}")
                logging.info(f"  chapter_name: {row.get('chapter_name', 'N/A')[:50]}...")
                logging.info(f"  embedding: {'<present>' if row.get('embedding') else '<empty>'}")
                logging.info(f"  id: {row.get('id', 'N/A')} (should be empty)")
                logging.info(f"  created_at: {row.get('created_at', 'N/A')} (should be empty)")
        
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