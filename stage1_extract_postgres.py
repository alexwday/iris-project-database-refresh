# Removed yaml import
import psycopg2
import pandas as pd
import sys
import os # Keep os for path checks if needed later, but not for config

def stage1_extract_postgres():
    """
    Connects to a PostgreSQL database, extracts data from the 'apg_content' table
    filtered by document_source='internal_esg', and saves it as a JSON file.
    Database credentials are hardcoded in this script.
    """
    output_file = '1A_files_in_postgres.json'

    # --- Hardcoded Database Configuration ---
    # TODO: Replace placeholders with actual credentials before running
    # Consider moving these to environment variables or a config file for production
    DB_PARAMS = {
        "host": "localhost",
        "port": "5432",
        "dbname": "maven-finance",
        "user": "your_username",  # Replace with your DB username
        "password": "your_password"   # Replace with your DB password
    }
    print("Using hardcoded database configuration.")

    conn = None  # Initialize conn to None
    try:
        # --- Database Connection ---
        print("Connecting to the PostgreSQL database...")
        conn = psycopg2.connect(**DB_PARAMS) # Use hardcoded params
        print("Database connection successful.")

        # --- Query Execution ---
        query = "SELECT * FROM apg_content WHERE document_source = 'internal_esg';"
        print(f"Executing query: {query}")
        
        # Use pandas to execute query and fetch data
        df = pd.read_sql_query(query, conn)
        print(f"Query executed successfully. Found {len(df)} records.")

        # --- Data Saving ---
        print(f"Saving data to '{output_file}'...")
        # Save as JSON, orient='records' creates a list of JSON objects
        df.to_json(output_file, orient='records', indent=4) 
        print(f"Data successfully saved to '{output_file}'.")

    except psycopg2.Error as db_err:
        print(f"Database error: {db_err}")
        sys.exit(1)
    except pd.errors.DatabaseError as pd_db_err:
        # Catch pandas-specific database errors which might wrap psycopg2 errors
        print(f"Pandas Database error during query execution: {pd_db_err}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
    finally:
        # --- Close Connection ---
        if conn is not None:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    print("--- Starting Stage 1: Extract Postgres Data ---")
    stage1_extract_postgres()
    print("--- Stage 1 Completed ---")
