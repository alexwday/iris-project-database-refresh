import yaml # Changed from configparser
import psycopg2
import pandas as pd
import sys
import os

def stage1_extract_postgres():
    """
    Connects to a PostgreSQL database, extracts data from the 'apg_content' table
    filtered by document_source='internal_esg', and saves it as a JSON file.
    Database credentials are read from 'stage1_config.yaml'.
    """
    config_file = 'stage1_config.yaml' # Changed from .ini
    output_file = '1A_files_in_postgres.json'
    
    # Check if config file exists
    if not os.path.exists(config_file):
        print(f"Error: Configuration file '{config_file}' not found.")
        sys.exit(1)

    # --- Read Configuration ---
    conn = None  # Initialize conn to None
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            if config is None or 'Database' not in config:
                 raise ValueError("Configuration file is empty or missing 'Database' section.")
            db_params = config['Database']
            # Ensure port is treated as a string if needed by psycopg2, though it often handles int
            if 'port' in db_params and isinstance(db_params['port'], int):
                db_params['port'] = str(db_params['port'])
        print("Successfully read database configuration from YAML.")
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration file '{config_file}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading configuration file '{config_file}': {e}")
        sys.exit(1)

    try:
        # --- Database Connection ---
        print("Connecting to the PostgreSQL database...")
        conn = psycopg2.connect(**db_params)
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
