# Catalog Search - Small Pipeline

This directory contains the Python scripts for the "Catalog Search - Small" automated data processing pipeline.

## Overview

This pipeline synchronizes documents from a Network Attached Storage (NAS) source with a PostgreSQL database. It processes standard documents (like PDFs, DOCX) by extracting their content, generating summaries using AI, and updating corresponding catalog and content tables in the database. The generated summaries (`usage` and `description`) are optimized for use by the Iris project's agentic retrieval system, allowing an AI agent to select relevant documents based on the `usage` field and retrieve full content.

The pipeline operates in five distinct stages, executed sequentially. It uses intermediate JSON files stored on the NAS within a directory specific to the `DOCUMENT_SOURCE` being processed (e.g., `path/to/your/output_folder/internal_esg/`) to pass data between stages and enable checkpointing/resumption.

## Workflow Stages

### Stage 1: Extract & Compare (`stage1_extract_postgres.py`)

1.  **Connect & Query DB:** Connects to the PostgreSQL database (`maven-finance` by default) and queries the catalog table (`apg_catalog`) to get the list of files already known for the specified `DOCUMENT_SOURCE`.
2.  **Connect & List NAS:** Connects to the NAS via SMB and lists all files present in the corresponding source directory (e.g., `//nas_ip/share/path/to/your/base_input_folder/internal_esg`).
3.  **Compare:** Compares the database list and the NAS list based on filenames and modification timestamps (UTC).
    *   Identifies **new files** (on NAS, not in DB).
    *   Identifies **updated files** (on NAS, newer than DB record).
    *   Implicitly identifies potentially deleted files (in DB, not on NAS - no action taken by default).
4.  **Output:** Saves the comparison results to the source-specific output directory on the NAS:
    *   `1A_catalog_in_postgres.json`: Snapshot of DB records for the source.
    *   `1B_files_in_nas.json`: Snapshot of files found on NAS for the source.
    *   `1C_nas_files_to_process.json`: List of new and updated files (with metadata) to be processed by subsequent stages.
    *   `1D_postgres_files_to_delete.json`: List of DB records (identified by `id`, `document_source`, `document_type`, `document_name`) corresponding to *updated* NAS files, which need to be deleted before re-insertion in Stage 4.
5.  **Flags:**
    *   Creates `_SKIP_SUBSEQUENT_STAGES.flag` if `1C_...json` is empty.
    *   Supports a `FULL_REFRESH` mode (boolean flag in script) which skips comparison, marks all NAS files for processing, and all existing DB records for deletion. Creates `_FULL_REFRESH.flag` if enabled.

### Stage 2: Process Documents (`stage2_process_documents.py`)

1.  **Check Skip Flag:** Checks for the `_SKIP_SUBSEQUENT_STAGES.flag` from Stage 1. If present, skips execution.
2.  **Load Input:** Reads the `1C_nas_files_to_process.json` file.
3.  **Process Files:** For each file in the list:
    *   Downloads the file from its NAS path to a local temporary directory.
    *   Uses **Azure Document Intelligence** (`prebuilt-layout` model) to analyze the document content.
    *   **Handles Large PDFs:** If a PDF exceeds a page limit (`PDF_PAGE_LIMIT`, default 2000), it splits the PDF into chunks (`PDF_CHUNK_SIZE`, default 1000 pages), processes each chunk with Document Intelligence, and concatenates the resulting Markdown.
    *   Extracts the content in **Markdown format**.
4.  **Output:** Saves the results for each processed file to a dedicated subfolder within the Stage 2 output directory (`2A_processed_files/`) on the NAS:
    *   `{base_filename}.md`: The extracted Markdown content (combined if chunked).
    *   `{base_filename}.json` (or `{base_filename}_chunk_N.json`): The raw JSON analysis result from Document Intelligence.

### Stage 3: Generate Summaries (`stage3_generate_summaries.py`)

1.  **Check Skip Flag:** Checks for the `_SKIP_SUBSEQUENT_STAGES.flag`. If present, skips execution.
2.  **Setup CA Bundle:** Downloads a specified CA bundle (`rbc-ca-bundle.cer`) from the NAS output root and sets environment variables (`REQUESTS_CA_BUNDLE`, `SSL_CERT_FILE`) to use it for HTTPS requests (primarily for OAuth/GPT calls).
3.  **Check Refresh Flag:** Checks for `_FULL_REFRESH.flag`. If present, deletes existing Stage 3 output files (`3A_...`, `3B_...`) before starting.
4.  **Load Metadata & Checkpoints:**
    *   Loads the original file metadata from `1C_nas_files_to_process.json` into a lookup dictionary.
    *   If not in full refresh mode, loads existing entries from `3A_catalog_entries.json` and `3B_content_entries.json` to allow resuming/checkpointing. Identifies already processed Markdown files.
5.  **Find MD Files:** Locates all `.md` files within the `2A_processed_files` directory on the NAS.
6.  **Process MD Files:** For each `.md` file (skipping if already processed in incremental mode):
    *   **Authenticate:** Obtains an OAuth 2.0 access token using client credentials flow.
    *   **Initialize GPT Client:** Sets up the `openai` client with the custom base URL and the obtained token.
    *   **Read Content:** Reads the Markdown content from the NAS file.
    *   **Call GPT:** Sends the Markdown content to the configured custom GPT model using a detailed system prompt and **tool calling**. The prompt instructs the model to act as a technical writer and generate two fields:
        *   `usage`: A detailed, structured summary for AI agent retrieval assessment (detail level configurable).
        *   `description`: A concise 1-2 sentence summary for human users.
    *   The model is forced to call the `generate_catalog_fields` tool, returning the summaries as arguments.
    *   **Combine Data:** Retrieves the original file metadata using the filename. Creates two dictionary entries:
        *   **Catalog Entry:** Contains source, type, name, generated description/usage, original file metadata (dates, size, path, link), and the path to the processed MD file (`processed_md_path`) for checkpointing.
        *   **Content Entry:** Contains source, type, name, section details (id=0, name=base_filename, summary=usage), the full Markdown content, and a creation timestamp.
7.  **Append & Save:** Appends the new catalog entry to the `catalog_entries` list and the content entry to the `content_entries` list. **Crucially, it saves both updated lists back to their respective JSON files (`3A_...`, `3B_...`) on the NAS *after processing each file*.** This provides checkpointing.

### Stage 4: Update PostgreSQL (`stage4_update_postgres.py`)

1.  **Check Skip Flag:** Checks for the `_SKIP_SUBSEQUENT_STAGES.flag`. If present, skips execution.
2.  **Load Inputs:** Reads the list of records to delete (`1D_postgres_files_to_delete.json`), the catalog entries (`3A_catalog_entries.json`), and the content entries (`3B_content_entries.json`) from the NAS.
3.  **Connect to DB:** Establishes a connection to the PostgreSQL database.
4.  **Validation (Before):** Counts existing records in `apg_catalog` and `apg_content` for the `DOCUMENT_SOURCE`.
5.  **Delete Records:** If the deletion list is not empty:
    *   Iterates through the unique keys (`document_source`, `document_type`, `document_name`) from the list.
    *   Executes `DELETE` statements against both `apg_content` and `apg_catalog` tables for each key.
    *   Commits the transaction.
6.  **Validation (After Deletion):** Counts records again after deletions.
7.  **Insert Records:**
    *   If the catalog entries list is not empty, uses `psycopg2.extras.execute_values` for efficient bulk insertion into `apg_catalog`. Commits.
    *   If the content entries list is not empty, uses `psycopg2.extras.execute_values` for efficient bulk insertion into `apg_content`. Commits.
8.  **Validation (After Insertion):** Counts records a final time and compares against expected counts based on initial counts, deletions, and insertions.

### Stage 5: Archive Results (`stage5_archive_results.py`)

1.  **Define Paths:** Constructs the path to the source-specific output directory on the NAS (e.g., `.../output_folder/internal_esg`) and the target archive directory (e.g., `.../output_folder/_archive`).
2.  **Ensure Archive Dir:** Creates the target archive directory if it doesn't exist.
3.  **Check Source Dir:** Verifies that the source-specific output directory exists and is a directory. Exits cleanly if not found.
4.  **Rename & Move:**
    *   Generates a timestamp (e.g., `20250425_101435`).
    *   Constructs a new name for the source directory using the source name and timestamp (e.g., `internal_esg_20250425_101435`).
    *   Uses `smbclient.rename()` to move the entire source directory into the archive directory with the new timestamped name.

## Configuration

Key configuration parameters (Database credentials, NAS details, Azure DI endpoint/key, OAuth details, GPT endpoint/model, Document Source, Document Type) are currently hardcoded within each script. For production use, consider externalizing these to a configuration file (e.g., `.env`, `config.json`) or environment variables.

## Dependencies

This pipeline relies on the following major Python libraries:

*   `psycopg2`: For PostgreSQL interaction.
*   `pandas`: Used in Stage 1 for data handling and comparison.
*   `smbclient`: For interacting with the NAS (SMB protocol).
*   `pypdf`: Used in Stage 2 for reading and splitting PDF files.
*   `azure-ai-documentintelligence`: For interacting with Azure Document Intelligence.
*   `requests`: Used in Stage 3 for OAuth token requests.
*   `openai`: Used in Stage 3 for interacting with the custom GPT API.

## Execution

The scripts are designed to be run in sequence (Stage 1 -> Stage 5) for a given `DOCUMENT_SOURCE`. The use of the `_SKIP_SUBSEQUENT_STAGES.flag` allows subsequent stages to exit cleanly if Stage 1 determines there are no new or updated files to process. The checkpointing in Stage 3 allows it to be resumed if interrupted.
