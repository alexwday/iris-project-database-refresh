# Catalog Search - Vision Pipeline

This directory contains the Python scripts for the "Catalog Search - Vision" automated data processing pipeline.

## Overview

This pipeline synchronizes **PDF infographic documents** from a Network Attached Storage (NAS) source with a PostgreSQL database. It is specifically designed to handle visually rich documents where layout, charts, diagrams, and images are crucial for understanding content.

It processes these PDFs by:
1.  Converting each page into an image.
2.  Performing a multi-pass analysis of each page image using a **local Vision Language Model (VLM)** to extract different layers of information (text, data visuals, layout, diagrams, symbols, metadata).
3.  Synthesizing the VLM outputs for all pages into a single comprehensive **Markdown document** using a Generative Pre-trained Transformer (GPT) model.
4.  Generating structured `usage` and `description` summaries from the synthesized Markdown using GPT (with tool calling).
5.  Updating corresponding catalog and content tables in the PostgreSQL database.

The generated summaries (`usage` and `description`) are optimized for use by the Iris project's agentic retrieval system, allowing an AI agent to select relevant documents based on the `usage` field and retrieve the full synthesized Markdown content.

The pipeline operates in five distinct stages, executed sequentially. It uses intermediate JSON files stored on the NAS within a directory specific to the `DOCUMENT_SOURCE` being processed (e.g., `path/to/your/output_folder/internal_cheatsheets/`) to pass data between stages and enable checkpointing/resumption.

## Workflow Stages

### Stage 1: Extract & Compare (`stage1_extract_postgres.py`)

*   **(Identical to Catalog Search - Small)**
*   Connects to PostgreSQL, queries `apg_catalog` for the `DOCUMENT_SOURCE` (e.g., 'internal_cheatsheets').
*   Connects to NAS, lists files in the source directory (e.g., `base_input_folder/internal_cheatsheets`).
*   Compares DB and NAS lists (filename, mod time) to find new/updated PDF files.
*   Outputs: `1A_catalog_in_postgres.json`, `1B_files_in_nas.json`, `1C_nas_files_to_process.json`, `1D_postgres_files_to_delete.json`.
*   Sets flags (`_SKIP_SUBSEQUENT_STAGES.flag`, `_FULL_REFRESH.flag`) as needed.

### Stage 2: Process Infographics (`stage2_process_infographics.py`)

*   **(Vision-Specific Implementation)**
1.  **Check Skip Flag:** Skips if `_SKIP_SUBSEQUENT_STAGES.flag` exists in the source output directory (e.g., `output_folder/internal_cheatsheets/`).
2.  **Load Input:** Reads `1C_nas_files_to_process.json`.
3.  **Process PDFs:** For each PDF file in the list:
    *   Downloads the PDF from NAS to a local temporary directory.
    *   Uses **PyMuPDF** (`fitz`) to iterate through PDF pages.
    *   For each page:
        *   Converts the page to a **JPEG image**.
        *   Performs **multi-pass analysis** using the configured local **Vision Language Model (VLM)** API. Each pass uses a specific CO-STAR formatted prompt (via XML tags) targeting different aspects:
            *   Pass 1: Holistic Impression & Core Message
            *   Pass 2: Verbatim Text Extraction
            *   Pass 3: Data Visualization & Numerical Extraction
            *   Pass 4: Layout, Flow & Visual Structure Analysis
            *   Pass 5: Diagram & Symbolic Visual Analysis
            *   Pass 6: Contextual & Attribution Information
        *   Collects the textual responses from all passes for the page.
    *   Cleans up temporary page images and the downloaded PDF.
4.  **Output:** Saves the structured results (containing responses from all passes for all pages) for each processed PDF to a dedicated subfolder on the NAS (`2A_vision_outputs/`) as a JSON file (e.g., `{base_filename}.json`).

### Stage 3: Generate Markdown and Summaries (`stage3_generate_markdown_and_summaries.py`)

*   **(Vision-Specific Implementation)**
1.  **Check Skip Flag:** Skips if `_SKIP_SUBSEQUENT_STAGES.flag` exists.
2.  **Setup:** Downloads CA bundle (from `NAS_BASE_OUTPUT_FOLDER`), sets environment variables, handles OAuth for GPT API.
3.  **Check Refresh Flag:** Deletes existing Stage 3 outputs (`3A_...`, `3B_...`, potentially `3C_...`) if `_FULL_REFRESH.flag` exists.
4.  **Load Metadata & Checkpoints:** Loads `1C...json` for metadata lookup. Loads existing `3A...json` to identify already processed files (based on `processed_md_path` stem) if not in full refresh.
5.  **Find JSON Files:** Locates all `.json` files in the `2A_vision_outputs` directory.
6.  **Process JSON Files:** For each `.json` file (skipping if already processed):
    *   **Load Vision Data:** Reads the structured multi-pass, multi-page vision analysis results.
    *   **Initialize GPT Client:** Sets up the `openai` client (using `AzureOpenAI` class) with the custom endpoint and OAuth token.
    *   **Synthesize Markdown:**
        *   Iterates through the pages in the JSON data.
        *   For each page: Combines the results from all vision passes into a structured input prompt for GPT.
        *   Calls the **Markdown Synthesis GPT model** (`MARKDOWN_SYNTHESIS_MODEL`) to generate a comprehensive Markdown representation of that single page.
        *   Collects the Markdown for all pages.
    *   **Combine & Save Markdown:** Joins the per-page Markdown strings (with separators) into one final Markdown document. Saves this to a new NAS directory (`3C_generated_markdown/{base_filename}.md`).
    *   **Generate Summaries:**
        *   Takes the *entire* synthesized Markdown content.
        *   Calls the **Summarization GPT model** (`SUMMARIZATION_MODEL`) using a detailed system prompt and **tool calling**. Forces the model to call the `generate_catalog_fields` tool, returning `usage` and `description` fields.
    *   **Combine Data & Save Checkpoints:** Retrieves original file metadata. Creates `catalog_entry` (with generated summaries, metadata, and path to the *generated* MD file) and `content_entry` (with the full *generated* Markdown content). Appends these to lists and saves both updated lists back to `3A_catalog_entries.json` and `3B_content_entries.json` on the NAS *after processing each file*.

### Stage 4: Update PostgreSQL (`stage4_update_postgres.py`)

*   **(Identical Logic to Catalog Search - Small, uses configured `DOCUMENT_SOURCE`)**
*   Checks skip flag.
*   Loads `1D...json` (deletions), `3A...json` (catalog entries), `3B...json` (content entries) from the source output directory.
*   Connects to DB.
*   Performs validation counts.
*   Executes `DELETE` statements based on `1D...json`.
*   Uses `psycopg2.extras.execute_values` for bulk `INSERT` into `apg_catalog` from `3A...json`.
*   Uses `psycopg2.extras.execute_values` for bulk `INSERT` into `apg_content` from `3B...json`.
*   Performs final validation counts.

### Stage 5: Archive Results (`stage5_archive_results.py`)

*   **(Identical Logic to Catalog Search - Small, uses configured `DOCUMENT_SOURCE`)**
*   Checks skip flag.
*   Constructs source output path (e.g., `.../output_folder/internal_cheatsheets`) and archive path (`.../output_folder/_archive`).
*   Renames the source output directory with a timestamp (e.g., `internal_cheatsheets_YYYYMMDD_HHMMSS`).
*   Moves the renamed directory into the archive directory using `smbclient.rename()`.

## Configuration

Key configuration parameters are currently hardcoded within each script. **Externalizing these to a configuration file (e.g., `.env`, `config.py`) or environment variables is highly recommended for production use.**

**Required Configuration Includes:**

*   **NAS:** `NAS_BASE_INPUT_FOLDER`, `NAS_BASE_OUTPUT_FOLDER`, `SMB_USER`, `SMB_PASSWORD`.
*   **Database:** PostgreSQL connection details (within Stage 1 and 4).
*   **Pipeline:** `DOCUMENT_SOURCE` (e.g., 'internal_cheatsheets', must be consistent across stages).
*   **Vision Model:** `VISION_API_URL`, `VISION_MODEL_NAME`, `MAX_TOKENS` (in Stage 2).
*   **GPT API:**
    *   OAuth: `TENANT_ID`, `CLIENT_ID`, `CLIENT_SECRET`, `SCOPE` (in Stage 3).
    *   Endpoint: `AZURE_OPENAI_ENDPOINT`, `API_VERSION` (in Stage 3).
    *   Models: `MARKDOWN_SYNTHESIS_MODEL`, `SUMMARIZATION_MODEL` (in Stage 3).
*   **CA Bundle:** `CA_BUNDLE_NAS_PATH` (in Stage 3).

## Dependencies

This pipeline relies on the following major Python libraries:

*   `psycopg2`: For PostgreSQL interaction.
*   `pandas`: Used in Stage 1 for data handling.
*   `smbclient`: For interacting with the NAS (SMB protocol).
*   `PyMuPDF` (package `pymupdf`): Used in Stage 2 for reading and converting PDF pages.
*   `Pillow`: Used implicitly by PyMuPDF or potentially for image handling (good to have).
*   `requests`: Used in Stage 2 (Vision API) and Stage 3 (OAuth).
*   `openai`: Used in Stage 3 for interacting with the Azure OpenAI API.
*   `backoff`: Used in Stage 3 for retry logic on API calls.

Ensure these are installed in your Python environment (e.g., via `pip install -r requirements.txt`).

## Execution

The scripts are designed to be run in sequence (Stage 1 -> Stage 5) for a given `DOCUMENT_SOURCE` (e.g., 'internal_cheatsheets'). The use of the `_SKIP_SUBSEQUENT_STAGES.flag` allows subsequent stages to exit cleanly if Stage 1 determines there are no new or updated files to process. The checkpointing in Stage 3 allows it to be resumed if interrupted.
