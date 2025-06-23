#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 6: Vector Search, Refinement & Response Generation

Goal: Perform vector search, refine results using relevance filtering,
importance reranking, token-based section expansion, and sequence-based gap filling,
then format results as "cards" for GPT to generate a cited response.

For use in Jupyter notebook cells.

Requires:
- `psycopg2` or `psycopg2-binary` library.
- `openai` library (`pip install openai`).
- `requests` library (`pip install requests`).
- `pgvector` extension enabled in PostgreSQL.
- Environment variables for OAuth/SSL configured.
"""

import os
import sys
import json
import time
import traceback
import requests # Needed for OAuth
from pathlib import Path
import psycopg2
import psycopg2.extras # For DictCursor
from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from pgvector.psycopg2 import register_vector
import json # For parsing GPT relevance response
import itertools # For grouping in gap filling
from tabulate import tabulate # For better table printing
try:
    import tiktoken
except ImportError:
    tiktoken = None
    print("WARNING: tiktoken not installed. Token counts will be estimated (chars/4). `pip install tiktoken`")


# --- Configuration ---
# --- Search & Refinement Configuration ---
# Source document ID to search within (optional, set to None to search all)
DOCUMENT_ID_TO_SEARCH = "EY_GUIDE_2024_PLACEHOLDER" # Updated placeholder
# Number of results to retrieve initially (before filtering/expansion)
INITIAL_K = 20
# Final number of results to pass to GPT (after all steps) - Set to None to keep all
# FINAL_K = None # REMOVED - No truncation step anymore
# API model for response generation
RESPONSE_MODEL = "gpt-4o"
# API model for summary relevance check (can be faster/cheaper if needed)
RELEVANCE_MODEL = "gpt-3.5-turbo"
# Importance factor for reranking
IMPORTANCE_FACTOR = 0.2
# Section expansion thresholds (token count)
SECTION_EXPANSION_TOP_K_RANK = 5 # Rank threshold for applying the higher token limit
SECTION_EXPANSION_TOP_K_TOKENS = 8000 # Token limit for top K ranks
SECTION_EXPANSION_GENERAL_TOKENS = 4000 # Token limit for ranks > TOP_K_RANK
# Gap filling threshold (sequence number gap)
GAP_FILL_MAX_SEQUENCE_GAP = 8
# Maximum tokens for the response
MAX_RESPONSE_TOKENS = 4000
# Temperature for response generation
RESPONSE_TEMPERATURE = 0.7

# --- Target Table ---
TARGET_TABLE = "guidance_sections" # Name of the table containing chunks

# --- Embedding Configuration ---
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 2000

# --- Database Configuration (Self-contained) --- # TODO: Align with Stage 4
DB_PARAMS = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": os.environ.get("DB_PORT", "5432"),
    "dbname": os.environ.get("DB_NAME", "ey_guidance"),
    "user": os.environ.get("DB_USER", "user"),
    "password": os.environ.get("DB_PASSWORD", "password"), # Be cautious storing passwords directly
}

# --- API Connection Configuration (Copied from Script 10) ---
# Ensure these match your environment/script 10 setup
BASE_URL = os.environ.get("RBC_LLM_ENDPOINT", "https://api.example.com/v1") # Use env var or placeholder
OAUTH_URL = os.environ.get("RBC_OAUTH_URL", "https://api.example.com/oauth/token") # Use env var or placeholder
CLIENT_ID = os.environ.get("RBC_OAUTH_CLIENT_ID", "your_client_id") # Placeholder
CLIENT_SECRET = os.environ.get("RBC_OAUTH_CLIENT_SECRET", "your_client_secret") # Placeholder
SSL_SOURCE_PATH = os.environ.get("RBC_SSL_SOURCE_PATH", "/path/to/your/rbc-ca-bundle.cer") # Placeholder
SSL_LOCAL_PATH = "/tmp/rbc-ca-bundle.cer" # Temporary local path for cert

# --- Internal Constants (Copied/Adapted from Script 10) ---
_SSL_CONFIGURED = False # Flag to avoid redundant SSL setup
_RETRY_ATTEMPTS = 3
_RETRY_DELAY = 5 # seconds

# --- Helper Functions (Tokenizer) ---
_TOKENIZER = None
if tiktoken:
    try:
        # Using cl100k_base as it's common for models like GPT-4, GPT-3.5
        _TOKENIZER = tiktoken.get_encoding("cl100k_base")
        print("Using 'cl100k_base' tokenizer for token counting.")
    except Exception as e:
        print(f"WARNING: Failed tokenizer init: {e}. Estimating tokens.")
        _TOKENIZER = None

def count_tokens(text: str) -> int:
    """Counts tokens using tiktoken or estimates if unavailable/fails."""
    if not text: return 0
    if _TOKENIZER:
        try:
            return len(_TOKENIZER.encode(text))
        except Exception as e:
            # Fallback to estimation on encoding error
            print(f"WARNING: tiktoken encode error: {e}. Estimating tokens for this text.")
            return len(text) // 4 # Estimate
    else:
        # Fallback if tiktoken is not installed
        return len(text) // 4 # Estimate

# --- Helper Functions (Database Connection) ---

def connect_to_db(params):
    """Connects to the PostgreSQL database."""
    conn = None
    try:
        print(f"Connecting to database '{params['dbname']}' on {params['host']}...")
        conn = psycopg2.connect(**params)
        # Register pgvector type handler
        register_vector(conn)
        print("Connection successful and pgvector registered.")
        return conn
    except Exception as e:
        print(f"ERROR: Database connection failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return None

# --- Helper Functions (Copied/Adapted from Script 10 for OpenAI Connection) ---

def _setup_ssl(source_path=SSL_SOURCE_PATH, local_path=SSL_LOCAL_PATH):
    """Copies SSL cert locally and sets environment variables."""
    global _SSL_CONFIGURED
    if _SSL_CONFIGURED:
        return True # Already configured

    print("Setting up SSL certificate...")
    try:
        source = Path(source_path)
        local = Path(local_path)

        if not source.is_file():
            print(f"ERROR: SSL source certificate not found at {source_path}", file=sys.stderr)
            return False

        local.parent.mkdir(parents=True, exist_ok=True)
        with open(source, "rb") as source_file:
            content = source_file.read()
        with open(local, "wb") as dest_file:
            dest_file.write(content)

        os.environ["SSL_CERT_FILE"] = str(local)
        os.environ["REQUESTS_CA_BUNDLE"] = str(local)
        print(f"SSL certificate configured successfully at: {local}")
        _SSL_CONFIGURED = True
        return True
    except Exception as e:
        print(f"ERROR: Error setting up SSL certificate: {e}", file=sys.stderr)
        return False

def _get_oauth_token(oauth_url=OAUTH_URL, client_id=CLIENT_ID, client_secret=CLIENT_SECRET, ssl_verify_path=SSL_LOCAL_PATH):
    """Retrieves OAuth token from the specified endpoint."""
    print("Attempting to get OAuth token...")
    payload = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret
    }
    try:
        # Ensure SSL is set up before making the request
        if not _SSL_CONFIGURED:
             print("ERROR: SSL not configured, cannot get OAuth token.", file=sys.stderr)
             return None

        response = requests.post(
            oauth_url,
            data=payload,
            timeout=30,
            verify=ssl_verify_path # Use the configured local path
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        token_data = response.json()
        oauth_token = token_data.get('access_token')
        if not oauth_token:
            print("ERROR: 'access_token' not found in OAuth response.", file=sys.stderr)
            return None
        print("OAuth token obtained successfully.")
        return oauth_token
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Error getting OAuth token: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"ERROR: Unexpected error during OAuth token retrieval: {e}", file=sys.stderr)
        traceback.print_exc()
        return None


def create_openai_client(base_url=BASE_URL):
    """Sets up SSL, gets OAuth token, and creates the OpenAI client."""
    if not _setup_ssl():
        print("ERROR: Aborting client creation due to SSL setup failure.", file=sys.stderr)
        return None # SSL setup failed

    api_key = _get_oauth_token()
    if not api_key:
        print("ERROR: Aborting client creation due to OAuth token failure.", file=sys.stderr)
        return None # Token retrieval failed

    try:
        # Pass http_client using httpx if needed for custom SSL context,
        # otherwise rely on REQUESTS_CA_BUNDLE env var set in _setup_ssl
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        print("OpenAI client created successfully using OAuth token.")
        return client
    except Exception as e:
        print(f"ERROR: Error creating OpenAI client: {e}", file=sys.stderr)
        traceback.print_exc()
        return None

def generate_query_embedding(client, query: str, model: str, dimensions: int) -> list[float] | None:
    """Generates embedding for the query string using the provided client."""
    if not client:
        print("ERROR: OpenAI client not available for embedding generation.", file=sys.stderr)
        return None
    print(f"Generating embedding for query: '{query}'...")
    last_exception = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            response = client.embeddings.create(
                input=[query], # API expects a list
                model=model,
                dimensions=dimensions
            )
            if response.data and response.data[0].embedding:
                print("Embedding generated successfully.")
                return response.data[0].embedding
            else:
                print(f"ERROR: No embedding data received from API (Attempt {attempt + 1}).", file=sys.stderr)
                last_exception = ValueError("No embedding data in API response")
                time.sleep(_RETRY_DELAY) # Wait before retry

        except APIError as e:
            print(f"WARNING: OpenAI API error during embedding generation (Attempt {attempt + 1}): {e}", file=sys.stderr)
            last_exception = e
            time.sleep(_RETRY_DELAY * (attempt + 1)) # Exponential backoff might be better
        except Exception as e:
            print(f"WARNING: Unexpected error during embedding generation (Attempt {attempt + 1}): {e}", file=sys.stderr)
            last_exception = e
            time.sleep(_RETRY_DELAY)

    print(f"ERROR: Failed to generate embedding after {_RETRY_ATTEMPTS} attempts.", file=sys.stderr)
    if last_exception:
        print(f"Last error: {last_exception}", file=sys.stderr)
    return None


# --- Search Functions ---

def perform_hybrid_search(cursor, query: str, query_embedding: list[float], initial_k: int, doc_id: str | None):
    """
    Performs hybrid search combining vector similarity and keyword search.
    Returns a combined ranked list of results.
    """
    print(f"\n--- Performing Initial Hybrid Search (Retrieving Top {initial_k}) ---") # Updated log message
    results = []

    if query_embedding is None:
        print("ERROR: Cannot perform vector component of hybrid search without embedding.", file=sys.stderr)
        return results

    try:
        # REMOVED: Text search preparation logic.

        # Vector-only search SQL - Simplified
        sql = f"""
            SELECT
                c.*, -- Select all columns from the target table
                1 - (c.embedding <=> %s::vector) AS vector_score -- Calculate vector score directly
            FROM {TARGET_TABLE} c -- Use config variable
            WHERE 1=1
            {" AND c.document_id = %s" if doc_id else ""}
            ORDER BY vector_score DESC -- Order by vector score
            LIMIT %s; -- Use initial_k for the limit
        """

        # Prepare parameters for vector search only
        params = [query_embedding] # Start with embedding
        if doc_id:
            params.append(doc_id) # Add doc_id if provided
        params.append(initial_k) # Add limit

        cursor.execute(sql, params)
        results = cursor.fetchall()
        print(f"Found {len(results)} results via vector search.")

    except Exception as e:
        print(f"ERROR: Vector search failed: {e}", file=sys.stderr)
        traceback.print_exc()

    return results


# --- Reranking & Filtering Functions ---

def filter_by_summary_relevance(client: OpenAI, query: str, results: list[dict], model: str = RELEVANCE_MODEL) -> list[dict]:
    """
    Uses GPT to classify chunk summaries as relevant (1) or irrelevant (0) to the query.
    Filters out irrelevant chunks.
    """
    print(f"\n--- Step 1: Filtering {len(results)} results by summary relevance using {model} ---")
    if not results:
        return []

    # Prepare summaries for GPT
    summaries_data = []
    for i, record in enumerate(results):
        # Use a unique identifier for each chunk in the prompt
        chunk_id = record.get('id')
        # WORKAROUND: Use 'chapter_summary' field which contains the section summary
        summary = record.get('chapter_summary', '')
        if chunk_id and summary:
            summaries_data.append({"id": chunk_id, "summary": summary})
        else:
            print(f"WARNING: Skipping result index {i} due to missing id or chapter_summary (used as section summary).", file=sys.stderr)

    if not summaries_data:
        print("WARNING: No valid summaries found to send for relevance check.", file=sys.stderr)
        return results # Return original results if none could be processed

    # Construct prompt for GPT
    prompt_summaries = "\n".join([f"ID: {item['id']}\nSummary: {item['summary']}\n---" for item in summaries_data])

    system_message = """You are an assistant tasked with evaluating the relevance of text summaries to a user's query.
Analyze the user's query and each provided summary.
For each summary ID, determine if the summary is:
- Directly relevant or highly related to the query (output 1)
- Completely irrelevant or unrelated to the query (output 0)

Respond ONLY with a valid JSON object where keys are the summary IDs (as strings) and values are either 1 (relevant) or 0 (irrelevant).
Example response format: {"chunk_id_1": 1, "chunk_id_2": 0, "chunk_id_3": 1}
Do not include any explanations or introductory text outside the JSON object."""

    user_message = f"""User Query: "{query}"

Evaluate the relevance of the following summaries to the query:
---
{prompt_summaries}
---
Provide your response as a single JSON object mapping each ID to 1 (relevant) or 0 (irrelevant)."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    relevance_map = {}
    last_exception = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            print(f"Calling {model} for summary relevance check (Attempt {attempt + 1}/{_RETRY_ATTEMPTS})...")
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2, # Low temperature for classification
                response_format={"type": "json_object"} # Request JSON output
            )
            # Add safety check for empty choices list
            if not response.choices:
                print(f"ERROR: API returned empty choices list during relevance check (Attempt {attempt + 1}).", file=sys.stderr)
                raise APIError("Empty choices list received from API", response=response, body=None) # Raise APIError to trigger retry

            response_content = response.choices[0].message.content
            relevance_map = json.loads(response_content)
            # Validate format (simple check)
            if not isinstance(relevance_map, dict) or not all(isinstance(k, str) and v in [0, 1] for k, v in relevance_map.items()):
                 raise ValueError("Invalid JSON format received from relevance check API.")
            print("Summary relevance check successful.")
            break # Success
        except json.JSONDecodeError as e:
            print(f"WARNING: Failed to decode JSON response from relevance check (Attempt {attempt + 1}): {e}. Response: {response_content}", file=sys.stderr)
            last_exception = e
            time.sleep(_RETRY_DELAY)
        except (APIError, RateLimitError, APITimeoutError) as e:
            print(f"WARNING: API error during relevance check (Attempt {attempt + 1}): {e}", file=sys.stderr)
            last_exception = e
            time.sleep(_RETRY_DELAY * (attempt + 1))
        except Exception as e:
            print(f"WARNING: Unexpected error during relevance check (Attempt {attempt + 1}): {e}", file=sys.stderr)
            last_exception = e
            time.sleep(_RETRY_DELAY)

    if not relevance_map:
        print("ERROR: Failed to get relevance classifications after multiple attempts. Skipping filtering.", file=sys.stderr)
        if last_exception: print(f"  Last error: {last_exception}", file=sys.stderr)
        return results # Return original results if API failed

    # Filter results based on relevance map
    filtered_results = []
    removed_count = 0
    for record in results:
        chunk_id = record.get('id')
        # Default to relevant if ID wasn't processed or returned by GPT
        is_relevant = relevance_map.get(str(chunk_id), 1)
        if is_relevant == 1:
            filtered_results.append(record)
        else:
            removed_count += 1
            print(f"  - Filtering out chunk ID {chunk_id} (summary deemed irrelevant).")

    print(f"Finished summary filtering. Kept {len(filtered_results)} results, removed {removed_count}.")
    print(f"Finished summary filtering. Kept {len(filtered_results)} results, removed {removed_count}.")
    # Return the filtered list and the map for logging
    return filtered_results, relevance_map


# Renamed function and updated parameters
def expand_sections_by_token_count(cursor, results: list[dict], top_k_rank: int, top_k_tokens: int, general_tokens: int) -> tuple[list[dict | list[dict]], set]:
    """
    Expands chunks belonging to sections below token thresholds by fetching all chunks for that section.
    Uses different thresholds based on the rank of the triggering chunk.
    Returns a tuple containing:
    - The processed list (with groups for expanded sections).
    - A set of chunk IDs that were added during expansion (excluding the original triggering chunk).
    """
    print(f"\n--- Step 4: Expanding sections by token count (Top {top_k_rank} < {top_k_tokens} tokens, Others < {general_tokens} tokens) ---") # Using new config names in log
    if not results: return [], set()

    processed_results = []
    expansion_log_data = []
    headers_expansion = ["Orig Chunk ID", "Rank", "Section Tokens", "Threshold", "Action", "Added Chunks"] # Updated header
    added_chunk_ids = set() # Track IDs added by this step

    # Identify sections already present more than once
    section_keys = set()
    multi_chunk_sections = set()
    for record in results:
        key = (record.get('document_id'), record.get('chapter_name'), record.get('section_hierarchy'))
        if key in section_keys:
            multi_chunk_sections.add(key)
        section_keys.add(key)

    # Keep track of sections we've already expanded to avoid redundant DB calls
    expanded_sections = set() # Store section_key tuples
    original_chunk_ids_in_results = {r.get('id') for r in results if isinstance(r, dict)} # IDs before expansion

    for record in results:
        orig_chunk_id = record.get('id')
        doc_id = record.get('document_id')
        chapter = record.get('chapter_name')
        hierarchy = record.get('section_hierarchy')
        # page_start = record.get('page_start') # Not needed for token expansion
        # page_end = record.get('page_end') # Not needed for token expansion
        section_tokens = record.get('section_token_count') # Use token count
        rank = record.get('rank') # Rank assigned earlier
        section_key = (doc_id, chapter, hierarchy)

        log_row = [orig_chunk_id or "N/A", rank or "N/A", section_tokens or "N/A", "N/A", "Keep Single", 0] # Default log row

        # Skip if this section has already been fully added by a previous expansion
        if section_key in expanded_sections:
            # If the *original* chunk is being processed again after its section was expanded, skip it.
            # This prevents adding the single chunk back after the group was added.
            # We don't log this skip action as it's internal cleanup.
            continue

        # Determine if expansion is needed based on token count and rank
        should_expand = False
        threshold = "N/A"
        if section_tokens is not None and rank is not None:
            # Determine threshold based on rank
            is_top_k = rank <= top_k_rank
            threshold = top_k_tokens if is_top_k else general_tokens
            log_row[3] = threshold

            if section_tokens <= threshold:
                should_expand = True
                # Log message handled by table below
            # else:
                # Log message handled by table below

        if should_expand:
            try:
                # Fetch all chunks for the section if expansion is needed
                sql = f"""
                    SELECT * FROM {TARGET_TABLE} -- Use config variable
                    WHERE document_id = %s AND chapter_name = %s AND section_hierarchy = %s
                    ORDER BY sequence_number;
                """
                cursor.execute(sql, (doc_id, chapter, hierarchy))
                section_chunks_raw = cursor.fetchall()
                num_found = len(section_chunks_raw)

                if num_found > 1: # Only expand if there are actually more chunks in DB
                    section_chunks = [dict(chunk) for chunk in section_chunks_raw]
                    # Assign the 'new_score' (calculated in the reranking step) to the group for sorting
                    # Also copy other relevant metadata from the triggering chunk
                    group_info = {
                        'type': 'group',
                        'original_rank': rank,
                        'original_vector_score': record.get('vector_score'), # Keep original vector score
                        'section_importance_score': record.get('section_importance_score'), # Keep importance
                        'new_score': record.get('new_score', 0.0), # Use the score calculated in reranking
                        'chunks': section_chunks,
                        # Add key metadata for sorting/identification if needed later
                        'document_id': doc_id,
                        'chapter_name': chapter,
                        'section_hierarchy': hierarchy,
                        'min_seq': section_chunks[0].get('sequence_number') if section_chunks else None # For gap filling sort
                    }
                    processed_results.append(group_info)
                    expanded_sections.add(section_key) # Mark section as expanded
                    log_row[4] = f"Expand Group ({num_found} total)"
                    log_row[5] = num_found - 1 # Number of *added* chunks (total - original)
                    # Track newly added chunk IDs (excluding the original trigger chunk ID)
                    for chunk in section_chunks:
                        chunk_id = chunk.get('id')
                        if chunk_id and chunk_id != orig_chunk_id:
                             added_chunk_ids.add(chunk_id)
                else:
                    # If only 1 chunk found in DB, just add the original record back
                    processed_results.append(record) # Keep original single chunk
                    log_row[4] = "Keep Single (1 in DB)"
                    log_row[5] = 0
            except Exception as e:
                print(f"ERROR: Failed to fetch or process expansion for section {section_key}: {e}", file=sys.stderr)
                traceback.print_exc()
                processed_results.append(record) # Add original back on error
                log_row[4] = "Error - Keep Single"
                log_row[5] = 0
        else:
            # If no expansion needed, just add the original record
            processed_results.append(record)
            # log_row already defaults to "Keep Single", 0

        expansion_log_data.append(log_row)

    # Print the expansion log table
    try:
        print(tabulate(expansion_log_data, headers=headers_expansion, tablefmt="grid"))
    except ImportError:
        print("WARN: 'tabulate' library not found. Skipping table format.")
        for row in expansion_log_data:
            print(f"ID: {row[0]}, Rank: {row[1]}, Tokens: {row[2]}, Threshold: {row[3]}, Action: {row[4]}, Added: {row[5]}") # Updated print

    print(f"Finished section expansion. Intermediate result count: {len(processed_results)} (items/groups). Added {len(added_chunk_ids)} new chunks.")

    # --- Second Pass: Filter out single chunks now contained within groups ---
    print("\n--- Filtering expanded chunks from results list ---")
    final_processed_results = []
    grouped_chunk_ids = set()

    # Identify all chunk IDs within groups
    for item in processed_results:
        if isinstance(item, dict) and item.get('type') == 'group':
            for chunk in item.get('chunks', []):
                if chunk.get('id'):
                    grouped_chunk_ids.add(chunk.get('id'))

    # Add items to final list, skipping single chunks that are now in groups
    skipped_singles = 0
    for item in processed_results:
        if isinstance(item, dict) and item.get('type') == 'group':
            final_processed_results.append(item) # Keep groups
        elif isinstance(item, dict): # It's a single chunk
            chunk_id = item.get('id')
            if chunk_id in grouped_chunk_ids:
                print(f"  - Filtering out single chunk ID {chunk_id} (Rank: {item.get('rank', 'N/A')}) as it's included in an expanded group.")
                skipped_singles += 1
                continue # Skip this single chunk
            else:
                final_processed_results.append(item) # Keep this single chunk
        else:
             print(f"WARNING: Unexpected item type during final expansion filtering: {type(item)}")
             final_processed_results.append(item) # Keep unknown items

    print(f"Finished filtering. Removed {skipped_singles} single chunks already covered by groups.")
    print(f"Final count after expansion and filtering: {len(final_processed_results)}")

    return final_processed_results, added_chunk_ids # Return the properly filtered list

# Function name already updated in previous step, ensure logic matches
def fill_sequence_gaps(cursor, results: list[dict | list[dict]], max_seq_gap: int) -> tuple[list[dict | list[dict]], set]:
    """
    Identifies small sequence number gaps between consecutive results and fetches missing chunks.
    Handles both single chunks and groups. Assigns a 'new_score' to gap chunks based on neighbors.
    Returns the updated list and a set of added chunk IDs.
    """
    print(f"\n--- Step 5: Filling sequence gaps (Max Gap: {max_seq_gap} sequences) ---") # Log uses correct config name
    if len(results) < 2:
        return results, set() # Need at least two items to have a gap

    items_with_sequences = []
    gap_log_data = []
    headers_gaps = ["Between Item (Seq)", "And Item (Seq)", "Sequence Gap", "Action", "Added Chunks"] # Updated header
    added_chunk_ids = set() # Track IDs added by this step

    # Extract sequence info for sorting and gap checking
    for item in results:
        if isinstance(item, dict) and item.get('type') == 'group':
            # It's a group
            if not item.get('chunks'): continue # Skip empty groups
            first_chunk = item['chunks'][0]
            last_chunk = item['chunks'][-1]
            doc_id = first_chunk.get('document_id')
            min_seq = first_chunk.get('sequence_number')
            max_seq = last_chunk.get('sequence_number')
            if all(v is not None for v in [doc_id, min_seq, max_seq]):
                items_with_sequences.append({
                    'item': item, 'doc_id': doc_id,
                    'min_seq': min_seq, 'max_seq': max_seq, 'is_group': True,
                    'id_repr': f"Group({min_seq}-{max_seq})" # Representation for logging
                })
        elif isinstance(item, dict):
             # It's a single chunk (original or from expansion/reranking)
             doc_id = item.get('document_id')
             seq = item.get('sequence_number')
             chunk_id = item.get('id')
             if all(v is not None for v in [doc_id, seq]):
                 items_with_sequences.append({
                    'item': item, 'doc_id': doc_id,
                    'min_seq': seq, 'max_seq': seq, 'is_group': False,
                    'id_repr': f"Chunk({chunk_id})" # Representation for logging
                 })

    if len(items_with_sequences) < 2:
        print("  Not enough items with sequence numbers to check for gaps.")
        return results, set()

    # Sort items by sequence number
    items_with_sequences.sort(key=lambda x: x['min_seq'])

    final_results_with_gaps = []
    last_item_info = None

    for current_item_info in items_with_sequences:
        if last_item_info:
            # Check for gap only if documents match
            if last_item_info['doc_id'] == current_item_info['doc_id']:
                # Calculate sequence gap
                seq_gap = current_item_info['min_seq'] - last_item_info['max_seq'] - 1
                log_row = [f"{last_item_info['id_repr']} ({last_item_info['max_seq']})", f"{current_item_info['id_repr']} ({current_item_info['min_seq']})", seq_gap, "None", 0]

                # Check if gap is within threshold and positive
                if 0 < seq_gap <= max_seq_gap:
                    try:
                        sql = f"""
                            SELECT * FROM {TARGET_TABLE} -- Use config variable
                            WHERE document_id = %s AND sequence_number > %s AND sequence_number < %s
                            ORDER BY sequence_number;
                        """
                        cursor.execute(sql, (current_item_info['doc_id'], last_item_info['max_seq'], current_item_info['min_seq']))
                        gap_chunks_raw = cursor.fetchall()
                        num_added = len(gap_chunks_raw)
                        if num_added > 0:
                            # Calculate average score of surrounding items based on 'new_score'
                            preceding_score = last_item_info.get('item', {}).get('new_score', 0.0)
                            following_score = current_item_info.get('item', {}).get('new_score', 0.0)
                            preceding_score = preceding_score if preceding_score is not None else 0.0
                            following_score = following_score if following_score is not None else 0.0
                            average_score = (preceding_score + following_score) / 2.0

                            gap_chunks = []
                            for chunk_raw in gap_chunks_raw:
                                chunk = dict(chunk_raw)
                                chunk['new_score'] = average_score # Assign average score
                                gap_chunks.append(chunk)
                                if chunk.get('id'): added_chunk_ids.add(chunk.get('id')) # Track added IDs

                            final_results_with_gaps.extend(gap_chunks) # Add gap chunks with scores
                            log_row[3] = f"Fill Gap ({num_added} chunks, Avg Score ~{average_score:.4f})"
                            log_row[4] = num_added
                        else:
                            log_row[3] = "No Chunks Found"
                            log_row[4] = 0
                    except Exception as e:
                        print(f"ERROR: Failed to fetch or process gap fill between seq {last_item_info['max_seq']} and {current_item_info['min_seq']}: {e}", file=sys.stderr)
                        traceback.print_exc()
                        log_row[3] = "Error Fetching"
                        log_row[4] = 0
                elif seq_gap > max_seq_gap:
                     log_row[3] = f"Gap > {max_seq_gap} sequences"
                     log_row[4] = 0
                else: # seq_gap <= 0
                     log_row[3] = "No Gap / Overlap"
                     log_row[4] = 0
                gap_log_data.append(log_row)

        # Add the current item (chunk or group)
        final_results_with_gaps.append(current_item_info['item'])
        last_item_info = current_item_info

    # Print the gap log table
    if gap_log_data:
        try:
            print(tabulate(gap_log_data, headers=headers_gaps, tablefmt="grid"))
        except ImportError:
            print("WARN: 'tabulate' library not found. Skipping table format.")
            for row in gap_log_data:
                print(f"Between: {row[0]}, And: {row[1]}, Seq Gap: {row[2]}, Action: {row[3]}, Added: {row[4]}") # Updated print
    else:
        print("  No gaps checked (less than 2 items with sequence numbers).")

    print(f"Finished sequence gap filling. Result count: {len(final_results_with_gaps)}. Added {len(added_chunk_ids)} new chunks.")
    return final_results_with_gaps, added_chunk_ids


def rerank_by_importance(results: list[dict | list[dict]], importance_factor: float) -> list[dict | list[dict]]:
    """
    Calculates 'new_score' based on vector score and section importance,
    then sorts the results based on this new score and assigns 'new_rank'.
    Handles only single chunks (items from initial search + filtering).
    Returns the reranked and sorted list.
    """
    print(f"\n--- Step 3: Reranking by Importance & Sorting (Factor: {importance_factor}) ---") # Updated title
    if not results: return []

    # 1. Calculate new_score for each item
    items_with_scores = []
    for item in results:
        if not isinstance(item, dict):
             print(f"WARNING: Skipping unexpected item type in reranking: {type(item)}", file=sys.stderr)
             continue

        new_score = 0.0
        original_score = item.get('vector_score', 0.0) or 0.0
        importance = item.get('section_importance_score', 0.0) or 0.0
        try:
            original_score = float(original_score)
            importance = float(importance)
            boost = 1.0 + (importance_factor * importance)
            new_score = original_score * boost
        except (TypeError, ValueError) as e:
            print(f"WARNING: Could not calculate score for Chunk {item.get('id', 'N/A')} due to invalid numeric values. Setting new_score to 0. Error: {e}", file=sys.stderr)
            new_score = 0.0

        item['new_score'] = new_score
        items_with_scores.append(item)

    # 2. Sort by new_score (descending)
    # Use original rank as a tie-breaker (ascending) for stability
    items_with_scores.sort(key=lambda x: (x.get('new_score', 0.0), -x.get('rank', float('inf'))), reverse=True)

    # 3. Assign new_rank and prepare log data
    rerank_log_data = []
    headers_rerank = ["New Rank", "Orig Rank", "Chunk ID", "Orig Score", "Importance", "New Score"]
    final_reranked_list = []
    # Store original ranks before sorting and overwriting
    original_ranks = {item.get('id'): item.get('rank') for item in items_with_scores if item.get('id')}

    for new_rank, item in enumerate(items_with_scores, 1):
        item_id = item.get('id')
        orig_rank = original_ranks.get(item_id, 'N/A') # Get original rank safely
        item['rank'] = new_rank # Overwrite original rank with new rank
        final_reranked_list.append(item)
        rerank_log_data.append([
            new_rank,
            orig_rank, # Use the stored original rank for logging
            item_id,
            f"{item.get('vector_score', 0.0):.4f}",
            f"{item.get('section_importance_score', 0.0):.2f}",
            f"{item.get('new_score', 0.0):.4f}"
        ])

    print(f"Finished reranking and sorting {len(final_reranked_list)} items.")
    print("\n--- Importance Reranking Results ---")
    try:
        print(tabulate(rerank_log_data, headers=headers_rerank, tablefmt="grid"))
    except ImportError:
        print("WARN: 'tabulate' library not found.")
        for row in rerank_log_data:
            print(f"NewRank: {row[0]}, OrigRank: {row[1]}, ID: {row[2]}, OrigScore: {row[3]}, Importance: {row[4]}, NewScore: {row[5]}")

    # Return the sorted list with 'new_score' and updated 'rank'
    return final_reranked_list


# --- Helper function for sorting ---
def get_min_sequence_number(item):
    """Gets the minimum sequence number for a chunk or a group."""
    if isinstance(item, dict) and item.get('type') == 'group':
        # For groups, find the min sequence number among its chunks
        try:
            return min(c.get('sequence_number') for c in item.get('chunks', []) if c.get('sequence_number') is not None)
        except (ValueError, TypeError):
            return float('inf') # Return infinity if no valid sequence number found
    elif isinstance(item, dict):
        # For single chunks (including gap-filled ones if they are dicts)
        seq = item.get('sequence_number')
        return seq if seq is not None else float('inf')
    elif isinstance(item, psycopg2.extras.DictRow):
         # Handle potential DictRow from gap filling if not converted earlier
         seq = item['sequence_number']
         return seq if seq is not None else float('inf')
    return float('inf') # Default for unexpected types


# --- Response Generation Functions ---

def format_chunks_as_cards(results: list[dict | list[dict]]):
    """
    Formats database results into "cards" for better GPT context understanding.
    
    Args:
        results: List of database result rows (as dict cursors)
    
    Returns:
        Formatted string with all chunks as cards
    """
    """
    Formats database results (including groups) into "cards" for GPT context.
    Includes only specified fields with labels.
    """
    print("\n--- Formatting Final Results as Cards for LLM ---")
    cards = []
    final_item_count = 0

    for i, item in enumerate(results):
        card_parts = []
        content_parts = []
        record_for_metadata = None
        is_group = False

        if isinstance(item, dict) and item.get('type') == 'group':
            # It's an expanded section group
            is_group = True
            if not item.get('chunks'): continue # Skip empty groups
            record_for_metadata = item['chunks'][0] # Use first chunk for metadata
            card_parts.append(f"--- CARD {i+1} (Reconstructed Section) ---")
            # Concatenate content from all chunks in the group
            for chunk in item['chunks']:
                 content_parts.append(chunk.get('content', ''))
            content = "\n\n".join(filter(None, content_parts))
            print(f"  - Formatting Card {i+1}: Group of {len(item['chunks'])} chunks (Section: {record_for_metadata.get('section_hierarchy', 'N/A')})")

        elif isinstance(item, dict):
             # It's a single chunk (original, or filled gap)
             # Handle potential DictRow from gap filling if not converted earlier
             if isinstance(item, psycopg2.extras.DictRow):
                 item = dict(item)
             record_for_metadata = item
             card_parts.append(f"--- CARD {i+1} ---")
             content = record_for_metadata.get('content', '')
             print(f"  - Formatting Card {i+1}: Single Chunk ID {record_for_metadata.get('id', 'N/A')}")
        else:
            print(f"WARNING: Skipping unexpected item type during formatting: {type(item)}", file=sys.stderr)
            continue

        if not record_for_metadata or not content:
            print(f"WARNING: Skipping Card {i+1} due to missing metadata or content.", file=sys.stderr)
            continue

        # Extract and format required fields with labels
        chapter_name = record_for_metadata.get('chapter_name', 'Unknown Chapter')
        section_title = record_for_metadata.get('section_title', 'Unknown Section')
        section_hierarchy = record_for_metadata.get('section_hierarchy', '')
        standard = record_for_metadata.get('standard')
        standard_codes = record_for_metadata.get('standard_codes')

        card_parts.append(f"Chapter: {chapter_name}")
        card_parts.append(f"Section Title: {section_title}")
        if section_hierarchy:
            card_parts.append(f"Section Hierarchy: {section_hierarchy}")
        # Use section_standard and section_standard_codes from schema
        standard = record_for_metadata.get('section_standard')
        standard_codes = record_for_metadata.get('section_standard_codes')
        if standard:
            card_parts.append(f"Standard: {standard}")
        if standard_codes and isinstance(standard_codes, list) and standard_codes:
            card_parts.append(f"Standard Codes: {', '.join(standard_codes)}")
        # Add chapter_tags (schema doesn't have section_tags)
        tags = record_for_metadata.get('chapter_tags')
        if tags and isinstance(tags, list) and tags:
             card_parts.append(f"Chapter Tags: {', '.join(tags)}")

        # Add the content
        card_parts.append("\nContent:") # Changed label slightly
        card_parts.append(content)

        cards.append("\n".join(card_parts))
        final_item_count += 1

    print(f"Formatted {final_item_count} cards.")
    # Join all cards with clear separation
    return "\n\n" + "\n\n".join(cards) + "\n\n"


def generate_response_from_chunks(client: OpenAI, query: str, formatted_chunks: str):
    """
    Generates a GPT response based on the query and formatted chunks.
    
    Args:
        client: OpenAI client
        query: User's query/question
        formatted_chunks: Chunks formatted as cards
    
    Returns:
        Generated response from GPT
    """
    """
    Generates a GPT response based on the query and formatted chunks.
    """
    print("\n--- Step 6: Generating Final Response from Processed Chunks ---")

    system_message = """You are a specialized accounting research assistant with expertise in IFRS and US GAAP standards.
Your task is to answer accounting questions based ONLY on the information provided in the context cards below.
Each card represents a relevant piece of text from the source document. Some cards might represent a reconstructed section containing multiple original text chunks.

Context Card Fields:
- Chapter: The name of the chapter the text belongs to.
- Section Title: The title of the specific section.
- Section Hierarchy: The structural path to the section (e.g., "Chapter 1 > Part A > Section 1.1").
- Standard: The primary accounting standard discussed (e.g., IFRS 16, ASC 842).
- Standard Codes: Specific codes or paragraph references within the standard.
- Section Content: The actual text content from the source document.

Instructions for Answering:
1. Rely EXCLUSIVELY on the "Section Content" provided in the cards. DO NOT use your external knowledge or training data.
2. Synthesize the information from the relevant cards to provide a comprehensive answer to the user's question.
3. You MUST cite your sources for every significant point or piece of information. Use the "Chapter" and "Section Title" or "Section Hierarchy" from the card(s) you used. Format citations clearly, e.g., [Source: Chapter Name, Section Title] or [Source: Section Hierarchy].
4. If multiple cards support a point, cite all relevant sources.
5. If the provided cards do not contain sufficient information to fully answer the question, clearly state what information is missing or cannot be determined from the context. Do not speculate or fabricate.
6. Structure your response logically, using headings or bullet points if helpful.
7. Provide a concise summary (2-3 sentences) at the end.

Remember: Accuracy and strict adherence to the provided context with proper citations are paramount."""

    user_message = f"""User Question: {query}

Context Cards:
{formatted_chunks}
---
Based ONLY on the context cards provided above, please answer the user's question with clear citations for each point, referencing the Chapter, Section Title, or Section Hierarchy."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    last_exception = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            print(f"Making API call to {RESPONSE_MODEL} for final response generation (Attempt {attempt + 1}/{_RETRY_ATTEMPTS})...")

            response = client.chat.completions.create(
                model=RESPONSE_MODEL,
                messages=messages,
                max_tokens=MAX_RESPONSE_TOKENS,
                temperature=RESPONSE_TEMPERATURE,
                stream=False
            )

            print("API call for response generation successful.")
            # Add safety check for empty choices list
            if not response.choices:
                print(f"ERROR: API returned empty choices list during final response generation (Attempt {attempt + 1}).", file=sys.stderr)
                raise APIError("Empty choices list received from API", response=response, body=None) # Raise APIError to trigger retry

            response_content = response.choices[0].message.content

            # Print token usage information
            usage_info = response.usage
            if usage_info:
                print(f"Token Usage - Prompt: {usage_info.prompt_tokens}, Completion: {usage_info.completion_tokens}, Total: {usage_info.total_tokens}")

            return response_content

        except (APIError, RateLimitError, APITimeoutError) as e:
            print(f"API Error on attempt {attempt + 1}: {e}", file=sys.stderr)
            last_exception = e
            time.sleep(_RETRY_DELAY * (attempt + 1))
        except Exception as e:
            print(f"Non-API Error on attempt {attempt + 1}: {e}", file=sys.stderr)
            last_exception = e
            time.sleep(_RETRY_DELAY)

    # If we get here, all attempts failed
    error_msg = f"ERROR: Failed to generate response after {_RETRY_ATTEMPTS} attempts."
    if last_exception:
        error_msg += f" Last error: {str(last_exception)}"
    print(error_msg, file=sys.stderr)
    return f"Error generating response: {error_msg}"


# --- Main Function ---

def generate_response(query: str):
    """
    Main function orchestrating the enhanced retrieval and response generation.
    """
    """
    Main function orchestrating the enhanced retrieval and response generation.
    """
    print("\n" + "="*80)
    print("--- Starting Enhanced Response Generation Process ---")
    print("="*80)
    print(f"\n>>> User Query:\n{query}\n")
    print(f"--- Configuration ---")
    print(f"Initial K Results: {INITIAL_K}")
    # print(f"Final K Results: {'All' if FINAL_K is None else FINAL_K}") # Removed FINAL_K
    print(f"Importance Factor: {IMPORTANCE_FACTOR}")
    print(f"Relevance Model: {RELEVANCE_MODEL}")
    print(f"Response Model: {RESPONSE_MODEL}")
    print(f"Expansion Thresholds: Top {SECTION_EXPANSION_TOP_K_RANK} < {SECTION_EXPANSION_TOP_K_TOKENS} tokens, Others < {SECTION_EXPANSION_GENERAL_TOKENS} tokens") # Updated expansion log
    print(f"Gap Fill Max Sequence: {GAP_FILL_MAX_SEQUENCE_GAP}") # Updated gap fill log
    print("-" * 80)


    # 1. Create OpenAI Client
    client = create_openai_client()
    if not client:
        return "ERROR: Failed to create OpenAI client."

    # 2. Generate Query Embedding
    query_embedding = generate_query_embedding(client, query, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS)
    if query_embedding is None:
        return "ERROR: Failed to generate query embedding."

    # 3. Connect to Database
    conn = connect_to_db(DB_PARAMS)
    if conn is None:
        return "ERROR: Database connection failed."

    response_text = "Error: Processing failed before response generation."
    cursor = None
    processed_results = [] # To hold results through the pipeline

    try:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # --- Stage 1: Initial Retrieval (Vector Search Only) ---
        print(f"\n--- Stage 1: Performing Initial Vector Search (Top {INITIAL_K}) ---") # Updated title
        # Call the modified search function (now vector-only)
        initial_results_raw = perform_hybrid_search( # Function name kept for now, but logic changed
            cursor=cursor,
            query=query,
            query_embedding=query_embedding,
            initial_k=INITIAL_K,
            doc_id=DOCUMENT_ID_TO_SEARCH
        )

        if not initial_results_raw:
            return "No relevant information found for your query in the initial database search."

        # Convert to list of dicts and add initial rank
        initial_results = []
        initial_chunk_ids = set() # Track initial IDs
        similarity_table_data = []
        headers = ["Rank", "Chunk ID", "Chapter", "Vector Score"] # Updated header

        for i, row in enumerate(initial_results_raw):
            record = dict(row)
            chunk_id = record.get('id')
            rank = i + 1
            record['rank'] = rank # Add rank based on initial vector score order
            initial_results.append(record)
            if chunk_id:
                initial_chunk_ids.add(chunk_id)
                similarity_table_data.append([
                    rank,
                    chunk_id,
                    record.get('chapter_name', 'N/A'),
                    f"{record.get('vector_score', 0.0):.4f}" # Use vector_score
                ])

        print(f"Retrieved {len(initial_results)} initial results.")
        print("\n--- Initial Vector Search Results (Top K) ---") # Clarified title
        try:
            print(tabulate(similarity_table_data, headers=headers, tablefmt="grid"))
        except ImportError:
            print("WARN: 'tabulate' library not found. Install with 'pip install tabulate' for table formatting.")
            for row in similarity_table_data:
                print(f"Rank: {row[0]}, ID: {row[1]}, Chapter: {row[2]}, Score: {row[3]}")
        print("-" * 80)

        # --- Log Unique Sections & Summaries from Top K ---
        print("\n--- Unique Sections & Summaries in Initial Top K Results ---") # Updated title
        unique_sections = {} # Use a set to store unique tuples
        for record in initial_results:
            chapter_name = record.get('chapter_name', 'N/A')
            section_hierarchy = record.get('section_hierarchy', 'N/A')
            # WORKAROUND: Use chapter_summary which holds section summary
            summary = record.get('chapter_summary', '') # Use empty string if None
            # Create a unique key based on chapter, hierarchy, and summary
            section_key = (chapter_name, section_hierarchy, summary)
            if section_key not in unique_sections:
                 # Store the full data for easy access later, only add if key is new
                 unique_sections[section_key] = {
                     'chapter': chapter_name,
                     'hierarchy': section_hierarchy,
                     'summary': summary
                 }

        # Prepare data for tabulation
        unique_section_data_for_table = [
            [data['chapter'], data['hierarchy'], data['summary'][:100] + ("..." if len(data['summary'])>100 else "")]
            for data in unique_sections.values()
        ]
        # Sort data for consistent output (optional, but helpful)
        unique_section_data_for_table.sort(key=lambda x: (x[0], x[1]))

        try:
            print(tabulate(unique_section_data_for_table, headers=["Chapter", "Section Hierarchy", "Section Summary (Truncated)"], tablefmt="grid"))
        except ImportError:
             print("WARN: 'tabulate' library not found.")
             for row in unique_section_data_for_table: print(f"- Ch: {row[0]}, Hier: {row[1]}, Sum: {row[2]}")
        print("-" * 80)


        processed_results = initial_results
        all_added_chunk_ids = set() # Track IDs added by expansion/gaps
        all_removed_chunk_ids = set() # Track IDs removed by filtering

        # --- Stage 2: Summary Relevance Filtering ---
        # Modify filter_by_summary_relevance to return the map
        filtered_results, relevance_map = filter_by_summary_relevance(client, query, processed_results)
        if not filtered_results: return "No relevant information remained after summary filtering."

        # Log relevance filtering results with more detail
        relevance_table_data = []
        headers_relevance = ["Orig Rank", "Chunk ID", "Vector Score", "Summary (Trunc)", "GPT Decision"]
        current_filtered_ids = set()
        for record in filtered_results:
             chunk_id = record.get('id')
             if chunk_id: current_filtered_ids.add(chunk_id)

        for record in initial_results: # Iterate original results to show all details
             chunk_id = record.get('id')
             gpt_decision_num = relevance_map.get(str(chunk_id), 1) # Default to Keep (1) if not in map
             gpt_decision_str = "Keep" if gpt_decision_num == 1 else "Remove"
             if gpt_decision_num == 0:
                 all_removed_chunk_ids.add(chunk_id) # Track removed IDs

             # WORKAROUND: Use chapter_summary which holds section summary
             summary = record.get('chapter_summary', '')
             summary_trunc = summary[:80] + ("..." if len(summary) > 80 else "")

             relevance_table_data.append([
                 record.get('rank', 'N/A'), # Original Rank
                 chunk_id,
                 f"{record.get('vector_score', 0.0):.4f}", # Original Score
                 summary_trunc,
                 gpt_decision_str
             ])

        print("\n--- Summary Relevance Filtering Results ---") # Updated title
        try:
            print(tabulate(relevance_table_data, headers=headers_relevance, tablefmt="grid"))
        except ImportError:
             print("WARN: 'tabulate' library not found. Skipping table format.")
             for row in relevance_table_data:
                 print(f"Rank: {row[0]}, ID: {row[1]}, Score: {row[2]}, Summary: {row[3]}, Decision: {row[4]}")
        print(f"Removed {len(all_removed_chunk_ids)} chunks based on summary relevance.")
        print("-" * 80)
        processed_results = filtered_results # Update results for next step

        # --- Stage 3: Importance Reranking (Moved Earlier) ---
        # This step calculates 'new_score' but does NOT re-sort the list yet.
        # Sorting happens at the end based on the calculated 'new_score'.
        processed_results = rerank_by_importance(processed_results, IMPORTANCE_FACTOR)
        print("-" * 80)

        # --- Stage 4: Section Expansion (by Token Count) ---
        # Pass the reranked results to expansion
        expanded_results, added_by_expansion_ids = expand_sections_by_token_count( # Use renamed function
            cursor=cursor,
            results=processed_results, # Pass reranked results
            top_k_rank=SECTION_EXPANSION_TOP_K_RANK,
            top_k_tokens=SECTION_EXPANSION_TOP_K_TOKENS,
            general_tokens=SECTION_EXPANSION_GENERAL_TOKENS
        )
        if not expanded_results: return "No results remained after section expansion."
        all_added_chunk_ids.update(added_by_expansion_ids)
        print("-" * 80)
        processed_results = expanded_results # Update results

        # --- Stage 5: Sequence Gap Filling ---
        # Pass the expanded results to sequence gap filling
        filled_results, added_by_gaps_ids = fill_sequence_gaps( # Function name already updated
            cursor=cursor,
            results=processed_results,
            max_seq_gap=GAP_FILL_MAX_SEQUENCE_GAP
        )
        if not filled_results: return "No results remained after sequence gap filling."
        all_added_chunk_ids.update(added_by_gaps_ids)
        print("-" * 80)
        processed_results = filled_results # Update results

        # --- Stage 6: Final Sorting (REMOVED - Sorting now done in Reranking Step 3) ---
        # print(f"\n--- Step 6: Sorting {len(processed_results)} final items by new_score (desc) and sequence_number (asc) ---")
        # processed_results.sort(key=lambda x: (
        #     x.get('new_score', 0.0) if isinstance(x, dict) else 0.0, # Primary key: score (desc)
        #     -get_min_sequence_number(x) # Secondary key: min sequence (asc, negate for reverse sort)
        #     ), reverse=True)
        # print("Sorting complete.")
        # print("-" * 80)

        # --- Stage 6: Truncation (REMOVED) --- (Renumbered)
        # if FINAL_K is not None and len(processed_results) > FINAL_K:
        #     print(f"\n--- Truncating results from {len(processed_results)} to {FINAL_K} ---")
        #     processed_results = processed_results[:FINAL_K]
        #     print("-" * 80)

        # --- Stage 6: Final Results Summary Log --- (Renumbered)
        print("\n--- Step 6: Final Results Summary ---") # Renumbered
        final_chunk_ids = set()
        total_context_tokens = 0
        # Corrected loop for token counting and ID collection
        for item in processed_results:
            if isinstance(item, dict) and item.get('type') == 'group':
                for chunk in item.get('chunks', []):
                    chunk_id = chunk.get('id')
                    if chunk_id: final_chunk_ids.add(chunk_id)
                    total_context_tokens += count_tokens(chunk.get('content', ''))
            elif isinstance(item, dict): # Single chunk (original or gap-filled)
                 chunk_id = item.get('id')
                 if chunk_id: final_chunk_ids.add(chunk_id)
                 total_context_tokens += count_tokens(item.get('content', ''))
            elif isinstance(item, psycopg2.extras.DictRow): # Handle gap-filled chunks if not dicts yet
                 # Convert DictRow to dict for consistent access
                 chunk_dict = dict(item)
                 chunk_id = chunk_dict.get('id')
                 if chunk_id: final_chunk_ids.add(chunk_id)
                 total_context_tokens += count_tokens(chunk_dict.get('content', ''))


        print(f"Initial Chunk IDs ({len(initial_chunk_ids)}): {sorted(list(initial_chunk_ids))}")
        print(f"Removed by Filtering ({len(all_removed_chunk_ids)}): {sorted(list(all_removed_chunk_ids))}")
        # Added IDs are those in the final set that weren't in the initial set AND weren't removed
        truly_added_ids = final_chunk_ids - (initial_chunk_ids - all_removed_chunk_ids)
        print(f"Added by Expansion/Gaps ({len(truly_added_ids)}): {sorted(list(truly_added_ids))}")
        # Verify all_added_chunk_ids matches truly_added_ids (debugging check)
        # print(f"DEBUG: Tracked Added IDs ({len(all_added_chunk_ids)}): {sorted(list(all_added_chunk_ids))}")
        print(f"Final Chunk IDs in Context ({len(final_chunk_ids)}): {sorted(list(final_chunk_ids))}")
        print(f"Estimated Total Tokens in Final Context: {total_context_tokens}") # Log token count
        print("-" * 80)

        # --- Stage 7: Format Cards --- (Renumbered)
        print("\n--- Step 7: Formatting Final Results as Cards ---") # Renumbered
        formatted_chunks = format_chunks_as_cards(processed_results)
        print("-" * 80)

        # --- Stage 8: Generate Final Response --- (Renumbered)
        print("\n--- Step 8: Generating Final Response ---") # Renumbered
        response_text = generate_response_from_chunks(client, query, formatted_chunks)

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"ERROR: Enhanced response generation process failed: {error}", file=sys.stderr)
        traceback.print_exc()
        response_text = f"Error during response generation process: {str(error)}"
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            print("\nDatabase connection closed.")

    print("\n--- Enhanced Response Generation Process Finished ---")
    return response_text

# Example of how to use in a Jupyter notebook cell:
# response = generate_response("Explain the accounting treatment for leases under IFRS 16")
# print(response)
