# -*- coding: utf-8 -*-
"""
Stage 3: Chunking & Final Assembly

Purpose:
Processes section data from Stage 2. It performs merging of small sections,
splitting of large sections (while mapping positions back to the original raw
markdown), a final merge pass for ultra-small chunks, generates embeddings for
the final content, assigns a final sequence number, and assembles records
matching the target database schema.

Input:
- JSON file from Stage 2 (e.g., 'pipeline_output/stage2/stage2_section_data.json').
- Original chapter markdown files (path configured, needed for position mapping).

Output:
- A JSON file in OUTPUT_DIR containing a list of final chunk dictionaries,
  ready for database insertion (e.g., 'pipeline_output/stage3/stage3_final_records.json').
"""

import os
import json
import traceback
import re
import time
import logging
import requests
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
from collections import defaultdict

# --- Dependencies Check ---
try:
    import tiktoken
except ImportError:
    tiktoken = None
    print("WARNING: tiktoken not installed. Token counts will be estimated (chars/4). `pip install tiktoken`")

try:
    import natsort
except ImportError:
    natsort = None
    print("INFO: natsort not installed. Sorting might not be natural. `pip install natsort`")

try:
    from openai import OpenAI, APIError
except ImportError:
    OpenAI = None
    APIError = None
    print("ERROR: openai library not installed. GPT/Embedding features unavailable. `pip install openai`")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x # Make tqdm optional
    print("INFO: tqdm not installed. Progress bars disabled. `pip install tqdm`")

# ==============================================================================
# Configuration
# ==============================================================================

# --- Directory Paths ---
# TODO: Adjust these paths as needed
STAGE2_OUTPUT_DIR = "pipeline_output/stage2"
STAGE2_FILENAME = "stage2_section_data.json"
# Directory containing the original source markdown files (needed for position mapping)
ORIGINAL_MD_DIR = "1_chapters_md"
OUTPUT_DIR = "pipeline_output/stage3"
OUTPUT_FILENAME = "stage3_final_records.json"
LOG_DIR = "pipeline_output/logs"

# --- API Configuration ---
# TODO: Load securely or replace placeholders (Ensure consistency)
BASE_URL = os.environ.get("OPENAI_API_BASE", "https://api.example.com/v1")
MODEL_NAME_EMBEDDING = os.environ.get("OPENAI_MODEL_EMBEDDING", "text-embedding-3-large")
OAUTH_URL = os.environ.get("OAUTH_URL", "https://api.example.com/oauth/token")
CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID", "your_client_id")
CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET", "your_client_secret")
SSL_SOURCE_PATH = os.environ.get("SSL_SOURCE_PATH", "/path/to/your/rbc-ca-bundle.cer")
SSL_LOCAL_PATH = "/tmp/rbc-ca-bundle.cer"

# --- API Parameters ---
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY = 5 # seconds
EMBEDDING_DIMENSIONS = 2000 # For text-embedding-3-large, adjust if using different model/size
EMBEDDING_BATCH_SIZE = 32 # Number of texts to send in one embedding API call

# --- Chunking/Merging Thresholds ---
# TODO: Load from config or adjust as needed
SECTION_MIN_TOKENS = 250 # Sections below this might be merged in Pass 1
SECTION_MAX_TOKENS = 750 # Sections above this trigger splitting (Aligned with CHUNK_SPLIT_MAX_TOKENS)
CHUNK_MERGE_THRESHOLD = 50 # Ultra-small chunks below this merged in Pass 2
CHUNK_SPLIT_MAX_TOKENS = 750 # Target max tokens for chunks *after* splitting

# --- Token Cost (Optional) ---
EMBEDDING_COST_PER_1K_TOKENS = 0.00013 # Example cost for text-embedding-3-large

# --- Logging Setup ---
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
log_file = Path(LOG_DIR) / 'stage3_chunking_assembly.log'
# Remove existing handlers if configuring multiple times in a notebook
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)

# ==============================================================================
# Utility Functions (Self-Contained)
# ==============================================================================

# --- Tokenizer ---
_TOKENIZER = None
if tiktoken:
    try: _TOKENIZER = tiktoken.get_encoding("cl100k_base"); logging.info("Using 'cl100k_base' tokenizer.")
    except Exception as e: logging.warning(f"Failed tokenizer init: {e}. Estimating tokens."); _TOKENIZER = None

def count_tokens(text: str) -> int:
    """Counts tokens using tiktoken or estimates."""
    if not text: return 0
    if _TOKENIZER:
        try: return len(_TOKENIZER.encode(text))
        except Exception: return len(text) // 4
    else: return len(text) // 4

# --- API Client ---
_SSL_CONFIGURED = False
_OPENAI_CLIENT = None

def _setup_ssl(source_path=SSL_SOURCE_PATH, local_path=SSL_LOCAL_PATH) -> bool:
    """Copies SSL cert locally."""
    global _SSL_CONFIGURED
    if _SSL_CONFIGURED: return True
    if not Path(source_path).is_file(): logging.warning(f"SSL source cert not found: {source_path}."); _SSL_CONFIGURED = True; return True
    logging.info("Setting up SSL certificate...")
    try:
        source = Path(source_path); local = Path(local_path)
        local.parent.mkdir(parents=True, exist_ok=True)
        with open(source, "rb") as sf, open(local, "wb") as df: df.write(sf.read())
        os.environ["SSL_CERT_FILE"] = str(local); os.environ["REQUESTS_CA_BUNDLE"] = str(local)
        logging.info(f"SSL certificate configured: {local}"); _SSL_CONFIGURED = True; return True
    except Exception as e: logging.error(f"Error setting up SSL: {e}", exc_info=True); return False

def _get_oauth_token(oauth_url=OAUTH_URL, client_id=CLIENT_ID, client_secret=CLIENT_SECRET, ssl_verify_path=SSL_LOCAL_PATH) -> Optional[str]:
    """Retrieves OAuth token."""
    verify_path = ssl_verify_path if Path(ssl_verify_path).exists() else True
    logging.info("Attempting OAuth token..."); payload = {'grant_type': 'client_credentials', 'client_id': client_id, 'client_secret': client_secret}
    try:
        response = requests.post(oauth_url, data=payload, timeout=30, verify=verify_path); response.raise_for_status()
        token_data = response.json(); oauth_token = token_data.get('access_token')
        if not oauth_token: logging.error("OAuth Error: 'access_token' not found."); return None
        logging.info("OAuth token obtained."); return oauth_token
    except requests.exceptions.RequestException as e: logging.error(f"OAuth Error: {e}", exc_info=True); return None

def get_openai_client(base_url=BASE_URL) -> Optional[OpenAI]:
    """Initializes and returns the OpenAI client."""
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT: return _OPENAI_CLIENT
    if not OpenAI: logging.error("OpenAI library not available."); return None
    if not _setup_ssl(): logging.warning("Proceeding without explicit SSL setup.")
    api_key = _get_oauth_token()
    if not api_key: logging.error("Aborting client creation: OAuth token failure."); return None
    try: _OPENAI_CLIENT = OpenAI(api_key=api_key, base_url=base_url); logging.info("OpenAI client created."); return _OPENAI_CLIENT
    except Exception as e: logging.error(f"Error creating OpenAI client: {e}", exc_info=True); return None

# --- API Call (Embeddings) ---
def get_embeddings_batch(texts: List[str], client: OpenAI, model: str = MODEL_NAME_EMBEDDING, dimensions: Optional[int] = EMBEDDING_DIMENSIONS) -> List[Optional[List[float]]]:
    """Generates embeddings for a batch of texts with retry logic."""
    if not client: logging.error("OpenAI client not available for embeddings."); return [None] * len(texts)
    if not texts: logging.warning("Empty list passed to get_embeddings_batch."); return []
    # Replace empty strings or None with a placeholder to avoid API errors, will return None embedding for these
    processed_texts = [t if t and t.strip() else " " for t in texts]
    original_indices_to_return_none = {i for i, t in enumerate(texts) if not t or not t.strip()}

    last_exception = None
    embedding_params = {"input": processed_texts, "model": model}
    if dimensions: embedding_params["dimensions"] = dimensions

    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            logging.debug(f"Requesting embeddings for batch size {len(processed_texts)} (Attempt {attempt + 1}/{API_RETRY_ATTEMPTS})...")
            response = client.embeddings.create(**embedding_params)
            # Sort embeddings by index to ensure correct order
            embeddings_data = sorted(response.data, key=lambda e: e.index)
            # Extract the embedding vectors
            batch_embeddings = [e.embedding for e in embeddings_data]

            usage = response.usage
            if usage:
                total_tokens = usage.total_tokens
                cost = (total_tokens / 1000) * EMBEDDING_COST_PER_1K_TOKENS
                logging.info(f"Embedding API Usage (Batch Size: {len(processed_texts)}) - Tokens: {total_tokens}, Cost: ${cost:.6f}")
            logging.debug(f"Embeddings received successfully for batch size {len(processed_texts)}.")

            # Ensure the number of embeddings matches the number of processed texts
            if len(batch_embeddings) != len(processed_texts):
                 logging.error(f"Mismatch between requested texts ({len(processed_texts)}) and received embeddings ({len(batch_embeddings)}).")
                 # Fallback: return None for all in case of mismatch
                 return [None] * len(texts)

            # Reconstruct the final list, inserting None for originally empty texts
            final_embeddings = []
            current_embedding_idx = 0
            for i in range(len(texts)):
                if i in original_indices_to_return_none:
                    final_embeddings.append(None)
                else:
                    final_embeddings.append(batch_embeddings[current_embedding_idx])
                    current_embedding_idx += 1
            return final_embeddings

        except APIError as e:
            logging.warning(f"API Error on embedding batch attempt {attempt + 1}: {e}"); last_exception = e; time.sleep(API_RETRY_DELAY * (attempt + 1))
        except Exception as e:
            logging.warning(f"Non-API Error on embedding batch attempt {attempt + 1}: {e}", exc_info=True); last_exception = e; time.sleep(API_RETRY_DELAY)

    logging.error(f"Embedding batch generation failed after {API_RETRY_ATTEMPTS} attempts for batch size {len(processed_texts)}.")
    if last_exception: logging.error(f"Last error: {last_exception}")
    # Return None for all texts in the batch if all attempts fail
    return [None] * len(texts)

# --- File/Path Utils ---
def create_directory(directory: str):
    """Creates the specified directory if it does not already exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)

# --- Position Mapping & Tag Handling ---
# Removing extract_tag_mapping and map_clean_to_raw_pos as the required
# raw_section_slice_start/end_pos data is missing from Stage 2 output.
AZURE_TAG_PATTERN = re.compile(r'<!--\s*Page(Footer|Number|Break|Header)=?(".*?"|\d+)?\s*-->\s*\n?')

# --- Content Splitting ---
def split_paragraph_by_sentences(paragraph: str, max_tokens: int = CHUNK_SPLIT_MAX_TOKENS) -> List[str]:
    """Splits a large paragraph into sentence-based chunks."""
    sentence_pattern = re.compile(r'(?<=[.!?])(?:\s+|\n+)') # Lookbehind for terminator
    sentences = [s.strip() for s in sentence_pattern.split(paragraph) if s and s.strip()]
    if not sentences: return [paragraph] if paragraph.strip() else []

    chunks = []; current_chunk_sentences = []; current_tokens = 0
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        if sentence_tokens > max_tokens: # Handle oversized sentences
            if current_chunk_sentences: chunks.append(' '.join(current_chunk_sentences))
            # Simple word split for oversized sentence
            words = sentence.split(); temp_chunk = []; temp_tokens = 0
            for word in words:
                word_tokens = count_tokens(word + ' ')
                if temp_tokens > 0 and temp_tokens + word_tokens > max_tokens:
                    chunks.append(' '.join(temp_chunk)); temp_chunk = [word]; temp_tokens = word_tokens
                else: temp_chunk.append(word); temp_tokens += word_tokens
            if temp_chunk: chunks.append(' '.join(temp_chunk))
            current_chunk_sentences = []; current_tokens = 0; continue
        elif current_tokens > 0 and current_tokens + sentence_tokens > max_tokens:
            chunks.append(' '.join(current_chunk_sentences))
            current_chunk_sentences = [sentence]; current_tokens = sentence_tokens
        else:
            current_chunk_sentences.append(sentence); current_tokens += sentence_tokens
    if current_chunk_sentences: chunks.append(' '.join(current_chunk_sentences))
    return [c for c in chunks if c]

def split_large_section_content(cleaned_content: str, max_tokens: int = CHUNK_SPLIT_MAX_TOKENS) -> List[Dict[str, Any]]:
    """Splits large section content by paragraphs, then sentences."""
    para_pattern = re.compile(r'\n\s*\n+')
    para_matches = list(para_pattern.finditer(cleaned_content))
    para_boundaries = [m.end() for m in para_matches]
    split_sub_chunks = []
    current_chunk_paras_clean_text = []; current_chunk_clean_start_pos = 0; current_chunk_tokens = 0

    for i in range(len(para_boundaries) + 1):
        para_slice_start = para_boundaries[i-1] if i > 0 else 0
        para_slice_end = para_boundaries[i] if i < len(para_boundaries) else len(cleaned_content)
        para_text_raw = cleaned_content[para_slice_start:para_slice_end]
        para_text_stripped = para_text_raw.strip()
        if not para_text_stripped: continue
        para_tokens = count_tokens(para_text_stripped)
        try: strip_offset = para_text_raw.index(para_text_stripped); para_clean_start_in_section = para_slice_start + strip_offset
        except ValueError: para_clean_start_in_section = para_slice_start

        if para_tokens > max_tokens: # Handle oversized paragraphs
            if current_chunk_paras_clean_text:
                chunk_clean_text = '\n\n'.join(current_chunk_paras_clean_text)
                chunk_clean_end_pos = para_slice_start
                split_sub_chunks.append({'content': chunk_clean_text, 'clean_start_pos': current_chunk_clean_start_pos, 'clean_end_pos': chunk_clean_end_pos, 'token_count': current_chunk_tokens})
            sentence_chunks_text = split_paragraph_by_sentences(para_text_stripped, max_tokens)
            current_sentence_offset_in_para = 0
            for sentence_chunk_text in sentence_chunks_text:
                sentence_chunk_tokens = count_tokens(sentence_chunk_text)
                chunk_clean_start = para_clean_start_in_section + current_sentence_offset_in_para
                chunk_clean_end = chunk_clean_start + len(sentence_chunk_text) # Approx end
                split_sub_chunks.append({'content': sentence_chunk_text, 'clean_start_pos': chunk_clean_start, 'clean_end_pos': chunk_clean_end, 'token_count': sentence_chunk_tokens})
                try: # Update offset for next sentence
                    next_start_in_para = para_text_stripped.index(sentence_chunk_text, current_sentence_offset_in_para) + len(sentence_chunk_text)
                    while next_start_in_para < len(para_text_stripped) and para_text_stripped[next_start_in_para].isspace(): next_start_in_para += 1
                    current_sentence_offset_in_para = next_start_in_para
                except ValueError: current_sentence_offset_in_para = len(para_text_stripped)
            current_chunk_clean_start_pos = para_slice_end; current_chunk_paras_clean_text = []; current_chunk_tokens = 0; continue
        elif current_chunk_tokens > 0 and current_chunk_tokens + para_tokens > max_tokens:
            chunk_clean_text = '\n\n'.join(current_chunk_paras_clean_text)
            chunk_clean_end_pos = para_slice_start
            split_sub_chunks.append({'content': chunk_clean_text, 'clean_start_pos': current_chunk_clean_start_pos, 'clean_end_pos': chunk_clean_end_pos, 'token_count': current_chunk_tokens})
            current_chunk_paras_clean_text = [para_text_stripped]; current_chunk_tokens = para_tokens; current_chunk_clean_start_pos = para_clean_start_in_section
        else:
            if not current_chunk_paras_clean_text: current_chunk_clean_start_pos = para_clean_start_in_section
            current_chunk_paras_clean_text.append(para_text_stripped); current_chunk_tokens += para_tokens

    if current_chunk_paras_clean_text:
        chunk_clean_text = '\n\n'.join(current_chunk_paras_clean_text)
        chunk_clean_end_pos = len(cleaned_content)
        split_sub_chunks.append({'content': chunk_clean_text, 'clean_start_pos': current_chunk_clean_start_pos, 'clean_end_pos': chunk_clean_end_pos, 'token_count': current_chunk_tokens})

    return split_sub_chunks

# --- Merging Logic ---
# Removing merge_sections_pass1 as it's redundant with Stage 2 merging.

def merge_chunks_pass2(chunks: List[Dict], small_threshold: int, max_tokens: int) -> List[Dict]:
    """Merges ultra-small chunks (Pass 2)."""
    if not chunks: return []
    final_chunks_list = []
    merged_flags = [False] * len(chunks)
    i = 0
    while i < len(chunks):
        if merged_flags[i]: i += 1; continue
        current_chunk = chunks[i]
        # Use 'chunk_token_count' for internal merging logic only
        # (this field will be removed from final output)
        current_chunk_tokens = current_chunk.get("chunk_token_count", 0)

        if current_chunk_tokens >= small_threshold:
            final_chunks_list.append(current_chunk); i += 1; continue

        logging.debug(f"Found small chunk (Seq: {current_chunk.get('sequence_number', 'N/A')}, Part: {current_chunk.get('part_number', 'N/A')}, Tokens: {current_chunk_tokens}). Attempting merge...")
        is_heading_only = current_chunk.get("content", "").strip().startswith("#")
        merge_occurred = False

        # Preferred: Heading forward, Content backward
        if is_heading_only:
            next_idx = i + 1
            while next_idx < len(chunks) and merged_flags[next_idx]: next_idx += 1
            if next_idx < len(chunks):
                next_chunk = chunks[next_idx]
                next_chunk_tokens = next_chunk.get("chunk_token_count", 0)
                if (current_chunk.get("chapter_number") == next_chunk.get("chapter_number") and
                    current_chunk.get("section_number") == next_chunk.get("section_number") and # Ensure same section
                    current_chunk_tokens + next_chunk_tokens <= max_tokens):
                    logging.debug(f"  Merging forward (heading) into chunk (Seq: {next_chunk.get('sequence_number', 'N/A')})")
                    next_chunk["content"] = f"{current_chunk.get('content', '')}\n\n{next_chunk.get('content', '')}"
                    next_chunk["chunk_token_count"] = count_tokens(next_chunk["content"])
                    next_chunk["start_pos"] = current_chunk["start_pos"] # Update start pos
                    merged_flags[i] = True; merge_occurred = True
        else: # Content prefers backward
            if final_chunks_list:
                prev_chunk = final_chunks_list[-1]
                prev_chunk_tokens = prev_chunk.get("chunk_token_count", 0)
                if (current_chunk.get("chapter_number") == prev_chunk.get("chapter_number") and
                    current_chunk.get("section_number") == prev_chunk.get("section_number") and # Ensure same section
                    prev_chunk_tokens + current_chunk_tokens <= max_tokens):
                    logging.debug(f"  Merging backward into chunk (Seq: {prev_chunk.get('sequence_number', 'N/A')})")
                    prev_chunk["content"] = f"{prev_chunk.get('content', '')}\n\n{current_chunk.get('content', '')}"
                    prev_chunk["chunk_token_count"] = count_tokens(prev_chunk["content"])
                    prev_chunk["end_pos"] = current_chunk["end_pos"] # Update end pos
                    merged_flags[i] = True; merge_occurred = True

        # Fallback: Heading backward, Content forward
        if not merge_occurred:
            if is_heading_only:
                 if final_chunks_list:
                    prev_chunk = final_chunks_list[-1]
                    prev_chunk_tokens = prev_chunk.get("chunk_token_count", 0)
                    if (current_chunk.get("chapter_number") == prev_chunk.get("chapter_number") and
                        current_chunk.get("section_number") == prev_chunk.get("section_number") and
                        prev_chunk_tokens + current_chunk_tokens <= max_tokens):
                        logging.debug(f"  Merging backward (heading fallback) into chunk (Seq: {prev_chunk.get('sequence_number', 'N/A')})")
                        prev_chunk["content"] = f"{prev_chunk.get('content', '')}\n\n{current_chunk.get('content', '')}"
                        prev_chunk["chunk_token_count"] = count_tokens(prev_chunk["content"])
                        prev_chunk["end_pos"] = current_chunk["end_pos"]
                        merged_flags[i] = True; merge_occurred = True
            else: # Content fallback forward
                next_idx = i + 1
                while next_idx < len(chunks) and merged_flags[next_idx]: next_idx += 1
                if next_idx < len(chunks):
                    next_chunk = chunks[next_idx]
                    next_chunk_tokens = next_chunk.get("chunk_token_count", 0)
                    if (current_chunk.get("chapter_number") == next_chunk.get("chapter_number") and
                        current_chunk.get("section_number") == next_chunk.get("section_number") and
                        current_chunk_tokens + next_chunk_tokens <= max_tokens):
                        logging.debug(f"  Merging forward (content fallback) into chunk (Seq: {next_chunk.get('sequence_number', 'N/A')})")
                        next_chunk["content"] = f"{current_chunk.get('content', '')}\n\n{next_chunk.get('content', '')}"
                        next_chunk["chunk_token_count"] = count_tokens(next_chunk["content"])
                        next_chunk["start_pos"] = current_chunk["start_pos"]
                        merged_flags[i] = True; merge_occurred = True

        if not merge_occurred:
            logging.warning(f"  Could not merge small chunk (Seq: {current_chunk.get('sequence_number', 'N/A')}). Keeping as is.")
            final_chunks_list.append(current_chunk)
        i += 1
    return final_chunks_list

# ==============================================================================
# Main Stage 3 Logic
# ==============================================================================

# Cache for raw chapter content to avoid repeated reads
raw_chapter_content_cache = {}

def get_raw_chapter_content(chapter_number: int, source_md_dir: str) -> Optional[str]:
    """Loads raw chapter content from cache or file."""
    if chapter_number in raw_chapter_content_cache:
        return raw_chapter_content_cache[chapter_number]

    # Find the corresponding markdown file (assuming naming convention like 1_intro.md)
    # This might need adjustment based on Stage 1's _source_filename if available
    md_path = Path(source_md_dir)
    found_file = None
    for item in md_path.glob(f"{chapter_number}_*.md"): # Simple pattern match
        found_file = item
        break # Take the first match

    if not found_file or not found_file.is_file():
        logging.error(f"Could not find original markdown file for chapter {chapter_number} in {source_md_dir}")
        return None

    try:
        with open(found_file, 'r', encoding='utf-8') as f:
            content = f.read()
        raw_chapter_content_cache[chapter_number] = content
        return content
    except Exception as e:
        logging.error(f"Error reading raw markdown file {found_file}: {e}", exc_info=True)
        return None


def run_stage3():
    """Main function to execute Stage 3 processing."""
    logging.info("--- Starting Stage 3: Chunking & Final Assembly ---")
    create_directory(OUTPUT_DIR)

    # --- Load Stage 2 Data ---
    stage2_output_file = Path(STAGE2_OUTPUT_DIR) / STAGE2_FILENAME
    if not stage2_output_file.exists():
        logging.error(f"Stage 2 output file not found: {stage2_output_file}"); return None
    try:
        with open(stage2_output_file, "r", encoding="utf-8") as f: all_section_data = json.load(f)
        logging.info(f"Loaded {len(all_section_data)} sections from {stage2_output_file}")
    except Exception as e:
        logging.error(f"Error loading Stage 2 data: {e}", exc_info=True); return None
    if not all_section_data: logging.warning("Stage 2 data is empty."); return []

    # --- Initialize OpenAI Client (for embeddings) ---
    client = get_openai_client()
    if not client: logging.warning("OpenAI client failed. Embeddings will not be generated.")

    # --- Process Sections: Merge(1) -> Split -> Assemble Chunks ---
    all_initial_chunks = []
    # Group sections by chapter first for easier processing
    sections_by_chapter = defaultdict(list)
    for section in all_section_data: sections_by_chapter[section['chapter_number']].append(section)

    # Sort chapters naturally if possible
    chapter_keys = list(sections_by_chapter.keys())
    if natsort: chapter_keys = natsort.natsorted(chapter_keys)
    else: chapter_keys.sort()

    for chapter_num in tqdm(chapter_keys, desc="Processing Chapters"):
        chapter_sections = sections_by_chapter[chapter_num]
        # Sort sections within chapter by section_number
        chapter_sections.sort(key=lambda s: s.get('section_number', 0))

        # 1. Merge Small Sections (Pass 1) - Removed this call as it's redundant with Stage 2.
        # merged_sections = merge_sections_pass1(chapter_sections, SECTION_MIN_TOKENS, SECTION_MAX_TOKENS)
        # logging.info(f"Chapter {chapter_num}: {len(chapter_sections)} sections -> {len(merged_sections)} after Pass 1 merge.")
        # Use the sections directly from Stage 2 output for splitting/chunking
        merged_sections = chapter_sections # Rename variable for clarity downstream

        # 2. Split Large Sections & Assemble Initial Chunks
        raw_chapter_content = get_raw_chapter_content(chapter_num, ORIGINAL_MD_DIR)
        if raw_chapter_content is None:
            logging.error(f"Skipping chapter {chapter_num} due to missing raw content for position mapping.")
            continue
        full_tag_mapping = extract_tag_mapping(raw_chapter_content)

        for section in tqdm(merged_sections, desc=f"Chapter {chapter_num} Splitting", leave=False):
            section_token_count = section.get("section_token_count", 0)
            cleaned_content = section.get("cleaned_section_content", "")
            # raw_start = section.get("raw_section_slice_start_pos") # Removed: Data missing from Stage 2
            # raw_end = section.get("raw_section_slice_end_pos") # Removed: Data missing from Stage 2

            if section_token_count <= SECTION_MAX_TOKENS:
                # Section doesn't need splitting, treat as one chunk
                # Set start/end pos to None as raw mapping is not possible
                chunk_data = section.copy() # Start with section metadata
                chunk_data["part_number"] = 1
                chunk_data["content"] = cleaned_content # Final content is the cleaned section content
                chunk_data["chunk_token_count"] = section_token_count
                chunk_data["start_pos"] = None # Set to None as mapping is not possible
                chunk_data["end_pos"] = None # Set to None as mapping is not possible
                chunk_data.pop("cleaned_section_content", None) # Remove intermediate field
                chunk_data.pop("raw_section_slice_start_pos", None) # Remove potentially existing key if present
                chunk_data.pop("raw_section_slice_end_pos", None) # Remove potentially existing key if present
                all_initial_chunks.append(chunk_data)
            else:
                # Section needs splitting
                logging.debug(f"Splitting Section {section.get('section_number')} (Tokens: {section_token_count})")
                # Removed check for raw_start/raw_end as they are not available/needed for this modified logic
                # Removed section_tag_mapping filtering as map_clean_to_raw_pos is removed

                split_sub_chunks = split_large_section_content(cleaned_content, CHUNK_SPLIT_MAX_TOKENS)
                logging.debug(f"  Split into {len(split_sub_chunks)} sub-chunks.")

                # Deduplicate based on content hash (within the same section split)
                unique_sub_chunks = []
                seen_hashes = set()
                for sub_chunk in split_sub_chunks:
                     content_hash = hashlib.md5(sub_chunk['content'].encode('utf-8')).hexdigest()
                     if content_hash not in seen_hashes:
                         unique_sub_chunks.append(sub_chunk)
                         seen_hashes.add(content_hash)
                     else:
                         logging.debug("  Skipping duplicate sub-chunk content.")

                for i, sub_chunk in enumerate(unique_sub_chunks):
                    part_num = i + 1
                    chunk_data = section.copy() # Inherit metadata
                    chunk_data["part_number"] = part_num
                    chunk_data["content"] = sub_chunk["content"]
                    chunk_data["chunk_token_count"] = sub_chunk["token_count"]
                    # Set start/end pos to None as mapping is not possible
                    chunk_data["start_pos"] = None
                    chunk_data["end_pos"] = None
                    chunk_data.pop("cleaned_section_content", None)
                    chunk_data.pop("raw_section_slice_start_pos", None) # Remove potentially existing key if present
                    chunk_data.pop("raw_section_slice_end_pos", None) # Remove potentially existing key if present
                    all_initial_chunks.append(chunk_data)

    logging.info(f"Generated {len(all_initial_chunks)} initial chunks before final merge.")

    # --- Sort all initial chunks globally for sequence numbering and Pass 2 merge ---
    # Sort primarily by chapter, then section, then part number
    all_initial_chunks.sort(key=lambda c: (
        c.get('chapter_number', 0),
        c.get('section_number', 0),
        c.get('part_number', 0)
    ))

    # --- Merge Ultra-Small Chunks (Pass 2) ---
    final_merged_chunks = merge_chunks_pass2(all_initial_chunks, CHUNK_MERGE_THRESHOLD, CHUNK_SPLIT_MAX_TOKENS)
    logging.info(f"Total chunks after final merge pass: {len(final_merged_chunks)}")

    # --- Final Assembly & Embedding Generation ---
    final_records = []
    # Process embeddings in batches
    num_chunks = len(final_merged_chunks)
    for i in range(0, num_chunks, EMBEDDING_BATCH_SIZE):
        batch_chunks = final_merged_chunks[i:min(i + EMBEDDING_BATCH_SIZE, num_chunks)]
        batch_texts = [chunk.get('content', '') for chunk in batch_chunks]

        batch_embeddings = []
        if client and batch_texts:
            logging.info(f"Generating embeddings for batch {i // EMBEDDING_BATCH_SIZE + 1} (size: {len(batch_texts)})...")
            batch_embeddings = get_embeddings_batch(batch_texts, client)
        elif not client:
            logging.debug(f"Skipping embedding generation for batch {i // EMBEDDING_BATCH_SIZE + 1} (no client).")
            batch_embeddings = [None] * len(batch_chunks)
        else: # No texts in batch
             batch_embeddings = []

        # Assign sequence numbers and assemble records for the current batch
        for j, chunk in enumerate(tqdm(batch_chunks, desc=f"Batch {i // EMBEDDING_BATCH_SIZE + 1} Assembly", leave=False)):
            # Calculate global sequence number
            sequence_num = i + j + 1
            chunk['sequence_number'] = sequence_num

            # Get the corresponding embedding from the batch result
            embedding_vector = batch_embeddings[j] if j < len(batch_embeddings) else None
            if embedding_vector is None and chunk.get('content'):
                 # Log warning only if content existed but embedding failed/was None
                 logging.warning(f"Embedding is None for chunk sequence {sequence_num}.")

            # Assemble final record matching schema (ensure all fields are present)
            record = {
                "document_id": chunk.get("document_id"),
                "chapter_number": chunk.get("chapter_number"),
                "section_number": chunk.get("section_number"),
                "part_number": chunk.get("part_number"),
                "sequence_number": sequence_num,
                "chapter_name": chunk.get("chapter_name"),
                "chapter_tags": chunk.get("chapter_tags"),
                # WORKAROUND: Use section_summary from input for the chapter_summary field in output/DB
                "chapter_summary": chunk.get("section_summary"), # Changed from chunk.get("chapter_summary")
                "chapter_token_count": chunk.get("chapter_token_count"),
                "section_start_page": chunk.get("section_start_page"),
                "section_end_page": chunk.get("section_end_page"),
                "section_importance_score": chunk.get("section_importance_score"),
                "section_token_count": chunk.get("section_token_count"), # Token count of original section
                "section_hierarchy": chunk.get("section_hierarchy"),
                "section_title": chunk.get("section_title"),
                "section_standard": chunk.get("section_standard"),
                "section_standard_codes": chunk.get("section_standard_codes"),
                "section_references": chunk.get("section_references"),
                "content": chunk.get("content"),
                "embedding": embedding_vector # This will be None if generation failed or skipped
                # DB handles: id, created_at, text_search_vector
            }
            # Remove any intermediate fields if they accidentally carried over
            record.pop("level", None)
            record.pop("chunk_token_count", None)  # Remove internal field not in final schema
            # Remove start/end pos if they exist from earlier steps (should be None anyway)
            record.pop("start_pos", None)
            record.pop("end_pos", None)
            for k in range(1, 7): record.pop(f"level_{k}", None) # Use k to avoid shadowing loop var

            final_records.append(record)

    # --- Save Output ---
    output_filepath = Path(OUTPUT_DIR) / OUTPUT_FILENAME
    try:
        with open(output_filepath, "w", encoding="utf-8") as f:
            # Convert embedding vectors to lists for JSON serialization if they are numpy arrays etc.
            # (OpenAI client typically returns lists, but good practice)
            serializable_records = []
            for record in final_records:
                if record.get("embedding") is not None and not isinstance(record["embedding"], list):
                     record["embedding"] = list(record["embedding"])
                serializable_records.append(record)
            json.dump(serializable_records, f, indent=2, ensure_ascii=False)
        logging.info(f"Successfully saved {len(final_records)} final records to {output_filepath}")
    except Exception as e:
        logging.error(f"Error saving final records JSON to {output_filepath}: {e}", exc_info=True)

    # --- Print Summary ---
    logging.info("--- Stage 3 Summary ---")
    logging.info(f"Total sections processed: {len(all_section_data)}")
    logging.info(f"Total final records generated: {len(final_records)}")
    logging.info(f"Output JSON file: {output_filepath}")
    logging.info("--- Stage 3 Finished ---")

    return final_records # Return data for potential chaining

# ==============================================================================
# Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    run_stage3()
