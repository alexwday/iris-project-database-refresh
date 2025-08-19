#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 4: Embedding Generation Pipeline
Processes chunks from Stage 3.5, removes HTML tags for clean embedding generation,
and adds embedding vectors to each chunk.

Key Features:
- Removes HTML page tags from content before embedding generation
- Preserves original chunk content with tags intact
- Generates embeddings using text-embedding-3-large with 2000 dimensions
- Processes embeddings in batches for efficiency
- Uses OAuth authentication and SSL configuration from NAS
- Saves output to NAS

Input: JSON file from Stage 3.5 output (stage3_5_corrected_chunks.json)
Output: JSON file with embeddings added (stage4_embedded_chunks.json)
"""

import os
import json
import re
import logging
import time
import requests
import tempfile
import socket
import io
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

# --- pysmb imports for NAS access ---
from smb.SMBConnection import SMBConnection
from smb import smb_structs

# --- Dependencies Check ---
try:
    from openai import OpenAI, APIError
except ImportError:
    OpenAI = None
    APIError = None
    print("ERROR: openai library not installed. Embedding features unavailable. `pip install openai`")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x
    print("INFO: tqdm not installed. Progress bars disabled. `pip install tqdm`")

# ==============================================================================
# Configuration (Hardcoded - update these values)
# ==============================================================================

# --- NAS Configuration ---
NAS_PARAMS = {
    "ip": "your_nas_ip",  # TODO: Replace with actual NAS IP
    "share": "your_share_name",  # TODO: Replace with actual share name
    "user": "your_nas_user",  # TODO: Replace with actual NAS username
    "password": "your_nas_password",  # TODO: Replace with actual NAS password
    "port": 445,  # Default SMB port (can be 139)
}

# --- Directory Paths (Relative to NAS Share) ---
NAS_INPUT_PATH = "semantic_search/pipeline_output/stage3_5/stage3_5_corrected_chunks.json"
NAS_OUTPUT_PATH = "semantic_search/pipeline_output/stage4"
NAS_LOG_PATH = "semantic_search/pipeline_output/logs"
OUTPUT_FILENAME = "stage4_embedded_chunks.json"

# --- CA Bundle Configuration ---
NAS_SSL_CERT_PATH = "certificates/rbc-ca-bundle.cer"
SSL_LOCAL_PATH = "/tmp/rbc-ca-bundle.cer"

# --- API Configuration ---
BASE_URL = "https://api.example.com/v1"  # TODO: Replace with actual API base URL
MODEL_NAME_EMBEDDING = "text-embedding-3-large"
OAUTH_URL = "https://api.example.com/oauth/token"  # TODO: Replace with actual OAuth URL
CLIENT_ID = "your_client_id"  # TODO: Replace with actual client ID
CLIENT_SECRET = "your_client_secret"  # TODO: Replace with actual client secret

# --- Embedding Configuration ---
EMBEDDING_DIMENSIONS = 2000  # Using 2000 dimensions for text-embedding-3-large
EMBEDDING_BATCH_SIZE = 32  # Number of texts to send in one embedding API call

# --- API Parameters ---
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY = 5  # seconds

# --- Token Cost (Optional) ---
EMBEDDING_COST_PER_1K_TOKENS = 0.00013  # Example cost for text-embedding-3-large

# --- pysmb Configuration ---
smb_structs.SUPPORT_SMB2 = True
smb_structs.MAX_PAYLOAD_SIZE = 65536
CLIENT_HOSTNAME = socket.gethostname()

# --- Logging Level Control ---
VERBOSE_LOGGING = False

# ==============================================================================
# Configuration Validation
# ==============================================================================

def validate_configuration():
    """Validates that configuration values have been properly set."""
    errors = []
    
    if "your_nas_ip" in NAS_PARAMS["ip"]:
        errors.append("NAS IP address not configured")
    if "your_share_name" in NAS_PARAMS["share"]:
        errors.append("NAS share name not configured")
    if "your_nas_user" in NAS_PARAMS["user"]:
        errors.append("NAS username not configured")
    if "your_nas_password" in NAS_PARAMS["password"]:
        errors.append("NAS password not configured")
    if "api.example.com" in BASE_URL:
        errors.append("API base URL not configured")
    if "api.example.com" in OAUTH_URL:
        errors.append("OAuth URL not configured")
    if "your_client_id" in CLIENT_ID:
        errors.append("Client ID not configured")
    if "your_client_secret" in CLIENT_SECRET:
        errors.append("Client secret not configured")
    
    if errors:
        print("❌ Configuration errors detected:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease update the configuration values in the script before running.")
        return False
    return True

# ==============================================================================
# NAS Helper Functions
# ==============================================================================

def create_nas_connection():
    """Creates and returns an authenticated SMBConnection object."""
    try:
        conn = SMBConnection(
            NAS_PARAMS["user"],
            NAS_PARAMS["password"],
            CLIENT_HOSTNAME,
            NAS_PARAMS["ip"],
            use_ntlm_v2=True,
            is_direct_tcp=(NAS_PARAMS["port"] == 445),
        )
        connected = conn.connect(NAS_PARAMS["ip"], NAS_PARAMS["port"], timeout=60)
        if not connected:
            logging.error("Failed to connect to NAS")
            return None
        return conn
    except Exception as e:
        logging.error(f"Exception creating NAS connection: {e}")
        return None

def ensure_nas_dir_exists(conn, share_name, dir_path_relative):
    """Ensures a directory exists on the NAS, creating it if necessary."""
    if not conn:
        return False
    
    path_parts = dir_path_relative.strip("/").split("/")
    current_path = ""
    try:
        for part in path_parts:
            if not part:
                continue
            current_path = os.path.join(current_path, part).replace("\\", "/")
            try:
                conn.listPath(share_name, current_path)
            except Exception:
                conn.createDirectory(share_name, current_path)
        return True
    except Exception as e:
        logging.error(f"Failed to ensure NAS directory: {e}")
        return False

def write_to_nas(share_name, nas_path_relative, content_bytes):
    """Writes bytes to a file path on the NAS using pysmb."""
    conn = None
    try:
        conn = create_nas_connection()
        if not conn:
            return False
        
        dir_path = os.path.dirname(nas_path_relative).replace("\\", "/")
        if dir_path and not ensure_nas_dir_exists(conn, share_name, dir_path):
            return False
        
        file_obj = io.BytesIO(content_bytes)
        bytes_written = conn.storeFile(share_name, nas_path_relative, file_obj)
        
        if bytes_written == 0 and len(content_bytes) > 0:
            logging.error(f"No bytes written to {nas_path_relative}")
            return False
        
        return True
    except Exception as e:
        logging.error(f"Error writing to NAS: {e}")
        return False
    finally:
        if conn:
            conn.close()

def read_from_nas(share_name, nas_path_relative):
    """Reads content (as bytes) from a file path on the NAS using pysmb."""
    conn = None
    file_obj = None
    try:
        conn = create_nas_connection()
        if not conn:
            return None
        
        file_obj = io.BytesIO()
        _, _ = conn.retrieveFile(share_name, nas_path_relative, file_obj)
        file_obj.seek(0)
        content_bytes = file_obj.read()
        return content_bytes
    except Exception as e:
        logging.error(f"Error reading from NAS: {e}")
        return None
    finally:
        if file_obj:
            try:
                file_obj.close()
            except Exception:
                pass
        if conn:
            conn.close()

# ==============================================================================
# Logging Setup
# ==============================================================================

def setup_logging():
    """Setup logging with controlled verbosity."""
    temp_log = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log")
    temp_log_path = temp_log.name
    temp_log.close()
    
    # Clear any existing handlers to prevent duplication
    logging.root.handlers = []
    
    log_level = logging.DEBUG if VERBOSE_LOGGING else logging.WARNING
    
    # Only add file handler to root logger (no console handler)
    root_file_handler = logging.FileHandler(temp_log_path)
    root_file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[root_file_handler],
    )
    
    # Progress logger handles all console output
    progress_logger = logging.getLogger("progress")
    progress_logger.setLevel(logging.INFO)
    progress_logger.propagate = False
    
    # Console handler for progress messages
    progress_console_handler = logging.StreamHandler()
    progress_console_handler.setFormatter(logging.Formatter("%(message)s"))
    progress_logger.addHandler(progress_console_handler)
    
    # File handler for progress messages
    progress_file_handler = logging.FileHandler(temp_log_path)
    progress_file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    progress_logger.addHandler(progress_file_handler)
    
    # If verbose logging is enabled, also show warnings/errors on console
    if VERBOSE_LOGGING:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        console_handler.setLevel(logging.WARNING)
        logging.root.addHandler(console_handler)
    
    return temp_log_path

def log_progress(message, end="\n"):
    """Log a progress message that always shows."""
    progress_logger = logging.getLogger("progress")
    
    if end == "":
        sys.stdout.write(message)
        sys.stdout.flush()
        for handler in progress_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.stream.write(f"{datetime.now().isoformat()} - {message}\n")
                handler.flush()
    else:
        progress_logger.info(message)

# ==============================================================================
# Token Counting
# ==============================================================================

def count_tokens(text: str) -> int:
    """Estimates token count using empirically-calibrated formula."""
    if not text:
        return 0
    
    char_count = len(text)
    estimated_tokens = int(char_count / 3.5)
    
    MIN_CHARS_PER_TOKEN = 2
    MAX_CHARS_PER_TOKEN = 10
    
    max_possible_tokens = char_count // MIN_CHARS_PER_TOKEN
    min_possible_tokens = char_count // MAX_CHARS_PER_TOKEN
    
    estimated_tokens = max(min_possible_tokens, min(estimated_tokens, max_possible_tokens))
    
    return estimated_tokens

# ==============================================================================
# API Client Setup
# ==============================================================================

_SSL_CONFIGURED = False
_OPENAI_CLIENT = None

def _setup_ssl_from_nas() -> bool:
    """Downloads SSL cert from NAS and sets environment variables."""
    global _SSL_CONFIGURED
    if _SSL_CONFIGURED:
        return True
    
    try:
        cert_bytes = read_from_nas(NAS_PARAMS["share"], NAS_SSL_CERT_PATH)
        if cert_bytes is None:
            logging.warning("SSL certificate not found on NAS, continuing without it")
            _SSL_CONFIGURED = True
            return True
        
        local_cert = Path(SSL_LOCAL_PATH)
        local_cert.parent.mkdir(parents=True, exist_ok=True)
        with open(local_cert, "wb") as f:
            f.write(cert_bytes)
        
        os.environ["SSL_CERT_FILE"] = str(local_cert)
        os.environ["REQUESTS_CA_BUNDLE"] = str(local_cert)
        _SSL_CONFIGURED = True
        return True
    except Exception as e:
        logging.error(f"Error setting up SSL: {e}")
        return False

def _get_oauth_token(
    oauth_url=OAUTH_URL, 
    client_id=CLIENT_ID, 
    client_secret=CLIENT_SECRET, 
    ssl_verify_path=SSL_LOCAL_PATH
) -> Optional[str]:
    """Retrieves OAuth token."""
    try:
        verify_path = ssl_verify_path if ssl_verify_path and Path(ssl_verify_path).exists() else True
    except (TypeError, OSError):
        verify_path = True
    
    payload = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    }
    
    try:
        response = requests.post(oauth_url, data=payload, timeout=30, verify=verify_path)
        response.raise_for_status()
        token_data = response.json()
        oauth_token = token_data.get("access_token")
        if not oauth_token:
            logging.error("No access token in OAuth response")
            return None
        return oauth_token
    except requests.exceptions.RequestException as e:
        logging.error(f"OAuth token request failed: {e}")
        return None

def get_openai_client(base_url=BASE_URL) -> Optional[OpenAI]:
    """Initializes and returns the OpenAI client."""
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT:
        return _OPENAI_CLIENT
    if not OpenAI:
        return None
    if not _setup_ssl_from_nas():
        pass
    
    api_key = _get_oauth_token()
    if not api_key:
        return None
    try:
        _OPENAI_CLIENT = OpenAI(api_key=api_key, base_url=base_url)
        return _OPENAI_CLIENT
    except Exception as e:
        logging.error(f"Failed to create OpenAI client: {e}")
        return None

# ==============================================================================
# HTML Tag Removal
# ==============================================================================

def remove_html_tags(content: str) -> str:
    """
    Remove HTML page header/footer tags from content.
    Preserves the actual text content while removing only the HTML comment tags.
    
    Args:
        content: The chunk content potentially containing HTML tags
    
    Returns:
        Clean content with HTML tags removed
    """
    if not content:
        return content
    
    # Pattern to match page headers and footers
    # Exact format from Stage 2: <!-- PageHeader PageNumber="X" PageReference="Y" -->
    # Exact format from Stage 2: <!-- PageFooter PageNumber="X" PageReference="Y" -->
    tag_pattern = re.compile(
        r'<!--\s*Page(?:Header|Footer)\s+PageNumber="\d+"\s+PageReference="[^"]*"\s*-->',
        re.IGNORECASE
    )
    
    # Remove all matching tags
    cleaned_content = tag_pattern.sub('', content)
    
    # Clean up any extra whitespace left behind
    # Replace multiple newlines with double newline
    cleaned_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_content)
    
    # Strip leading/trailing whitespace
    cleaned_content = cleaned_content.strip()
    
    return cleaned_content

# ==============================================================================
# Embedding Generation
# ==============================================================================

def get_embeddings_batch(
    texts: List[str], 
    client: OpenAI, 
    model: str = MODEL_NAME_EMBEDDING, 
    dimensions: Optional[int] = EMBEDDING_DIMENSIONS
) -> List[Optional[List[float]]]:
    """
    Generate embeddings for a batch of texts with retry logic.
    
    Args:
        texts: List of text strings to embed
        client: OpenAI client instance
        model: Embedding model name
        dimensions: Number of dimensions for the embedding
    
    Returns:
        List of embedding vectors (or None for failed embeddings)
    """
    if not client:
        logging.error("OpenAI client not available for embeddings.")
        return [None] * len(texts)
    
    if not texts:
        logging.warning("Empty list passed to get_embeddings_batch.")
        return []
    
    # Replace empty strings with a space to avoid API errors
    processed_texts = [t if t and t.strip() else " " for t in texts]
    original_indices_to_return_none = {i for i, t in enumerate(texts) if not t or not t.strip()}
    
    last_exception = None
    embedding_params = {
        "input": processed_texts,
        "model": model
    }
    
    if dimensions:
        embedding_params["dimensions"] = dimensions
    
    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            logging.debug(f"Requesting embeddings for batch size {len(processed_texts)} (Attempt {attempt + 1}/{API_RETRY_ATTEMPTS})...")
            
            response = client.embeddings.create(**embedding_params)
            
            # Sort embeddings by index to ensure correct order
            embeddings_data = sorted(response.data, key=lambda e: e.index)
            
            # Extract the embedding vectors
            batch_embeddings = [e.embedding for e in embeddings_data]
            
            # Log usage information
            usage = response.usage
            if usage:
                total_tokens = usage.total_tokens
                cost = (total_tokens / 1000) * EMBEDDING_COST_PER_1K_TOKENS
                log_progress(f"  Embedding API Usage - Tokens: {total_tokens}, Cost: ${cost:.6f}")
            
            logging.debug(f"Embeddings received successfully for batch size {len(processed_texts)}.")
            
            # Ensure the number of embeddings matches the number of texts
            if len(batch_embeddings) != len(processed_texts):
                logging.error(f"Mismatch between requested texts ({len(processed_texts)}) and received embeddings ({len(batch_embeddings)}).")
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
            logging.warning(f"API Error on embedding batch attempt {attempt + 1}: {e}")
            last_exception = e
            time.sleep(API_RETRY_DELAY * (attempt + 1))
        except Exception as e:
            logging.warning(f"Non-API Error on embedding batch attempt {attempt + 1}: {e}", exc_info=True)
            last_exception = e
            time.sleep(API_RETRY_DELAY)
    
    logging.error(f"Embedding batch generation failed after {API_RETRY_ATTEMPTS} attempts for batch size {len(processed_texts)}.")
    if last_exception:
        logging.error(f"Last error: {last_exception}")
    
    # Return None for all texts if all attempts fail
    return [None] * len(texts)

# ==============================================================================
# Main Processing
# ==============================================================================

def process_chunks_with_embeddings(chunks: List[Dict], client: OpenAI) -> List[Dict]:
    """
    Process all chunks to add embeddings.
    
    Args:
        chunks: List of chunk dictionaries from Stage 3.5
        client: OpenAI client instance
    
    Returns:
        List of chunks with embeddings added
    """
    log_progress(f"\n📊 Processing {len(chunks)} chunks for embedding generation...")
    
    # Process chunks with progress bar
    processed_chunks = []
    total_batches = (len(chunks) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
    total_cost = 0.0
    total_tokens = 0
    
    for batch_idx in tqdm(range(0, len(chunks), EMBEDDING_BATCH_SIZE), 
                         desc="Generating embeddings", 
                         total=total_batches):
        batch_chunks = chunks[batch_idx:min(batch_idx + EMBEDDING_BATCH_SIZE, len(chunks))]
        
        # Extract content and remove HTML tags for embedding generation
        batch_texts = []
        for chunk in batch_chunks:
            content = chunk.get('chunk_content', '')
            # Remove HTML tags for embedding but keep original content
            clean_content = remove_html_tags(content)
            batch_texts.append(clean_content)
        
        # Generate embeddings for the batch
        batch_embeddings = []
        if client and batch_texts:
            log_progress(f"  Batch {(batch_idx // EMBEDDING_BATCH_SIZE) + 1}/{total_batches} (size: {len(batch_texts)})...")
            batch_embeddings = get_embeddings_batch(batch_texts, client)
            
            # Estimate tokens and cost for this batch
            batch_tokens = sum(count_tokens(text) for text in batch_texts)
            batch_cost = (batch_tokens / 1000) * EMBEDDING_COST_PER_1K_TOKENS
            total_tokens += batch_tokens
            total_cost += batch_cost
        else:
            logging.warning(f"Skipping embedding generation for batch {(batch_idx // EMBEDDING_BATCH_SIZE) + 1} (no client or texts).")
            batch_embeddings = [None] * len(batch_chunks)
        
        # Add embeddings to chunks
        for j, chunk in enumerate(batch_chunks):
            chunk_copy = chunk.copy()
            
            # Add the embedding
            embedding_vector = batch_embeddings[j] if j < len(batch_embeddings) else None
            chunk_copy['embedding'] = embedding_vector
            
            # Log warning if embedding failed for non-empty content
            if embedding_vector is None and chunk.get('chunk_content'):
                logging.warning(f"Embedding is None for chunk {j + 1} in batch starting at {batch_idx + 1}")
            
            processed_chunks.append(chunk_copy)
    
    log_progress(f"\n✅ Successfully processed {len(processed_chunks)} chunks with embeddings.")
    log_progress(f"   Total estimated tokens: {total_tokens:,}")
    log_progress(f"   Total estimated cost: ${total_cost:.4f}")
    
    return processed_chunks

def main():
    """Main processing function for Stage 4"""
    
    # Validate configuration
    if not validate_configuration():
        return 1
    
    # Setup logging
    temp_log_path = setup_logging()
    log_progress("\n" + "="*60)
    log_progress("🚀 Starting Stage 4: Embedding Generation Pipeline")
    log_progress("="*60)
    
    # Load input data from NAS
    log_progress(f"\n📥 Loading chunks from NAS: {NAS_INPUT_PATH}")
    try:
        input_bytes = read_from_nas(NAS_PARAMS["share"], NAS_INPUT_PATH)
        if input_bytes is None:
            log_progress("❌ Failed to read input file from NAS")
            return 1
        
        chunks = json.loads(input_bytes.decode('utf-8'))
        log_progress(f"✅ Loaded {len(chunks)} chunks from input file")
    except Exception as e:
        log_progress(f"❌ Error loading input file: {e}")
        logging.error(f"Error loading input file: {e}", exc_info=True)
        return 1
    
    if not chunks:
        log_progress("⚠️ No chunks found in input file")
        processed_chunks = []
    else:
        # Initialize OpenAI client
        log_progress("\n🔧 Initializing OpenAI client...")
        client = get_openai_client()
        if not client:
            log_progress("❌ Failed to initialize OpenAI client. Cannot generate embeddings.")
            return 1
        log_progress("✅ OpenAI client initialized successfully")
        
        # Process chunks with embeddings
        processed_chunks = process_chunks_with_embeddings(chunks, client)
    
    # Save output to NAS
    output_path = os.path.join(NAS_OUTPUT_PATH, OUTPUT_FILENAME).replace("\\", "/")
    log_progress(f"\n💾 Saving processed chunks to NAS: {output_path}")
    
    try:
        output_json = json.dumps(processed_chunks, indent=2, ensure_ascii=False)
        output_bytes = output_json.encode('utf-8')
        
        if write_to_nas(NAS_PARAMS["share"], output_path, output_bytes):
            log_progress(f"✅ Successfully saved {len(processed_chunks)} chunks with embeddings")
        else:
            log_progress("❌ Failed to write output to NAS")
            return 1
    except Exception as e:
        log_progress(f"❌ Error saving output file: {e}")
        logging.error(f"Error saving output file: {e}", exc_info=True)
        return 1
    
    # Upload log file to NAS
    try:
        with open(temp_log_path, 'r') as f:
            log_content = f.read()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nas_log_filename = f"stage4_embedding_{timestamp}.log"
        nas_log_path = os.path.join(NAS_LOG_PATH, nas_log_filename).replace("\\", "/")
        
        if write_to_nas(NAS_PARAMS["share"], nas_log_path, log_content.encode('utf-8')):
            log_progress(f"📝 Log file uploaded to NAS: {nas_log_path}")
        else:
            log_progress("⚠️ Could not upload log file to NAS")
    except Exception as e:
        log_progress(f"⚠️ Error uploading log: {e}")
    
    # Print summary statistics
    log_progress("\n" + "="*60)
    log_progress("📊 Stage 4 Processing Summary:")
    log_progress(f"  Total chunks processed: {len(processed_chunks)}")
    
    if processed_chunks:
        chunks_with_embeddings = sum(1 for c in processed_chunks if c.get('embedding') is not None)
        log_progress(f"  Chunks with embeddings: {chunks_with_embeddings}")
        log_progress(f"  Chunks without embeddings: {len(processed_chunks) - chunks_with_embeddings}")
        
        # Sample the first chunk to show structure
        sample_chunk = processed_chunks[0]
        log_progress(f"\n  Sample chunk fields: {list(sample_chunk.keys())}")
        if sample_chunk.get('embedding'):
            log_progress(f"  Embedding dimensions: {len(sample_chunk['embedding'])}")
    
    log_progress("="*60)
    log_progress("✅ Stage 4 processing completed successfully!")
    log_progress("="*60 + "\n")
    
    # Clean up temp log file
    try:
        os.unlink(temp_log_path)
    except:
        pass
    
    return 0

if __name__ == "__main__":
    exit(main())