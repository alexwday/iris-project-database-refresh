# Stage 00: PDF to Markdown Converter - Technical Documentation

## Overview
The PDF to Markdown Converter is a specialized preprocessing tool designed for converting EY PDF documents into structured JSON format using Azure Document Intelligence. It processes PDF files stored on a NAS (Network Attached Storage) system, converts each page to markdown format, and outputs a flat JSON array suitable for further processing in the semantic search pipeline.

## Purpose
This tool serves as the initial preprocessing stage (Stage 00) for EY documents, converting raw PDF content into machine-readable markdown format while preserving document structure through Azure tags. It's specifically designed for enterprise environments where documents are stored on network drives.

## Architecture

### Core Components

1. **NAS Connection Manager**
   - SMB protocol implementation using pysmb
   - Authenticated connection handling
   - Directory creation and file I/O operations

2. **PDF Processor**
   - Page extraction using pypdf
   - Individual page PDF generation
   - Batch processing coordination

3. **Azure Document Intelligence Integration**
   - Layout analysis using prebuilt-layout model
   - Markdown format output
   - Retry logic for API resilience

4. **Incremental Writer**
   - Sequential JSON writing
   - Memory-efficient processing
   - Order preservation despite concurrent execution

## Input Processing

### NAS Configuration
```python
NAS_PARAMS = {
    "ip": "your_nas_ip",           # NAS server IP address
    "share": "your_share_name",     # SMB share name
    "user": "your_nas_user",        # Authentication username
    "password": "your_nas_password", # Authentication password
    "port": 445                     # SMB port (445 for SMB2/3, 139 for SMB1)
}
```

### Input Requirements
- **Location**: `NAS_INPUT_PATH = "semantic_search/source_documents/ey"`
- **Format**: Single PDF file (errors if multiple PDFs found)
- **Access**: Read permissions via SMB protocol
- **Validation**: Exactly one PDF file must be present

### Input PDF Characteristics:
- Standard PDF format
- May contain headers/footers
- Can include page numbers in various formats (numeric, roman numerals)
- No specific page limit (memory managed through streaming)

## Processing Workflow

### 1. NAS Connection and File Discovery
```python
def run_ey_prep():
    # Establish NAS connection
    conn = create_nas_connection()
    
    # List files in input directory
    nas_files = list_nas_directory(share_name, NAS_INPUT_PATH)
    
    # Filter for PDF files
    pdf_files = [f for f in nas_files if f.filename.lower().endswith('.pdf')]
    
    # Validate single file requirement
    if len(pdf_files) != 1:
        error("Expects exactly one PDF file")
```

### 2. PDF Download and Extraction
```python
def extract_individual_pages(local_pdf_path, temp_dir):
    reader = PdfReader(local_pdf_path)
    page_files = []
    
    for page_num in range(len(reader.pages)):
        # Create individual PDF for each page
        writer = PdfWriter()
        writer.add_page(reader.pages[page_num])
        
        # Save to temp file
        page_filename = f"{base_name}_page_{page_num + 1}.pdf"
        page_path = os.path.join(temp_dir, page_filename)
        writer.write(page_path)
        
        page_files.append({
            'page_number': page_num + 1,
            'file_path': page_path,
            'original_pdf': local_pdf_path
        })
```

### 3. Azure Document Intelligence Processing
```python
def analyze_document_with_di(di_client, page_file_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            with open(page_file_path, "rb") as f:
                document_bytes = f.read()
            
            # Analyze with prebuilt-layout model
            poller = di_client.begin_analyze_document(
                "prebuilt-layout",
                document_bytes,
                output_content_format=DocumentContentFormat.MARKDOWN
            )
            result = poller.result()
            return result
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise
```

### 4. PageNumber Tag Processing

#### Extraction Pattern:
```python
# Pattern to extract PageNumber value
PAGE_NUMBER_EXTRACT_PATTERN = re.compile(
    r'<!--\s*PageNumber\s*[=:]\s*["\']?([^"\'>\s]+)["\']?\s*-->', 
    re.IGNORECASE
)
```

#### Processing Logic:
```python
def extract_and_clean_page_number(content):
    # Extract page reference value
    page_reference = None
    match = PAGE_NUMBER_EXTRACT_PATTERN.search(content)
    if match:
        page_reference = match.group(1).strip()
    
    # Remove all PageNumber tags from content
    cleaned_content = PAGE_NUMBER_REMOVE_PATTERN.sub('', content)
    
    # Clean up extra newlines
    cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)
    
    return cleaned_content, page_reference
```

### 5. Concurrent Processing with Order Preservation
```python
def process_pages_batch_incremental(di_client, page_files, output_file_path, ...):
    results_dict = {}  # Store by page number for ordering
    next_page_to_write = 1  # Track sequential writing
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PAGES) as executor:
        # Submit all pages for concurrent processing
        futures = {
            executor.submit(process_single_page, di_client, page): page
            for page in page_files
        }
        
        # Process results as they complete
        for future in as_completed(futures):
            result = future.result()
            page_num = result['page_number']
            results_dict[page_num] = result
            
            # Write any sequential pages that are ready
            while next_page_to_write in results_dict:
                page_data = results_dict[next_page_to_write]
                write_to_json(page_data)
                del results_dict[next_page_to_write]
                next_page_to_write += 1
```

### 6. Incremental JSON Writing
```python
def write_incrementally(output_file_path, page_data, first_record):
    with open(output_file_path, 'a', encoding='utf-8') as f:
        if not first_record:
            f.write(',\n')
        json.dump({
            'document_id': document_id,
            'filename': filename,
            'filepath': filepath,
            'page_number': page_number,
            'page_reference': page_reference,  # Extracted from PageNumber tag
            'content': content  # PageNumber tags removed
        }, f, indent=2, ensure_ascii=False)
```

## Azure Tag Handling

### Tags Preserved in Content:
- `<!--PageHeader="...">` - Document headers
- `<!--PageFooter="...">` - Document footers
- `<!--SectionHeader="...">` - Section headers
- Other Azure DI structural tags

### Tags Processed and Removed:
- `<!--PageNumber="...">` - Extracted to `page_reference` field, removed from content

### Tag Processing Rationale:
- PageNumber tags are redundant with the `page_reference` field
- Other tags provide structural information useful for downstream processing
- Header/Footer tags help identify repeated content

## Output Generation

### Output Location:
- **NAS Path**: `semantic_search/prep_output/ey/`
- **Filename**: `ey_prep_output.json`
- **Logs**: `semantic_search/prep_output/ey/logs/`

### Output JSON Structure:
```json
[
  {
    "document_id": "EY_GUIDE_2024",
    "filename": "ey_document.pdf",
    "filepath": "semantic_search/source_documents/ey/ey_document.pdf",
    "page_number": 1,
    "page_reference": "i",  // Extracted from PageNumber tag
    "content": "# Table of Contents\n\n## Chapter 1: Introduction\n..."
  },
  {
    "document_id": "EY_GUIDE_2024",
    "filename": "ey_document.pdf",
    "filepath": "semantic_search/source_documents/ey/ey_document.pdf",
    "page_number": 2,
    "page_reference": "ii",
    "content": "## Overview\n\nThis document provides..."
  }
]
```

### Field Definitions:

#### Standard Fields:
- `document_id`: Configured identifier for the document (e.g., "EY_GUIDE_2024")
- `filename`: Original PDF filename
- `filepath`: Full NAS path to source PDF
- `page_number`: Sequential page number (1-based)
- `page_reference`: Page identifier extracted from PageNumber tag (may be numeric, roman, or alphanumeric)
- `content`: Markdown-formatted page content with PageNumber tags removed

## Error Handling and Recovery

### Retry Mechanisms:
```python
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY = 5  # seconds

# Retry logic for Azure DI API calls
for attempt in range(API_RETRY_ATTEMPTS):
    try:
        result = analyze_document()
        break
    except Exception as e:
        if attempt < API_RETRY_ATTEMPTS - 1:
            time.sleep(API_RETRY_DELAY)
        else:
            log_error(f"Failed after {API_RETRY_ATTEMPTS} attempts")
```

### Failed Page Handling:
- Failed pages logged with error details
- Processing continues for remaining pages
- Failed pages tracked in `failed_pages` list
- Summary report generated at completion

### NAS Connection Resilience:
```python
def create_nas_connection():
    try:
        conn = SMBConnection(...)
        connected = conn.connect(ip, port, timeout=60)
        if not connected:
            return None
        return conn
    except Exception as e:
        logging.error(f"NAS connection failed: {e}")
        return None
```

## Performance Optimizations

### Concurrent Processing:
- `MAX_CONCURRENT_PAGES = 5` - Configurable parallelism
- ThreadPoolExecutor for managed concurrency
- API rate limiting through worker pool

### Memory Management:
```python
# Incremental writing prevents memory buildup
while next_page_to_write in results_dict:
    write_page(results_dict[next_page_to_write])
    del results_dict[next_page_to_write]  # Free memory
    next_page_to_write += 1

# Warning for high memory usage
if len(results_dict) > 500:
    log_warning(f"High memory: {len(results_dict)} pages buffered")
```

### Temporary File Cleanup:
```python
# Automatic cleanup with context manager
with tempfile.TemporaryDirectory() as temp_dir:
    # Process files
    pass  # Files automatically deleted on exit

# Manual cleanup for page PDFs
for page_info in page_files:
    if os.path.exists(page_info['file_path']):
        os.remove(page_info['file_path'])
```

## Network Configuration

### SMB Protocol Settings:
```python
# Enable SMB2/3 support
smb_structs.SUPPORT_SMB2 = True
smb_structs.MAX_PAYLOAD_SIZE = 65536

# Connection parameters
use_ntlm_v2 = True  # Modern authentication
is_direct_tcp = (port == 445)  # Direct TCP for SMB2/3
timeout = 60  # Connection timeout in seconds
```

### SSL Certificate Handling:
```python
# Certificate path for secure connections
NAS_SSL_CERT_PATH = "certificates/rbc-ca-bundle.cer"
SSL_LOCAL_PATH = "/tmp/rbc-ca-bundle.cer"

# Download and use certificate for Azure connections
cert_content = read_from_nas(share_name, NAS_SSL_CERT_PATH)
```

## Logging Configuration

### Log Levels:
```python
# Application logging
logging.basicConfig(level=logging.INFO)

# Suppress verbose Azure SDK logging
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core").setLevel(logging.WARNING)

# Suppress pypdf warnings
logging.getLogger("pypdf").setLevel(logging.ERROR)

# Suppress SMB connection details
logging.getLogger("SMB").setLevel(logging.WARNING)
```

### Log Output:
- Console output for real-time monitoring
- Temporary local file during processing
- Upload to NAS on completion: `logs/ey_prep_YYYYMMDD_HHMMSS.log`

## Dependencies

### Required Libraries:
- **pypdf**: PDF reading and page extraction
- **azure-ai-documentintelligence**: Azure DI client
- **azure-core**: Azure authentication
- **pysmb**: SMB/CIFS protocol for NAS access
- **json**: Data serialization
- **tempfile**: Temporary file management
- **concurrent.futures**: Parallel processing

### Optional Libraries:
- **tqdm**: Progress bar display (graceful fallback)

## Configuration Requirements

### Azure Document Intelligence:
```python
AZURE_DI_ENDPOINT = "YOUR_DI_ENDPOINT"  # Azure endpoint URL
AZURE_DI_KEY = "YOUR_DI_KEY"           # API key for authentication
```

### NAS Access:
- Valid SMB credentials
- Network connectivity to NAS
- Read permissions on input directory
- Write permissions on output directory

### System Requirements:
- Python 3.8+
- Sufficient disk space for temporary files
- Network bandwidth for NAS operations
- Azure API quota for page processing

## Usage Workflow

1. **Configure Settings**:
   - Update NAS_PARAMS with network credentials
   - Set Azure DI endpoint and key
   - Configure document ID

2. **Prepare Input**:
   - Place single PDF in NAS input directory
   - Ensure no other PDFs present

3. **Execute Script**:
   ```bash
   python stage_00_pdf_to_markdown.py
   ```

4. **Monitor Progress**:
   - Console output shows processing status
   - Progress bars if tqdm installed
   - Log messages for debugging

5. **Retrieve Output**:
   - JSON file in NAS output directory
   - Log file in logs subdirectory
   - Summary statistics in console

## Integration with Pipeline

### Prerequisites:
- PDF document available on NAS
- Azure DI service configured and accessible
- Network connectivity established

### Output Compatibility:
- JSON format compatible with stage_00_chapter_splitter
- Preserves all Azure tags except PageNumber
- Page reference field for mapping

### Next Stage (Chapter Splitter):
- Loads ey_prep_output.json
- Displays markdown content
- Assigns chapter boundaries
- Splits PDF and updates JSON

## Known Limitations

1. **Single PDF Requirement**: Processes exactly one PDF at a time
2. **Network Dependency**: Requires stable NAS connection
3. **API Rate Limits**: Subject to Azure DI quotas
4. **Page Size**: Very large pages may exceed API limits
5. **Memory Usage**: Buffering for sequential write order

## Error Scenarios and Solutions

### Scenario 1: Multiple PDFs in Input Directory
- **Error**: "Multiple PDF files found"
- **Solution**: Remove extra PDFs, keep only target document

### Scenario 2: NAS Connection Failure
- **Error**: "Failed to connect to NAS"
- **Solution**: Verify network connectivity, credentials, and share name

### Scenario 3: Azure DI API Failure
- **Error**: "DI analysis failed after 3 attempts"
- **Solution**: Check API key, endpoint, and quota limits

### Scenario 4: Out of Memory
- **Warning**: "High memory usage: X pages buffered"
- **Solution**: Reduce MAX_CONCURRENT_PAGES setting

## Best Practices

1. **Test Connection**: Verify NAS access before processing
2. **Monitor Logs**: Check for warnings and errors
3. **Validate Output**: Ensure all pages processed successfully
4. **Clean Temporary Files**: Verify temp directory cleanup
5. **Backup Source**: Keep original PDF unchanged
6. **Check API Limits**: Monitor Azure DI usage and quotas

## Security Considerations

1. **Credentials**: Store NAS credentials securely (consider environment variables)
2. **Network**: Use SMB2/3 with encryption when possible
3. **Certificates**: Validate SSL certificates for secure connections
4. **API Keys**: Protect Azure DI keys from exposure
5. **Temporary Files**: Ensure cleanup of sensitive data

## Performance Metrics

### Typical Processing Rates:
- Page extraction: ~10 pages/second
- Azure DI analysis: ~2-3 pages/second (with 5 concurrent)
- JSON writing: Near instantaneous
- NAS upload: Depends on network speed

### Resource Usage:
- CPU: Low (mainly I/O bound)
- Memory: ~100-500 MB depending on buffering
- Disk: Temporary space for page PDFs
- Network: Continuous for NAS and Azure API