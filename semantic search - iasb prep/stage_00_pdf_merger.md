# Stage 00: PDF Merger - Technical Documentation

## Overview
The PDF Merger is a batch processing tool designed to consolidate related IASB standards PDFs based on naming conventions. It merges base documents with their B- and C- prefixed variants into single consolidated PDFs, preparing them for Azure Document Intelligence processing.

## Purpose
This tool serves as the first preprocessing stage (Stage 00) for IASB standards documents (IAS, IFRS, IFRIC, SIC), consolidating multiple PDF variants that represent different sections or updates of the same standard into unified documents.

## Architecture

### Core Components

1. **NAS Connection Manager**
   - SMB protocol implementation using pysmb (matching EY prep)
   - Authenticated connection with retry logic
   - Directory creation and file I/O operations

2. **Filename Parser**
   - Regex-based pattern matching
   - Component extraction (prefix, standard, number, name)
   - Grouping key generation

3. **PDF Merger Engine**
   - pypdf-based page concatenation
   - Priority-based ordering (base → B → C)
   - Memory-efficient processing

## Input Processing

### NAS Configuration (Matching EY Prep)
```python
NAS_PARAMS = {
    "ip": "your_nas_ip",
    "share": "your_share_name", 
    "user": "your_nas_user",
    "password": "your_nas_password",
    "port": 445  # SMB2/3
}
```

### Input Directory Structure
```
semantic_search/source_documents/iasb/{standard}/
├── ias-1-presentation-of-financial-statements.pdf
├── B-ias-1-presentation-of-financial-statements.pdf
├── C-ias-1-presentation-of-financial-statements.pdf
├── ias-2-inventories.pdf
├── B-ias-2-inventories.pdf
├── ifrs-16-leases.pdf
├── B-ifrs-16-leases.pdf
└── C-ifrs-16-leases.pdf
```

### Filename Pattern
- **Format**: `[prefix-]standard-number-name.pdf`
- **Components**:
  - `prefix`: Optional (B or C)
  - `standard`: ias, ifrs, ifric, or sic
  - `number`: Standard number (integer)
  - `name`: Descriptive name (hyphenated)

### Examples:
- Base: `ias-2-inventories.pdf`
- B variant: `B-ias-2-inventories.pdf`
- C variant: `C-ias-2-inventories.pdf`

## Processing Workflow

### 1. Configuration and Setup
```python
def setup_logging(standard_type):
    # Create temp log file
    # Configure handlers (file + console)
    # Suppress pypdf and SMB warnings
    # Return temp log path for later upload
```

### 2. File Discovery
```python
def list_nas_directory(share_name, dir_path_relative):
    # Connect to NAS
    # List all files in directory
    # Filter for PDFs
    # Return file list
```

### 3. Filename Parsing and Grouping
```python
def parse_filename(filename):
    # Extract: prefix, standard, number, name
    # Pattern: ^(B-|C-)?([a-z]+)-(\d+)-(.+)\.pdf$
    # Generate base_key for grouping
    # Return parsed components
    
def group_files_by_standard(pdf_files, standard_type):
    # Parse all filenames
    # Filter by standard type
    # Group by base_key (standard-number-name)
    # Sort within groups: base → B → C
    # Return grouped dictionary
```

### 4. PDF Merging Logic
```python
def merge_pdf_group(pdf_paths, output_path):
    writer = PdfWriter()
    
    # Process in priority order
    for pdf_path in pdf_paths:  # [base, B, C]
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            writer.add_page(page)
    
    # Write merged PDF
    writer.write(output_path)
```

### 5. Batch Processing
```python
def process_all_pdfs(nas_input_path, nas_output_path, standard_type):
    # List all PDFs on NAS
    # Group by standard-number-name
    # Download to temp directory
    
    for base_key, file_group in groups:
        # Get files in priority order
        # Merge into single PDF
        # Upload to NAS output directory
```

## Output Generation

### Output Directory Structure
```
semantic_search/prep_output/iasb/{standard}/merged/
├── ias-1-presentation-of-financial-statements.pdf (merged)
├── ias-2-inventories.pdf (merged)
└── ifrs-16-leases.pdf (merged from base + B + C)
```

### Merge Rules:
1. **Complete Set** (base + B + C):
   - Output: base filename
   - Order: base pages → B pages → C pages

2. **Partial Set** (B + C only):
   - Output: base filename (prefix removed)
   - Order: B pages → C pages

3. **Single File** (base only):
   - Output: unchanged (copy)

### Logging Output:
```
semantic_search/prep_output/iasb/{standard}/logs/
└── {standard}_merger_YYYYMMDD_HHMMSS.log
```

## Network Configuration (Matching EY Prep)

### SMB Protocol Settings:
```python
smb_structs.SUPPORT_SMB2 = True
smb_structs.MAX_PAYLOAD_SIZE = 65536
CLIENT_HOSTNAME = socket.gethostname()

# Connection parameters
use_ntlm_v2 = True
is_direct_tcp = (port == 445)
timeout = 60
```

### SSL Certificate Configuration:
```python
NAS_SSL_CERT_PATH = "certificates/rbc-ca-bundle.cer"
SSL_LOCAL_PATH = "/tmp/rbc-ca-bundle.cer"
```

## Error Handling

### File-Level Errors:
- **Missing Base File**: Process B/C variants only
- **Invalid Filename**: Log warning, skip file
- **Download Failure**: Log error, continue with remaining files

### Group-Level Errors:
- **No Files in Group**: Skip group
- **Merge Failure**: Log error, continue with next group
- **Upload Failure**: Log error, retain local copy

### Connection Errors:
- **NAS Connection Failed**: Exit with error
- **Directory Creation Failed**: Log warning, attempt to continue

## Performance Optimizations

### Memory Management:
- Process one group at a time
- Clean up temp files after upload
- Use streaming for large PDFs

### Network Optimization:
- Single connection for directory listing
- Batch downloads before processing
- Parallel uploads where possible

### Processing Efficiency:
- Pre-sort files to minimize memory usage
- Use priority ordering to maintain consistency
- Progress bars with tqdm (if available)

## Dependencies

### Required Libraries:
- **pypdf**: PDF reading and merging
- **pysmb**: NAS/SMB protocol support
- **logging**: Error tracking
- **tempfile**: Temporary file management
- **socket**: Network hostname resolution

### Optional Libraries:
- **tqdm**: Progress bar display

## Configuration Requirements

### Standard Type Selection:
```python
STANDARD_TYPE = "ias"  # Options: "ias", "ifrs", "ifric", "sic"
```

### Path Templates:
```python
NAS_INPUT_PATH_TEMPLATE = "semantic_search/source_documents/iasb/{standard}"
NAS_OUTPUT_PATH_TEMPLATE = "semantic_search/prep_output/iasb/{standard}/merged"
NAS_LOG_PATH_TEMPLATE = "semantic_search/prep_output/iasb/{standard}/logs"
```

### System Requirements:
- Python 3.8+
- Network access to NAS
- Sufficient temp disk space
- SMB protocol support

## Usage Workflow

1. **Set Configuration**:
   ```python
   STANDARD_TYPE = "ias"  # or "ifrs", "ifric", "sic"
   ```

2. **Update NAS Credentials**:
   - Set IP, share name, username, password

3. **Execute Script**:
   ```bash
   python stage_00_pdf_merger.py
   ```

4. **Monitor Progress**:
   - Console output shows processing status
   - Progress bars for batch operations
   - Detailed logging to file

5. **Verify Output**:
   - Check merged PDFs in output directory
   - Review log file for any errors
   - Confirm page counts match expectations

## Integration with Pipeline

### Prerequisites:
- Individual PDF files organized by standard type
- Consistent naming convention
- NAS connectivity established

### Output Compatibility:
- Merged PDFs ready for Azure DI processing
- Maintains original naming structure (without prefix)
- Compatible with stage_00_standards_processor.py

### Next Stage (Standards Processor):
- Reads merged PDFs from output directory
- Processes through Azure Document Intelligence
- Creates stage1_input.json

## Validation Checks

### Pre-Processing:
1. Verify standard type configuration
2. Check NAS connectivity
3. Validate input directory exists

### During Processing:
1. Filename pattern validation
2. Group consistency checks
3. Page count verification

### Post-Processing:
1. Output file existence
2. Merged PDF integrity
3. Upload confirmation

## Known Limitations

1. **Single Standard Type**: Processes one type per run
2. **Sequential Processing**: Groups processed one at a time
3. **Memory Usage**: Large PDFs may require significant RAM
4. **Network Dependency**: Requires stable NAS connection

## Best Practices

1. **Test with Small Set**: Verify configuration with few files first
2. **Monitor Logs**: Check for warnings about missing files
3. **Verify Merges**: Spot-check merged PDFs for correctness
4. **Backup Originals**: Keep source files unchanged
5. **Clean Temp Files**: Ensure temp directory cleanup

## Troubleshooting

### Common Issues:

1. **"No PDF files found"**
   - Check input path configuration
   - Verify standard type matches files
   - Ensure NAS connectivity

2. **"Failed to merge PDFs"**
   - Check file corruption
   - Verify pypdf installation
   - Review memory availability

3. **"Failed to upload to NAS"**
   - Check write permissions
   - Verify network stability
   - Ensure output directory exists

4. **Incorrect Merge Order**
   - Verify filename patterns
   - Check prefix priority logic
   - Review group sorting

## Example Processing Log

```
2024-01-15 10:00:00 - INFO - Starting IASB PDF Merger for IAS standards
2024-01-15 10:00:01 - INFO - Found 15 PDF files to process
2024-01-15 10:00:02 - INFO - Found 5 unique IAS standards to process
2024-01-15 10:00:03 - INFO - Processing group: ias-1-presentation-of-financial-statements
2024-01-15 10:00:03 - INFO -   Adding: ias-1-presentation-of-financial-statements.pdf
2024-01-15 10:00:04 - INFO -   Adding: B-ias-1-presentation-of-financial-statements.pdf
2024-01-15 10:00:05 - INFO -   Adding: C-ias-1-presentation-of-financial-statements.pdf
2024-01-15 10:00:06 - INFO -   Merged 3 files into ias-1-presentation-of-financial-statements.pdf (150 total pages)
2024-01-15 10:00:10 - INFO - Successfully uploaded 5/5 merged PDFs
```