# Stage 00: Standards Processor - Technical Documentation

## Overview
The Standards Processor is an automated batch processing tool that converts merged IASB standards PDFs into structured JSON format using Azure Document Intelligence. It assigns chapter numbers based on standard numbers, processes each page to markdown, and creates output matching the EY prep pipeline format.

## Purpose
This tool serves as the second preprocessing stage (Stage 00) for IASB standards, taking merged PDFs from the PDF Merger and producing a stage1_input.json file ready for the semantic search pipeline.

## Architecture

### Core Components

1. **NAS Connection Manager** (Matching EY Prep)
   - Identical SMB implementation from EY prep
   - Connection pooling and retry logic
   - Secure file transfer operations

2. **PDF Sorter and Chapter Assigner**
   - Standard number extraction
   - Numeric sorting algorithm
   - Sequential chapter numbering

3. **Azure DI Processor** (From EY Prep)
   - Prebuilt-layout model usage
   - Concurrent page processing
   - PageNumber tag extraction

4. **JSON Generator**
   - EY-compatible format output
   - Incremental writing capability
   - Field mapping and transformation

## Input Processing

### Input Source
```
semantic_search/prep_output/iasb/{standard}/merged/
├── ias-1-presentation-of-financial-statements.pdf
├── ias-2-inventories.pdf
├── ias-7-statement-of-cash-flows.pdf
├── ias-8-accounting-policies.pdf
└── ias-12-income-taxes.pdf
```

### Filename Parsing
```python
def parse_merged_filename(filename):
    # Pattern: standard-number-name.pdf
    # Example: ias-2-inventories.pdf
    
    pattern = r'^([a-z]+)-(\d+)-(.+)\.pdf$'
    # Returns:
    {
        'standard': 'ias',
        'number': 2,
        'name': 'inventories',
        'name_formatted': 'Inventories'
    }
```

## Processing Workflow

### 1. PDF Discovery and Sorting
```python
def sort_merged_pdfs(pdf_files):
    # Parse all filenames
    parsed_files = [parse_merged_filename(f) for f in pdf_files]
    
    # Sort by standard number
    parsed_files.sort(key=lambda x: x['number'])
    
    # Use actual standard number as chapter number
    for parsed in parsed_files:
        chapter_number = parsed['number']  # e.g., 2, 7, 12
        chapter_name = f"{standard.upper()} {number} - {name_formatted}"
        # Returns: (filename, chapter_number, chapter_name)
```

### Example Sorting:
```
Input Order:              Sorted Order (Chapter #):
ias-8-accounting.pdf  →   Chapter 1: IAS 1 - Presentation
ias-2-inventories.pdf →   Chapter 2: IAS 2 - Inventories
ias-1-presentation.pdf →  Chapter 7: IAS 7 - Cash Flows
ias-12-income-taxes.pdf → Chapter 8: IAS 8 - Accounting
ias-7-cash-flows.pdf  →   Chapter 12: IAS 12 - Income Taxes
```

Note: Chapter numbers match the actual standard numbers (not sequential)

### 2. Page Extraction (From EY Prep)
```python
def extract_individual_pages(local_pdf_path, temp_dir):
    reader = PdfReader(local_pdf_path)
    page_files = []
    
    for page_num in range(len(reader.pages)):
        writer = PdfWriter()
        writer.add_page(reader.pages[page_num])
        
        # Save individual page PDF
        page_path = f"{base_name}_page_{page_num + 1}.pdf"
        writer.write(page_path)
        
        page_files.append({
            'page_number': page_num + 1,
            'file_path': page_path
        })
```

### 3. Azure Document Intelligence Processing (From EY Prep)
```python
def analyze_document_with_di(di_client, page_file_path):
    # Retry logic (3 attempts, 5 second delay)
    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            poller = di_client.begin_analyze_document(
                "prebuilt-layout",
                document_bytes,
                output_content_format=DocumentContentFormat.MARKDOWN
            )
            return poller.result()
        except Exception:
            if attempt < API_RETRY_ATTEMPTS - 1:
                time.sleep(API_RETRY_DELAY)
```

### 4. PageNumber Tag Processing (From EY Prep)
```python
def extract_and_clean_page_number(content):
    # Extract PageNumber value
    match = PAGE_NUMBER_EXTRACT_PATTERN.search(content)
    page_reference = match.group(1) if match else None
    
    # Remove PageNumber tags from content
    cleaned_content = PAGE_NUMBER_REMOVE_PATTERN.sub('', content)
    
    return cleaned_content, page_reference
```

### 5. Concurrent Processing with ThreadPoolExecutor
```python
def process_chapter_pdf(di_client, local_pdf_path, chapter_info):
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PAGES) as executor:
        # Submit all pages for processing
        futures = {
            executor.submit(process_single_page, di_client, page): page
            for page in page_files
        }
        
        # Collect results maintaining order
        for future in as_completed(futures):
            result = future.result()
            all_results.append(result)
    
    # Sort by page number for sequential output
    all_results.sort(key=lambda x: x['page_number'])
```

### 6. JSON Record Creation
```python
def create_json_record(page_result, chapter_info, page_idx):
    return {
        'document_id': 'IAS_2024',
        'filename': '01_ias_1_presentation.pdf',  # Chapter PDF
        'filepath': 'path/to/01_ias_1_presentation.pdf',
        'page_number': page_idx,  # Sequential within chapter
        'page_reference': page_result['page_reference'],  # From Azure
        'content': page_result['content'],  # Markdown
        'chapter_number': chapter_info['number'],
        'chapter_name': chapter_info['name'],
        'source_filename': 'ias-1-presentation.pdf',  # Merged PDF
        'source_page_number': page_result['original_page']
    }
```

## Output Generation

### Output Directory Structure
```
semantic_search/prep_output/iasb/{standard}/chapters_YYYYMMDD_HHMMSS/
├── 01_ias_1_presentation_of_financial_statements.pdf
├── 02_ias_2_inventories.pdf
├── 03_ias_7_statement_of_cash_flows.pdf
├── ...
└── stage1_input.json
```

### Chapter PDF Naming Convention:
- **Format**: `{chapter_num:02d}_{standard}_{number}_{name}.pdf`
- **Example**: `01_ias_1_presentation_of_financial_statements.pdf` (for IAS 1)
- **Example**: `07_ias_7_statement_of_cash_flows.pdf` (for IAS 7)
- **Example**: `12_ias_12_income_taxes.pdf` (for IAS 12)

### JSON Output Format (Matching EY Prep):
```json
[
  {
    "document_id": "IAS_2024",
    "filename": "01_ias_1_presentation_of_financial_statements.pdf",
    "filepath": "semantic_search/prep_output/iasb/ias/chapters_20240115_100000/01_ias_1_presentation_of_financial_statements.pdf",
    "page_number": 1,
    "page_reference": "1",
    "content": "# IAS 1\n## Presentation of Financial Statements\n\n...",
    "chapter_number": 1,
    "chapter_name": "IAS 1 - Presentation of Financial Statements",
    "source_filename": "ias-1-presentation-of-financial-statements.pdf",
    "source_page_number": 1
  },
  {
    "document_id": "IAS_2024",
    "filename": "01_ias_1_presentation_of_financial_statements.pdf",
    "filepath": "semantic_search/prep_output/iasb/ias/chapters_20240115_100000/01_ias_1_presentation_of_financial_statements.pdf",
    "page_number": 2,
    "page_reference": "2",
    "content": "## Objective\n\nThis Standard prescribes...",
    "chapter_number": 1,
    "chapter_name": "IAS 1 - Presentation of Financial Statements",
    "source_filename": "ias-1-presentation-of-financial-statements.pdf",
    "source_page_number": 2
  }
]
```

### Field Mappings:

#### Standard Fields (From EY Prep):
- `document_id`: "{STANDARD}_2024" (e.g., "IAS_2024", "IFRS_2024")
- `page_reference`: Extracted from PageNumber Azure tag
- `content`: Markdown with PageNumber tags removed

#### Chapter Fields (New Logic):
- `chapter_number`: Actual standard number (e.g., 1, 2, 7, 12)
- `chapter_name`: Formatted as "{STANDARD} {number} - {name}"
- `filename`: Chapter-specific PDF with standard number prefix
- `page_number`: Sequential within chapter (1, 2, 3...)

#### Source Tracking (Matching EY Prep):
- `source_filename`: Original merged PDF name
- `source_page_number`: Page number in merged PDF

## Azure Configuration (Matching EY Prep)

### Client Initialization:
```python
di_client = DocumentIntelligenceClient(
    endpoint=AZURE_DI_ENDPOINT,
    credential=AzureKeyCredential(AZURE_DI_KEY)
)
```

### Processing Parameters:
```python
MAX_CONCURRENT_PAGES = 5
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY = 5  # seconds
```

### Model Configuration:
- Model: `prebuilt-layout`
- Output Format: `DocumentContentFormat.MARKDOWN`

## Network Configuration (Identical to EY Prep)

### SMB Settings:
```python
smb_structs.SUPPORT_SMB2 = True
smb_structs.MAX_PAYLOAD_SIZE = 65536
CLIENT_HOSTNAME = socket.gethostname()
use_ntlm_v2 = True
is_direct_tcp = (port == 445)
```

### SSL Certificate:
```python
NAS_SSL_CERT_PATH = "certificates/rbc-ca-bundle.cer"
SSL_LOCAL_PATH = "/tmp/rbc-ca-bundle.cer"
```

## Error Handling

### Page-Level Failures:
- Failed pages logged but don't stop processing
- Retry logic for Azure DI calls
- Continue with remaining pages

### Chapter-Level Failures:
- Skip chapter if all pages fail
- Log warning and continue
- Track failed chapters in summary

### Connection Failures:
- NAS: Retry with exponential backoff
- Azure DI: 3 retries with 5-second delay
- Exit gracefully with error logging

## Performance Optimizations

### Concurrent Processing:
- `MAX_CONCURRENT_PAGES = 5` for Azure DI calls
- ThreadPoolExecutor for managed concurrency
- Sequential processing per chapter

### Memory Management:
- Process one chapter at a time
- Clean up temp page files immediately
- Stream large files when possible

### Network Efficiency:
- Batch NAS operations
- Reuse connections where possible
- Local caching during processing

## Dependencies

### Required Libraries (Matching EY Prep):
- **pypdf**: PDF manipulation
- **azure-ai-documentintelligence**: Azure DI client
- **azure-core**: Azure authentication
- **pysmb**: NAS/SMB protocol
- **concurrent.futures**: Parallel processing

### Optional Libraries:
- **tqdm**: Progress bars

## Configuration Requirements

### Standard Type:
```python
STANDARD_TYPE = "ias"  # Options: "ias", "ifrs", "ifric", "sic"
DOCUMENT_ID_TEMPLATE = "{STANDARD}_2024"
```

### Path Configuration:
```python
NAS_INPUT_PATH_TEMPLATE = "semantic_search/prep_output/iasb/{standard}/merged"
NAS_OUTPUT_PATH_TEMPLATE = "semantic_search/prep_output/iasb/{standard}"
```

### Azure Credentials:
```python
AZURE_DI_ENDPOINT = "YOUR_ENDPOINT"
AZURE_DI_KEY = "YOUR_KEY"
```

## Usage Workflow

1. **Ensure PDF Merger Has Run**:
   - Verify merged PDFs exist in input directory

2. **Configure Settings**:
   ```python
   STANDARD_TYPE = "ias"  # Match merger configuration
   ```

3. **Set Azure Credentials**:
   - Update endpoint and key

4. **Execute Script**:
   ```bash
   python stage_00_standards_processor.py
   ```

5. **Monitor Progress**:
   - Progress bars per chapter
   - Console output for status
   - Detailed logging to file

6. **Verify Output**:
   - Check stage1_input.json
   - Verify chapter PDFs
   - Review processing log

## Integration with Pipeline

### Prerequisites:
- Merged PDFs from stage_00_pdf_merger.py
- Azure DI service configured
- NAS connectivity established

### Output Compatibility:
- JSON format matches EY prep exactly
- Compatible with Stage 1 processing
- Maintains field consistency

### Next Stage (Stage 1):
- Reads stage1_input.json
- Processes chapter-assigned content
- Continues semantic search pipeline

## Validation Checks

### Input Validation:
1. Verify merged PDFs exist
2. Check filename patterns
3. Validate standard type

### Processing Validation:
1. Page count preservation
2. Content extraction success rate
3. Chapter assignment accuracy

### Output Validation:
1. JSON schema compliance
2. All required fields present
3. Chapter sequence integrity

## Known Limitations

1. **Sequential Chapter Processing**: One chapter at a time
2. **API Rate Limits**: Subject to Azure DI quotas
3. **Memory Usage**: Large chapters may require significant RAM
4. **Network Dependency**: Requires stable connections

## Best Practices

1. **Run After Merger**: Ensure PDF merger completes first
2. **Monitor API Usage**: Track Azure DI quota consumption
3. **Verify Sorting**: Check chapter number assignments
4. **Test Small Set**: Validate with few standards first
5. **Review Logs**: Check for warnings and failed pages

## Troubleshooting

### Common Issues:

1. **"No PDF files found"**
   - Check merger output directory
   - Verify standard type configuration
   - Ensure path templates correct

2. **"Azure DI client initialization failed"**
   - Verify endpoint URL
   - Check API key validity
   - Ensure network connectivity

3. **"Failed to process chapter"**
   - Check PDF corruption
   - Review Azure DI limits
   - Verify memory availability

4. **Incorrect Chapter Numbers**
   - Review sort algorithm
   - Check standard number parsing
   - Verify filename patterns

## Performance Metrics

### Typical Processing Rates:
- Page extraction: ~10 pages/second
- Azure DI analysis: ~2-3 pages/second (with concurrency)
- JSON writing: Near instantaneous
- NAS operations: Network dependent

### Resource Usage:
- CPU: Moderate (Azure DI calls)
- Memory: 200-500 MB typical
- Disk: Temp space for page PDFs
- Network: Continuous for NAS and Azure

## Example Processing Log

```
2024-01-15 10:30:00 - INFO - Starting IASB Standards Processor for IAS
2024-01-15 10:30:01 - INFO - Found 5 merged PDFs to process
2024-01-15 10:30:02 - INFO - Chapter 1: IAS 1 - Presentation of Financial Statements
2024-01-15 10:30:02 - INFO - Chapter 2: IAS 2 - Inventories
2024-01-15 10:30:03 - INFO - Chapter 3: IAS 7 - Statement of Cash Flows
2024-01-15 10:30:10 - INFO - Processing Chapter 1: IAS 1 - Presentation
2024-01-15 10:30:11 - INFO - Extracting 45 pages from PDF...
2024-01-15 10:31:30 - INFO - Chapter 1: 45/45 pages successful
2024-01-15 10:35:00 - INFO - Creating stage1_input.json with 225 records
2024-01-15 10:35:05 - INFO - Successfully uploaded stage1_input.json
2024-01-15 10:35:05 - INFO - Output location: chapters_20240115_103000/
```