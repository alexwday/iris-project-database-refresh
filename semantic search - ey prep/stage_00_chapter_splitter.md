# Stage 00: Chapter Splitter - Technical Documentation

## Overview
The Chapter Splitter is a PyQt6-based GUI application designed for manual chapter assignment and PDF splitting. It provides a streamlined interface for viewing markdown content extracted from PDFs, assigning chapter boundaries, and splitting the source PDF into individual chapter files while maintaining data integrity through JSON record management.

## Purpose
This tool serves as a preprocessing stage (Stage 00) for documents that require manual chapter identification before entering the main semantic search pipeline. It's specifically designed for EY documents where automatic chapter detection may be insufficient.

## Architecture

### Core Components

1. **ChapterAssignmentTool (Main Window)**
   - Central application controller
   - Manages UI state and data flow
   - Coordinates between JSON data, PDF operations, and user interactions

2. **ChapterDefinition (Data Model)**
   - Dataclass for chapter metadata
   - Fields: `chapter_number`, `start_page`, `end_page`, `chapter_name`
   - Includes validation logic for data integrity

3. **ChapterEditDialog**
   - Modal dialog for adding/editing chapter definitions
   - Provides validation and auto-suggestions for page ranges

## Input Processing

### Primary Input: JSON File
- **Format**: Flat JSON array from Azure Document Intelligence processing
- **Required Fields**:
  - `content`: Markdown content of each page
  - `page_number`: Sequential page number
  - `filename`: Source PDF filename
  - `document_id`: Document identifier
  - `page_reference`: Optional page reference extracted from content

### Secondary Input: Source PDF File
- **Purpose**: Used for splitting into chapter-specific PDFs
- **Validation**: Page count must match JSON record count
- **Format**: Standard PDF file that corresponds to the JSON data

### Input JSON Structure Example:
```json
[
  {
    "document_id": "EY_GUIDE_2024",
    "filename": "source_document.pdf",
    "filepath": "/path/to/source.pdf",
    "page_number": 1,
    "page_reference": "i",
    "content": "# Introduction\n\nThis is the content..."
  }
]
```

## Processing Workflow

### 1. Data Loading Phase
```python
load_json_file(file_path):
    - Parse JSON array
    - Extract document metadata
    - Count total pages
    - Check for existing chapter assignments
    - Initialize UI controls with page ranges
```

### 2. Content Display and Navigation
```python
go_to_page(page_num):
    - Load page data from JSON array
    - Extract and display PageHeader/PageFooter tags
    - Clean content for main display
    - Update navigation indicators
    - Highlight current chapter assignment
```

### 3. Azure Tag Handling
The tool specifically processes Azure Document Intelligence tags:

#### PageHeader/PageFooter Extraction:
```python
# Pattern for extracting header/footer values
header_pattern = r'<!--\s*PageHeader\s*[=:]\s*["\']?([^"\'>\n]+)["\']?\s*-->'
footer_pattern = r'<!--\s*PageFooter\s*[=:]\s*["\']?([^"\'>\n]+)["\']?\s*-->'

# These are displayed in fixed UI elements
# Removed from main content display for clarity
```

#### PageNumber Tag Processing:
- PageNumber tags are preserved in the content
- Used for page reference mapping
- Not removed during display

### 4. Chapter Assignment Process

#### Manual Chapter Creation:
```python
quick_add_chapter():
    - Validate chapter boundaries
    - Check for overlaps with existing chapters
    - Create ChapterDefinition object
    - Sort chapters by start_page
    - Update UI table display
```

#### Auto-Detection Logic:
```python
auto_detect_chapters():
    - Scan first 500 characters of each page
    - Look for patterns: '# Chapter', '## Chapter', 'CHAPTER'
    - Extract chapter names from markdown headers
    - Skip Azure comment tags during extraction
    - Create chapter definitions for detected boundaries
```

### 5. Chapter Name Extraction
```python
extract_chapter_name(content):
    - Skip Azure comment tags (<!--...-->)
    - Find markdown headers (#, ##, ###)
    - Remove "Chapter X:" prefixes
    - Clean numeric prefixes
    - Return meaningful title (max 100 chars)
```

### 6. PDF Splitting and Data Update

#### Phase 1: Apply Chapter Assignments
```python
apply_chapter_assignments():
    # Only modifies chapter_number and chapter_name fields
    for record in json_data:
        page_num = record.get('source_page_number')  # Use source as reference
        for chapter in chapters:
            if chapter.start_page <= page_num <= chapter.end_page:
                record['chapter_number'] = chapter.chapter_number
                record['chapter_name'] = chapter.chapter_name
```

#### Phase 2: Split PDF by Chapters
```python
split_pdf_by_chapters():
    for chapter in chapters:
        writer = PdfWriter()
        # Add pages from start_page to end_page (0-indexed)
        for page_num in range(chapter.start_page - 1, chapter.end_page):
            writer.add_page(pdf_reader.pages[page_num])
        
        # Generate filename: 00_Front_Matter.pdf, 01_Introduction.pdf
        filename = f"{chapter.chapter_number:02d}_{safe_chapter_name}.pdf"
```

#### Phase 3: Update JSON Records
```python
update_json_with_chapter_files(created_files, output_dir):
    # Preserves source_filename and source_page_number
    # Updates filename, filepath, page_number for chapter PDFs
    
    for record in json_data:
        # Preserve original reference (one-time initialization)
        if 'source_filename' not in record:
            record['source_filename'] = record['filename']
            record['source_page_number'] = record['page_number']
        
        # Update with chapter-specific file
        chapter_num = record['chapter_number']
        if chapter_num in chapter_files:
            record['filename'] = chapter_files[chapter_num]['filename']
            record['filepath'] = chapter_files[chapter_num]['filepath']
            # Sequential numbering within chapter (1, 2, 3...)
            record['page_number'] = sequential_page_in_chapter
```

## Output Generation

### Output Directory Structure:
```
chapters_YYYYMMDD_HHMMSS/
├── 00_Front_Matter.pdf
├── 01_Introduction.pdf
├── 02_Chapter_Title.pdf
├── ...
└── stage1_input.json
```

### Output JSON Structure:
```json
[
  {
    "document_id": "EY_GUIDE_2024",
    "filename": "01_Introduction.pdf",
    "filepath": "/path/to/chapters/01_Introduction.pdf",
    "page_number": 1,  // Sequential within chapter PDF
    "page_reference": "1",
    "content": "# Introduction\n\n...",
    "chapter_number": 1,
    "chapter_name": "Introduction",
    "source_filename": "original_document.pdf",  // Preserved
    "source_page_number": 15  // Original page in source PDF
  }
]
```

### Field Definitions:

#### Modified Fields:
- `filename`: Updated to chapter-specific PDF filename
- `filepath`: Updated to chapter PDF location
- `page_number`: Reset to sequential numbering within chapter (1, 2, 3...)

#### Added Fields:
- `chapter_number`: Integer chapter identifier (0 for front matter)
- `chapter_name`: Human-readable chapter title
- `source_filename`: Original PDF filename (preserved for reference)
- `source_page_number`: Original page number in source PDF

#### Preserved Fields:
- `document_id`: Document identifier
- `page_reference`: Page reference from content (e.g., "i", "ii", "1", "2")
- `content`: Markdown content (unchanged)

## Validation and Error Handling

### Chapter Validation Rules:
1. **Boundary Checks**:
   - Start page must be >= 1
   - End page must be <= total pages
   - Start page must be <= end page

2. **Overlap Detection**:
   - No two chapters can have overlapping page ranges
   - Warns about gaps between chapters

3. **Coverage Analysis**:
   - Identifies pages not assigned to any chapter
   - Warns about missing page coverage

### Data Integrity Checks:
```python
# Verify sequential order within chapters
for chapter_records in grouped_by_chapter:
    prev_source_page = None
    for record in chapter_records:
        current_source_page = record['source_page_number']
        if prev_source_page and current_source_page < prev_source_page:
            log_warning("Pages out of sequential order")
```

## UI Features

### Navigation Controls:
- Page spinner with current/total display
- First/Previous/Next/Last navigation buttons
- Go to page dialog (Ctrl+G)
- Clickable page numbers in chapter table

### Quick Add Section:
- Chapter number auto-increment
- Page range selection with count display
- "Use Current Page" button for context-aware assignment
- Auto-extraction of chapter name from current page

### Chapter Management Table:
- Columns: #, Name, Start, End, Pages
- Color coding for validation errors
- Double-click to edit
- Click on page numbers to jump

### Content Viewer:
- Fixed header/footer display for Azure tags
- Main content area with markdown text
- Search functionality with highlighting
- Chapter indicator showing current assignment

## Performance Optimizations

### Incremental Writing:
- JSON results written sequentially as pages complete
- Prevents memory overflow with large documents
- Maintains page order despite concurrent processing

### Memory Management:
- Individual page PDFs created temporarily
- Cleanup after processing each page
- Results buffered only until sequential write

### Concurrent Processing:
- ThreadPoolExecutor for parallel page analysis
- Configurable MAX_CONCURRENT_PAGES setting
- Progress tracking with tqdm

## Dependencies

### Required Libraries:
- **PyQt6**: GUI framework
- **pypdf**: PDF manipulation (reading/writing)
- **json**: Data serialization
- **logging**: Error tracking and debugging
- **tempfile**: Temporary file management
- **pathlib**: File path operations

### Optional Libraries:
- **tqdm**: Progress bar display (graceful fallback if missing)

## Configuration Requirements

### File Paths:
- Input JSON must be accessible via file dialog
- Source PDF must correspond to JSON data
- Output directory created with timestamp

### System Requirements:
- Python 3.8+
- PyQt6 installation
- Sufficient disk space for temporary files
- Write permissions for output directory

## Error Recovery

### Failed Page Handling:
- Failed pages logged but don't stop processing
- Null entries maintain sequence integrity
- Summary report of failed pages

### Transaction Safety:
- Chapter assignments applied in memory first
- PDF splitting completed before JSON update
- Original files never modified

## Usage Workflow

1. **Load JSON File**: Open JSON from Azure DI processing
2. **Load Source PDF**: Open corresponding PDF file
3. **Review Content**: Navigate pages to understand structure
4. **Define Chapters**: 
   - Use auto-detect for initial suggestions
   - Manually adjust boundaries
   - Add chapter names
5. **Validate**: Check for overlaps and gaps
6. **Save**: Creates chapter PDFs and updated JSON
7. **Output**: Ready for Stage 1 semantic search processing

## Integration with Pipeline

### Prerequisites:
- Azure Document Intelligence processing completed
- PDF and JSON files available locally or on network

### Output Compatibility:
- JSON format compatible with Stage 1 chapter processing
- Chapter PDFs maintain page integrity
- Source references preserved for traceability

### Next Stage (Stage 1):
- Reads stage1_input.json
- Processes chapter-assigned content
- Continues semantic search pipeline

## Known Limitations

1. **Manual Process**: Requires human intervention for chapter identification
2. **Single Document**: Processes one document at a time
3. **Memory Usage**: Large PDFs may require significant RAM
4. **Sequential Writing**: Output must be written in page order

## Best Practices

1. **Always validate** chapters before saving
2. **Use auto-detect** as starting point, then refine
3. **Check page coverage** to ensure no content is lost
4. **Preserve source fields** for audit trail
5. **Test with small PDFs** before processing large documents