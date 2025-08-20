# Stage 2.5: Page Boundary Correction - Input/Output Schema Documentation

## 1. Overview and Purpose

Stage 2.5 corrects page boundaries for sections after Stage 2 processing by:

- **Page Boundary Correction**: Uses embedded HTML page tags to determine actual page ranges for each section
- **Position-Based Mapping**: Builds a position map of sections in concatenated content to accurately assign pages
- **Page Reference Extraction**: Extracts and adds page references (e.g., "1-1", "xv") from HTML tags
- **Page Count Validation**: Ensures section_page_count accurately reflects (end_page - start_page + 1)
- **Large Section Detection**: Identifies sections >5 pages that may need splitting for better chunking
- **Continuity Validation**: Detects and fixes gaps/overlaps between consecutive sections

**Key Transformation**: Corrects potentially incorrect page boundaries from Stage 2 while preserving all content and adding page reference metadata.

## 2. Input Schema

Stage 2.5 expects JSON input from Stage 2 output (`stage2_section_records.json`).

### Input Fields (per record):
```json
{
  "document_id": "EY_GUIDE_2024",                     // Document identifier
  "filename": "01_Chapter_Name.pdf",                  // Chapter-specific PDF filename  
  "filepath": "/path/to/chapters/01_Chapter_Name.pdf", // Full path to chapter PDF
  "source_filename": "ey_original.pdf",               // Original PDF filename
  "chapter_number": 1,                                // Chapter number
  "chapter_name": "Introduction to IFRS",             // Chapter name/title
  "chapter_summary": "This chapter provides...",      // GPT-generated chapter summary
  "chapter_page_count": 25,                           // Total pages in chapter
  "section_number": 2,                                // Sequential section number within chapter
  "section_summary": "Introduction to IFRS > Recognition Criteria: Explains...", // Hierarchical summary
  "section_start_page": 3,                            // First page number of section (may be incorrect)
  "section_end_page": 7,                              // Last page number of section (may be incorrect)  
  "section_page_count": 5,                            // Total pages in section (may be incorrect)
  "section_content": "<!-- PageHeader PageNumber=\"3\" PageReference=\"1-5\" -->\n# Recognition Criteria\n..." // Content with embedded page tags
}
```

### Input Requirements:
- All records must have completed Stage 2 processing (includes `section_content` with embedded page tags)
- Section content must contain HTML page boundary tags in the format:
  - `<!-- PageHeader PageNumber="N" PageReference="ref" -->`
  - `<!-- PageFooter PageNumber="N" PageReference="ref" -->`
- Records should be grouped by chapter for optimal processing
- Page boundaries (`section_start_page`, `section_end_page`) may be incorrect and need correction

## 3. Output Schema

Stage 2.5 outputs section-level records with corrected page boundaries and added page references.

### Output Fields (per record):
```json
{
  "document_id": "EY_GUIDE_2024",                     // [Preserved] Document identifier
  "filename": "01_Chapter_Name.pdf",                  // [Preserved] Chapter filename  
  "filepath": "/path/to/chapters/01_Chapter_Name.pdf", // [Preserved] Chapter filepath
  "source_filename": "ey_original.pdf",               // [Preserved] Source filename
  "chapter_number": 1,                                // [Preserved] Chapter number
  "chapter_name": "Introduction to IFRS",             // [Preserved] Chapter name
  "chapter_summary": "This chapter provides...",      // [Preserved] Chapter summary
  "chapter_page_count": 25,                           // [Preserved] Chapter page count
  "section_number": 2,                                // [Preserved] Section number
  "section_summary": "Introduction to IFRS > Recognition Criteria: Explains...", // [Preserved] Section summary
  "section_start_page": 3,                            // [CORRECTED] Actual first page based on tags
  "section_end_page": 5,                              // [CORRECTED] Actual last page based on tags
  "section_page_count": 3,                            // [CORRECTED] Recalculated: end - start + 1
  "section_start_reference": "1-5",                   // [NEW] Page reference for start page
  "section_end_reference": "1-7",                     // [NEW] Page reference for end page
  "section_content": "<!-- PageHeader PageNumber=\"3\" PageReference=\"1-5\" -->\n# Recognition Criteria\n..." // [Preserved] Content unchanged
}
```

## 4. New and Modified Fields

### New Fields Added by Stage 2.5:

1. **`section_start_reference`** (string)
   - Page reference extracted from the page tag for the section's start page
   - Examples: "1-1", "2-15", "xv", "A-1", "Introduction-3"
   - Empty string if no page tag found for the start page

2. **`section_end_reference`** (string)
   - Page reference extracted from the page tag for the section's end page
   - Examples: "1-5", "2-20", "xvii", "B-12", "Appendix-7"
   - Empty string if no page tag found for the end page

### Corrected Fields:

1. **`section_start_page`** (integer)
   - Corrected based on actual page tags found in content
   - Uses position-based mapping to determine which pages the section overlaps
   - First section always starts at page 1 (first section rule)

2. **`section_end_page`** (integer)
   - Corrected based on actual page tags found in content
   - Determined by the last page tag appearing within the section's content

3. **`section_page_count`** (integer)
   - Always recalculated as: `section_end_page - section_start_page + 1`
   - Ensures consistency after all corrections

### Optional Debug Fields (when DEBUG_MODE = True):

1. **`original_start_page`** (integer or null)
   - The original `section_start_page` value from Stage 2 input
   - Used for comparison and debugging

2. **`original_end_page`** (integer or null)
   - The original `section_end_page` value from Stage 2 input
   - Used for comparison and debugging

3. **`page_boundary_method`** (string)
   - Method used to determine page boundaries:
     - `"tag_found"`: Pages determined from actual page tags in content
     - `"position_inferred"`: Pages inferred from neighboring sections
     - `"first_section_rule"`: First section set to page 1
     - `"continuity_fixed"`: Adjusted to fix overlap with next section

4. **`page_correction_applied`** (boolean)
   - `true` if page boundaries were changed from input values
   - `false` if boundaries were already correct

## 5. Processing Logic

### Page Boundary Correction Algorithm

1. **Position Mapping**: 
   - Concatenates all section content within a chapter
   - Maps each section to its character position range in concatenated content

2. **Tag Extraction**:
   - Extracts all page header/footer tags with their positions
   - Records page numbers and references from tags

3. **Page Range Building**:
   - Processes tags sequentially to build page ranges
   - Each page spans from its first tag to its last tag
   - Handles duplicate headers (ignores) and missing footers gracefully

4. **Section-to-Page Mapping**:
   - Determines which pages each section overlaps based on position ranges
   - Assigns `section_start_page` = minimum overlapping page
   - Assigns `section_end_page` = maximum overlapping page

5. **Inference for Untagged Sections**:
   - Sections without direct page overlaps are inferred from neighbors
   - Uses position between previous and next sections to assign pages
   - First section always starts at page 1 if no tags found

6. **Continuity Validation**:
   - Checks for gaps between consecutive sections
   - Fixes overlaps by adjusting section end pages
   - Ensures monotonic page progression through chapter

### Page Reference Extraction

- Extracts `PageReference` attribute from HTML tags
- Prefers header references over footer references for the same page
- Stores references in a dictionary mapped by page number
- Applied to all sections, even those not needing correction

## 6. Edge Case Handling

### Duplicate Page Tags
- **Detection**: Identifies back-to-back identical headers
- **Behavior**: Ignores duplicate headers for the same page number
- **Impact**: Prevents incorrect page range extension

### Sections Without Tags
- **Detection**: Sections with no page header/footer tags in content
- **Behavior**: Infers pages based on position between neighboring sections
- **Fallback**: If first section, starts at page 1; if last, continues from previous

### Missing Headers or Footers
- **Page with only footer**: Start position calculated from previous page's end or content beginning
- **Page with only header**: End position calculated from next page's start or content end
- **Robust detection**: Uses direct search for "-->" to find tag ends (fallback to regex)

### Content Between Pages
- **Detection**: Content appearing after a footer but before next header
- **Behavior**: Not assigned to any specific page (correctly represents inter-page content)
- **Inference**: Sections in these gaps are assigned based on neighboring sections

### Large Sections (>5 pages)
- **Detection**: Identifies sections with `section_page_count > 5`
- **Reporting**: Lists affected chapters and section details
- **Purpose**: Flags potential candidates for splitting in downstream processing

### First Section Rule
- **Behavior**: First section of each chapter always starts at page 1
- **Override**: Even if tags suggest otherwise, logs warning but maintains document structure

## 7. Statistics and Reporting

### Processing Statistics
- `total_chapters`: Number of chapters processed
- `total_sections`: Total sections across all chapters
- `sections_corrected`: Sections with changed page boundaries
- `sections_inferred`: Sections with pages determined by inference
- `sections_with_tags`: Sections with page tags found in content
- `sections_without_tags`: Sections with no page tags
- `chapters_without_tags`: Chapters with no page tags in any section

### Large Section Detection
- `large_sections`: List of sections with >5 pages
- `chapters_with_large_sections`: Chapters containing large sections
- Each large section entry includes:
  - `chapter`: Chapter number
  - `section`: Section number
  - `page_count`: Number of pages
  - `pages`: Range string (e.g., "3-10")

## 8. Configuration Parameters

### File Paths (Local Version)
- **Input**: Configurable via `-i` flag (default: `stage2_section_records.json`)
- **Output**: Configurable via `-o` flag (default: `stage2_5_corrected_sections.json`)

### File Paths (NAS Version)
- **Input Path**: `semantic_search/pipeline_output/stage2/stage2_section_records.json`
- **Output Path**: `semantic_search/pipeline_output/stage2_5/stage2_5_corrected_sections.json`
- **Log Path**: `semantic_search/pipeline_output/logs/`

### Processing Options
- `VERBOSE_LOGGING`: Enable detailed debug output (default: False)
- `DEBUG_MODE`: Include debug fields in output (default: False)
- `SAMPLE_CHAPTER_LIMIT`: Process only first N chapters (default: None)

### NAS Configuration (NAS Version)
- Same NAS parameters as Stage 2 (IP, share, credentials, port)
- Automatic directory creation if output path doesn't exist
- Log file upload with timestamp

## 9. Example Records

### Input Record (from Stage 2):
```json
{
  "document_id": "EY_GUIDE_2024",
  "filename": "04_Financial_Instruments.pdf",
  "filepath": "/nas/chapters/04_Financial_Instruments.pdf",
  "source_filename": "ey_ifrs_guide_2024.pdf",
  "chapter_number": 4,
  "chapter_name": "Financial Instruments Under IFRS 9",
  "chapter_summary": "This chapter addresses IFRS 9 requirements...",
  "chapter_page_count": 67,
  "section_number": 3,
  "section_summary": "Financial Instruments > Impairment: Explains ECL model...",
  "section_start_page": 25,  // Incorrect - should be 23
  "section_end_page": 28,    // Incorrect - should be 29
  "section_page_count": 4,   // Will be recalculated to 7
  "section_content": "<!-- PageHeader PageNumber=\"23\" PageReference=\"4-35\" -->\n## Impairment Requirements\n...\n<!-- PageFooter PageNumber=\"29\" PageReference=\"4-41\" -->"
}
```

### Output Record (after Stage 2.5):
```json
{
  "document_id": "EY_GUIDE_2024",
  "filename": "04_Financial_Instruments.pdf",
  "filepath": "/nas/chapters/04_Financial_Instruments.pdf",
  "source_filename": "ey_ifrs_guide_2024.pdf",
  "chapter_number": 4,
  "chapter_name": "Financial Instruments Under IFRS 9",
  "chapter_summary": "This chapter addresses IFRS 9 requirements...",
  "chapter_page_count": 67,
  "section_number": 3,
  "section_summary": "Financial Instruments > Impairment: Explains ECL model...",
  "section_start_page": 23,         // Corrected based on tags
  "section_end_page": 29,           // Corrected based on tags
  "section_page_count": 7,          // Recalculated: 29 - 23 + 1
  "section_start_reference": "4-35", // NEW: Extracted from page 23 header
  "section_end_reference": "4-41",   // NEW: Extracted from page 29 footer
  "section_content": "<!-- PageHeader PageNumber=\"23\" PageReference=\"4-35\" -->\n## Impairment Requirements\n...\n<!-- PageFooter PageNumber=\"29\" PageReference=\"4-41\" -->"
}
```

## 10. Integration Notes

### What Stage 3 Should Expect
- **Corrected Page Boundaries**: Accurate `section_start_page` and `section_end_page` values
- **Page References**: New fields `section_start_reference` and `section_end_reference` for citation purposes
- **Validated Page Counts**: `section_page_count` guaranteed to equal (end - start + 1)
- **Large Section Warnings**: Sections >5 pages flagged in processing statistics
- **Clean Schema**: Output matches input schema (no debug fields by default)

### Important Considerations for Stage 3
1. **Page References**: Can be used for precise citation and cross-referencing
2. **Large Sections**: Consider splitting sections with >5 pages for optimal chunking
3. **Content Integrity**: Section content is unchanged - only metadata is corrected
4. **Continuity**: Sections within a chapter have continuous, non-overlapping page ranges

### Data Quality Guarantees
- All sections have valid, corrected page boundaries
- Page counts are mathematically consistent
- First section of each chapter starts at page 1
- No gaps in page coverage within chapters (after continuity fixes)
- Page references preserved for downstream citation needs

## 11. Error Handling and Fallbacks

### No Page Tags Found
- **Chapter Level**: If no tags in entire chapter, preserves original boundaries
- **Section Level**: Infers from neighboring sections or applies first section rule
- **Logging**: Warns about sections/chapters without tags

### Invalid Page Numbers
- **Non-numeric**: Skipped during processing (relies on integer page numbers)
- **Duplicate resolution**: Prefers header references over footer references

### Position Mapping Failures
- **Empty content**: Treats as zero-length segment in position map
- **Missing sections**: Logs warning and continues with available sections

### Large Section Detection
- **Threshold**: Hardcoded at 5 pages (not configurable)
- **Reporting limit**: Shows first 10 large sections in summary
- **No automatic splitting**: Only reports for manual review

## 12. Performance Characteristics

- **Processing Speed**: ~1-2 seconds per chapter (depends on content size)
- **Memory Usage**: Proportional to largest chapter (content held in memory)
- **Scalability**: Linear with number of sections
- **No API Calls**: Pure algorithmic processing (no external dependencies)
- **Parallel Potential**: Chapters can be processed independently