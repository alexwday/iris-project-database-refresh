# Stage 2: Section Processing - Input/Output Schema Documentation

## 1. Overview and Purpose

Stage 2 transforms page-level data from Stage 1 into section-level records by:

- **Section Identification**: Uses heading patterns (levels 1-3) to identify distinct sections within chapters
- **Content Aggregation**: Combines pages that belong to the same section while preserving natural page boundaries
- **Page Tag Embedding**: Embeds HTML page tags to show actual page transitions within section content
- **GPT Summary Generation**: Creates hierarchical section summaries using GPT-4 with contextual awareness
- **Data Enrichment**: Adds section-specific metadata including page ranges and token counts

**Key Transformation**: Converts multiple page-level records per chapter into fewer section-level records, where each section may span multiple pages.

## 2. Input Schema

Stage 2 expects JSON input from Stage 1 output (`stage1_page_records.json`).

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
  "page_number": 1,                                   // Page number within chapter PDF
  "page_reference": "2-15",                           // Original page reference
  "source_page_number": 45,                           // Original document page number
  "content": "# Section Title\nMarkdown content..."   // Page content in markdown
}
```

### Input Requirements:
- All input records must have completed Stage 1 processing (includes `chapter_summary` and `chapter_page_count`)
- Records are grouped by `chapter_number` for section processing
- Pages should be ordered by `source_page_number` within each chapter
- Unassigned pages (`chapter_number: null`) are skipped

### Example Stage 1 Output Record:
```json
{
  "document_id": "EY_GUIDE_2024",
  "filename": "03_Lease_Accounting.pdf", 
  "filepath": "/nas/chapters/03_Lease_Accounting.pdf",
  "source_filename": "ey_ifrs_guide_2024.pdf",
  "chapter_number": 3,
  "chapter_name": "Lease Accounting Under IFRS 16",
  "chapter_summary": "This chapter covers IFRS 16 lease accounting requirements including lease identification, initial measurement, and subsequent measurement. It addresses key implementation challenges for lessees and lessors.",
  "chapter_page_count": 42,
  "page_number": 15,
  "page_reference": "3-47",
  "source_page_number": 89,
  "content": "## Lease Modification Accounting\n\nLease modifications under IFRS 16 require specific accounting treatment..."
}
```

## 3. Output Schema

Stage 2 outputs section-level records with embedded page information and GPT-generated summaries.

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
  "section_number": 2,                                // [NEW] Sequential section number within chapter
  "section_summary": "Introduction to IFRS > Recognition Criteria: Explains IFRS recognition criteria including asset definition requirements and measurement reliability thresholds. Details the fundamental principles for determining when items qualify for balance sheet recognition.", // [NEW] Hierarchical summary with GPT content
  "section_start_page": 3,                            // [NEW] First page number of section
  "section_end_page": 7,                              // [NEW] Last page number of section  
  "section_page_count": 5,                            // [NEW] Total pages in section
  "section_content": "<!-- PageHeader PageNumber=\"3\" PageReference=\"1-5\" -->\n# Recognition Criteria\n\nThe fundamental principle...\n<!-- PageFooter PageNumber=\"3\" PageReference=\"1-5\" -->\n<!-- PageHeader PageNumber=\"4\" PageReference=\"1-6\" -->\n\nAdditional guidance on recognition...\n<!-- PageFooter PageNumber=\"4\" PageReference=\"1-6\" -->" // [NEW] Content with embedded page tags
}
```

## 4. Special Field Formats

### a. section_content Field Format

The `section_content` field contains markdown content with embedded HTML page boundary tags:

```html
<!-- PageHeader PageNumber="15" PageReference="xv" -->
[actual markdown content of the page here]
<!-- PageFooter PageNumber="15" PageReference="xv" -->
<!-- PageHeader PageNumber="16" PageReference="xvi" -->  
[content from next page continues here]
<!-- PageFooter PageNumber="16" PageReference="xvi" -->
```

**Format Details**:
- **PageNumber**: Sequential page number within chapter (integer) - matches `page_number` from Stage 1
- **PageReference**: Original page reference from source document - matches `page_reference` from Stage 1 
- **Content Preservation**: All existing markdown formatting is preserved between tags
- **Multiple Pages**: When sections span multiple pages, all pages are concatenated with their respective header/footer tags
- **HTML Escaping**: Page reference values are HTML-escaped in attributes (e.g., `"3-5"` remains unescaped, special characters get escaped)

### b. section_summary Field Format

The `section_summary` field follows a strict hierarchical breadcrumb format:

```
"Chapter Name > Section Title: [2-3 sentence summary generated by GPT-4]. [Second sentence providing more detail]."
```

**Format Examples**:
- `"Lease Accounting Under IFRS 16 > Initial Recognition: Explains the requirements for initial recognition of lease liabilities and right-of-use assets under IFRS 16. Details the measurement approaches and practical expedients available to lessees."`
- `"Revenue Recognition > Performance Obligations > Identification Methods: Describes the five-step model for identifying distinct performance obligations in customer contracts. Covers bundling criteria and stand-alone selling price considerations."`

**Format Rules**:
- **Hierarchy Separator**: ` > ` (space-greater-than-space)
- **Summary Separator**: `: ` (colon-space) between hierarchy and summary content
- **Length**: Always 2-3 sentences generated by GPT-4
- **Content**: Naturally embeds accounting standards (e.g., "IFRS 16", "ASC 842") and key technical concepts
- **Tense**: Present tense, third person
- **Standards Integration**: Technical references integrated naturally into narrative (not as appendices)

### c. section_number Field

- **Format**: Integer starting at 1
- **Scope**: Sequential within each chapter only  
- **Reset Behavior**: Resets to 1 for each new chapter
- **Ordering**: Reflects document order based on first heading appearance

## 5. Processing Logic

### Section Identification Algorithm
1. **Pattern Matching**: Uses regex `^(#{1,3})\s+(.+)$` to find headings (levels 1-3 only)
2. **Introduction Section**: If no heading found at start of chapter, creates "Introduction" section with chapter name as title
3. **Content Segmentation**: Splits chapter content at each heading boundary
4. **Level Detection**: Heading level determines hierarchy (# = level 1, ## = level 2, ### = level 3)

### Page Tag Embedding Process
1. **Tag Cleaning**: Removes any existing Azure page tags from input content
2. **Tag Generation**: Creates HTML comment tags with page number and reference data
3. **Content Wrapping**: Wraps each page's content between PageHeader and PageFooter tags
4. **Concatenation**: Combines all pages within a section while preserving tag boundaries
5. **HTML Escaping**: Escapes special characters in page references for valid HTML attributes

### Summary Generation Approach
1. **Hierarchy Building**: Creates breadcrumb path by tracking heading levels and previous sections
2. **Context Gathering**: Provides GPT-4 with chapter summary and previous section summaries for continuity
3. **GPT Processing**: Uses structured prompts with CO-STAR framework for consistent, technical summaries
4. **Format Standardization**: Combines hierarchy string with GPT-generated summary using `: ` separator

## 6. Data Transformations

### Page-Level to Section-Level Conversion

| Aspect | Stage 1 (Page-Level) | Stage 2 (Section-Level) |
|--------|---------------------|-------------------------|
| **Granularity** | One record per page | One record per section (spanning multiple pages) |
| **Content Scope** | Single page markdown | Multiple pages with embedded page tags |
| **Identification** | Page number + reference | Section number + title + page range |
| **Summary Level** | Chapter summary only | Chapter + hierarchical section summary |

### Field Mapping Table

| Stage 1 Field | Stage 2 Field | Transformation |
|---------------|---------------|----------------|
| `document_id` | `document_id` | **Preserved** - Direct copy |
| `filename` | `filename` | **Preserved** - Direct copy |
| `filepath` | `filepath` | **Preserved** - Direct copy |
| `source_filename` | `source_filename` | **Preserved** - Direct copy |
| `chapter_number` | `chapter_number` | **Preserved** - Direct copy |
| `chapter_name` | `chapter_name` | **Preserved** - Direct copy |
| `chapter_summary` | `chapter_summary` | **Preserved** - Direct copy |
| `chapter_page_count` | `chapter_page_count` | **Preserved** - Direct copy |
| `page_number` | *Embedded in page tags* | **Transformed** - Used in PageHeader/PageFooter tags |
| `page_reference` | *Embedded in page tags* | **Transformed** - Used in PageHeader/PageFooter tags |
| `source_page_number` | *Used for ordering* | **Transformed** - Used for processing order, not in output |
| `content` | `section_content` | **Aggregated** - Combined with page tags across multiple pages |
| *N/A* | `section_number` | **Generated** - Sequential numbering within chapter |
| *N/A* | `section_summary` | **Generated** - Hierarchy + GPT summary |
| *N/A* | `section_start_page` | **Calculated** - From page tag analysis |
| *N/A* | `section_end_page` | **Calculated** - From page tag analysis |
| *N/A* | `section_page_count` | **Calculated** - Count of unique pages in section |

## 7. Edge Case Handling

### Empty Section Content Handling
- **Detection**: Checks if section content is empty or contains only whitespace
- **Behavior**: Uses fallback summary format: `"Section covering {section_title}"`
- **Logging**: Warns about empty content with section number for debugging
- **Processing**: Section is still included in output with minimal summary

### Zero Token Handling  
- **Context**: Applied during content processing and API calls
- **Validation**: Checks token count before GPT API calls
- **Fallback**: Uses hierarchy-only summary if content cannot be tokenized
- **Cost Protection**: Prevents unnecessary API calls on empty/invalid content

### Division by Zero Protection
- **Context**: During section merging calculations 
- **Protection**: Ensures positive denominators in token-based calculations
- **Fallback**: Uses minimum threshold values to prevent mathematical errors
- **Error Handling**: Logs errors and continues processing with safe defaults

### Missing or Malformed Data
- **Missing Fields**: Uses default values (e.g., "Introduction" for missing chapter names)
- **Invalid References**: HTML-escapes special characters in page references
- **Malformed Content**: Cleans existing page tags before processing
- **Graceful Degradation**: Processing continues with warnings logged for investigation

## 8. Configuration Parameters

### MAX_TOKENS Settings
- `GPT_INPUT_TOKEN_LIMIT`: 80,000 - Maximum tokens for GPT input
- `MAX_COMPLETION_TOKENS`: 4,000 - Maximum tokens for GPT responses  
- `TOKEN_BUFFER`: 2,000 - Safety buffer for token calculations

### API Configuration
- `MODEL_NAME_CHAT`: "gpt-4-turbo-nonp" - GPT model for summary generation
- `TEMPERATURE`: 0.3 - Controls randomness in GPT responses
- `API_RETRY_ATTEMPTS`: 3 - Number of retry attempts for failed API calls
- `API_RETRY_DELAY`: 5 seconds - Delay between retry attempts

### Section Processing Thresholds
- `MIN_SECTION_TOKENS`: 250 - Minimum tokens to avoid merging with adjacent sections
- `MAX_HEADING_LEVEL`: 3 - Maximum heading depth for section identification
- `TOOL_RESPONSE_RETRIES`: 5 - Retries for structured GPT responses
- `TOOL_RESPONSE_RETRY_DELAY`: 3 seconds - Delay between tool response retries

### File Paths
- **Input Path**: `semantic_search/pipeline_output/stage1/stage1_page_records.json`
- **Output Path**: `semantic_search/pipeline_output/stage2/stage2_section_records.json`
- **Log Path**: `semantic_search/pipeline_output/logs/`

## 9. Example Records

### Complete Example Input Record from Stage 1:
```json
{
  "document_id": "EY_GUIDE_2024",
  "filename": "04_Financial_Instruments.pdf",
  "filepath": "/nas/chapters/04_Financial_Instruments.pdf", 
  "source_filename": "ey_ifrs_guide_2024.pdf",
  "chapter_number": 4,
  "chapter_name": "Financial Instruments Under IFRS 9",
  "chapter_summary": "This chapter addresses IFRS 9 financial instrument classification, measurement, and impairment requirements. Covers expected credit loss models and hedge accounting principles for complex financial products.",
  "chapter_page_count": 67,
  "page_number": 23,
  "page_reference": "4-35",
  "source_page_number": 156,
  "content": "## Impairment Requirements\n\n### Expected Credit Loss Model\n\nIFRS 9 requires entities to recognize expected credit losses (ECL) rather than incurred losses...\n\n#### Staging Approach\n\nThe three-stage approach categorizes financial instruments based on credit risk:\n\n- **Stage 1**: 12-month ECL for newly originated assets\n- **Stage 2**: Lifetime ECL when credit risk has increased significantly\n- **Stage 3**: Lifetime ECL for credit-impaired assets"
}
```

### Complete Example Output Record from Stage 2:
```json
{
  "document_id": "EY_GUIDE_2024", 
  "filename": "04_Financial_Instruments.pdf",
  "filepath": "/nas/chapters/04_Financial_Instruments.pdf",
  "source_filename": "ey_ifrs_guide_2024.pdf",
  "chapter_number": 4,
  "chapter_name": "Financial Instruments Under IFRS 9",
  "chapter_summary": "This chapter addresses IFRS 9 financial instrument classification, measurement, and impairment requirements. Covers expected credit loss models and hedge accounting principles for complex financial products.",
  "chapter_page_count": 67,
  "section_number": 3,
  "section_summary": "Financial Instruments Under IFRS 9 > Impairment Requirements: Explains the IFRS 9 expected credit loss model and three-stage approach for recognizing impairment losses on financial assets. Details the criteria for assessing significant increases in credit risk and the calculation methodologies for 12-month versus lifetime expected credit losses.",
  "section_start_page": 23,
  "section_end_page": 29,  
  "section_page_count": 7,
  "section_content": "<!-- PageHeader PageNumber=\"23\" PageReference=\"4-35\" -->\n## Impairment Requirements\n\n### Expected Credit Loss Model\n\nIFRS 9 requires entities to recognize expected credit losses (ECL) rather than incurred losses...\n\n#### Staging Approach\n\nThe three-stage approach categorizes financial instruments based on credit risk:\n\n- **Stage 1**: 12-month ECL for newly originated assets\n- **Stage 2**: Lifetime ECL when credit risk has increased significantly  \n- **Stage 3**: Lifetime ECL for credit-impaired assets\n<!-- PageFooter PageNumber=\"23\" PageReference=\"4-35\" -->\n<!-- PageHeader PageNumber=\"24\" PageReference=\"4-36\" -->\n\n### Significant Increase in Credit Risk\n\nDetermining whether credit risk has increased significantly requires:\n\n1. **Quantitative Analysis**: Comparison of probability of default over expected life\n2. **Qualitative Indicators**: Delinquency, restructuring, or other risk factors\n3. **Backstop Requirements**: Automatic Stage 2 classification when >30 days past due\n\n<!-- PageFooter PageNumber=\"24\" PageReference=\"4-36\" -->\n<!-- PageHeader PageNumber=\"25\" PageReference=\"4-37\" -->\n\n### ECL Measurement Approaches\n\n#### General Approach\nApplies to most financial assets and requires:\n- Probability of default (PD)\n- Loss given default (LGD)  \n- Exposure at default (EAD)\n\n<!-- PageFooter PageNumber=\"25\" PageReference=\"4-37\" -->"
}
```

### Transformation Example:
**Input**: 7 page-level records (pages 23-29) for the "Impairment Requirements" section  
**Output**: 1 section-level record spanning pages 23-29 with embedded page tags and GPT summary

## 10. Integration Notes for Stage 3

### What Stage 3 Should Expect
- **Input Format**: JSON array of section-level records from `stage2_section_records.json`
- **Content Structure**: Section content with embedded page tags showing natural page boundaries
- **Summary Hierarchy**: Section summaries with breadcrumb navigation for context
- **Page Metadata**: Accurate page ranges (`section_start_page`, `section_end_page`, `section_page_count`)

### Important Considerations for Stage 3
1. **Page Tag Preservation**: Stage 3 should preserve or properly handle page boundary tags during chunking
2. **Summary Utilization**: Rich hierarchical summaries provide valuable context for chunk-level processing
3. **Section Boundaries**: Natural section boundaries may inform chunking strategy decisions
4. **Token Awareness**: Section-level token counts can guide chunk size optimization
5. **Chapter Context**: Chapter summaries remain available for broader context understanding

### Data Quality Expectations
- All section records will have valid `section_number` (sequential within chapters)
- Section content will be properly formatted with consistent page tag structure
- Summaries will follow standardized format with hierarchy and GPT-generated content
- Page ranges will be accurate and reflect actual content boundaries
- No orphaned or incomplete sections (all content assigned to valid sections)

## Key Implementation Notes

### Date/Time Formats
- All timestamps use ISO 8601 format: `YYYY-MM-DDTHH:MM:SS.sssZ`
- Processing timestamps logged in local timezone with UTC offset

### ID Generation Methods
- `section_number`: Sequential integer generation within each chapter (resets per chapter)
- No UUIDs or complex ID schemes used - relies on document_id + chapter_number + section_number for uniqueness

### Character Encoding
- All text content processed as UTF-8 encoding
- HTML escaping applied to page reference values in tag attributes
- Markdown formatting preserved throughout processing
- No additional character escaping beyond HTML attribute requirements

### Performance Characteristics
- Processing time scales with number of chapters and GPT API response times
- Memory usage proportional to largest chapter size (content held in memory during processing)
- Token costs apply per section for GPT summary generation
- Typical processing: ~2-5 seconds per section including GPT API calls