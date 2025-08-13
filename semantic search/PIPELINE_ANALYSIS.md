# Semantic Search Pipeline Analysis - Page Tracking Requirements

## Current Problem
When we merge pages in Stage 1, we lose the ability to track which specific page a chunk of text came from. This is critical for citations in the final retrieval system.

## Retrieval Requirements
The final retrieval system needs to provide citations with:
1. **Original filename** (e.g., "ey_guide_2024.pdf")
2. **Actual page number** from the PDF (e.g., page 42)
3. **Page reference** from the footer (e.g., "EY-42" or "ii")

These citations must appear at the end of each paragraph in the synthesized response.

## Current Pipeline Data Flow

### Stage 1 V2 (Current Implementation)
**Input:** JSON array with page-level records
```json
{
  "document_id": "EY_GUIDE_2024",
  "filename": "ey_guide.pdf",
  "filepath": "path/to/file",
  "page_number": 42,
  "page_reference": "EY-42",
  "content": "page content...",
  "chapter_number": 3,
  "chapter_name": "Revenue Recognition"
}
```

**Current Processing:**
- Groups pages by chapter_number
- Concatenates content from all pages
- Loses page-level granularity

**Output:** One record per chapter
```json
{
  "document_id": "EY_GUIDE_2024",
  "filename": "ey_guide.pdf",
  "chapter_number": 3,
  "chapter_name": "Revenue Recognition",
  "chapter_page_start": 40,
  "chapter_page_end": 60,
  "content": "concatenated content from all pages...",
  "page_references": ["EY-40", "EY-41", "EY-42", ...]  // Just a list, no mapping
}
```

### Stage 2 (Section Processing)
**Input:** Chapter records from Stage 1
**Processing:**
- Identifies sections based on markdown headers
- Currently uses PageNumber tags to determine section page ranges (which we don't have anymore)
- Generates section summaries

**Output:** Section records with:
- `section_start_page`, `section_end_page` (currently derived from tags)
- Section content (slice of chapter content)

### Stage 3 (Chunking)
**Input:** Section records from Stage 2
**Processing:**
- Splits large sections into chunks
- Merges small chunks
- Currently tries to map positions back to original markdown (but can't with concatenated content)

**Output:** Final chunks with:
- `section_start_page`, `section_end_page` (inherited from Stage 2)
- But no way to know which specific page within the section the chunk came from

## The Core Issue
Once we concatenate pages in Stage 1, we lose the ability to track:
1. Which page a specific piece of text came from
2. Where page boundaries are within the concatenated content
3. The page_reference for each specific page

## Proposed Solutions

### Solution 1: Preserve Page Boundaries in Content
Instead of simple concatenation, preserve page metadata within the content:
```python
# In Stage 1 V2
pages_data = []
for page in pages:
    pages_data.append({
        'page_number': page['page_number'],
        'page_reference': page.get('page_reference'),
        'start_offset': current_offset,
        'content': page['content'],
        'end_offset': current_offset + len(page['content'])
    })
    current_offset += len(page['content']) + 2  # +2 for \n\n

chapter_record = {
    ...
    'content': concatenated_content,
    'page_mappings': pages_data  # Preserve page boundary information
}
```

### Solution 2: Keep Page-Level Records Through Pipeline
Don't merge pages at all in Stage 1. Instead:
1. Stage 1: Add chapter summary/tags to each page record
2. Stage 2: Process sections across multiple page records
3. Stage 3: Create chunks that reference specific page records

### Solution 3: Hybrid Approach (Recommended)
Keep both page-level and chapter-level data:

#### Stage 1 V2 Output:
```json
{
  "document_id": "EY_GUIDE_2024",
  "filename": "ey_guide.pdf",
  "chapter_number": 3,
  "chapter_name": "Revenue Recognition",
  "chapter_summary": "...",
  "chapter_tags": [...],
  "pages": [
    {
      "page_number": 40,
      "page_reference": "EY-40",
      "content": "page 40 content...",
      "char_start": 0,
      "char_end": 2500
    },
    {
      "page_number": 41,
      "page_reference": "EY-41",
      "content": "page 41 content...",
      "char_start": 2502,
      "char_end": 5100
    }
  ],
  "full_content": "concatenated content..."  // For section identification
}
```

#### Stage 2 Enhancement:
- When identifying sections, track character positions
- Map section boundaries to page numbers using char_start/char_end

#### Stage 3 Enhancement:
- When creating chunks, track character positions
- Map chunk positions back to specific pages
- Include page_number and page_reference in final chunk record

## Recommended Implementation Plan

### Phase 1: Modify Stage 1 V2
1. Preserve individual page data within chapter records
2. Track character offsets for each page
3. Include both concatenated content and page mappings

### Phase 2: Modify Stage 2
1. Update section identification to use character positions
2. Map section boundaries to pages using offset tracking
3. Pass page mapping data forward

### Phase 3: Modify Stage 3
1. Track character positions when creating chunks
2. Determine source page(s) for each chunk
3. Include page_number and page_reference in final records

### Phase 4: Update Retrieval
1. Use page_number and page_reference from chunks
2. Format citations as: (filename, page X, ref: Y)

## Benefits of This Approach
1. **Preserves page-level metadata** through entire pipeline
2. **Enables accurate citations** in retrieval
3. **Maintains chapter/section context** for better retrieval
4. **Backward compatible** with existing GPT processing
5. **Supports multi-page chunks** (can reference page range)

## Next Steps
1. Update Stage 1 V2 to implement hybrid approach
2. Create utility functions for position-to-page mapping
3. Test with sample data to verify citation accuracy
4. Update downstream stages incrementally