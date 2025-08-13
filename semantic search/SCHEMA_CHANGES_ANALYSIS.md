# Database Schema Changes Analysis

## Current Database Schema (iris_textbook_database)

The current schema expects these fields for each chunk:

### Structural Fields
- `document_id` (TEXT) - e.g., "EY_GUIDE_2024"
- `chapter_number` (INT)
- `section_number` (INT)
- `part_number` (INT) - for chunks within sections
- `sequence_number` (INT) - global ordering

### Chapter Metadata
- `chapter_name` (TEXT)
- `chapter_tags` (TEXT[])
- `chapter_summary` (TEXT)
- `chapter_token_count` (INT)

### Section Metadata
- **`section_start_page` (INT)** - Current: derived from tags, needs specific page
- **`section_end_page` (INT)** - Current: derived from tags, needs specific page
- `section_importance_score` (FLOAT)
- `section_token_count` (INT)
- `section_hierarchy` (TEXT)
- `section_title` (TEXT)
- `section_standard` (TEXT)
- `section_standard_codes` (TEXT[])
- `section_references` (TEXT[])

### Content
- `content` (TEXT) - the actual chunk text
- `embedding` (VECTOR)

## Missing Fields for Citation Requirements

For proper citations, we need to know for EACH CHUNK:
1. **Actual page number(s)** where the chunk content appears
2. **Page reference(s)** from the footer (e.g., "EY-42")
3. **Original filename** for citation

## Proposed Schema Changes

### Option 1: Add New Fields to Existing Schema
Add these fields to `iris_textbook_database`:
```sql
-- Add chunk-specific page tracking
ALTER TABLE iris_textbook_database ADD COLUMN chunk_start_page INT;
ALTER TABLE iris_textbook_database ADD COLUMN chunk_end_page INT;
ALTER TABLE iris_textbook_database ADD COLUMN chunk_page_references TEXT[];  -- Array of page refs
ALTER TABLE iris_textbook_database ADD COLUMN source_filename TEXT;  -- Original PDF filename
```

### Option 2: Keep Schema, Track in Pipeline
Keep the database schema as-is, but ensure:
- `section_start_page` and `section_end_page` are accurate
- Add logic to determine chunk position within section
- Store page mappings separately or derive at retrieval time

## Impact Analysis

### With Option 1 (Schema Changes):
**Pros:**
- Direct access to page info for each chunk
- Simple retrieval queries
- Accurate citations guaranteed

**Cons:**
- Requires database migration
- Changes to Stage 4 population script
- More storage per record

### With Option 2 (No Schema Change):
**Pros:**
- No database migration needed
- Backward compatible

**Cons:**
- Complex logic to derive chunk pages
- Potential inaccuracy if chunks span pages
- No page_reference storage

## Recommended Approach

### Phase 1: Immediate (No Schema Change)
1. Ensure `section_start_page` and `section_end_page` are accurate
2. Track page boundaries in Stage 1-3 pipeline
3. Use section pages as approximation for chunks

### Phase 2: Future Enhancement (Schema Change)
1. Add new fields for chunk-level page tracking
2. Update pipeline to populate these fields
3. Migrate existing data with best-effort page mapping

## Pipeline Changes Required (No Schema Change)

### Stage 1 V2 Changes:
```python
{
  "document_id": "EY_GUIDE_2024",
  "filename": "ey_guide.pdf",  # Preserve for all stages
  "chapter_number": 3,
  "pages": [
    {
      "page_number": 40,
      "page_reference": "EY-40",
      "content": "...",
      "char_start": 0,
      "char_end": 2500
    }
  ],
  "full_content": "...",
  "page_mappings": {  # Character offset to page mapping
    0: {"page": 40, "ref": "EY-40"},
    2502: {"page": 41, "ref": "EY-41"}
  }
}
```

### Stage 2 Changes:
- Use character positions to determine section page ranges
- Pass page_mappings forward
- Calculate accurate `section_start_page` and `section_end_page`

### Stage 3 Changes:
- Track character positions when creating chunks
- Use page_mappings to determine chunk pages
- Store as close to actual pages as possible in section fields

### Stage 4 Changes:
- Ensure `section_start_page` and `section_end_page` are populated
- Consider adding filename to `document_id` if not present

## Citation Strategy Without Schema Changes

During retrieval:
1. Use `section_start_page` and `section_end_page` as page range
2. Format citation as: `(document_id, pages X-Y)`
3. If we track page_references in document_id or metadata, use those

Example citation:
```
"This guidance applies to all leases except..." [EY_GUIDE_2024, pages 42-45]
```

## Conclusion

The current schema can work WITHOUT changes if we:
1. Accurately populate `section_start_page` and `section_end_page`
2. Accept that chunks will reference section page ranges, not exact pages
3. Include filename info in `document_id` or track separately

For perfect citations with exact page numbers and references, we would need the schema changes in Option 1.