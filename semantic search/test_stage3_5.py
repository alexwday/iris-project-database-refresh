#!/usr/bin/env python3
"""
Test script for Stage 3.5 chunk page boundary correction
"""

import json
from stage3_5_chunk_page_correction import (
    extract_page_tags_with_positions,
    build_page_ranges,
    build_position_map,
    determine_pages_for_position_range,
    process_section_chunks
)

def create_test_chunks():
    """Create test chunks with realistic content and page tags"""
    
    chunks = [
        {
            'document_id': 'TEST_DOC',
            'filename': 'test.pdf',
            'filepath': '/test/test.pdf',
            'source_filename': 'original.pdf',
            'chapter_number': 1,
            'chapter_name': 'Test Chapter',
            'chapter_summary': 'Test chapter summary',
            'chapter_page_count': 10,
            'section_number': 1,
            'section_summary': 'Test section summary',
            'section_start_page': 5,
            'section_end_page': 8,
            'section_page_count': 4,
            'section_start_reference': '1-5',
            'section_end_reference': '1-8',
            'chunk_number': 1,
            'chunk_content': '''<!-- PageHeader PageNumber="5" PageReference="1-5" -->
## Introduction

This is the first chunk content that starts on page 5.

We have some important information here that continues...
<!-- PageFooter PageNumber="5" PageReference="1-5" -->'''
        },
        {
            'document_id': 'TEST_DOC',
            'filename': 'test.pdf',
            'filepath': '/test/test.pdf',
            'source_filename': 'original.pdf',
            'chapter_number': 1,
            'chapter_name': 'Test Chapter',
            'chapter_summary': 'Test chapter summary',
            'chapter_page_count': 10,
            'section_number': 1,
            'section_summary': 'Test section summary',
            'section_start_page': 5,
            'section_end_page': 8,
            'section_page_count': 4,
            'section_start_reference': '1-5',
            'section_end_reference': '1-8',
            'chunk_number': 2,
            'chunk_content': '''<!-- PageHeader PageNumber="6" PageReference="1-6" -->

### Detailed Information

This chunk spans from page 6 to page 7.

More content here with various details and explanations.

<!-- PageFooter PageNumber="6" PageReference="1-6" -->
<!-- PageHeader PageNumber="7" PageReference="1-7" -->

Continuing on page 7 with additional information.

<!-- PageFooter PageNumber="7" PageReference="1-7" -->'''
        },
        {
            'document_id': 'TEST_DOC',
            'filename': 'test.pdf',
            'filepath': '/test/test.pdf',
            'source_filename': 'original.pdf',
            'chapter_number': 1,
            'chapter_name': 'Test Chapter',
            'chapter_summary': 'Test chapter summary',
            'chapter_page_count': 10,
            'section_number': 1,
            'section_summary': 'Test section summary',
            'section_start_page': 5,
            'section_end_page': 8,
            'section_page_count': 4,
            'section_start_reference': '1-5',
            'section_end_reference': '1-8',
            'chunk_number': 3,
            'chunk_content': '''<!-- PageHeader PageNumber="7" PageReference="1-7" -->

### Final Section

This is the last chunk that ends on page 8.

<!-- PageFooter PageNumber="7" PageReference="1-7" -->
<!-- PageHeader PageNumber="8" PageReference="1-8" -->

## Conclusion

Final thoughts and summary for this section.

<!-- PageFooter PageNumber="8" PageReference="1-8" -->'''
        }
    ]
    
    return chunks

def test_page_tag_extraction():
    """Test extraction of page tags from content"""
    print("Testing page tag extraction...")
    
    content = '''<!-- PageHeader PageNumber="5" PageReference="1-5" -->
Some content here
<!-- PageFooter PageNumber="5" PageReference="1-5" -->
<!-- PageHeader PageNumber="6" PageReference="1-6" -->
More content
<!-- PageFooter PageNumber="6" PageReference="1-6" -->'''
    
    tags = extract_page_tags_with_positions(content)
    
    print(f"  Found {len(tags)} tags")
    for pos, tag_type, page_num, page_ref in tags:
        print(f"    Position {pos}: {tag_type} for page {page_num} (ref: {page_ref})")
    
    assert len(tags) == 4, "Should find 4 tags"
    assert tags[0][1] == 'header' and tags[0][2] == 5
    assert tags[1][1] == 'footer' and tags[1][2] == 5
    print("  ✓ Page tag extraction works correctly\n")

def test_page_range_building():
    """Test building page ranges from tags"""
    print("Testing page range building...")
    
    content = '''<!-- PageHeader PageNumber="5" PageReference="1-5" -->
Content for page 5
<!-- PageFooter PageNumber="5" PageReference="1-5" -->
<!-- PageHeader PageNumber="6" PageReference="1-6" -->
Content for page 6
<!-- PageFooter PageNumber="6" PageReference="1-6" -->'''
    
    tags = extract_page_tags_with_positions(content)
    ranges, references = build_page_ranges(tags, content)
    
    print(f"  Built {len(ranges)} page ranges")
    for start, end, page_num in ranges:
        print(f"    Page {page_num}: positions {start}-{end}")
    
    print(f"  Found {len(references)} page references")
    for page_num, ref in references.items():
        print(f"    Page {page_num}: reference '{ref}'")
    
    assert len(ranges) == 2, "Should have 2 page ranges"
    assert references[5] == "1-5", "Page 5 reference should be '1-5'"
    print("  ✓ Page range building works correctly\n")

def test_chunk_correction():
    """Test the full chunk correction process"""
    print("Testing chunk correction process...")
    
    chunks = create_test_chunks()
    print(f"  Created {len(chunks)} test chunks")
    
    # Process chunks
    corrected_chunks, stats = process_section_chunks(chunks)
    
    print(f"\n  Statistics:")
    for key, value in stats.items():
        if value > 0:
            print(f"    {key}: {value}")
    
    # Verify corrections
    print(f"\n  Chunk page assignments:")
    for chunk in corrected_chunks:
        chunk_num = chunk['chunk_number']
        start_page = chunk.get('chunk_start_page')
        end_page = chunk.get('chunk_end_page')
        start_ref = chunk.get('chunk_start_reference', '')
        end_ref = chunk.get('chunk_end_reference', '')
        print(f"    Chunk {chunk_num}: pages {start_page}-{end_page} (refs: {start_ref} to {end_ref})")
    
    # Verify expected results
    assert corrected_chunks[0]['chunk_start_page'] == 5, "Chunk 1 should start at page 5"
    assert corrected_chunks[0]['chunk_end_page'] == 5, "Chunk 1 should end at page 5"
    assert corrected_chunks[1]['chunk_start_page'] == 6, "Chunk 2 should start at page 6"
    assert corrected_chunks[1]['chunk_end_page'] == 7, "Chunk 2 should end at page 7"
    assert corrected_chunks[2]['chunk_start_page'] == 7, "Chunk 3 should start at page 7"
    assert corrected_chunks[2]['chunk_end_page'] == 8, "Chunk 3 should end at page 8"
    
    print("  ✓ Chunk correction works correctly\n")

def test_missing_tags():
    """Test handling of chunks without page tags"""
    print("Testing chunks without page tags...")
    
    chunks = [
        {
            'document_id': 'TEST_DOC',
            'chapter_number': 1,
            'section_number': 1,
            'section_start_page': 10,
            'section_end_page': 12,
            'chunk_number': 1,
            'chunk_content': 'Content without any page tags'
        },
        {
            'document_id': 'TEST_DOC',
            'chapter_number': 1,
            'section_number': 1,
            'section_start_page': 10,
            'section_end_page': 12,
            'chunk_number': 2,
            'chunk_content': 'More content without tags'
        }
    ]
    
    corrected_chunks, stats = process_section_chunks(chunks)
    
    print(f"  Stats: {stats}")
    print(f"  Chunk 1: pages {corrected_chunks[0].get('chunk_start_page')}-{corrected_chunks[0].get('chunk_end_page')}")
    print(f"  Chunk 2: pages {corrected_chunks[1].get('chunk_start_page')}-{corrected_chunks[1].get('chunk_end_page')}")
    
    assert stats['no_tags_found'] > 0, "Should detect no tags found"
    assert stats['inferred'] > 0, "Should infer page boundaries"
    print("  ✓ Handles missing tags correctly\n")

def save_test_output(corrected_chunks):
    """Save test output for inspection"""
    with open('test_stage3_5_output.json', 'w', encoding='utf-8') as f:
        json.dump(corrected_chunks, f, indent=2, ensure_ascii=False)
    print("Test output saved to: test_stage3_5_output.json")

if __name__ == "__main__":
    print("=" * 60)
    print("Stage 3.5 Chunk Page Correction Test Suite")
    print("=" * 60)
    print()
    
    # Run tests
    test_page_tag_extraction()
    test_page_range_building()
    test_chunk_correction()
    test_missing_tags()
    
    # Create and save full test output
    chunks = create_test_chunks()
    corrected_chunks, _ = process_section_chunks(chunks)
    save_test_output(corrected_chunks)
    
    print("=" * 60)
    print("✅ All tests passed successfully!")
    print("=" * 60)