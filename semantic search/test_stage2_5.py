#!/usr/bin/env python3
"""
Test script for Stage 2.5 Page Boundary Correction
Creates sample data to test various edge cases
"""

import json
import os
from pathlib import Path

# Test data with various edge cases
test_data = [
    # Chapter 1: Normal case with all page tags
    {
        "document_id": "TEST_DOC",
        "filename": "01_test.pdf",
        "filepath": "/test/01_test.pdf",
        "source_filename": "test.pdf",
        "chapter_number": 1,
        "chapter_name": "Test Chapter 1",
        "chapter_summary": "Test chapter summary",
        "chapter_page_count": 5,
        "section_number": 1,
        "section_summary": "Test Chapter 1 > Introduction: Test section 1",
        "section_start_page": 2,  # WRONG - should be 1
        "section_end_page": 5,     # WRONG - should be 1
        "section_page_count": 4,
        "section_content": 'Introduction text <!-- PageFooter PageNumber="1" PageReference="1-1" -->'
    },
    {
        "document_id": "TEST_DOC",
        "filename": "01_test.pdf",
        "filepath": "/test/01_test.pdf",
        "source_filename": "test.pdf",
        "chapter_number": 1,
        "chapter_name": "Test Chapter 1",
        "chapter_summary": "Test chapter summary",
        "chapter_page_count": 5,
        "section_number": 2,
        "section_summary": "Test Chapter 1 > Middle Section: Test section 2",
        "section_start_page": None,  # Missing page info
        "section_end_page": None,
        "section_page_count": 0,
        "section_content": "Content with no page tags at all"
    },
    {
        "document_id": "TEST_DOC",
        "filename": "01_test.pdf",
        "filepath": "/test/01_test.pdf",
        "source_filename": "test.pdf",
        "chapter_number": 1,
        "chapter_name": "Test Chapter 1",
        "chapter_summary": "Test chapter summary",
        "chapter_page_count": 5,
        "section_number": 3,
        "section_summary": "Test Chapter 1 > Main Content: Test section 3",
        "section_start_page": 1,  # WRONG - should be 2
        "section_end_page": 1,     # WRONG - should be 2
        "section_page_count": 1,
        "section_content": '<!-- PageHeader PageNumber="2" PageReference="1-2" --> Main content for page 2 <!-- PageFooter PageNumber="2" PageReference="1-2" -->'
    },
    
    # Chapter 2: Sections spanning multiple pages
    {
        "document_id": "TEST_DOC",
        "filename": "02_test.pdf",
        "filepath": "/test/02_test.pdf",
        "source_filename": "test.pdf",
        "chapter_number": 2,
        "chapter_name": "Test Chapter 2",
        "chapter_summary": "Test chapter 2 summary",
        "chapter_page_count": 10,
        "section_number": 1,
        "section_summary": "Test Chapter 2 > Overview: Multi-page section",
        "section_start_page": 3,  # WRONG - should be 1-3
        "section_end_page": 5,
        "section_page_count": 3,
        "section_content": '''<!-- PageHeader PageNumber="1" PageReference="2-1" -->
Chapter 2 starts here
<!-- PageFooter PageNumber="1" PageReference="2-1" -->
<!-- PageHeader PageNumber="2" PageReference="2-2" -->
Continues on page 2
<!-- PageFooter PageNumber="2" PageReference="2-2" -->
<!-- PageHeader PageNumber="3" PageReference="2-3" -->
And page 3
<!-- PageFooter PageNumber="3" PageReference="2-3" -->'''
    },
    {
        "document_id": "TEST_DOC",
        "filename": "02_test.pdf",
        "filepath": "/test/02_test.pdf",
        "source_filename": "test.pdf",
        "chapter_number": 2,
        "chapter_name": "Test Chapter 2",
        "chapter_summary": "Test chapter 2 summary",
        "chapter_page_count": 10,
        "section_number": 2,
        "section_summary": "Test Chapter 2 > Details: Another section",
        "section_start_page": 2,  # WRONG - should be 4-5
        "section_end_page": 3,
        "section_page_count": 2,
        "section_content": '''<!-- PageHeader PageNumber="4" PageReference="2-4" -->
Section 2 content
<!-- PageFooter PageNumber="4" PageReference="2-4" -->
<!-- PageHeader PageNumber="5" PageReference="2-5" -->
More content
<!-- PageFooter PageNumber="5" PageReference="2-5" -->'''
    },
    
    # Chapter 3: Missing headers/footers
    {
        "document_id": "TEST_DOC",
        "filename": "03_test.pdf",
        "filepath": "/test/03_test.pdf",
        "source_filename": "test.pdf",
        "chapter_number": 3,
        "chapter_name": "Test Chapter 3",
        "chapter_summary": "Test chapter 3 summary",
        "chapter_page_count": 3,
        "section_number": 1,
        "section_summary": "Test Chapter 3 > Intro: Missing footer",
        "section_start_page": 2,  # Should be 1
        "section_end_page": 2,
        "section_page_count": 1,
        "section_content": '<!-- PageHeader PageNumber="1" PageReference="3-1" -->\nIntro with no footer'
    },
    {
        "document_id": "TEST_DOC",
        "filename": "03_test.pdf",
        "filepath": "/test/03_test.pdf",
        "source_filename": "test.pdf",
        "chapter_number": 3,
        "chapter_name": "Test Chapter 3",
        "chapter_summary": "Test chapter 3 summary",
        "chapter_page_count": 3,
        "section_number": 2,
        "section_summary": "Test Chapter 3 > Middle: No tags",
        "section_start_page": None,
        "section_end_page": None,
        "section_page_count": 0,
        "section_content": 'Middle section with absolutely no page tags'
    },
    {
        "document_id": "TEST_DOC",
        "filename": "03_test.pdf",
        "filepath": "/test/03_test.pdf",
        "source_filename": "test.pdf",
        "chapter_number": 3,
        "chapter_name": "Test Chapter 3",
        "chapter_summary": "Test chapter 3 summary",
        "chapter_page_count": 3,
        "section_number": 3,
        "section_summary": "Test Chapter 3 > End: Missing header",
        "section_start_page": 1,  # Should be 2
        "section_end_page": 1,
        "section_page_count": 1,
        "section_content": 'Content with no header\n<!-- PageFooter PageNumber="2" PageReference="3-2" -->'
    }
]

# Save test data
output_file = "test_stage2_input.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)

print(f"✅ Created test data file: {output_file}")
print(f"   - {len(test_data)} sections across 3 chapters")
print("\n📋 Test cases included:")
print("   1. Section with page footer only (should infer page 1 start)")
print("   2. Section with no page tags (should infer from neighbors)")
print("   3. Section with complete page tags (should correct boundaries)")
print("   4. Multi-page sections (should span pages 1-3 and 4-5)")
print("   5. Missing headers/footers (should handle gracefully)")
print("\n🚀 Run the correction script:")
print(f"   python stage2_5_page_boundary_correction.py -i {output_file} -o test_stage2_5_output.json -v")