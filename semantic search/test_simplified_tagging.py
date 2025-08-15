#!/usr/bin/env python3
"""
Test script for simplified page tagging approach
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stage2_section_processing_v3 import (
    embed_page_tags,
    identify_sections_from_pages,
    extract_page_range_from_content,
    clean_existing_page_tags
)

# Test data with sections that span pages
test_pages = [
    {
        'page_number': 1,
        'page_reference': '2-15',
        'content': '''# Introduction to IFRS 16

This chapter provides comprehensive guidance on implementing IFRS 16 lease accounting.

## Recognition and Measurement

The recognition criteria for leases under IFRS 16 require careful consideration of several factors.'''
    },
    {
        'page_number': 2,
        'page_reference': '2-16',
        'content': '''The measurement of lease liabilities involves discounting future lease payments using the implicit rate.

Key considerations include:
- Variable lease payments
- Residual value guarantees
- Purchase options

## Disclosure Requirements'''
    },
    {
        'page_number': 3,
        'page_reference': '2-17',
        'content': '''Entities must provide detailed disclosures about their leasing activities including:

1. Maturity analysis of lease liabilities
2. Total cash outflow for leases
3. Short-term lease expense

## Transition Approaches

IFRS 16 offers two transition methods.'''
    }
]

def test_simple_page_tagging():
    """Test the simplified page tagging approach"""
    print("Testing simplified page tagging approach...")
    print("=" * 70)
    
    # Test 1: Embed page tags
    print("\nTest 1: Embedding page tags")
    print("-" * 40)
    tagged_content = embed_page_tags(test_pages)
    
    # Check that each page has header and footer
    lines = tagged_content.split('\n')
    page_headers = [l for l in lines if 'PageHeader' in l]
    page_footers = [l for l in lines if 'PageFooter' in l]
    
    print(f"Found {len(page_headers)} page headers")
    print(f"Found {len(page_footers)} page footers")
    
    # Verify we have the right number
    assert len(page_headers) == 3, f"Expected 3 headers, got {len(page_headers)}"
    assert len(page_footers) == 3, f"Expected 3 footers, got {len(page_footers)}"
    print("✅ All pages have headers and footers")
    
    # Test 2: Section identification
    print("\nTest 2: Section identification with embedded tags")
    print("-" * 40)
    
    chapter_metadata = {
        'chapter_number': 1,
        'chapter_name': 'IFRS 16 Leases',
        'chapter_summary': 'This chapter covers IFRS 16 implementation.'
    }
    
    sections = identify_sections_from_pages(test_pages, chapter_metadata)
    
    print(f"Found {len(sections)} sections:")
    for section in sections:
        print(f"  Section {section['section_number']}: {section['title']}")
        print(f"    Pages: {section['start_page']}-{section['end_page']}")
        
        # Check that content contains page tags
        content = section['content']
        has_tags = 'PageHeader' in content or 'PageFooter' in content
        print(f"    Contains page tags: {has_tags}")
    
    # Test 3: Verify section spanning pages has correct tags
    print("\nTest 3: Checking section that spans pages 1-2")
    print("-" * 40)
    
    # Find the "Recognition and Measurement" section
    recognition_section = next((s for s in sections if 'Recognition' in s['title']), None)
    
    if recognition_section:
        content = recognition_section['content']
        
        # Extract all page numbers from the content
        import re
        page_nums = re.findall(r'PageNumber="(\d+)"', content)
        unique_pages = sorted(set(int(p) for p in page_nums))
        
        print(f"Section spans pages: {unique_pages}")
        print(f"Section reports pages: {recognition_section['start_page']}-{recognition_section['end_page']}")
        
        # Show a snippet of the content
        print("\nFirst 150 characters:")
        print(repr(content[:150]))
        print("\nLast 150 characters:")
        print(repr(content[-150:]))
        
        # Verify it contains tags from both pages
        assert '1' in page_nums and '2' in page_nums, "Section should contain tags from pages 1 and 2"
        print("✅ Section correctly contains tags from both pages it spans")
    
    print("\n" + "=" * 70)
    print("All tests passed! Simplified approach works correctly.")

if __name__ == "__main__":
    test_simple_page_tagging()