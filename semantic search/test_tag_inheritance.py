#!/usr/bin/env python3
"""
Test script for page tag inheritance approach
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stage2_section_processing_v3 import (
    embed_page_tags,
    identify_sections_from_pages,
    extract_page_range_from_content
)

# Test data with sections completely within pages
test_pages = [
    {
        'page_number': 1,
        'page_reference': '2-15',
        'content': '''Introduction text before any heading.

## Section A

This is Section A, completely within page 1.

## Section B

This is the start of Section B, also on page 1.'''
    },
    {
        'page_number': 2,
        'page_reference': '2-16',
        'content': '''Section B continues onto page 2.

It has more content here.

## Section C

Section C starts and ends on page 2.'''
    },
    {
        'page_number': 3,
        'page_reference': '2-17',
        'content': '''## Section D

Section D is entirely on page 3.

## Section E

Section E also on page 3.'''
    }
]

def test_tag_inheritance():
    """Test that all sections get page tags through inheritance"""
    print("Testing page tag inheritance approach...")
    print("=" * 70)
    
    chapter_metadata = {
        'chapter_number': 1,
        'chapter_name': 'Test Chapter',
        'chapter_summary': 'Testing tag inheritance'
    }
    
    # Get sections
    sections = identify_sections_from_pages(test_pages, chapter_metadata)
    
    print(f"\nFound {len(sections)} sections:\n")
    
    all_have_tags = True
    for i, section in enumerate(sections):
        print(f"Section {section['section_number']}: {section['title']}")
        print(f"  Pages: {section['start_page']}-{section['end_page']}")
        
        content = section['content']
        
        # Check for header at start
        has_header = content.strip().startswith('<!-- PageHeader')
        print(f"  Has PageHeader at start: {has_header}")
        
        # Check for footer at end
        has_footer = content.strip().endswith('-->')
        if has_footer:
            # Double check it's actually a footer
            import re
            has_footer = bool(re.search(r'<!-- PageFooter[^>]*-->$', content.strip()))
        print(f"  Has PageFooter at end: {has_footer}")
        
        if not has_header or not has_footer:
            all_have_tags = False
            print(f"  ❌ Missing tags!")
            
        # Show snippets
        print(f"  First 100 chars: {repr(content[:100])}")
        print(f"  Last 100 chars: {repr(content[-100:])}")
        print()
    
    # Test specific cases
    print("Specific test cases:")
    print("-" * 40)
    
    # Section A should be completely within page 1
    section_a = next((s for s in sections if 'Section A' in s['title']), None)
    if section_a:
        print("Section A (completely within page 1):")
        print(f"  Expected pages: 1-1")
        print(f"  Actual pages: {section_a['start_page']}-{section_a['end_page']}")
        assert section_a['start_page'] == 1 and section_a['end_page'] == 1
        print("  ✅ Correct page range")
    
    # Section B should span pages 1-2
    section_b = next((s for s in sections if 'Section B' in s['title']), None)
    if section_b:
        print("\nSection B (spans pages 1-2):")
        print(f"  Expected pages: 1-2")
        print(f"  Actual pages: {section_b['start_page']}-{section_b['end_page']}")
        assert section_b['start_page'] == 1 and section_b['end_page'] == 2
        print("  ✅ Correct page range")
    
    # Section D should be completely within page 3
    section_d = next((s for s in sections if 'Section D' in s['title']), None)
    if section_d:
        print("\nSection D (completely within page 3):")
        print(f"  Expected pages: 3-3")
        print(f"  Actual pages: {section_d['start_page']}-{section_d['end_page']}")
        assert section_d['start_page'] == 3 and section_d['end_page'] == 3
        print("  ✅ Correct page range")
    
    print("\n" + "=" * 70)
    if all_have_tags:
        print("✅ SUCCESS: All sections have both PageHeader and PageFooter tags!")
    else:
        print("❌ FAILURE: Some sections are missing tags")
        
    return all_have_tags

if __name__ == "__main__":
    success = test_tag_inheritance()
    sys.exit(0 if success else 1)