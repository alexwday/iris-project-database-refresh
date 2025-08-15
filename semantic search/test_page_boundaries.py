#!/usr/bin/env python3
"""
Test script to verify position-based page boundary tracking
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stage2_section_processing_v3 import (
    build_tagged_content_with_map,
    find_page_at_position,
    fix_section_boundaries_with_map,
    clean_existing_page_tags
)

# Test data
test_pages = [
    {
        'page_number': 1,
        'page_reference': '2-15',
        'content': '# Chapter 1: Introduction\n\nThis is the beginning of chapter 1.\n\n## Section 1.1\n\nContent of section 1.1 on page 1.'
    },
    {
        'page_number': 2,
        'page_reference': '2-16',
        'content': 'Continuation of section 1.1 from page 1.\n\n## Section 1.2\n\nThis is section 1.2 starting on page 2.'
    },
    {
        'page_number': 3,
        'page_reference': '2-17',
        'content': 'Still in section 1.2.\n\nMore content for section 1.2.\n\n## Section 1.3\n\nSection 1.3 starts here on page 3.'
    }
]

def test_position_tracking():
    """Test that position-based tracking correctly identifies page boundaries"""
    print("Testing position-based page boundary tracking...")
    print("=" * 70)
    
    # Build tagged content with position map
    tagged_content, page_boundaries = build_tagged_content_with_map(test_pages)
    
    print(f"Total content length: {len(tagged_content)} characters")
    print(f"Number of page boundaries: {len(page_boundaries)}")
    print()
    
    # Print page boundaries
    print("Page boundaries:")
    for i, boundary in enumerate(page_boundaries):
        print(f"  Page {boundary['page_num']}: positions {boundary['start_pos']}-{boundary['end_pos']}")
    print()
    
    # Test finding pages at various positions
    test_positions = [0, 100, 300, 500, len(tagged_content) - 1]
    print("Testing find_page_at_position:")
    for pos in test_positions:
        page = find_page_at_position(pos, page_boundaries)
        if page:
            print(f"  Position {pos}: Page {page['page_num']}")
        else:
            print(f"  Position {pos}: No page found")
    print()
    
    # Test fixing section boundaries
    print("Testing fix_section_boundaries_with_map:")
    
    # Simulate a section that spans pages 2-3
    section_start = 250  # Somewhere in page 2
    section_end = 450    # Somewhere in page 3
    
    if section_start < len(tagged_content) and section_end <= len(tagged_content):
        section_content = tagged_content[section_start:section_end]
        print(f"  Original section ({section_start}-{section_end}):")
        print(f"    First 50 chars: {section_content[:50]!r}")
        print(f"    Last 50 chars: {section_content[-50:]!r}")
        
        # Fix boundaries
        fixed_content = fix_section_boundaries_with_map(
            section_content, section_start, section_end, page_boundaries
        )
        
        print(f"  Fixed section:")
        print(f"    First 80 chars: {fixed_content[:80]!r}")
        print(f"    Last 80 chars: {fixed_content[-80:]!r}")
        
        # Check if it starts with header and ends with footer
        has_header = fixed_content.strip().startswith('<!-- PageHeader')
        has_footer = fixed_content.strip().endswith('-->')
        print(f"    Has PageHeader at start: {has_header}")
        print(f"    Has PageFooter at end: {has_footer}")
    
    print()
    print("=" * 70)
    print("Test complete!")
    
    # Return the tagged content for inspection
    return tagged_content, page_boundaries

if __name__ == "__main__":
    test_position_tracking()