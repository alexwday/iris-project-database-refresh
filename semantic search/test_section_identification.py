#!/usr/bin/env python3
"""
Test script to verify section identification with position-based tracking
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stage2_section_processing_v3 import identify_sections_from_pages

# Test data with a section that splits across pages
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

def test_section_identification():
    """Test that sections are correctly identified with proper page tags"""
    print("Testing section identification with position-based tracking...")
    print("=" * 70)
    
    chapter_metadata = {
        'chapter_number': 1,
        'chapter_name': 'IFRS 16 Leases',
        'chapter_summary': 'This chapter covers the implementation of IFRS 16.'
    }
    
    # Identify sections
    sections = identify_sections_from_pages(test_pages, chapter_metadata)
    
    print(f"Found {len(sections)} sections\n")
    
    for section in sections:
        print(f"Section {section['section_number']}: {section['title']}")
        print(f"  Level: {section['level']}")
        print(f"  Pages: {section['start_page']}-{section['end_page']}")
        print(f"  Token count: {section['token_count']}")
        
        # Check for page tags
        content = section['content']
        has_header = content.strip().startswith('<!-- PageHeader')
        has_footer = content.strip().endswith('-->')
        
        print(f"  Has PageHeader: {has_header}")
        print(f"  Has PageFooter: {has_footer}")
        
        # Show first and last parts of content
        if len(content) > 160:
            print(f"  First 80 chars: {content[:80]!r}")
            print(f"  Last 80 chars: {content[-80:]!r}")
        else:
            print(f"  Content: {content!r}")
        print()
    
    # Verify critical section that spans pages 1-2
    recognition_section = next((s for s in sections if 'Recognition' in s['title']), None)
    if recognition_section:
        print("Checking 'Recognition and Measurement' section (spans pages 1-2):")
        content = recognition_section['content']
        
        # This section should start with page 1 header and end with page 2 footer
        if '<!-- PageHeader PageNumber="1"' in content[:100]:
            print("  ✅ Has Page 1 header at start")
        else:
            print("  ❌ Missing Page 1 header at start")
            
        if '<!-- PageFooter PageNumber="2"' in content[-100:]:
            print("  ✅ Has Page 2 footer at end")
        else:
            print("  ❌ Missing Page 2 footer at end")
    
    print("\n" + "=" * 70)
    print("Test complete!")

if __name__ == "__main__":
    test_section_identification()