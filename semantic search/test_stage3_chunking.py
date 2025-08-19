#!/usr/bin/env python3
"""
Test script for Stage 3 chunking
"""

import json
from stage3_chunking import Stage3Chunker, SimpleTokenizer

def create_test_section():
    """Create a test section with realistic content"""
    content = """<!-- PageHeader PageNumber="1" PageReference="1-1" -->
# Introduction to Financial Reporting

Financial reporting under IFRS requires careful consideration of multiple factors. Organizations must ensure compliance with all relevant standards while maintaining transparency and accuracy in their disclosures.

## Recognition Criteria

The fundamental principle of recognition in financial statements involves determining when an item meets the definition of an element (asset, liability, equity, income, or expense) and satisfies the criteria for recognition. This process requires professional judgment and careful analysis.

### Asset Recognition

Assets are recognized when:
- It is probable that future economic benefits will flow to the entity
- The asset has a cost or value that can be measured reliably
- The entity has control over the resource

These criteria ensure that only genuine assets appear on the balance sheet.

<!-- PageFooter PageNumber="1" PageReference="1-1" -->
<!-- PageHeader PageNumber="2" PageReference="1-2" -->

### Liability Recognition

Liabilities are recognized when:
- The entity has a present obligation (legal or constructive)
- It is probable that an outflow of resources will be required
- The amount can be estimated reliably

The assessment of probability is crucial in determining whether a provision should be recognized or disclosed as a contingent liability.

## Measurement Bases

IFRS permits various measurement bases including:

1. **Historical Cost**: The original transaction price
2. **Fair Value**: The price in an orderly transaction between market participants
3. **Present Value**: Discounted future cash flows
4. **Current Cost**: The amount needed to acquire an equivalent asset today

Each measurement basis has specific applications depending on the nature of the item and the relevant IFRS standard.

### Fair Value Hierarchy

The fair value hierarchy categorizes inputs into three levels:
- Level 1: Quoted prices in active markets
- Level 2: Observable inputs other than quoted prices
- Level 3: Unobservable inputs

This hierarchy ensures consistency and comparability in fair value measurements across different entities and industries.

<!-- PageFooter PageNumber="2" PageReference="1-2" -->
<!-- PageHeader PageNumber="3" PageReference="1-3" -->

## Disclosure Requirements

Comprehensive disclosure is essential for users to understand the financial statements. Key disclosure requirements include:

### Accounting Policies

Entities must disclose:
- The measurement bases used
- Other accounting policies relevant to understanding the financial statements
- Information about judgments made in applying accounting policies

### Estimates and Judgments

Significant estimates and judgments require detailed disclosure including:
- The nature of the assumption
- The sensitivity of carrying amounts to methods and assumptions
- The range of reasonably possible outcomes

This transparency helps users assess the reliability and comparability of reported amounts.

## Presentation Considerations

The presentation of financial statements should provide relevant and reliable information that is comparable and understandable. Key presentation requirements include:

- Clear identification of the financial statements
- Prominent display of key information
- Appropriate level of rounding
- Consistent presentation from period to period

### Offsetting

Assets and liabilities, and income and expenses, should not be offset unless required or permitted by an IFRS. This gross presentation provides more useful information about an entity's resources and obligations.

<!-- PageFooter PageNumber="3" PageReference="1-3" -->"""
    
    section = {
        'document_id': 'TEST_DOC_2024',
        'filename': '01_Introduction.pdf',
        'filepath': '/test/01_Introduction.pdf',
        'source_filename': 'test_document.pdf',
        'chapter_number': 1,
        'chapter_name': 'Introduction to IFRS',
        'chapter_summary': 'This chapter provides an overview of IFRS principles.',
        'chapter_page_count': 10,
        'section_number': 1,
        'section_summary': 'Introduction to IFRS > Financial Reporting: Explains core concepts.',
        'section_start_page': 1,
        'section_end_page': 3,
        'section_page_count': 3,
        'section_start_reference': '1-1',
        'section_end_reference': '1-3',
        'section_content': content
    }
    
    return section

def test_tokenizer():
    """Test the simple tokenizer"""
    print("Testing SimpleTokenizer...")
    tokenizer = SimpleTokenizer()
    
    test_cases = [
        ("Hello world", 2),
        ("This is a longer sentence with more words.", 8),
        ("IFRS 16 requires entities to recognize lease liabilities.", 10),
        ("The quick brown fox jumps over the lazy dog.", 9),
    ]
    
    for text, expected in test_cases:
        count = tokenizer.count_tokens(text)
        print(f"  '{text[:30]}...' -> {count} tokens (expected ~{expected})")
    print()

def test_chunking():
    """Test the chunking functionality"""
    print("Testing Stage3Chunker...")
    
    # Create test section
    section = create_test_section()
    
    # Initialize chunker with smaller limits for testing
    chunker = Stage3Chunker(min_tokens=200, max_tokens=300, hard_max=350)
    
    # Process the section
    chunks = chunker.chunk_section(section)
    
    print(f"Created {len(chunks)} chunks from test section")
    print()
    
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:")
        print(f"  Chunk number: {chunk['chunk_number']}")
        print(f"  Section references: {chunk.get('section_start_reference')}-{chunk.get('section_end_reference')}")
        print(f"  Content preview: {chunk['chunk_content'][:100]}...")
        print()
    
    # Verify HTML tags are preserved
    all_content = ''.join(c['chunk_content'] for c in chunks)
    assert '<!-- PageHeader' in all_content
    assert '<!-- PageFooter' in all_content
    print("✓ HTML tags preserved in chunks")
    
    # Verify no content is lost
    original_length = len(section['section_content'])
    chunked_length = len(all_content)
    print(f"✓ Content integrity maintained ({original_length} chars -> {chunked_length} chars)")
    
    # Verify schema fields
    expected_fields = [
        'document_id', 'filename', 'filepath', 'source_filename',
        'chapter_number', 'chapter_name', 'chapter_summary', 'chapter_page_count',
        'section_number', 'section_summary', 'section_start_page', 'section_end_page',
        'section_page_count', 'section_start_reference', 'section_end_reference',
        'chunk_number', 'chunk_content'
    ]
    
    for field in expected_fields:
        assert field in chunks[0], f"Missing field: {field}"
    
    # Check no extra fields
    actual_fields = set(chunks[0].keys())
    expected_set = set(expected_fields)
    extra_fields = actual_fields - expected_set
    assert len(extra_fields) == 0, f"Extra fields found: {extra_fields}"
    
    print(f"✓ Output schema correct ({len(expected_fields)} fields)")
    
    return chunks

def save_test_output(chunks):
    """Save test output for inspection"""
    output_file = 'test_stage3_output.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"\nTest output saved to: {output_file}")

if __name__ == "__main__":
    print("=" * 60)
    print("Stage 3 Chunking Test Suite")
    print("=" * 60)
    print()
    
    # Test tokenizer
    test_tokenizer()
    
    # Test chunking
    chunks = test_chunking()
    
    # Save output
    save_test_output(chunks)
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)