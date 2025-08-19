-- Stage 3: Database Schema for Semantic Search
-- Table Name: iris_semantic_search
-- Purpose: Stores chunked content with embeddings for semantic search

-- VERSION 1: 3072 Dimensions with HalfVec (Try this first)
CREATE TABLE iris_semantic_search (
    -- Primary Key
    id SERIAL PRIMARY KEY, -- Auto-incrementing unique identifier for each record
    
    -- Document Fields
    document_id VARCHAR(255) NOT NULL, -- Unique identifier for the document (e.g., "EY_GUIDE_2024")
    filename VARCHAR(255) NOT NULL, -- Chapter-specific PDF filename (e.g., "03_Lease_Accounting.pdf")
    filepath TEXT, -- Full path to the chapter PDF file
    source_filename VARCHAR(255), -- Original source PDF filename before splitting into chapters
    
    -- Chapter Fields
    chapter_number INTEGER, -- Chapter number within the document
    chapter_name VARCHAR(500), -- Name/title of the chapter
    chapter_summary TEXT, -- GPT-generated summary of the entire chapter (2-3 sentences)
    chapter_page_count INTEGER, -- Total number of pages in the chapter
    
    -- Section Fields  
    section_number INTEGER, -- Sequential section number within the chapter
    section_summary TEXT, -- Hierarchical section summary with breadcrumb format (Chapter > Section: Summary)
    section_start_page INTEGER, -- First page number where this section begins
    section_end_page INTEGER, -- Last page number where this section ends
    section_page_count INTEGER, -- Total number of pages in the section
    
    -- Chunk Fields
    chunk_number INTEGER NOT NULL, -- Sequential chunk number within each section (starts at 1 for each section)
    chunk_content TEXT NOT NULL, -- The actual text content of this chunk (400-500 tokens)
    chunk_start_page INTEGER, -- First page number this chunk spans
    chunk_end_page INTEGER, -- Last page number this chunk spans
    
    -- Embedding Field
    embedding HALFVEC(3072), -- 3072-dimensional embedding vector using half precision (uses 2 bytes per dimension)
    
    -- Extra Fields
    extra1 TEXT, -- Flexible field for future use (can store any text data)
    extra2 TEXT, -- Flexible field for future use (can store any text data)
    extra3 TEXT, -- Flexible field for future use (can store any text data)
    
    -- System Fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP, -- Timestamp when the record was created
    last_modified TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP -- Timestamp when the record was last modified
);

-- VERSION 2: 2000 Dimensions (Fallback if HalfVec doesn't work)
/*
CREATE TABLE iris_semantic_search (
    -- Primary Key
    id SERIAL PRIMARY KEY, -- Auto-incrementing unique identifier for each record
    
    -- Document Fields
    document_id VARCHAR(255) NOT NULL, -- Unique identifier for the document (e.g., "EY_GUIDE_2024")
    filename VARCHAR(255) NOT NULL, -- Chapter-specific PDF filename (e.g., "03_Lease_Accounting.pdf")
    filepath TEXT, -- Full path to the chapter PDF file
    source_filename VARCHAR(255), -- Original source PDF filename before splitting into chapters
    
    -- Chapter Fields
    chapter_number INTEGER, -- Chapter number within the document
    chapter_name VARCHAR(500), -- Name/title of the chapter
    chapter_summary TEXT, -- GPT-generated summary of the entire chapter (2-3 sentences)
    chapter_page_count INTEGER, -- Total number of pages in the chapter
    
    -- Section Fields  
    section_number INTEGER, -- Sequential section number within the chapter
    section_summary TEXT, -- Hierarchical section summary with breadcrumb format (Chapter > Section: Summary)
    section_start_page INTEGER, -- First page number where this section begins
    section_end_page INTEGER, -- Last page number where this section ends
    section_page_count INTEGER, -- Total number of pages in the section
    
    -- Chunk Fields
    chunk_number INTEGER NOT NULL, -- Sequential chunk number within each section (starts at 1 for each section)
    chunk_content TEXT NOT NULL, -- The actual text content of this chunk (400-500 tokens)
    chunk_start_page INTEGER, -- First page number this chunk spans
    chunk_end_page INTEGER, -- Last page number this chunk spans
    
    -- Embedding Field
    embedding VECTOR(2000), -- 2000-dimensional embedding vector (uses 4 bytes per dimension)
    
    -- Extra Fields
    extra1 TEXT, -- Flexible field for future use (can store any text data)
    extra2 TEXT, -- Flexible field for future use (can store any text data)
    extra3 TEXT, -- Flexible field for future use (can store any text data)
    
    -- System Fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP, -- Timestamp when the record was created
    last_modified TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP -- Timestamp when the record was last modified
);
*/

-- ============================================================================
-- NOTES FOR IT TEAM
-- ============================================================================

/*
VERSION 1 (HalfVec) Requirements:
- PostgreSQL with pgvector extension version 0.5.0 or higher
- HalfVec support was added in pgvector 0.5.0
- Check version: SELECT extversion FROM pg_extension WHERE extname = 'vector';
- If version < 0.5.0, uncomment and use Version 2 instead

VERSION 2 (Standard Vector) - Use if:
- pgvector version is below 0.5.0
- HalfVec type is not available
- Any compatibility issues arise

Storage Impact:
- Version 1 (HalfVec 3072): ~6KB per embedding
- Version 2 (Vector 2000): ~8KB per embedding

How HalfVec Works:
- Uses 16-bit floating point (half precision) instead of 32-bit
- 50% storage reduction compared to standard vectors
- Minimal precision loss (<1% impact on similarity scores)
- PostgreSQL automatically converts float32 embeddings to float16 on insert

Embedding Generation:
- Both versions work with OpenAI text-embedding-3-large
- Version 1: Use full 3072 dimensions
- Version 2: Request 2000 dimensions from the API using dimensions parameter

Table Usage Pattern:
- This table is truncated and reloaded on each refresh
- IDs will reset to 1 with each reload
- No need for complex versioning or update tracking
*/