/*
# Database Schema Documentation (`docs/database-schema.sql`)

This file contains the complete PostgreSQL database schema for the IRIS system, defining all tables, indexes, and relationships required for system operation.

## Overview

The IRIS database schema supports document management, content storage, and process monitoring for the intelligent retrieval and interaction system. The schema includes four primary tables designed to handle document cataloging, content storage, external textbook management, and operational monitoring. The design supports vector embeddings for semantic search, comprehensive process tracking, and efficient content retrieval across multiple document sources.

## Key Components

* **apg_catalog**: Document metadata and catalog information with vector embeddings
* **apg_content**: Document content storage with section-based organization
* **iris_textbook_database**: External textbook content with embeddings for semantic search
* **process_monitor_logs**: Comprehensive process monitoring and execution tracking

## Core Functions/Classes

### Document Catalog Table (apg_catalog)

#### Purpose
Stores document metadata, descriptions, and vector embeddings for semantic search and document management across internal and external sources.

#### Key Fields
* **System Fields**: Unique identifiers and creation timestamps
* **Document Identification**: Source, type, and name classification
* **Scope Fields**: AI-generated descriptions and usage guidance
* **Embedding Fields**: Vector embeddings for semantic search capabilities
* **Refresh Metadata**: File information and modification tracking

### Content Storage Table (apg_content)

#### Purpose
Stores actual document content organized by sections for efficient retrieval and processing by database subagents.

#### Key Fields
* **System Fields**: Unique identifiers and creation timestamps
* **Document References**: Links to catalog entries via source, type, and name
* **Content Organization**: Section-based content storage with summaries
* **Navigation Fields**: Page numbers and section ordering for content location

### External Textbook Table (iris_textbook_database)

#### Purpose
Stores external finance guidance content from providers like EY, PwC, KPMG, and IASB with detailed structural metadata and vector embeddings for semantic search.

#### Key Fields
* **System Fields**: Unique identifiers and creation timestamps
* **Structural Positioning**: Document hierarchy with chapter, section, part, and sequence numbering
* **Chapter Metadata**: Names, tags, summaries, and token counts for chapter-level organization
* **Section Metadata**: Pagination, importance scoring, hierarchy, titles, and standards references
* **Content & Embeddings**: Actual textbook content with vector embeddings and full-text search support

### Process Monitoring Table (process_monitor_logs)

#### Purpose
Comprehensive tracking of system execution including timing, token usage, costs, and operational details for performance monitoring and debugging.

#### Key Fields
* **Core Tracking**: Run UUIDs, stage names, timing, and status information
* **LLM Monitoring**: Token usage, costs, and response times for language model calls
* **Error Handling**: Status tracking and detailed error message storage
* **Metadata Storage**: Flexible JSON fields for environment and custom data

## Configuration

Database configuration requirements:

* **PostgreSQL Version**: Requires PostgreSQL with vector extension support (pgvector)
* **Vector Extension**: pgvector extension for embedding storage and similarity search
* **JSON Support**: JSONB support for flexible metadata and LLM call tracking
* **Timezone Support**: TIMESTAMPTZ fields for accurate time tracking across environments
* **Indexing Strategy**: Recommended indexes for performance optimization on lookup fields

## Usage Examples

### Table Creation
```sql
-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create all tables
\i database-schema.sql
```

### Document Insertion
```sql
INSERT INTO apg_catalog (document_source, document_type, document_name, document_description)
VALUES ('internal_capm', 'policy', 'Revenue Recognition Policy', 'Comprehensive policy for revenue recognition procedures');
```

### External Textbook Query
```sql
SELECT document_id, chapter_name, section_title, content 
FROM iris_textbook_database 
WHERE section_standard_codes && ARRAY['IFRS 16'] 
ORDER BY chapter_number, section_number, sequence_number;
```

### Process Monitoring Query
```sql
SELECT run_uuid, stage_name, duration_ms, total_cost 
FROM process_monitor_logs 
WHERE model_name = 'iris' 
ORDER BY stage_start_time DESC;
```

## Integration Points

The database schema integrates with multiple IRIS system components:

* **Database Subagents**: Query apg_catalog, apg_content, and iris_textbook_database for document retrieval
* **External Subagents**: PwC, EY, KPMG, and IASB subagents use iris_textbook_database for textbook searches
* **Vector Search**: Embedding fields support semantic similarity searches across all content tables
* **Process Monitoring**: All system components log execution details to process_monitor_logs
* **Content Management**: Document upload and refresh processes update catalog and content tables
* **Analytics**: Monitoring data supports performance analysis and system optimization

## Dependencies

* **PostgreSQL 12+**: Database engine with JSON and vector support
* **pgvector Extension**: Vector embedding storage and similarity operations
* **JSONB Support**: Flexible metadata storage for LLM calls and custom data
* **UUID Support**: Unique run identification across system executions

## Error Handling

Database schema includes error handling considerations:

* **Constraint Validation**: NOT NULL constraints on critical identification fields
* **Flexible Error Storage**: TEXT fields for detailed error messages and debugging information
* **Status Tracking**: Enumerated status fields for process outcome tracking
* **Data Integrity**: Foreign key relationships implicit through naming conventions
* **Recovery Support**: Comprehensive logging enables system state reconstruction

## Security Considerations

* **Access Control**: Database user permissions should restrict access based on component needs
* **Data Privacy**: Sensitive content stored with appropriate access controls
* **Audit Trail**: Complete process monitoring provides audit capabilities
* **Connection Security**: Database connections secured through SSL/TLS encryption
* **Credential Management**: Database credentials managed through secure environment configuration

## Performance Notes

* **Vector Indexing**: Vector fields should be indexed for efficient similarity searches
* **Partitioning Strategy**: Large tables may benefit from time-based partitioning
* **Query Optimization**: Indexes recommended on frequently queried fields (document_source, document_type)
* **Connection Pooling**: Connection pooling recommended for high-throughput environments
* **Monitoring Efficiency**: Process monitoring designed to minimize performance impact

---

This database schema provides comprehensive support for IRIS system operations including document management, content storage, and detailed process monitoring.
*/

-- =============================================================================
-- IRIS DATABASE SCHEMA
-- =============================================================================

-- 1. apg_catalog Table
CREATE TABLE apg_catalog (
    -- SYSTEM fields
    id SERIAL PRIMARY KEY,                        -- Auto-incrementing unique identifier
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(), -- When the record was added to the database
    
    -- DOCUMENT identification fields
    document_source VARCHAR(100) NOT NULL,        -- Source of the document (e.g., 'internal_capm', 'external_iasb')
    document_type VARCHAR(100) NOT NULL,          -- Type of document (e.g., 'capm', 'infographic', 'memo')
    document_name VARCHAR(255) NOT NULL,          -- Formatted document name (e.g., 'IFRS 9 - Financial Instruments')
    
    -- SCOPE fields
    document_description TEXT,                    -- Original AI-generated description of document usage/scope
    document_usage TEXT,                          -- New field for LLM selection/usage
    
    -- EMBEDDING fields
    document_usage_embedding vector(2000),        -- Vector embedding for document_usage
    document_description_embedding vector(2000),  -- Vector embedding for document_description
    
    -- REFRESH metadata fields
    date_created TIMESTAMP WITH TIME ZONE,        -- Original document creation date
    date_last_modified TIMESTAMP WITH TIME ZONE,  -- Date the document was last modified
    file_name VARCHAR(255),                       -- Full filename with extension (e.g., 'IFRS9_Financial_Instruments.pdf')
    file_type VARCHAR(50),                        -- File extension/type (e.g., '.pdf', '.docx', '.xlsx')
    file_size BIGINT,                             -- Size of the file in bytes
    file_path VARCHAR(1000),                      -- Full system path to the original file
    file_link VARCHAR(1000)                       -- URL or NAS path to the file
);

-- 2. apg_content Table
CREATE TABLE apg_content (
    -- SYSTEM fields
    id SERIAL PRIMARY KEY,                        -- Auto-incrementing unique identifier
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(), -- When the record was added to the database
    
    -- DOCUMENT reference fields (matching catalog)
    document_source VARCHAR(100) NOT NULL,        -- Source of the document (matches apg_catalog)
    document_type VARCHAR(100) NOT NULL,          -- Type of document (matches apg_catalog)
    document_name VARCHAR(255) NOT NULL,          -- Document name (matches apg_catalog)
    
    -- CONTENT fields
    section_id INTEGER NOT NULL,                  -- Ordered sequence number within the document
    section_name VARCHAR(500),                    -- Title of the section/chapter
    section_summary TEXT,                         -- AI-generated summary of the section
    section_content TEXT NOT NULL,                -- The actual content of the section
    page_number INTEGER                           -- Page number for content breakdown
);

-- 3. iris_textbook_database Table
CREATE TABLE iris_textbook_database (
  -- SYSTEM FIELDS
  id SERIAL PRIMARY KEY,  -- Unique identifier for each chunk
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),  -- Record creation timestamp

  -- STRUCTURAL POSITIONING FIELDS
  document_id TEXT,              -- E.g., "IFRS_Handbook_2023" or "EY_GAAP_Guide_2024"
  chapter_number INT,           -- E.g., 4
  section_number INT,           -- E.g., 2
  part_number INT,              -- E.g., 1
  sequence_number INT,          -- E.g., 14 (position in chunk order)

  -- CHAPTER-LEVEL METADATA
  chapter_name TEXT,            -- E.g., "Leases" or "Revenue Recognition"
  chapter_tags TEXT[],          -- E.g., {"Financial_Instruments", "Disclosure_Requirements"}
  chapter_summary TEXT,         -- E.g., "This chapter explains revenue recognition..."
  chapter_token_count INT,      -- Total token count across the full chapter

  -- SECTION-LEVEL PAGINATION & IMPORTANCE
  section_start_page INT,       -- E.g., 142 (start page of section)
  section_end_page INT,         -- E.g., 143 (end page of section)
  section_importance_score FLOAT,  -- E.g., 0.85 (importance of this section)
  section_token_count INT,      -- Total token count for this section

  -- SECTION-LEVEL METADATA
  section_hierarchy TEXT,       -- E.g., "Chapter 4 > Section 4.2 > Subsection 4.2.3"
  section_title TEXT,           -- E.g., "Identification of Separate Performance Obligations"
  section_standard TEXT,        -- E.g., "IFRS", "US_GAAP", "AASB"
  section_standard_codes TEXT[], -- E.g., {"IFRS 16", "IAS 17", "IFRS 9"}
  section_references TEXT[],    -- E.g., {"Section 3.4", "IAS 36 Para 12-15"}

  -- CONTENT & EMBEDDING
  content TEXT NOT NULL,        -- The actual textbook content in this chunk
  embedding VECTOR(2000),       -- OpenAI's text-embedding-3-large model vector
  text_search_vector TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

-- 4. process_monitor_logs Table
CREATE TABLE IF NOT EXISTS process_monitor_logs (
    -- Core Fields --
    log_id BIGSERIAL PRIMARY KEY,                         -- Auto-incrementing unique ID for each log entry
    run_uuid UUID NOT NULL,                               -- Unique ID generated for each complete model invocation/run
    model_name VARCHAR(100) NOT NULL,                     -- Identifier for the model (e.g., 'iris', 'model_b')
    stage_name VARCHAR(100) NOT NULL,                     -- Name of the specific process stage (e.g., 'SSL_Setup', 'Router_Processing')
    stage_start_time TIMESTAMPTZ NOT NULL,                 -- Timestamp when the stage began
    stage_end_time TIMESTAMPTZ,                            -- Timestamp when the stage ended
    duration_ms INT,                                       -- Duration of the stage in milliseconds (calculated: end_time - start_time)
    llm_calls JSONB,                                       -- JSON array storing details for LLM calls within this stage
    total_tokens INT,                                      -- Sum of total tokens from all llm_calls in this stage (calculated)
    total_cost DECIMAL(12, 6),                             -- Sum of costs from all llm_calls in this stage (calculated)
    status VARCHAR(255),                                   -- Outcome/Status of the stage (e.g., 'Success', 'Failure', 'Clarification')
    decision_details TEXT,                                -- Text field for specific outputs or decisions (e.g., Router's chosen agent)
    error_message TEXT,                                   -- Detailed error message if the stage failed
    log_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,   -- Timestamp when this specific log row was created

    -- Optional Extra Fields for Future Use --
    user_id VARCHAR(255),                                 -- Optional: Identifier for the user initiating the request (if applicable)
    environment VARCHAR(50),                              -- Optional: Environment identifier (e.g., 'production', 'staging', 'development')
    custom_metadata JSONB,                                -- Optional: Flexible JSONB field for any other structured metadata
    notes TEXT                                             -- Optional: Free-form text field for additional notes or context
);

-- Comments for clarity --
COMMENT ON COLUMN process_monitor_logs.llm_calls IS 'JSON array storing details for LLM calls: [{"model": str, "input_tokens": int, "output_tokens": int, "cost": float, "response_time_ms": int}]';
COMMENT ON COLUMN process_monitor_logs.custom_metadata IS 'Flexible JSONB field for any other structured metadata specific to the invocation or environment.';
