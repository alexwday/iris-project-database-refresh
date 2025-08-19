# Stage 3: Database Schema for Chunk Storage

## Table Name: `semantic_chunks`

### Core Fields

| Field Name | Data Type | Description | Constraints |
|------------|-----------|-------------|-------------|
| `id` | UUID | Primary key, auto-generated | PRIMARY KEY, DEFAULT gen_random_uuid() |
| `document_id` | VARCHAR(100) | Document identifier (e.g., "EY_GUIDE_2024") | NOT NULL, INDEX |
| `filename` | VARCHAR(255) | Source chapter filename | NOT NULL |
| `source_filename` | VARCHAR(255) | Original document filename | NOT NULL |

### Chapter Context Fields

| Field Name | Data Type | Description | Constraints |
|------------|-----------|-------------|-------------|
| `chapter_number` | INTEGER | Chapter number | INDEX |
| `chapter_name` | VARCHAR(500) | Chapter title/name | |
| `chapter_summary` | TEXT | GPT-generated chapter summary | |
| `chapter_page_count` | INTEGER | Total pages in chapter | |

### Section Context Fields

| Field Name | Data Type | Description | Constraints |
|------------|-----------|-------------|-------------|
| `section_number` | INTEGER | Section number within chapter | INDEX |
| `section_summary` | TEXT | Hierarchical section summary with breadcrumbs | |
| `section_start_page` | INTEGER | First page of section | |
| `section_end_page` | INTEGER | Last page of section | |
| `section_page_count` | INTEGER | Total pages in section | |
| `section_references` | JSONB | Array of cross-references from section | DEFAULT '[]'::jsonb |

### Chunk-Specific Fields

| Field Name | Data Type | Description | Constraints |
|------------|-----------|-------------|-------------|
| `chunk_number` | INTEGER | Sequential chunk number within section | NOT NULL, INDEX |
| `chunk_sequence` | INTEGER | Global sequence number across document | UNIQUE, NOT NULL, INDEX |
| `chunk_content` | TEXT | The actual chunk text content | NOT NULL |
| `chunk_token_count` | INTEGER | Token count for this chunk | NOT NULL |
| `chunk_start_page` | INTEGER | First page this chunk spans | |
| `chunk_end_page` | INTEGER | Last page this chunk spans | |
| `chunk_pages` | INTEGER[] | Array of page numbers this chunk spans | |

### Embedding Fields (Choose One Version)

#### Version A: 2000 Dimensions (text-embedding-3-small with dimension parameter)
| Field Name | Data Type | Description | Constraints |
|------------|-----------|-------------|-------------|
| `embedding` | VECTOR(2000) | 2000-dimensional embedding vector | |
| `embedding_model` | VARCHAR(50) | Model used for embedding | DEFAULT 'text-embedding-3-small-2000' |

#### Version B: 3072 Dimensions with halfvec (text-embedding-3-large)
| Field Name | Data Type | Description | Constraints |
|------------|-----------|-------------|-------------|
| `embedding` | HALFVEC(3072) | 3072-dimensional embedding using half precision | |
| `embedding_model` | VARCHAR(50) | Model used for embedding | DEFAULT 'text-embedding-3-large' |

### Metadata & Placeholder Fields

| Field Name | Data Type | Description | Constraints |
|------------|-----------|-------------|-------------|
| `metadata_1` | JSONB | Placeholder for additional structured data | DEFAULT '{}'::jsonb |
| `metadata_2` | JSONB | Placeholder for additional structured data | DEFAULT '{}'::jsonb |
| `custom_field_1` | VARCHAR(500) | Placeholder for future string data | |
| `custom_field_2` | VARCHAR(500) | Placeholder for future string data | |
| `custom_field_3` | TEXT | Placeholder for future long text | |
| `score_field_1` | DECIMAL(10,4) | Placeholder for future scoring/ranking | |
| `score_field_2` | DECIMAL(10,4) | Placeholder for future scoring/ranking | |
| `flag_field_1` | BOOLEAN | Placeholder for future boolean flags | DEFAULT FALSE |
| `flag_field_2` | BOOLEAN | Placeholder for future boolean flags | DEFAULT FALSE |
| `tags` | TEXT[] | Array of tags for categorization | DEFAULT ARRAY[]::text[] |

### System Fields

| Field Name | Data Type | Description | Constraints |
|------------|-----------|-------------|-------------|
| `created_at` | TIMESTAMP WITH TIME ZONE | Record creation timestamp | DEFAULT CURRENT_TIMESTAMP |
| `updated_at` | TIMESTAMP WITH TIME ZONE | Last update timestamp | DEFAULT CURRENT_TIMESTAMP |
| `processing_version` | VARCHAR(20) | Pipeline version that created this record | |
| `is_active` | BOOLEAN | Soft delete flag | DEFAULT TRUE |

### Full-Text Search Field

| Field Name | Data Type | Description | Constraints |
|------------|-----------|-------------|-------------|
| `text_search_vector` | TSVECTOR | Full-text search vector | GENERATED ALWAYS AS (to_tsvector('english', chunk_content)) STORED |

## Indexes

```sql
-- Primary indexes
CREATE INDEX idx_semantic_chunks_document_id ON semantic_chunks(document_id);
CREATE INDEX idx_semantic_chunks_chapter_number ON semantic_chunks(chapter_number);
CREATE INDEX idx_semantic_chunks_section_number ON semantic_chunks(section_number);
CREATE INDEX idx_semantic_chunks_chunk_number ON semantic_chunks(chunk_number);
CREATE INDEX idx_semantic_chunks_chunk_sequence ON semantic_chunks(chunk_sequence);

-- Composite indexes for common queries
CREATE INDEX idx_semantic_chunks_doc_chapter_section ON semantic_chunks(document_id, chapter_number, section_number);
CREATE INDEX idx_semantic_chunks_doc_sequence ON semantic_chunks(document_id, chunk_sequence);

-- Full-text search index
CREATE INDEX idx_semantic_chunks_text_search ON semantic_chunks USING GIN(text_search_vector);

-- Vector similarity search index (choose based on embedding version)
-- For Version A (2000 dimensions):
CREATE INDEX idx_semantic_chunks_embedding ON semantic_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- For Version B (3072 dimensions halfvec):
CREATE INDEX idx_semantic_chunks_embedding ON semantic_chunks USING ivfflat (embedding halfvec_cosine_ops) WITH (lists = 100);

-- JSONB indexes for metadata fields
CREATE INDEX idx_semantic_chunks_metadata_1 ON semantic_chunks USING GIN(metadata_1);
CREATE INDEX idx_semantic_chunks_metadata_2 ON semantic_chunks USING GIN(metadata_2);
CREATE INDEX idx_semantic_chunks_section_references ON semantic_chunks USING GIN(section_references);
```

## Example Record

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "document_id": "EY_GUIDE_2024",
  "filename": "03_Lease_Accounting.pdf",
  "source_filename": "ey_ifrs_guide_2024.pdf",
  
  "chapter_number": 3,
  "chapter_name": "Lease Accounting Under IFRS 16",
  "chapter_summary": "This chapter covers IFRS 16 lease accounting requirements including lease identification, initial measurement, and subsequent measurement. It addresses key implementation challenges for lessees and lessors.",
  "chapter_page_count": 42,
  
  "section_number": 2,
  "section_summary": "Lease Accounting Under IFRS 16 > Initial Recognition: Explains the requirements for initial recognition of lease liabilities and right-of-use assets under IFRS 16. Details the measurement approaches and practical expedients available to lessees.",
  "section_start_page": 8,
  "section_end_page": 15,
  "section_page_count": 8,
  "section_references": ["IFRS 16.22-24", "IFRS 16.26", "Example 3.2"],
  
  "chunk_number": 3,
  "chunk_sequence": 145,
  "chunk_content": "The initial measurement of the lease liability includes the present value of lease payments not yet paid at the commencement date. This comprises fixed payments (including in-substance fixed payments), variable lease payments that depend on an index or rate, amounts expected to be payable under residual value guarantees...",
  "chunk_token_count": 448,
  "chunk_start_page": 10,
  "chunk_end_page": 11,
  "chunk_pages": [10, 11],
  
  "embedding": [0.0234, -0.0156, 0.0891, ...], // 2000 or 3072 dimensions
  "embedding_model": "text-embedding-3-small-2000",
  
  "metadata_1": {"processing_notes": "merged_small_section"},
  "metadata_2": {},
  "custom_field_1": null,
  "custom_field_2": null,
  "custom_field_3": null,
  "score_field_1": null,
  "score_field_2": null,
  "flag_field_1": false,
  "flag_field_2": false,
  "tags": ["IFRS16", "lease", "initial_measurement"],
  
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z",
  "processing_version": "v2.0.0",
  "is_active": true,
  
  "text_search_vector": "'initi':1 'measur':2 'leas':3,8 'liabil':4 ..."
}
```

## PostgreSQL Table Creation Scripts

### Version A: 2000 Dimensions

```sql
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Create table with 2000-dimensional embeddings
CREATE TABLE semantic_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id VARCHAR(100) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    source_filename VARCHAR(255) NOT NULL,
    
    chapter_number INTEGER,
    chapter_name VARCHAR(500),
    chapter_summary TEXT,
    chapter_page_count INTEGER,
    
    section_number INTEGER,
    section_summary TEXT,
    section_start_page INTEGER,
    section_end_page INTEGER,
    section_page_count INTEGER,
    section_references JSONB DEFAULT '[]'::jsonb,
    
    chunk_number INTEGER NOT NULL,
    chunk_sequence INTEGER UNIQUE NOT NULL,
    chunk_content TEXT NOT NULL,
    chunk_token_count INTEGER NOT NULL,
    chunk_start_page INTEGER,
    chunk_end_page INTEGER,
    chunk_pages INTEGER[],
    
    embedding VECTOR(2000),
    embedding_model VARCHAR(50) DEFAULT 'text-embedding-3-small-2000',
    
    metadata_1 JSONB DEFAULT '{}'::jsonb,
    metadata_2 JSONB DEFAULT '{}'::jsonb,
    custom_field_1 VARCHAR(500),
    custom_field_2 VARCHAR(500),
    custom_field_3 TEXT,
    score_field_1 DECIMAL(10,4),
    score_field_2 DECIMAL(10,4),
    flag_field_1 BOOLEAN DEFAULT FALSE,
    flag_field_2 BOOLEAN DEFAULT FALSE,
    tags TEXT[] DEFAULT ARRAY[]::text[],
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processing_version VARCHAR(20),
    is_active BOOLEAN DEFAULT TRUE,
    
    text_search_vector TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', chunk_content)) STORED
);
```

### Version B: 3072 Dimensions with halfvec

```sql
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Create table with 3072-dimensional halfvec embeddings
CREATE TABLE semantic_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id VARCHAR(100) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    source_filename VARCHAR(255) NOT NULL,
    
    chapter_number INTEGER,
    chapter_name VARCHAR(500),
    chapter_summary TEXT,
    chapter_page_count INTEGER,
    
    section_number INTEGER,
    section_summary TEXT,
    section_start_page INTEGER,
    section_end_page INTEGER,
    section_page_count INTEGER,
    section_references JSONB DEFAULT '[]'::jsonb,
    
    chunk_number INTEGER NOT NULL,
    chunk_sequence INTEGER UNIQUE NOT NULL,
    chunk_content TEXT NOT NULL,
    chunk_token_count INTEGER NOT NULL,
    chunk_start_page INTEGER,
    chunk_end_page INTEGER,
    chunk_pages INTEGER[],
    
    embedding HALFVEC(3072),
    embedding_model VARCHAR(50) DEFAULT 'text-embedding-3-large',
    
    metadata_1 JSONB DEFAULT '{}'::jsonb,
    metadata_2 JSONB DEFAULT '{}'::jsonb,
    custom_field_1 VARCHAR(500),
    custom_field_2 VARCHAR(500),
    custom_field_3 TEXT,
    score_field_1 DECIMAL(10,4),
    score_field_2 DECIMAL(10,4),
    flag_field_1 BOOLEAN DEFAULT FALSE,
    flag_field_2 BOOLEAN DEFAULT FALSE,
    tags TEXT[] DEFAULT ARRAY[]::text[],
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processing_version VARCHAR(20),
    is_active BOOLEAN DEFAULT TRUE,
    
    text_search_vector TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', chunk_content)) STORED
);
```

## Notes for IT Team

1. **Extension Requirements**:
   - PostgreSQL with `pgvector` extension for vector similarity search
   - `uuid-ossp` extension for UUID generation
   - Version 0.5.0+ of pgvector recommended for halfvec support

2. **Embedding Dimensions**:
   - **Option A (2000)**: Uses `text-embedding-3-small` with dimension reduction for efficiency
   - **Option B (3072)**: Uses `text-embedding-3-large` with halfvec for storage optimization while maintaining higher dimensionality

3. **Storage Considerations**:
   - VECTOR(2000) = ~8KB per embedding (4 bytes × 2000)
   - HALFVEC(3072) = ~6KB per embedding (2 bytes × 3072)
   - Halfvec provides 50% storage reduction with minimal accuracy loss

4. **Placeholder Fields**:
   - `metadata_1/2`: JSONB fields for future structured data
   - `custom_field_1/2/3`: String/text fields for future requirements
   - `score_field_1/2`: Decimal fields for ranking/scoring algorithms
   - `flag_field_1/2`: Boolean fields for future filtering needs
   - `tags`: Array field for categorization and filtering

5. **Performance Optimization**:
   - IVFFlat index on embedding column for fast similarity search
   - Adjust `lists` parameter based on dataset size (100 is a good starting point)
   - Consider HNSW index for better recall if query performance is critical

6. **Future Considerations**:
   - Add partition by `document_id` if multiple documents will be stored
   - Consider adding a `version` field if chunk versioning is needed
   - May want to add `chunk_hash` field for deduplication detection