-- =============================================================================
-- SCHEMA UPDATE: Add document_source and document_name to iris_textbook_database
-- =============================================================================

-- Description:
-- This script adds two new fields to the iris_textbook_database table:
-- - document_source: Source identifier (e.g., 'external_sic', 'external_ifrs')
-- - document_name: Original document filename (e.g., 'ias-1-presentation-of-financial-statements.pdf')

-- Usage:
-- Run this script against your PostgreSQL database to add the new columns.
-- Existing data will have NULL values for these fields initially.

-- =============================================================================
-- Add new columns to iris_textbook_database table
-- =============================================================================

BEGIN;

-- Add document_source column
ALTER TABLE iris_textbook_database 
ADD COLUMN IF NOT EXISTS document_source TEXT;

-- Add document_name column  
ALTER TABLE iris_textbook_database 
ADD COLUMN IF NOT EXISTS document_name TEXT;

-- Add comments to describe the new columns
COMMENT ON COLUMN iris_textbook_database.document_source IS 'Source identifier for the document (e.g., external_sic, external_ifrs, external_ey)';
COMMENT ON COLUMN iris_textbook_database.document_name IS 'Original document filename (e.g., ias-1-presentation-of-financial-statements.pdf)';

-- Optional: Add indexes for performance (uncomment if needed)
-- CREATE INDEX IF NOT EXISTS idx_iris_textbook_document_source ON iris_textbook_database(document_source);
-- CREATE INDEX IF NOT EXISTS idx_iris_textbook_document_name ON iris_textbook_database(document_name);

-- Optional: Add combined index for common queries (uncomment if needed)
-- CREATE INDEX IF NOT EXISTS idx_iris_textbook_source_name ON iris_textbook_database(document_source, document_name);

COMMIT;

-- =============================================================================
-- Verification Query
-- =============================================================================

-- Run this query after the update to verify the new columns exist:
-- SELECT column_name, data_type, is_nullable, column_default 
-- FROM information_schema.columns 
-- WHERE table_name = 'iris_textbook_database' 
-- AND column_name IN ('document_source', 'document_name')
-- ORDER BY column_name;

-- =============================================================================
-- Example Usage After Update
-- =============================================================================

-- Example of inserting data with the new fields:
/*
INSERT INTO iris_textbook_database (
    document_source,
    document_name,
    document_id,
    chapter_number,
    section_number,
    part_number,
    sequence_number,
    chapter_name,
    content
) VALUES (
    'external_ifrs',
    'ias-1-presentation-of-financial-statements.pdf',
    'IAS_1_2024',
    1,
    1,
    1,
    1,
    'Introduction',
    'This chapter introduces the presentation of financial statements...'
);
*/

-- Example of querying with the new fields:
/*
SELECT document_source, document_name, chapter_name, section_title 
FROM iris_textbook_database 
WHERE document_source = 'external_ifrs'
ORDER BY chapter_number, section_number, sequence_number;
*/