-- Main Postgres schema file containing all table schemas for the project

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

-- 3. process_monitor_logs Table
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
