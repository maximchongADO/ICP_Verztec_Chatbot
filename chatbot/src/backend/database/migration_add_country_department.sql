-- Migration script to add country, department, and faiss_index_path columns
-- to the cleaned_texts table for the enhanced upload system

-- Add new columns to cleaned_texts table
ALTER TABLE cleaned_texts 
ADD COLUMN country VARCHAR(50) DEFAULT NULL,
ADD COLUMN department VARCHAR(50) DEFAULT NULL,
ADD COLUMN faiss_index_path VARCHAR(500) DEFAULT NULL;

-- Add indices for better query performance
CREATE INDEX idx_cleaned_texts_country ON cleaned_texts(country);
CREATE INDEX idx_cleaned_texts_department ON cleaned_texts(department);
CREATE INDEX idx_cleaned_texts_country_department ON cleaned_texts(country, department);

-- Update existing records to have NULL values (they will be migrated separately if needed)
-- No UPDATE needed as we're using DEFAULT NULL

-- Add comments to document the new columns
ALTER TABLE cleaned_texts 
MODIFY COLUMN country VARCHAR(50) DEFAULT NULL COMMENT 'Country where document belongs (china, singapore, etc.)',
MODIFY COLUMN department VARCHAR(50) DEFAULT NULL COMMENT 'Department where document belongs (hr, it, etc.)',
MODIFY COLUMN faiss_index_path VARCHAR(500) DEFAULT NULL COMMENT 'Path to the FAISS index where this document is stored';

-- Display the updated table structure
DESCRIBE cleaned_texts;
