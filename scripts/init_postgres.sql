-- Initialize PostgreSQL for Kimera SWM
-- This script sets up the pgvector extension and initial schema

-- Create pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Grant all privileges to kimera user
GRANT ALL PRIVILEGES ON DATABASE kimera_swm TO kimera;

-- Create indexes for better performance
-- These will be created automatically by SQLAlchemy, but we can pre-create them
-- for better initial performance

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Kimera SWM PostgreSQL initialization complete!';
    RAISE NOTICE 'pgvector extension enabled for vector similarity search';
END $$;