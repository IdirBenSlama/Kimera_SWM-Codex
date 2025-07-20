-- Kimera PostgreSQL Initialization Script

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- Create optimized indexes function
CREATE OR REPLACE FUNCTION create_optimized_indexes() RETURNS void AS $$
BEGIN
    -- Create GIN indexes for JSON fields
    CREATE INDEX IF NOT EXISTS idx_geoids_symbolic_state_gin ON geoids USING gin (symbolic_state);
    CREATE INDEX IF NOT EXISTS idx_geoids_metadata_gin ON geoids USING gin (metadata_json);
    CREATE INDEX IF NOT EXISTS idx_scars_geoids_gin ON scars USING gin (geoids);
    
    -- Create vector indexes for similarity search
    CREATE INDEX IF NOT EXISTS idx_geoids_semantic_vector ON geoids USING ivfflat (semantic_vector vector_cosine_ops) WITH (lists = 100);
    CREATE INDEX IF NOT EXISTS idx_scars_vector ON scars USING ivfflat (scar_vector vector_cosine_ops) WITH (lists = 100);
    
    -- Create B-tree indexes for frequently queried columns
    CREATE INDEX IF NOT EXISTS idx_scars_timestamp ON scars (timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_scars_vault_id ON scars (vault_id);
    CREATE INDEX IF NOT EXISTS idx_insights_created_at ON insights (created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_insights_type ON insights (insight_type);
    
    -- Create partial indexes for performance
    CREATE INDEX IF NOT EXISTS idx_scars_unresolved ON scars (scar_id) WHERE resolved_by IS NULL;
    CREATE INDEX IF NOT EXISTS idx_insights_provisional ON insights (insight_id) WHERE status = 'provisional';
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE kimera_swm TO kimera;
GRANT CREATE ON SCHEMA public TO kimera;

-- Performance settings
ALTER DATABASE kimera_swm SET shared_buffers = '256MB';
ALTER DATABASE kimera_swm SET effective_cache_size = '1GB';
ALTER DATABASE kimera_swm SET maintenance_work_mem = '64MB';
ALTER DATABASE kimera_swm SET work_mem = '16MB';

-- Connection settings
ALTER DATABASE kimera_swm SET max_connections = 100;
ALTER DATABASE kimera_swm SET idle_in_transaction_session_timeout = '30min';

-- Logging settings for development
ALTER DATABASE kimera_swm SET log_statement = 'mod';
ALTER DATABASE kimera_swm SET log_duration = on;
