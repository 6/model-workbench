-- =============================================================================
-- PostgreSQL Initialization Script
-- =============================================================================
-- Creates databases for each service.
-- Run automatically on first postgres container start.
-- =============================================================================

-- Create databases
CREATE DATABASE langfuse;
CREATE DATABASE mlflow;
CREATE DATABASE litellm;

-- Grant privileges (postgres user owns all)
GRANT ALL PRIVILEGES ON DATABASE langfuse TO postgres;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO postgres;
GRANT ALL PRIVILEGES ON DATABASE litellm TO postgres;

-- Performance tuning (applied to default postgres database)
-- These settings are also set via POSTGRES_INITDB_ARGS in docker-compose
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET work_mem = '16MB';
ALTER SYSTEM SET maintenance_work_mem = '128MB';
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;
