# Kimera SWM Database Architecture

## Overview

The Kimera SWM system utilizes a sophisticated database architecture designed for high-dimensional vector operations, cognitive state persistence, and efficient retrieval of complex data structures. The primary database system is PostgreSQL 15.12 with the pgvector extension for vector embedding operations.

## Database Engine Specifications

### PostgreSQL Configuration

- **Version**: PostgreSQL 15.12
- **Extensions**: pgvector (for vector operations), uuid-ossp, jsonb
- **Connection Pooling**: Implemented via SQLAlchemy
  - Pool Size: 5 (configurable)
  - Max Overflow: 10
  - Pool Recycle: 3600 seconds
  - Pool Pre-Ping: Enabled
- **Performance Optimizations**:
  - Shared Buffers: 2GB (25% of available RAM)
  - Effective Cache Size: 6GB (75% of available RAM)
  - Work Memory: 128MB per connection
  - Maintenance Work Memory: 512MB
  - Random Page Cost: 1.1 (optimized for SSD)
  - Effective I/O Concurrency: 200 (optimized for SSD)

## Schema Design

### Core Tables

#### 1. Geoid States

Stores cognitive state representations as high-dimensional vectors.

```sql
CREATE TABLE geoid_states (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    state_vector VECTOR(1024) NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    entropy DOUBLE PRECISION NOT NULL,
    coherence_factor DOUBLE PRECISION NOT NULL,
    energy_level DOUBLE PRECISION NOT NULL,
    creation_context TEXT,
    tags TEXT[]
);

CREATE INDEX idx_geoid_vector ON geoid_states USING ivfflat (state_vector vector_cosine_ops);
CREATE INDEX idx_geoid_timestamp ON geoid_states (timestamp);
CREATE INDEX idx_geoid_entropy ON geoid_states (entropy);
CREATE INDEX idx_geoid_tags ON geoid_states USING GIN (tags);
```

#### 2. Cognitive Transitions

Records transitions between cognitive states with associated metrics.

```sql
CREATE TABLE cognitive_transitions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL REFERENCES geoid_states(id),
    target_id UUID NOT NULL REFERENCES geoid_states(id),
    transition_energy DOUBLE PRECISION NOT NULL,
    conservation_error DOUBLE PRECISION NOT NULL,
    transition_type TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_transitions_source ON cognitive_transitions (source_id);
CREATE INDEX idx_transitions_target ON cognitive_transitions (target_id);
CREATE INDEX idx_transitions_type ON cognitive_transitions (transition_type);
CREATE INDEX idx_transitions_timestamp ON cognitive_transitions (timestamp);
```

#### 3. Semantic Embeddings

Stores text embeddings for semantic operations.

```sql
CREATE TABLE semantic_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text_content TEXT NOT NULL,
    embedding VECTOR(1024) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    source TEXT,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_embedding_vector ON semantic_embeddings USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_embedding_source ON semantic_embeddings (source);
```

#### 4. Portal Configurations

Stores configurations for interdimensional portals.

```sql
CREATE TABLE portal_configurations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_dimension INTEGER NOT NULL,
    target_dimension INTEGER NOT NULL,
    radius DOUBLE PRECISION NOT NULL,
    energy_requirement DOUBLE PRECISION NOT NULL,
    stability_factor DOUBLE PRECISION NOT NULL,
    creation_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_used_timestamp TIMESTAMP WITH TIME ZONE,
    configuration_parameters JSONB NOT NULL,
    status TEXT NOT NULL
);

CREATE INDEX idx_portal_dimensions ON portal_configurations (source_dimension, target_dimension);
CREATE INDEX idx_portal_status ON portal_configurations (status);
```

#### 5. System Metrics

Records system performance metrics for monitoring and optimization.

```sql
CREATE TABLE system_metrics (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metric_name TEXT NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    component TEXT NOT NULL,
    context JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_metrics_timestamp ON system_metrics (timestamp);
CREATE INDEX idx_metrics_name ON system_metrics (metric_name);
CREATE INDEX idx_metrics_component ON system_metrics (component);
```

## Vector Operations

### Vector Indexing

The system utilizes IVFFlat indexing for efficient similarity searches:

```sql
CREATE INDEX idx_geoid_vector ON geoid_states USING ivfflat (state_vector vector_cosine_ops) WITH (lists = 100);
```

This index configuration provides:
- Lists: 100 (partitions for vectors)
- Distance Metric: Cosine similarity
- Approximate Nearest Neighbor (ANN) search capability

### Vector Queries

Example vector similarity query:

```sql
SELECT id, state_vector, entropy, coherence_factor,
       1 - (state_vector <=> $1) as similarity
FROM geoid_states
ORDER BY state_vector <=> $1
LIMIT 10;
```

Where `<=>` is the cosine distance operator provided by pgvector.

## Connection Management

### Connection Strategy Pattern

The database connection management system implements a sophisticated strategy pattern with multiple authentication approaches:

1. **Primary Strategy**: Kimera-specific credentials
   ```python
   def _connect_with_kimera_credentials(self):
       """Connect using Kimera-specific credentials."""
       url = "postgresql+psycopg2://kimera:kimera_secure_pass_2025@localhost:5432/kimera_swm"
       return self._create_engine(url)
   ```

2. **Secondary Strategy**: Environment variable configuration
   ```python
   def _connect_with_env_variables(self):
       """Connect using environment variables."""
       url = os.environ.get("DATABASE_URL")
       if not url:
           raise ValueError("DATABASE_URL environment variable not set")
       return self._create_engine(url)
   ```

3. **Tertiary Strategy**: SQLite fallback for development
   ```python
   def _connect_with_sqlite_fallback(self):
       """Connect using SQLite as fallback."""
       url = "sqlite:///kimera_development.db"
       return self._create_engine(url, sqlite_fallback=True)
   ```

### Engine Creation with Optimized Parameters

```python
def _create_engine(self, url, sqlite_fallback=False):
    """Create SQLAlchemy engine with optimized parameters."""
    connect_args = {}
    engine_args = {
        "pool_pre_ping": True,
        "pool_recycle": 3600,
        "pool_size": self.pool_size,
        "max_overflow": self.max_overflow,
        "pool_timeout": self.pool_timeout
    }
    
    if sqlite_fallback:
        connect_args["check_same_thread"] = False
        # Remove PostgreSQL-specific parameters
        engine_args.pop("pool_size", None)
        engine_args.pop("max_overflow", None)
    else:
        # PostgreSQL-specific optimizations
        connect_args["keepalives"] = 1
        connect_args["keepalives_idle"] = 30
        connect_args["keepalives_interval"] = 10
        connect_args["keepalives_count"] = 5
        connect_args["application_name"] = "Kimera SWM"
    
    return create_engine(
        url,
        connect_args=connect_args,
        **engine_args
    )
```

## Data Access Layer

### Session Management

The system implements a session factory pattern for database access:

```python
class DatabaseSessionManager:
    def __init__(self, engine):
        self.engine = engine
        self.Session = sessionmaker(bind=engine)
        self.scoped_session = scoped_session(self.Session)
    
    def get_session(self):
        """Get a new database session."""
        return self.scoped_session()
    
    def close_all_sessions(self):
        """Close all sessions."""
        self.scoped_session.remove()
```

### Repository Pattern

Data access is implemented using the repository pattern for clean separation of concerns:

```python
class GeoidRepository:
    def __init__(self, session):
        self.session = session
    
    def save(self, geoid):
        """Save a geoid state to the database."""
        self.session.add(geoid)
        self.session.commit()
        return geoid
    
    def find_by_id(self, geoid_id):
        """Find a geoid state by its ID."""
        return self.session.query(GeoidState).filter(GeoidState.id == geoid_id).first()
    
    def find_similar(self, vector, limit=10):
        """Find similar geoid states by vector similarity."""
        return self.session.query(
            GeoidState,
            func.cosine_similarity(GeoidState.state_vector, vector).label('similarity')
        ).order_by(desc('similarity')).limit(limit).all()
```

## Backup and Recovery

### Backup Strategy

The system implements a comprehensive backup strategy:

1. **Full Database Dumps**:
   ```bash
   pg_dump -U kimera -d kimera_swm -F c -f /backups/kimera_full_$(date +%Y%m%d_%H%M%S).dump
   ```

2. **Incremental WAL Archiving**:
   ```
   # In postgresql.conf
   wal_level = replica
   archive_mode = on
   archive_command = 'cp %p /backups/wal_archive/%f'
   ```

3. **Automated Backup Schedule**:
   - Full backups: Daily at 01:00 UTC
   - WAL archiving: Continuous
   - Retention policy: 30 days

### Recovery Procedures

1. **Full Database Restoration**:
   ```bash
   pg_restore -U postgres -d kimera_swm -c /backups/kimera_full_20250704_010000.dump
   ```

2. **Point-in-Time Recovery**:
   ```bash
   # In recovery.conf
   restore_command = 'cp /backups/wal_archive/%f %p'
   recovery_target_time = '2025-07-04 08:15:00 UTC'
   ```

## Performance Considerations

### Query Optimization

1. **Vector Search Optimization**:
   - Use of IVFFlat indexes for approximate nearest neighbor searches
   - Precomputed embeddings for frequently accessed entities
   - Batch processing for large vector operations

2. **Index Strategy**:
   - B-tree indexes on frequently queried scalar fields
   - GIN indexes for array and JSONB fields
   - Partial indexes for filtered queries

3. **Connection Pooling**:
   - Optimized pool size based on available resources
   - Connection recycling to prevent stale connections
   - Connection validation with pre-ping

## Monitoring and Maintenance

### Monitoring Queries

1. **Index Usage**:
   ```sql
   SELECT relname, idx_scan, idx_tup_read, idx_tup_fetch
   FROM pg_stat_user_indexes
   JOIN pg_index USING (indexrelid)
   JOIN pg_class ON pg_class.oid = pg_index.indexrelid
   WHERE schemaname = 'public'
   ORDER BY idx_scan DESC;
   ```

2. **Table Statistics**:
   ```sql
   SELECT relname, n_live_tup, n_dead_tup, last_vacuum, last_analyze
   FROM pg_stat_user_tables
   WHERE schemaname = 'public';
   ```

3. **Query Performance**:
   ```sql
   SELECT query, calls, total_time, mean_time, rows
   FROM pg_stat_statements
   ORDER BY total_time DESC
   LIMIT 10;
   ```

### Maintenance Tasks

1. **Vacuum Schedule**:
   - Regular VACUUM ANALYZE on all tables
   - Automated by autovacuum with optimized parameters

2. **Index Maintenance**:
   - Periodic REINDEX on vector indexes
   - Monitoring of index bloat

3. **Statistics Collection**:
   - Regular ANALYZE to update statistics
   - Custom statistics for vector columns

## References

1. PostgreSQL Documentation: https://www.postgresql.org/docs/15/index.html
2. pgvector Documentation: https://github.com/pgvector/pgvector
3. SQLAlchemy Documentation: https://docs.sqlalchemy.org/
4. Database Indexing Strategies: Schütze, H., Manning, C.D., & Raghavan, P. (2008). Introduction to Information Retrieval. Cambridge University Press.
5. Vector Similarity Search: Jégou, H., Douze, M., & Schmid, C. (2011). Product Quantization for Nearest Neighbor Search. IEEE Transactions on Pattern Analysis and Machine Intelligence, 33(1), 117-128. 