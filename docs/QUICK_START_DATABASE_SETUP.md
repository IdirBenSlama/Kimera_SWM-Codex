# Kimera SWM Database & Environment Quick Start Guide

## 🚀 One-Click Installation

For the simplest setup experience, run the one-click installer:

```bash
python scripts/install_kimera.py
```

This will automatically:
- Check prerequisites (Python, Docker, Git)
- Install Python dependencies  
- Set up all databases and configurations
- Create proper directory structure
- Run verification checks

## 📋 Manual Setup (Advanced)

If you prefer step-by-step control or the one-click installer fails:

### Step 1: Prerequisites

Ensure you have:
- **Python 3.8+** with pip
- **Docker** and Docker Compose
- **Git** (for cloning/updates)

### Step 2: Install Dependencies

```bash
# Basic dependencies
pip install -r requirements_consolidated/base.txt

# Or if that file doesn't exist:
pip install fastapi uvicorn sqlalchemy psycopg2-binary redis neo4j numpy pandas
```

### Step 3: Run Complete Setup

```bash
python scripts/database_setup/complete_environment_setup.py
```

This script will:
- ✅ Create unified database configurations
- ✅ Set up environment-specific configs (dev/test/staging/prod)
- ✅ Install database services via Docker (if needed)
- ✅ Create SQLite fallback databases
- ✅ Generate comprehensive setup report

### Step 4: Verify Setup

```bash
python scripts/verification/verify_setup.py
```

## 🗃️ Database Services

Kimera SWM uses multiple databases for different purposes:

### PostgreSQL (Primary Database)
- **Purpose**: Main relational data with vector support
- **Port**: 5432
- **Database**: `kimera_swm`
- **Features**: pgvector extension for embeddings

### Redis (Cache & Pub/Sub)
- **Purpose**: High-speed caching and message queuing
- **Port**: 6379
- **Features**: Persistence, memory optimization

### Neo4j (Graph Database)
- **Purpose**: Symbolic relationships and graph operations
- **Ports**: 7474 (HTTP), 7687 (Bolt)
- **Features**: APOC plugins, Graph Data Science

### SQLite (Fallback)
- **Purpose**: Local fallback when other databases unavailable
- **Location**: `data/databases/kimera_swm.db`
- **Features**: All essential tables with indexes

## 🐳 Docker Setup

### Start All Database Services

```bash
cd configs/docker
docker-compose -f database-services-compose.yml up -d
```

### Stop All Database Services

```bash
cd configs/docker  
docker-compose -f database-services-compose.yml down
```

### View Service Logs

```bash
cd configs/docker
docker-compose -f database-services-compose.yml logs -f
```

### Individual Service Management

```bash
# Start only PostgreSQL
docker-compose -f database-services-compose.yml up -d postgres

# Start only Redis
docker-compose -f database-services-compose.yml up -d redis

# Start only Neo4j
docker-compose -f database-services-compose.yml up -d neo4j
```

## 🔧 Configuration

### Environment Selection

Set the environment using:

```bash
export KIMERA_ENV=development  # or testing, staging, production
```

### Database Configuration Files

- **Unified Config**: `configs/database/unified_database_config.json`
- **PostgreSQL**: `configs/database/postgresql_config.json`
- **Redis**: `configs/database/redis_config.json`
- **Neo4j**: `configs/database/neo4j_config.json`

### Environment-Specific Configs

- **Development**: `configs/environments/development.yaml`
- **Testing**: `configs/environments/testing.yaml`
- **Staging**: `configs/environments/staging.yaml`
- **Production**: `configs/environments/production.yaml`

## 🔍 Health Checks

### Quick Health Check

```bash
python scripts/verification/verify_setup.py
```

### Comprehensive Health Check

```bash
python scripts/health_check/comprehensive_health_check.py
```

### Manual Service Checks

```bash
# Check PostgreSQL
psql -h localhost -U kimera_user -d kimera_swm -c "SELECT version();"

# Check Redis
redis-cli ping

# Check Neo4j (requires neo4j Python driver)
cypher-shell -u neo4j -p kimera_neo4j "RETURN 1;"
```

## 🚦 Start the Application

Once setup is complete:

```bash
# Start the main application
python src/main.py

# Or with specific environment
KIMERA_ENV=development python src/main.py
```

Access the API at: http://localhost:8000/docs

## 🛠️ Troubleshooting

### Common Issues

#### "Port already in use"
```bash
# Find what's using the port
lsof -i :5432  # PostgreSQL
lsof -i :6379  # Redis  
lsof -i :7687  # Neo4j

# Kill the process or change the port in config
```

#### "Docker not running"
```bash
# Start Docker Desktop (Windows/Mac)
# Or start Docker daemon (Linux)
sudo systemctl start docker
```

#### "Permission denied"
```bash
# Make scripts executable
chmod +x scripts/install_kimera.py
chmod +x scripts/verification/verify_setup.py
```

#### "Module not found"
```bash
# Install missing dependencies
pip install -r requirements_consolidated/base.txt

# Or install specific modules
pip install psycopg2-binary redis neo4j
```

### Fallback Mode

If external databases fail, Kimera SWM automatically falls back to SQLite:

- ✅ **SQLite database** created at `data/databases/kimera_swm.db`
- ✅ **All essential tables** with proper indexes
- ✅ **No external dependencies** required
- ⚠️ **Limited performance** compared to PostgreSQL/Redis

### Reset Everything

To completely reset the setup:

```bash
# Stop all Docker services
cd configs/docker
docker-compose -f database-services-compose.yml down -v

# Remove Docker volumes (WARNING: This deletes all data!)
docker volume prune

# Re-run setup
python scripts/database_setup/complete_environment_setup.py
```

## 📊 Monitoring

### Database Access Tools

- **PostgreSQL**: pgAdmin at http://localhost:5050 (if enabled)
  - Email: admin@kimera.ai
  - Password: kimera_admin

- **Redis**: Use redis-cli or Redis Commander
  ```bash
  redis-cli monitor  # Watch all commands
  ```

- **Neo4j**: Browser at http://localhost:7474
  - Username: neo4j
  - Password: kimera_neo4j

### Performance Monitoring

The system includes built-in performance monitoring. Check:

- **Reports**: `docs/reports/performance/`
- **Logs**: `logs/` and `data/logs/`
- **Health Reports**: `docs/reports/health/`

## 📈 Next Steps

1. **Explore the API**: http://localhost:8000/docs
2. **Run the test suite**: `python -m pytest tests/`
3. **Check the documentation**: `docs/`
4. **Review configuration**: `configs/`
5. **Monitor health**: Regular health checks

## 🆘 Getting Help

- **Setup Reports**: Check `docs/reports/health/` for detailed setup logs
- **Verification Reports**: Run verification script for current status
- **Log Files**: Check `logs/` for application logs
- **Configuration**: Review `configs/` for all settings

---

**Generated by Kimera SWM Setup v3.0**  
**Following KIMERA Protocol - Aerospace-Grade Database Setup**
