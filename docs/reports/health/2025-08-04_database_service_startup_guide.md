# KIMERA SWM DATABASE SERVICE STARTUP GUIDE
**Date**: 2025-08-04  
**Purpose**: Complete database service restoration for full system functionality  
**Classification**: OPERATIONAL PROCEDURES

## OVERVIEW

Emergency repairs have been applied to make Kimera SWM crash-safe. To restore **FULL FUNCTIONALITY**, database services must be started manually.

## REQUIRED SERVICES

### Primary Services (Required)
1. **PostgreSQL** - Main persistent storage
2. **Redis** - High-speed caching and session storage

### Optional Services (Enhanced Features)
3. **QuestDB** - High-performance time-series analytics

## STARTUP OPTIONS

### Option A: System Services (Recommended for Development)

#### PostgreSQL
```bash
# Windows (if installed as service)
net start postgresql-x64-13

# Linux/macOS
sudo systemctl start postgresql
# OR
sudo service postgresql start

# Verify
psql -U postgres -c "SELECT version();"
```

#### Redis
```bash
# Windows (if installed as service)
net start redis

# Linux/macOS  
sudo systemctl start redis
# OR
sudo service redis-server start

# Verify
redis-cli ping
```

### Option B: Docker Services (Recommended for Isolation)

#### PostgreSQL with pgvector
```bash
# Start PostgreSQL with pgvector extension
docker run -d \
  --name kimera-postgres \
  -p 5432:5432 \
  -e POSTGRES_DB=kimera \
  -e POSTGRES_USER=kimera \
  -e POSTGRES_PASSWORD=kimera \
  -v kimera_postgres_data:/var/lib/postgresql/data \
  pgvector/pgvector:pg16

# Verify
docker exec kimera-postgres psql -U kimera -d kimera -c "SELECT version();"
```

#### Redis
```bash
# Start Redis  
docker run -d \
  --name kimera-redis \
  -p 6379:6379 \
  -v kimera_redis_data:/data \
  redis:latest redis-server --appendonly yes

# Verify
docker exec kimera-redis redis-cli ping
```

#### QuestDB (Optional)
```bash
# Start QuestDB
docker run -d \
  --name kimera-questdb \
  -p 9000:9000 \
  -p 9009:9009 \
  -v kimera_questdb_data:/var/lib/questdb \
  questdb/questdb:latest

# Verify  
curl http://localhost:9000
```

### Option C: Complete Docker Compose (Easiest)

Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: kimera-postgres
    environment:
      POSTGRES_DB: kimera
      POSTGRES_USER: kimera  
      POSTGRES_PASSWORD: kimera
    ports:
      - "5432:5432"
    volumes:
      - kimera_postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U kimera -d kimera"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:latest
    container_name: kimera-redis
    ports:
      - "6379:6379"
    volumes:
      - kimera_redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  questdb:
    image: questdb/questdb:latest
    container_name: kimera-questdb
    ports:
      - "9000:9000"
      - "9009:9009"  
    volumes:
      - kimera_questdb_data:/var/lib/questdb
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  kimera_postgres_data:
  kimera_redis_data:
  kimera_questdb_data:
```

Start all services:
```bash
docker-compose up -d
```

## VERIFICATION PROCEDURE

### Step 1: Service Health Check
```bash
# Run Kimera's database verification
python scripts/health_check/database_setup_verification.py
```

Expected output:
```
✅ PostgreSQL connection successful
✅ Redis connection successful  
✅ QuestDB connection successful (optional)
```

### Step 2: Kimera System Startup
```bash
# Start Kimera SWM
python src/main.py
```

Expected log output:
```
🗄️ Initializing database subsystem...
✅ Database initialized successfully
✅ Understanding Engine initialized successfully  
✅ Ethical Reasoning Engine initialized successfully
```

### Step 3: Full System Test
```bash
# Test API endpoints
curl http://localhost:8000/health
curl http://localhost:8000/docs

# Check engine status
curl http://localhost:8000/api/v1/system/status
```

## TROUBLESHOOTING

### PostgreSQL Connection Issues
```bash
# Check if service is running
docker ps | grep postgres
# OR
sudo systemctl status postgresql

# Check logs
docker logs kimera-postgres
# OR  
sudo journalctl -u postgresql

# Common fixes
- Ensure port 5432 is not blocked
- Verify credentials match configuration
- Check PostgreSQL is accepting connections
```

### Redis Connection Issues  
```bash
# Check if service is running
docker ps | grep redis
# OR
sudo systemctl status redis

# Check logs
docker logs kimera-redis
# OR
sudo journalctl -u redis

# Common fixes
- Ensure port 6379 is not blocked
- Verify Redis is bound to correct interface
- Check Redis configuration
```

### Database Configuration Issues
```bash
# Check Kimera database configuration
cat config/shared/database.json

# Verify environment variables
echo $DATABASE_URL
echo $REDIS_URL

# Update configuration if needed
```

## CONFIGURATION VERIFICATION

### Environment Variables
```bash
# Required environment variables
export DATABASE_URL="postgresql://kimera:kimera@localhost:5432/kimera"
export REDIS_URL="redis://localhost:6379/0"
export QUESTDB_URL="http://localhost:9000"  # Optional
```

### Configuration Files
Verify these files have correct database settings:
- `config/shared/database.json`
- `config/development.yaml`
- `.env` (if present)

## SUCCESS INDICATORS

### ✅ Full Recovery Achieved When:
1. All database services respond to health checks
2. Kimera starts without database initialization warnings
3. All engines initialize successfully
4. API endpoints respond correctly
5. Vault operations work properly

### Sample Success Log Output:
```
2025-08-04 14:15:00 - INFO - 🗄️ Initializing database subsystem...
2025-08-04 14:15:01 - INFO - ✅ Database initialized successfully
2025-08-04 14:15:02 - INFO - ✅ Understanding Engine initialized successfully
2025-08-04 14:15:03 - INFO - ✅ Ethical Reasoning Engine initialized successfully
2025-08-04 14:15:04 - INFO - ✅ Complexity Analysis Engine initialized successfully
2025-08-04 14:15:05 - INFO - 🚀 Kimera SWM System fully operational
```

## EMERGENCY CONTACT

If database startup fails:
1. Check the troubleshooting section above
2. Review emergency repair logs in `docs/reports/health/`
3. Ensure emergency fixes were applied correctly
4. Kimera system should still start in fallback mode

**Remember**: Even without databases, Kimera is now crash-safe and will operate in degraded mode thanks to the emergency repairs applied.

---

**Next**: After database services are running, restart Kimera SWM to achieve full functionality.
