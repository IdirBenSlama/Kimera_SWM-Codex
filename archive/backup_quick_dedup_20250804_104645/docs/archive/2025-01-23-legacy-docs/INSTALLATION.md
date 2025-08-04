# Kimera SWM Installation Guide

## System Requirements

### Hardware Requirements

- **CPU**: 4+ cores recommended (8+ for optimal performance)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 20GB available space
- **GPU**: CUDA-compatible NVIDIA GPU (RTX 2000 series or newer recommended)
  - VRAM: 8GB minimum, 12GB+ recommended
  - CUDA Toolkit: 11.7 or newer

### Software Requirements

- **Operating System**:
  - Linux: Ubuntu 20.04+, Debian 11+, or CentOS 8+
  - Windows: Windows 10/11 with WSL2 recommended
  - macOS: Monterey (12.0) or newer
- **Python**: 3.9+ (3.10 recommended)
- **PostgreSQL**: 15.x with pgvector extension
- **Docker**: 20.10+ and Docker Compose v2+ (for containerized deployment)
- **CUDA Toolkit**: 11.7+ (for GPU acceleration)
- **Git**: 2.30+

## Installation Methods

### Method 1: Standard Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/organization/kimera-swm.git
cd kimera-swm
```

#### 2. Create and Activate Virtual Environment

```bash
# Using venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# OR using conda
conda create -n kimera python=3.10
conda activate kimera
```

#### 3. Install Dependencies

```bash
# Install base requirements
pip install -r requirements/base.txt

# Install development tools (optional)
pip install -r requirements/dev.txt

# Install data science packages (optional)
pip install -r requirements/data.txt
```

#### 4. PostgreSQL Setup

```bash
# Install PostgreSQL (Ubuntu/Debian)
sudo apt update
sudo apt install postgresql-15 postgresql-contrib-15

# Install pgvector extension
sudo apt install postgresql-15-pgvector

# Create database and user
sudo -u postgres psql -c "CREATE USER kimera WITH PASSWORD 'kimera_secure_pass_2025';"
sudo -u postgres psql -c "CREATE DATABASE kimera_swm OWNER kimera;"
sudo -u postgres psql -d kimera_swm -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Configure connection
export DATABASE_URL="postgresql+psycopg2://kimera:kimera_secure_pass_2025@localhost:5432/kimera_swm"
```

#### 5. Environment Configuration

Create a `.env` file in the project root:

```bash
# Create .env file
cat > .env << EOL
# Database Configuration
DATABASE_URL=postgresql+psycopg2://kimera:kimera_secure_pass_2025@localhost:5432/kimera_swm
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10
DATABASE_POOL_TIMEOUT=30

# System Configuration
LOG_LEVEL=INFO
ENVIRONMENT=development
DEBUG=False

# GPU Configuration
USE_GPU=True
MIXED_PRECISION=True
MEMORY_LIMIT_GB=8

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
EOL
```

#### 6. Initialize the Database Schema

```bash
# Run database initialization script
python scripts/initialize_database.py
```

#### 7. Start the System

```bash
# Start the Kimera SWM system
python start_kimera.py
```

### Method 2: Docker Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/organization/kimera-swm.git
cd kimera-swm
```

#### 2. Configure Environment

Create a `.env` file for Docker:

```bash
# Create .env file for Docker
cat > .env << EOL
# PostgreSQL Configuration
POSTGRES_USER=kimera
POSTGRES_PASSWORD=kimera_secure_pass_2025
POSTGRES_DB=kimera_swm

# Kimera Configuration
DATABASE_URL=postgresql+psycopg2://kimera:kimera_secure_pass_2025@postgres:5432/kimera_swm
ENVIRONMENT=development
LOG_LEVEL=INFO
DEBUG=False
EOL
```

#### 3. Build and Start Docker Containers

```bash
# Build and start containers
docker-compose up -d
```

#### 4. Verify Installation

```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs -f kimera-api
```

## Advanced Configuration

### GPU Configuration

For optimal GPU utilization, configure the following parameters in `.env`:

```bash
# GPU Configuration
USE_GPU=True
MIXED_PRECISION=True
MEMORY_LIMIT_GB=8  # Adjust based on your GPU VRAM
CUDA_VISIBLE_DEVICES=0  # Specify GPU index if multiple GPUs are available
```

### Database Optimization

For production environments, optimize PostgreSQL with the following settings:

```bash
# Edit postgresql.conf
sudo nano /etc/postgresql/15/main/postgresql.conf

# Recommended settings
shared_buffers = 2GB  # 25% of RAM for dedicated server
effective_cache_size = 6GB  # 75% of RAM for dedicated server
maintenance_work_mem = 512MB
work_mem = 128MB
max_connections = 100
random_page_cost = 1.1  # For SSD storage
effective_io_concurrency = 200  # For SSD storage
max_worker_processes = 8  # Number of CPU cores
max_parallel_workers_per_gather = 4  # Half of CPU cores
max_parallel_workers = 8  # Number of CPU cores
```

### Security Configuration

For production deployments, implement the following security measures:

```bash
# Generate secure random password
export SECURE_PASSWORD=$(openssl rand -base64 32)

# Update database user password
sudo -u postgres psql -c "ALTER USER kimera WITH PASSWORD '$SECURE_PASSWORD';"

# Update .env file with new password
sed -i "s/kimera_secure_pass_2025/$SECURE_PASSWORD/g" .env

# Configure API authentication
cat >> .env << EOL
# API Security
API_KEY_REQUIRED=True
API_KEY=$(openssl rand -base64 32)
ALLOWED_ORIGINS=https://your-domain.com
EOL
```

## Verification and Testing

### System Verification

```bash
# Run system verification script
python scripts/verify_system.py

# Expected output:
# ✓ Database connection successful
# ✓ PostgreSQL version: 15.12
# ✓ pgvector extension available
# ✓ GPU acceleration available: NVIDIA GeForce RTX 2080 Ti
# ✓ CUDA version: 11.8
# ✓ All components initialized successfully
```

### Component Testing

```bash
# Run component tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run specific test
pytest tests/unit/test_thermodynamic_engine.py -v
```

## Troubleshooting

### Database Connection Issues

If you encounter database connection errors:

1. Verify PostgreSQL service is running:
   ```bash
   sudo systemctl status postgresql
   ```

2. Check connection parameters:
   ```bash
   psql -U kimera -h localhost -d kimera_swm
   ```

3. Verify pgvector extension:
   ```bash
   psql -U kimera -d kimera_swm -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
   ```

### GPU Acceleration Issues

If GPU acceleration is not working:

1. Check CUDA installation:
   ```bash
   nvcc --version
   nvidia-smi
   ```

2. Verify PyTorch CUDA compatibility:
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"
   ```

3. Set environment variables:
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   ```

## Deployment Considerations

### Production Deployment

For production environments:

1. Use a process manager like Supervisor or systemd:
   ```bash
   # Example systemd service file
   cat > /etc/systemd/system/kimera.service << EOL
   [Unit]
   Description=Kimera SWM System
   After=network.target postgresql.service
   
   [Service]
   User=kimera
   WorkingDirectory=/opt/kimera-swm
   Environment="PATH=/opt/kimera-swm/.venv/bin"
   ExecStart=/opt/kimera-swm/.venv/bin/python start_kimera.py
   Restart=always
   RestartSec=5
   
   [Install]
   WantedBy=multi-user.target
   EOL
   ```

2. Configure a reverse proxy with Nginx:
   ```bash
   # Example Nginx configuration
   cat > /etc/nginx/sites-available/kimera << EOL
   server {
       listen 80;
       server_name kimera.example.com;
       
       location / {
           proxy_pass http://localhost:8000;
           proxy_set_header Host \$host;
           proxy_set_header X-Real-IP \$remote_addr;
       }
   }
   EOL
   ```

3. Enable HTTPS with Let's Encrypt:
   ```bash
   sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx -d kimera.example.com
   ```

### Monitoring and Logging

For comprehensive monitoring:

1. Configure Prometheus metrics:
   ```bash
   # Add to .env
   ENABLE_METRICS=True
   METRICS_PORT=9090
   ```

2. Set up Grafana dashboards:
   ```bash
   # Start Grafana container
   docker-compose -f docker-compose.monitoring.yml up -d
   ```

3. Configure structured logging:
   ```bash
   # Add to .env
   LOG_FORMAT=json
   LOG_FILE=/var/log/kimera/kimera.log
   ```

## References

1. PostgreSQL Documentation: https://www.postgresql.org/docs/15/index.html
2. pgvector Documentation: https://github.com/pgvector/pgvector
3. CUDA Toolkit Documentation: https://docs.nvidia.com/cuda/
4. Docker Documentation: https://docs.docker.com/
5. PyTorch Installation Guide: https://pytorch.org/get-started/locally/ 