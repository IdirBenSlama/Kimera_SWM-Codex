# 🚀 KIMERA SWM - Complete Deployment Guide

This guide provides foolproof instructions for deploying Kimera SWM on any PC. Follow these steps to get Kimera running quickly and reliably.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Setup (Recommended)](#quick-setup-recommended)
3. [Manual Setup](#manual-setup)
4. [Docker Setup](#docker-setup)
5. [Troubleshooting](#troubleshooting)
6. [Post-Installation](#post-installation)

---

## 🔧 Prerequisites

### System Requirements

- **Operating System**: Windows 10/11, Linux, or macOS
- **Python**: 3.10 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB free space
- **Network**: Internet connection for downloading dependencies

### Check Python Version

```bash
python --version
# or
python3 --version
```

If Python is not installed or version is below 3.10, download from [python.org](https://www.python.org/downloads/).

---

## 🚀 Quick Setup (Recommended)

This is the easiest way to get Kimera running on any PC:

### Step 1: Download Kimera

```bash
# If you have git:
git clone <repository_url>
cd Kimera_SWM_Alpha_Prototype

# Or download and extract the ZIP file
```

### Step 2: Run the Universal Deployment Script

```bash
python deploy_kimera.py
```

**That's it!** The script will:
- Check your system compatibility
- Create a virtual environment
- Install all dependencies
- Configure the environment
- Create startup scripts
- Verify the installation

### Step 3: Start Kimera

After successful deployment, you can start Kimera in any of these ways:

**Windows:**
```bash
# Double-click the file or run:
start_kimera.bat
```

**Linux/macOS:**
```bash
./start_kimera.sh
```

**Universal:**
```bash
python kimera.py
```

### Step 4: Access Kimera

Once started, Kimera will be available at:
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## 🔧 Manual Setup

If you prefer manual control or the automatic setup fails:

### Step 1: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install base requirements (faster)
pip install -r requirements/base.txt

# Install all requirements (comprehensive)
pip install -r requirements.txt
```

### Step 3: Configure Environment

```bash
# Copy the environment template
cp .env.example .env

# Edit .env file with your settings
# Use any text editor to modify the configuration
```

### Step 4: Start Kimera

```bash
python kimera.py
```

---

## 🐳 Docker Setup

For containerized deployment:

### Step 1: Install Docker

Download and install Docker from [docker.com](https://www.docker.com/get-started).

### Step 2: Build and Run

```bash
# Build the Docker image
docker build -t kimera:latest .

# Run the container
docker run -p 8000:8000 kimera:latest

# Or use Docker Compose
docker-compose up
```

### Step 3: Access Kimera

Kimera will be available at http://localhost:8000

---

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. Python Version Too Old
```bash
# Check Python version
python --version

# Install Python 3.10+ from python.org
# Or use pyenv (Linux/macOS):
pyenv install 3.10.0
pyenv global 3.10.0
```

#### 2. Permission Errors (Windows)
```bash
# Run Command Prompt as Administrator
# Or use PowerShell with execution policy:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 3. Port Already in Use
```bash
# Find process using port 8000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/macOS:
lsof -i :8000
kill -9 <PID>
```

#### 4. Import Errors
```bash
# Clear pip cache
pip cache purge

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### 5. Memory Issues
```bash
# Reduce memory usage in .env:
KIMERA_MAX_THREADS=4
KIMERA_GPU_MEMORY_FRACTION=0.5

# Close other applications
# Use swap space on Linux
```

#### 6. Network/Firewall Issues
```bash
# Allow Python through firewall
# Windows: Windows Defender Firewall > Allow an app
# Linux: sudo ufw allow 8000
```

### Dependency Installation Issues

If you encounter dependency conflicts:

```bash
# Clean install approach
pip uninstall -y -r requirements.txt
pip install -r requirements.txt

# Or use conda
conda create -n kimera python=3.10
conda activate kimera
pip install -r requirements.txt
```

### GPU/CUDA Issues

If you have GPU-related errors:

```bash
# CPU-only mode - edit .env:
KIMERA_USE_GPU=false

# Or install CPU-only PyTorch:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## 📊 Post-Installation

### Verify Installation

```bash
# Check system health
curl http://localhost:8000/health

# Check API documentation
# Visit http://localhost:8000/docs in your browser
```

### Performance Tuning

Edit `.env` file to optimize performance:

```ini
# For low-end systems
KIMERA_MAX_THREADS=4
KIMERA_GPU_MEMORY_FRACTION=0.5

# For high-end systems
KIMERA_MAX_THREADS=16
KIMERA_GPU_MEMORY_FRACTION=0.8
```

### Monitoring

```bash
# View logs
tail -f logs/kimera.log

# Monitor system resources
# Install htop/task manager
```

### Security

```bash
# Change default API key in .env
KIMERA_API_KEY=your-secure-api-key-here

# Use environment variables for sensitive data
export OPENAI_API_KEY=your-openai-key
```

---

## 📝 Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KIMERA_ENV` | `development` | Environment mode |
| `KIMERA_HOST` | `127.0.0.1` | Server host |
| `KIMERA_PORT` | `8000` | Server port |
| `KIMERA_LOG_LEVEL` | `INFO` | Logging level |
| `KIMERA_MAX_THREADS` | `8` | Maximum threads |
| `KIMERA_GPU_MEMORY_FRACTION` | `0.8` | GPU memory usage |
| `DATABASE_URL` | `sqlite:///kimera.db` | Database URL |

### API Keys

Add your API keys to `.env`:

```ini
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
```

---

## 🔍 System Health Checks

### Quick Health Check

```bash
# Test basic functionality
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "timestamp": "2025-01-XX",
  "service": "kimera-swm-api"
}
```

### Performance Test

```bash
# Run performance test
python performance_test_kimera.py

# Monitor real-time performance
python monitor_kimera_realtime.py
```

---

## 📞 Support

If you encounter issues not covered in this guide:

1. **Check the logs**: Look in `logs/` directory
2. **Verify system requirements**: Ensure Python 3.10+, adequate RAM
3. **Try clean installation**: Remove `venv/` and run `python deploy_kimera.py`
4. **Check network connectivity**: Ensure internet access for downloads
5. **Review error messages**: Most errors include helpful hints

### Diagnostic Information

When reporting issues, include:

```bash
# System information
python -c "import platform; print(platform.platform())"
python --version

# Kimera status
curl http://localhost:8000/health

# Error logs
cat logs/kimera.log | tail -50
```

---

## 🎯 Success Indicators

Kimera is successfully deployed when:

✅ **Server starts without errors**
✅ **Health check returns 200 OK**
✅ **API documentation loads at /docs**
✅ **No critical errors in logs**
✅ **System responds to basic requests**

**You're ready to use Kimera! 🚀**

---

*This guide ensures Kimera can be reliably deployed on any PC with minimal effort. For advanced configurations, consult the technical documentation.* 