# 🚀 KIMERA SWM - QUICK START GUIDE

**Get Kimera running in 5 minutes or less!**

## 📦 One-Command Setup

```bash
# Download and setup Kimera automatically
python deploy_kimera.py
```

**That's it!** The script handles everything automatically:
- ✅ Checks your system
- ✅ Creates virtual environment
- ✅ Installs all dependencies
- ✅ Configures environment
- ✅ Creates startup scripts

## 🏃 Start Kimera

After setup, start Kimera with any of these methods:

### Windows
```bash
# Double-click the file, or run:
start_kimera.bat
```

### Linux/macOS
```bash
./start_kimera.sh
```

### Universal (any platform)
```bash
python kimera.py
# or
python launch_kimera.py
```

## 🌐 Access Kimera

Once started, open your browser to:
- **Main Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🆘 Having Issues?

### First Time Setup Problems?
```bash
# Make sure you have Python 3.10+
python --version

# If version is too old, download from python.org
```

### Dependencies Not Installing?
```bash
# Try manual installation
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Port Already in Use?
```bash
# Kill the process using port 8000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/macOS:
lsof -i :8000
kill -9 <PID>
```

### Still Not Working?
1. **Check Python version**: Must be 3.10+
2. **Run as administrator** (Windows) or with `sudo` (Linux/macOS)
3. **Check firewall**: Allow Python through firewall
4. **Try Docker**: See [Docker Setup](#docker-setup) below

## 🐳 Docker Setup (Alternative)

If you prefer containerized deployment:

```bash
# Build and run with Docker
docker build -t kimera .
docker run -p 8000:8000 kimera

# Or use Docker Compose
docker-compose up
```

## 📝 Manual Setup (If Needed)

If automatic setup fails, try manual installation:

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Kimera
python kimera.py
```

## 🎯 Success Indicators

Kimera is working correctly when you see:
- ✅ Server starts without errors
- ✅ http://localhost:8000 loads
- ✅ http://localhost:8000/docs shows API documentation
- ✅ No error messages in terminal

## 🔧 System Requirements

- **Python**: 3.10 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB free space
- **OS**: Windows 10+, Linux, or macOS

---

**🎉 Congratulations! You now have Kimera running on your PC!**

For advanced configuration and troubleshooting, see the [Complete Deployment Guide](DEPLOYMENT_GUIDE.md). 