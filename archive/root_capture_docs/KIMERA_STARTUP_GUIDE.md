# 🚀 KIMERA STARTUP GUIDE - ULTIMATE SOLUTION

**Never wonder where to run KIMERA again!**

This guide provides multiple foolproof ways to start KIMERA from anywhere. Choose the method that works best for you.

---

## 🎯 QUICK START (Recommended)

### For Windows Users:
1. **Double-click** `start_kimera.bat`
2. Choose option [1] to start KIMERA
3. That's it! 🎉

### For Linux/macOS Users:
1. **Double-click** `start_kimera.sh` or run `./start_kimera.sh`
2. Choose option [1] to start KIMERA  
3. That's it! 🎉

### Universal Python Method:
```bash
python run_kimera.py
```

---

## 🔧 FIRST TIME SETUP

If this is your first time running KIMERA:

### Windows:
1. Double-click `start_kimera.bat`
2. Choose option [3] for "First time setup"
3. Wait for installation to complete
4. Choose option [1] to start KIMERA

### Linux/macOS:
1. Run `./start_kimera.sh`
2. Choose option [3] for "First time setup"
3. Wait for installation to complete
4. Choose option [1] to start KIMERA

### Python Command:
```bash
python run_kimera.py --setup
python run_kimera.py
```

---

## 📁 WHERE TO RUN KIMERA

**The beauty of our solution:** You can run KIMERA from **anywhere** in the project!

✅ **These all work:**
- From project root: `python run_kimera.py`
- From any subdirectory: `python ../run_kimera.py`
- From scripts folder: `python ../run_kimera.py`
- From backend folder: `python ../run_kimera.py`

The startup script **automatically finds** the KIMERA project directory!

---

## 🚀 ALL STARTUP METHODS

### Method 1: Batch/Shell Scripts (Easiest)
```bash
# Windows
start_kimera.bat

# Linux/macOS
./start_kimera.sh
```

### Method 2: Python Script (Universal)
```bash
# Normal mode
python run_kimera.py

# Development mode (auto-reload)
python run_kimera.py --dev

# First time setup
python run_kimera.py --setup

# Check system
python run_kimera.py --help
```

### Method 3: Direct uvicorn (Advanced)
```bash
# Make sure you're in project root
cd "path/to/Kimera_SWM_Alpha_Prototype V0.1 140625"
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8001
```

### Method 4: Docker (If available)
```bash
docker-compose up -d
```

---

## 🌐 ACCESS KIMERA

Once started, KIMERA is available at:

- **Main Interface:** http://localhost:8001
- **API Documentation:** http://localhost:8001/docs
- **Health Check:** http://localhost:8001/system/health
- **System Status:** http://localhost:8001/system/status

---

## 🛠️ TROUBLESHOOTING

### Problem: "Python not found"
**Solution:**
- Install Python 3.10+ from https://www.python.org/downloads/
- Make sure Python is in your system PATH

### Problem: "KIMERA project files not found"
**Solution:**
- Make sure you're in the KIMERA project directory
- Look for folders: `backend/`, `requirements.txt`, `README.md`
- If not found, navigate to the correct directory

### Problem: "Dependencies missing"
**Solution:**
```bash
# Windows
start_kimera.bat → Choose option [3]

# Linux/macOS  
./start_kimera.sh → Choose option [3]

# Or directly
python run_kimera.py --setup
```

### Problem: "Port 8001 already in use"
**Solutions:**
1. Kill existing process using port 8001
2. Restart your computer
3. Use different port: `python -m uvicorn backend.api.main:app --port 8002`

### Problem: "Permission denied" (Linux/macOS)
**Solution:**
```bash
chmod +x start_kimera.sh
```

### Problem: "Module not found" or import errors
**Solution:**
1. Make sure virtual environment is activated
2. Run setup: `python run_kimera.py --setup`
3. Install requirements: `pip install -r requirements.txt`

---

## 🏗️ DEVELOPMENT MODE

For developers who want auto-reload when code changes:

```bash
# Windows
start_kimera.bat → Choose option [2]

# Linux/macOS
./start_kimera.sh → Choose option [2]

# Direct command
python run_kimera.py --dev
```

---

## 📋 SYSTEM REQUIREMENTS

- **Python:** 3.10 or higher
- **RAM:** 8GB minimum (16GB+ recommended)
- **Storage:** 5GB free space
- **GPU:** NVIDIA GPU recommended (optional)
- **OS:** Windows 10+, Linux, or macOS

---

## 🎉 SUCCESS INDICATORS

When KIMERA starts successfully, you'll see:

```
🚀 KIMERA SYSTEM LAUNCHER - ULTIMATE SOLUTION
   Kinetic Intelligence for Multidimensional Analysis
================================================================================
✅ Found KIMERA project at: /path/to/project
🐍 Using Python: python
📦 Checking dependencies...
✅ All dependencies available
🚀 Starting KIMERA...
🌐 KIMERA will be available at: http://localhost:8001
📚 API docs at: http://localhost:8001/docs
🔍 Health check: http://localhost:8001/system/health
Press Ctrl+C to stop the server
```

---

## 📞 STILL NEED HELP?

If you're still having trouble:

1. **Check the logs:** Look at `kimera_startup.log`
2. **Run diagnostics:** 
   ```bash
   python run_kimera.py --help
   ```
3. **Check system status:**
   ```bash
   # Windows
   start_kimera.bat → Choose option [4]
   
   # Linux/macOS
   ./start_kimera.sh → Choose option [4]
   ```

---

## 🏆 WHAT MAKES THIS SOLUTION ULTIMATE?

✅ **Works from anywhere** - Auto-detects project directory  
✅ **Multiple methods** - Choose what works for you  
✅ **Auto-setup** - Handles virtual environment and dependencies  
✅ **Cross-platform** - Windows, Linux, macOS support  
✅ **User-friendly** - Clear menus and error messages  
✅ **Developer-friendly** - Development mode with auto-reload  
✅ **Comprehensive** - Covers all possible scenarios  

**You'll never wonder "where should I run KIMERA?" again!** 🎯

---

## 📝 QUICK REFERENCE CARD

| What you want to do | Command |
|-------------------|---------|
| Start KIMERA (Windows) | Double-click `start_kimera.bat` |
| Start KIMERA (Linux/macOS) | `./start_kimera.sh` |
| Start KIMERA (any OS) | `python run_kimera.py` |
| First time setup | `python run_kimera.py --setup` |
| Development mode | `python run_kimera.py --dev` |
| Get help | `python run_kimera.py --help` |
| Check if working | Visit http://localhost:8001 |

---

**🎉 Congratulations! You now have the ultimate KIMERA startup solution!** 