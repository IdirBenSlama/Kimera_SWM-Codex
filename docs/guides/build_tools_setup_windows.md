# Windows Build Tools Setup for KIMERA SWM

## 🎯 Quick Fix for Missing Build Tools

The analysis detected missing build tools (gcc, g++, make) on Windows. Here's how to fix this:

## ✅ Option 1: Visual Studio Build Tools (Recommended)

### Download and Install
1. Go to [Visual Studio Downloads](https://visualstudio.microsoft.com/downloads/)
2. Download **"Build Tools for Visual Studio 2022"** (Free)
3. Run the installer

### Installation Options
When the installer opens, select:
- **C++ build tools** workload
- **Windows 10/11 SDK** (latest version)
- **CMake tools for Visual Studio** (optional, you already have CMake)

### Verify Installation
Open a new Command Prompt and test:
```cmd
cl
# Should show Microsoft C/C++ Compiler

where cl
# Should show path to compiler
```

## ✅ Option 2: MinGW-w64 (Alternative)

### Install via MSYS2
1. Download [MSYS2](https://www.msys2.org/)
2. Install and run MSYS2
3. Run these commands in MSYS2 terminal:
```bash
pacman -S mingw-w64-x86_64-gcc
pacman -S mingw-w64-x86_64-make
pacman -S mingw-w64-x86_64-cmake
```

### Add to PATH
Add `C:\msys64\mingw64\bin` to your Windows PATH environment variable.

## 🚀 Quick Setup Script

Run this after installing build tools:

```cmd
# Run the automated setup
scripts\installation\setup_build_env_windows.bat
```

## ✅ Verify Complete Setup

After installation, run the compatibility check again:
```cmd
python scripts/analysis/python_313_compatibility_analysis.py
```

You should see:
- ✅ gcc: Found
- ✅ g++: Found  
- ✅ make: Found

## 🔧 Install Python 3.13 Compatible Dependencies

Once build tools are ready:
```cmd
pip install -r requirements/python313_compatible.txt
```

## ⚡ Performance Libraries

The following libraries will now compile properly:
- **torch**: Neural networks with CUDA support
- **cupy**: GPU arrays (already working)
- **fastrlock**: Fast threading locks  
- **numba**: JIT compilation

## 🛠️ Troubleshooting

### If compilation still fails:
1. **Restart your command prompt** after installing build tools
2. **Check PATH**: Ensure build tools are in your PATH
3. **Update pip**: `python -m pip install --upgrade pip setuptools wheel`
4. **Try one library at a time**: `pip install fastrlock` to test

### Common errors:
- `error: Microsoft Visual C++ 14.0 is required` → Install Visual Studio Build Tools
- `gcc not found` → Install MinGW-w64 or add to PATH
- `Permission denied` → Run as Administrator

## 🎉 Success!

After setup, your system will have:
- ✅ Python 3.13.3 (already working)
- ✅ All performance libraries (already installed) 
- ✅ Build tools for compilation
- ✅ CUDA support (already working)

Your KIMERA SWM system is now fully optimized for Python 3.13!
