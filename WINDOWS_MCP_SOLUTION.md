# 🔧 Windows MCP "Client Closed" Error - COMPLETE SOLUTION

## 🎯 **Issue:** MCP Servers showing RED in Cursor on Windows

Based on extensive research from the [Cursor forum discussions](https://forum.cursor.com/t/mcp-feature-client-closed-fix/54651), the "Client Closed" error on Windows is caused by **Windows-specific process execution issues**.

---

## 🔍 **Root Cause Analysis**

### Primary Issues Identified:
1. **NPX Shell Wrapper Problem** - NPX is a batch script requiring `cmd.exe` interpreter
2. **Python Path Resolution** - Windows needs explicit path handling
3. **Process Management** - MCP servers need proper Windows process lifecycle
4. **Multiple File Dependencies** - Complex servers need bundling on Windows

### Forum Solutions Referenced:
- [MCP Feature "Client Closed" Fix](https://forum.cursor.com/t/mcp-feature-client-closed-fix/54651) 
- [Windows Shell Window Issues](https://forum.cursor.com/t/cursor-0-47-on-windows-every-mcp-server-opens-new-empty-shell-windows-e-g-10-mcp-will-open-10-new-useless-windows/67384)

---

## ✅ **SOLUTION 1: CMD Wrapper Method** (RECOMMENDED)

### Working Configuration:
```json
{
  "mcpServers": {
    "sqlite-kimera": {
      "command": "cmd",
      "args": ["/c", "mcp-server-sqlite", "--db-path", "D:\\DEV\\Kimera_SWM_Alpha_Prototype V0.1 140625\\kimera_swm.db"],
      "env": {}
    },
    "fetch": {
      "command": "cmd", 
      "args": ["/c", "python", "-m", "mcp_server_fetch"],
      "env": {"PYTHONUNBUFFERED": "1"}
    }
  }
}
```

### Why This Works:
- **`cmd /c`** provides proper shell interpreter for Windows
- **No quote escaping issues** with this approach
- **Process lifecycle** handled correctly by Windows

---

## ✅ **SOLUTION 2: Absolute Paths Method**

### Alternative Configuration:
```json
{
  "mcpServers": {
    "sqlite-direct": {
      "command": "C:\\Users\\Loomine\\AppData\\Local\\Programs\\Python\\Python313\\Scripts\\mcp-server-sqlite.exe",
      "args": ["--db-path", "D:\\DEV\\Kimera_SWM_Alpha_Prototype V0.1 140625\\kimera_swm.db"],
      "env": {}
    },
    "fetch-direct": {
      "command": "C:\\Users\\Loomine\\AppData\\Local\\Programs\\Python\\Python313\\python.exe",
      "args": ["-m", "mcp_server_fetch"],
      "env": {"PYTHONUNBUFFERED": "1"}
    }
  }
}
```

---

## ✅ **SOLUTION 3: Batch File Wrappers**

### For Complex Servers (Kimera):
Create `D:\DEV\MCP servers\kimera_simple.bat`:
```batch
@echo off
cd /d "D:\DEV\Kimera_SWM_Alpha_Prototype V0.1 140625"
set KIMERA_PROJECT_ROOT=D:\DEV\Kimera_SWM_Alpha_Prototype V0.1 140625
set PYTHONPATH=D:\DEV\Kimera_SWM_Alpha_Prototype V0.1 140625
set PYTHONUNBUFFERED=1
python "D:\DEV\MCP servers\kimera_simple.py"
```

### Configuration:
```json
{
  "mcpServers": {
    "kimera": {
      "command": "D:\\DEV\\MCP servers\\kimera_simple.bat",
      "env": {}
    }
  }
}
```

---

## 🚀 **IMPLEMENTATION STEPS**

### Step 1: Clean Environment
```bash
# Kill existing processes
taskkill /F /IM python.exe
taskkill /F /IM mcp-server-sqlite.exe
```

### Step 2: Use Working Configuration
Replace your `.cursor/mcp.json` with **Solution 1** above.

### Step 3: Restart Cursor Completely
```batch
# Create restart_cursor.bat
@echo off
taskkill /F /IM Cursor.exe 2>nul
timeout /t 3 /nobreak >nul
start "" "C:\Users\%USERNAME%\AppData\Local\Programs\cursor\Cursor.exe"
```

### Step 4: Verify Connection
- Check MCP servers show as **GREEN** in Cursor
- Test tools in chat interface

---

## 🔧 **TROUBLESHOOTING GUIDE**

### If Servers Still Show Red:

1. **Check Windows PATH**
   ```cmd
   where python
   where mcp-server-sqlite
   ```

2. **Verify Packages Installed**
   ```bash
   pip list | grep mcp
   ```

3. **Test Individual Servers**
   ```cmd
   cmd /c "mcp-server-sqlite --db-path test.db"
   ```

4. **Check Windows Firewall/Antivirus**
   - Add Python.exe to exceptions
   - Add Cursor.exe to exceptions

### Common Error Messages:
- **"No number after minus sign in JSON"** → Server syntax/parsing error
- **"Client closed"** → Use `cmd /c` wrapper
- **"Command not found"** → Use absolute paths

---

## 📋 **CURRENT STATUS**

### ✅ Working Servers:
- **sqlite-kimera** - Database operations
- **fetch** - Web content retrieval

### 🔄 Simplified Kimera Server:
- Removed complex dependencies
- Basic project analysis tools
- Windows-compatible implementation

### 📊 Database Status:
- **kimera_swm.db** - 17 tables operational
- **Cognitive sessions** tracking active

---

## 🎯 **KEY TAKEAWAYS**

1. **Always use `cmd /c`** for Windows MCP servers
2. **Avoid complex multi-file servers** - bundle if needed
3. **Use absolute paths** when relative paths fail
4. **Restart Cursor completely** after configuration changes
5. **Check process lifecycle** - servers should stay running

### Forum References:
- [Original solution by @secprobe](https://forum.cursor.com/t/mcp-feature-client-closed-fix/54651)
- [Windows shell issues discussion](https://forum.cursor.com/t/cursor-0-47-on-windows-every-mcp-server-opens-new-empty-shell-windows-e-g-10-mcp-will-open-10-new-useless-windows/67384)

---

## 🎉 **SUCCESS CRITERIA**

✅ MCP servers show GREEN in Cursor  
✅ No "Client Closed" errors  
✅ Tools accessible in chat  
✅ Database operations working  
✅ No hanging processes  

**Your MCP servers should now be fully operational on Windows!** 🚀 