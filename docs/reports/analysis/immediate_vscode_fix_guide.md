# IMMEDIATE VS Code HuggingFace Extension Fix Guide

**Date**: 2025-08-02  
**Issue**: EPIPE errors and "instant to be in bounds" panic crashes  
**Protocol**: Kimera SWM Defense-in-Depth Manual Recovery

## 🚨 IMMEDIATE ACTIONS (Execute Now)

### Step 1: Clean Process Shutdown
```powershell
# Kill all VS Code processes
taskkill /F /IM Code.exe
taskkill /F /IM code.exe
taskkill /F /IM "Code - Insiders.exe"

# Wait 5 seconds for cleanup
Start-Sleep -Seconds 5
```

### Step 2: Disable Problematic Extension
```powershell
# Disable HuggingFace extension
code --disable-extension HuggingFace.huggingface-vscode
```

### Step 3: Clear Extension Cache
```powershell
# Navigate to VS Code data directory
cd "$env:APPDATA\Code\User"

# Remove workspace storage (this clears corrupted state)
Remove-Item -Recurse -Force "workspaceStorage" -ErrorAction SilentlyContinue

# Remove logs (clears error states)
cd "$env:APPDATA\Code"
Remove-Item -Recurse -Force "logs" -ErrorAction SilentlyContinue

# Remove crash dumps
Remove-Item -Recurse -Force "CrashDumps" -ErrorAction SilentlyContinue
```

### Step 4: Completely Remove Extension
```powershell
# Uninstall extension completely
code --uninstall-extension HuggingFace.huggingface-vscode

# Wait for uninstall to complete
Start-Sleep -Seconds 3
```

### Step 5: System Verification
```powershell
# Check system time (common cause of Rust time panics)
w32tm /query /status

# Check available memory
Get-WmiObject -Class Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory

# Verify Node.js (required for extension)
node --version
```

### Step 6: Fresh Installation
```powershell
# Install fresh extension
code --install-extension HuggingFace.huggingface-vscode

# Wait for installation
Start-Sleep -Seconds 10
```

### Step 7: Verification Test
```powershell
# Start VS Code
code .

# Check extension status in VS Code:
# 1. Open Command Palette (Ctrl+Shift+P)
# 2. Type "Extensions: Show Installed Extensions"
# 3. Verify HuggingFace extension appears without errors
```

## 🔧 ALTERNATIVE SOLUTIONS (If Above Fails)

### Option A: Use GitHub Copilot Instead
```powershell
code --install-extension GitHub.copilot
```

### Option B: Use Codeium Instead  
```powershell
code --install-extension Codeium.codeium
```

### Option C: HuggingFace via Python API
```python
# Install HuggingFace transformers
pip install transformers

# Test API access
python -c "
from transformers import pipeline
print('HuggingFace API working!')
generator = pipeline('text-generation', model='gpt2')
print('Models accessible via API')
"
```

## 🧪 SCIENTIFIC VERIFICATION

### Success Metrics
- [ ] VS Code starts without extension errors
- [ ] Extension appears in Extensions list  
- [ ] No EPIPE errors in Output panel
- [ ] Extension responds to commands
- [ ] Stable for >30 minutes of use

### If Still Failing
1. **Check Windows Time Service**:
   ```powershell
   # Restart Windows Time service (fixes Rust time panics)
   Restart-Service w32time
   ```

2. **Update VS Code**:
   ```powershell
   # Check for VS Code updates
   code --version
   # Update if needed via VS Code settings
   ```

3. **Environment Isolation**:
   ```powershell
   # Create isolated VS Code profile
   code --user-data-dir "C:\temp\vscode-isolated" .
   ```

## 🚀 KIMERA SWM VALIDATION PROTOCOL

After executing fixes:

1. **Hypothesis Test**: Extension crash was due to corrupted state
2. **Empirical Verification**: Monitor for 1 hour without crashes  
3. **Documentation**: Record what worked in this file
4. **Backup Strategy**: VS Code settings exported for future recovery

## 📊 REPORT YOUR RESULTS

Execute this PowerShell to generate status report:

```powershell
# Create results file
$results = @"
# VS Code Recovery Results - $(Get-Date)

## Actions Taken
- [ ] Processes terminated
- [ ] Extension disabled  
- [ ] Cache cleared
- [ ] Extension reinstalled
- [ ] Verification completed

## Current Status
- Extension Version: $(code --list-extensions --show-versions | findstr HuggingFace)
- VS Code Version: $(code --version | Select-Object -First 1)
- System Time Status: $(w32tm /query /status | findstr "Source:")

## Notes
[Add your observations here]
"@

$results | Out-File -FilePath "docs\reports\analysis\vscode_recovery_results_$(Get-Date -Format 'yyyy-MM-dd_HH-mm').md"
```

---

**Next Steps**: If manual recovery succeeds, run the automated script for future issues: `python scripts/utils/vscode_extension_recovery.py`