# CODEBASE CLEANUP PROMPT
## Systematic Codebase Organization Task

---

### OBJECTIVE
Perform systematic codebase organization following the Kimera SWM Protocol v3.0 standards.

### PROCEDURE

#### 1. Initial Analysis
```markdown
Perform systematic codebase organization:
1. Scan all files recursively
2. Identify duplicates by content hash and AST similarity  
3. Classify by purpose (experimental/production/test/doc)
4. Generate migration script (no deletions, only archival)
5. Update all imports and references
6. Create cleanup report with metrics
```

#### 2. Classification Criteria

**Production Code** (`/backend/`, `/src/`)
- Fully tested and documented
- Used in main system operations
- Has proper error handling
- Follows coding standards

**Experimental Code** (`/experiments/`)
- Contains hypothesis documentation
- May have incomplete features
- Includes results and analysis
- Dated folder structure

**Archive Code** (`/archive/`)
- Deprecated functionality
- Historical implementations
- Must include DEPRECATED.md
- Dated folder structure

**Test Code** (`/tests/`)
- Unit tests
- Integration tests
- Performance benchmarks
- Test utilities

#### 3. Duplication Detection

```python
# Example duplication detection approach
import hashlib
import ast
from pathlib import Path

def find_duplicates(root_path):
    """Find duplicate files by content and AST similarity"""
    file_hashes = {}
    ast_signatures = {}
    duplicates = []
    
    for file_path in Path(root_path).rglob("*.py"):
        # Content hash
        content = file_path.read_bytes()
        content_hash = hashlib.sha256(content).hexdigest()
        
        # AST signature for Python files
        try:
            tree = ast.parse(content)
            ast_sig = generate_ast_signature(tree)
            
            # Check for duplicates
            if content_hash in file_hashes:
                duplicates.append((file_path, file_hashes[content_hash]))
            elif ast_sig in ast_signatures:
                duplicates.append((file_path, ast_signatures[ast_sig]))
            else:
                file_hashes[content_hash] = file_path
                ast_signatures[ast_sig] = file_path
        except:
            pass
    
    return duplicates
```

#### 4. Migration Script Template

```python
#!/usr/bin/env python
"""
Codebase Migration Script
Generated: {timestamp}
Protocol: Kimera SWM v3.0
"""

import shutil
from pathlib import Path
import json

MIGRATIONS = [
    {
        "source": "old/path/file.py",
        "destination": "new/path/file.py",
        "reason": "Reorganization for clarity",
        "update_imports": True
    }
]

def migrate():
    """Execute codebase migrations"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "migrations": [],
        "errors": []
    }
    
    for migration in MIGRATIONS:
        try:
            # Create backup
            backup_path = Path("archive") / f"{datetime.now():%Y%m%d}" / migration["source"]
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(migration["source"], backup_path)
            
            # Move file
            dest = Path(migration["destination"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(migration["source"], dest)
            
            # Update imports if needed
            if migration["update_imports"]:
                update_imports(migration["source"], migration["destination"])
            
            report["migrations"].append(migration)
        except Exception as e:
            report["errors"].append({
                "migration": migration,
                "error": str(e)
            })
    
    # Save report
    with open("migration_report.json", "w") as f:
        json.dump(report, f, indent=2)
```

#### 5. Cleanup Report Format

```yaml
cleanup_report:
  timestamp: "2025-01-23T10:00:00"
  statistics:
    total_files: 1234
    duplicates_found: 45
    files_archived: 23
    imports_updated: 156
    
  duplicates:
    - original: "path/to/original.py"
      duplicate: "path/to/copy.py"
      similarity: 98.5%
      action: "archived duplicate"
      
  reorganizations:
    - from: "scattered/location/file.py"
      to: "organized/module/file.py"
      reason: "Consolidating related functionality"
      
  quality_improvements:
    - file: "backend/core/engine.py"
      issues_fixed:
        - "Added missing docstrings"
        - "Fixed import order"
        - "Removed unused variables"
        
  recommendations:
    - "Consider merging similar modules X and Y"
    - "Module Z has high complexity, consider refactoring"
    - "Test coverage low in module W"
```

### EXECUTION CHECKLIST

- [ ] Create backup of entire codebase
- [ ] Run duplication detection
- [ ] Classify all files
- [ ] Generate migration plan
- [ ] Review migration plan
- [ ] Execute migrations
- [ ] Update all imports
- [ ] Run test suite
- [ ] Generate cleanup report
- [ ] Commit changes with detailed message

### SAFETY RULES

1. **NO DELETIONS** - Only archive with explanations
2. **MAINTAIN HISTORY** - Use git mv when possible
3. **TEST CONTINUOUSLY** - Run tests after each migration
4. **DOCUMENT EVERYTHING** - Every move needs justification
5. **REVERSIBILITY** - Ensure all changes can be rolled back

### OUTPUT ARTIFACTS

1. `migration_report.json` - Detailed migration log
2. `cleanup_report.yaml` - High-level summary
3. `archive/{date}/ARCHIVED.md` - Explanation of archived files
4. Updated `README.md` with new structure
5. Updated import statements across codebase

---

*Remember: Clean code is not just organized code—it's code that tells a clear story about its purpose and evolution.* 