#!/usr/bin/env python3
"""
Kimera Enhanced MCP Server
Advanced cognitive and development tools for Kimera architecture.
"""

import os
import json
import sqlite3
import subprocess
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import re

from fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("kimera-enhanced")

# Kimera project paths
KIMERA_ROOT = Path(os.getenv("KIMERA_PROJECT_ROOT", "D:/DEV/Kimera_SWM_Alpha_Prototype V0.1 140625"))
KIMERA_LOGS = KIMERA_ROOT / "logs"
KIMERA_REPORTS = KIMERA_ROOT / "reports"
KIMERA_BACKEND = KIMERA_ROOT / "backend"
KIMERA_DOCS = KIMERA_ROOT / "docs"

@mcp.tool()
def analyze_kimera_codebase(analysis_type: str = "structure", target_dir: str = "backend") -> Dict[str, Any]:
    """
    Comprehensive analysis of Kimera codebase structure, dependencies, and patterns.
    
    Args:
        analysis_type: Type of analysis ('structure', 'dependencies', 'patterns', 'metrics')
        target_dir: Directory to analyze (relative to Kimera root)
    
    Returns:
        Dictionary containing codebase analysis results
    """
    try:
        target_path = KIMERA_ROOT / target_dir
        if not target_path.exists():
            return {"error": f"Target directory {target_dir} not found"}
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "analysis_type": analysis_type,
            "target_directory": target_dir,
            "results": {}
        }
        
        if analysis_type == "structure":
            # Analyze directory structure
            structure = {}
            for item in target_path.rglob("*"):
                if item.is_file() and not any(skip in str(item) for skip in ['.git', '__pycache__', '.venv']):
                    rel_path = item.relative_to(target_path)
                    file_info = {
                        "size": item.stat().st_size,
                        "extension": item.suffix,
                        "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                    }
                    structure[str(rel_path)] = file_info
            
            analysis["results"]["file_structure"] = structure
            analysis["results"]["summary"] = {
                "total_files": len(structure),
                "file_types": list(set(info["extension"] for info in structure.values())),
                "total_size": sum(info["size"] for info in structure.values())
            }
            
        elif analysis_type == "dependencies":
            # Analyze Python imports and dependencies
            dependencies = set()
            for py_file in target_path.rglob("*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        # Extract import statements
                        imports = re.findall(r'^(?:from\s+(\S+)\s+import|import\s+(\S+))', content, re.MULTILINE)
                        for imp in imports:
                            dep = imp[0] if imp[0] else imp[1]
                            if dep and not dep.startswith('.'):
                                dependencies.add(dep.split('.')[0])
                except Exception as e:
                    # Could fail on weirdly encoded files, just skip them.
                    continue
            
            analysis["results"]["dependencies"] = sorted(list(dependencies))
            analysis["results"]["dependency_count"] = len(dependencies)
            
        elif analysis_type == "patterns":
            # Analyze code patterns and architecture
            patterns = {
                "classes": [],
                "functions": [],
                "decorators": [],
                "async_functions": []
            }
            
            for py_file in target_path.rglob("*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                        # Find classes
                        classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
                        patterns["classes"].extend([(cls, str(py_file.relative_to(target_path))) for cls in classes])
                        
                        # Find functions
                        functions = re.findall(r'^def\s+(\w+)', content, re.MULTILINE)
                        patterns["functions"].extend([(func, str(py_file.relative_to(target_path))) for func in functions])
                        
                        # Find decorators
                        decorators = re.findall(r'^@(\w+)', content, re.MULTILINE)
                        patterns["decorators"].extend(list(set(decorators)))
                        
                        # Find async functions
                        async_funcs = re.findall(r'^async\s+def\s+(\w+)', content, re.MULTILINE)
                        patterns["async_functions"].extend([(func, str(py_file.relative_to(target_path))) for func in async_funcs])
                        
                except Exception as e:
                    # Could fail on weirdly encoded files, just skip them.
                    continue
            
            analysis["results"]["patterns"] = patterns
            analysis["results"]["pattern_summary"] = {
                "total_classes": len(patterns["classes"]),
                "total_functions": len(patterns["functions"]),
                "unique_decorators": len(set(patterns["decorators"])),
                "async_functions": len(patterns["async_functions"])
            }
        
        return analysis
        
    except Exception as e:
        return {"error": f"Codebase analysis failed: {str(e)}"}

@mcp.tool()
def kimera_performance_monitor(metric_type: str = "system", time_range: str = "1h") -> Dict[str, Any]:
    """
    Monitor Kimera system performance and resource usage.
    
    Args:
        metric_type: Type of metrics ('system', 'memory', 'cpu', 'disk')
        time_range: Time range for analysis ('1h', '24h', '7d')
    
    Returns:
        Dictionary containing performance metrics
    """
    try:
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "metric_type": metric_type,
            "time_range": time_range,
            "system_info": {},
            "performance_data": {}
        }
        
        # Basic system information
        try:
            import psutil
            metrics["system_info"] = {
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "disk_usage": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent
            }
            
            if metric_type in ["system", "cpu"]:
                metrics["performance_data"]["cpu_percent"] = psutil.cpu_percent(interval=1)
                metrics["performance_data"]["cpu_per_core"] = psutil.cpu_percent(interval=1, percpu=True)
            
            if metric_type in ["system", "memory"]:
                memory = psutil.virtual_memory()
                metrics["performance_data"]["memory"] = {
                    "percent": memory.percent,
                    "available": memory.available,
                    "used": memory.used
                }
            
            if metric_type in ["system", "disk"]:
                disk = psutil.disk_usage('C:' if os.name == 'nt' else '/')
                metrics["performance_data"]["disk"] = {
                    "percent": disk.percent,
                    "free": disk.free,
                    "used": disk.used
                }
                
        except ImportError:
            metrics["system_info"]["note"] = "psutil not available for detailed metrics"
            
        # Analyze log file sizes and growth
        if KIMERA_LOGS.exists():
            log_metrics = {}
            for log_file in KIMERA_LOGS.glob("*.log"):
                stat = log_file.stat()
                log_metrics[log_file.name] = {
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
            metrics["performance_data"]["log_files"] = log_metrics
        
        return metrics
        
    except Exception as e:
        return {"error": f"Performance monitoring failed: {str(e)}"}

@mcp.tool()
def kimera_cognitive_session(action: str, session_data: str = "", session_id: str = "") -> Dict[str, Any]:
    """
    Manage cognitive sessions for context continuity and learning.
    
    Args:
        action: Action to perform ('start', 'update', 'retrieve', 'list', 'end')
        session_data: Data to store in session
        session_id: Unique session identifier
    
    Returns:
        Dictionary containing session management results
    """
    try:
        # Use SQLite for session storage
        session_db = Path("D:/DEV/MCP servers/kimera_sessions.db")
        
        with sqlite3.connect(session_db) as conn:
            cursor = conn.cursor()
            
            # Initialize sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cognitive_sessions (
                    id TEXT PRIMARY KEY,
                    session_data TEXT,
                    context_summary TEXT,
                    start_time TEXT,
                    last_update TEXT,
                    status TEXT DEFAULT 'active'
                )
            ''')
            
            result = {
                "action": action,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
            
            if action == "start":
                if not session_id:
                    session_id = f"kimera_session_{int(datetime.now().timestamp())}"
                
                cursor.execute('''
                    INSERT OR REPLACE INTO cognitive_sessions 
                    (id, session_data, start_time, last_update, status)
                    VALUES (?, ?, ?, ?, 'active')
                ''', (session_id, session_data, result["timestamp"], result["timestamp"]))
                
                result["session_id"] = session_id
                result["status"] = "started"
                
            elif action == "update":
                cursor.execute('''
                    UPDATE cognitive_sessions 
                    SET session_data = ?, last_update = ?
                    WHERE id = ?
                ''', (session_data, result["timestamp"], session_id))
                
                if cursor.rowcount > 0:
                    result["status"] = "updated"
                else:
                    result["status"] = "session_not_found"
                    
            elif action == "retrieve":
                cursor.execute('''
                    SELECT session_data, context_summary, start_time, last_update, status
                    FROM cognitive_sessions WHERE id = ?
                ''', (session_id,))
                
                row = cursor.fetchone()
                if row:
                    result["status"] = "found"
                    result["session_data"] = row[0]
                    result["context_summary"] = row[1]
                    result["start_time"] = row[2]
                    result["last_update"] = row[3]
                    result["session_status"] = row[4]
                else:
                    result["status"] = "not_found"
                    
            elif action == "list":
                cursor.execute('''
                    SELECT id, start_time, last_update, status
                    FROM cognitive_sessions 
                    ORDER BY last_update DESC LIMIT 20
                ''')
                
                sessions = []
                for row in cursor.fetchall():
                    sessions.append({
                        "id": row[0],
                        "start_time": row[1],
                        "last_update": row[2],
                        "status": row[3]
                    })
                
                result["status"] = "listed"
                result["sessions"] = sessions
                result["count"] = len(sessions)
                
            elif action == "end":
                cursor.execute('''
                    UPDATE cognitive_sessions 
                    SET status = 'ended', last_update = ?
                    WHERE id = ?
                ''', (result["timestamp"], session_id))
                
                if cursor.rowcount > 0:
                    result["status"] = "ended"
                else:
                    result["status"] = "session_not_found"
            
            conn.commit()
            return result
            
    except Exception as e:
        return {"error": f"Session management failed: {str(e)}"}

@mcp.tool()
def kimera_git_operations(operation: str, path: str = ".", params: str = "") -> Dict[str, Any]:
    """
    Advanced Git operations for Kimera project management.
    
    Args:
        operation: Git operation ('status', 'log', 'diff', 'branch', 'commit_stats')
        path: Path to operate on (relative to Kimera root)
        params: Additional parameters for the operation
    
    Returns:
        Dictionary containing Git operation results
    """
    try:
        target_path = KIMERA_ROOT / path if path != "." else KIMERA_ROOT
        
        result = {
            "operation": operation,
            "path": str(target_path),
            "timestamp": datetime.now().isoformat()
        }
        
        if operation == "status":
            # Get git status
            proc = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=target_path,
                capture_output=True,
                text=True
            )
            
            if proc.returncode == 0:
                status_lines = proc.stdout.strip().split('\n') if proc.stdout.strip() else []
                result["status"] = "success"
                result["changes"] = {
                    "modified": [line[3:] for line in status_lines if line.startswith(' M')],
                    "added": [line[3:] for line in status_lines if line.startswith('A ')],
                    "deleted": [line[3:] for line in status_lines if line.startswith(' D')],
                    "untracked": [line[3:] for line in status_lines if line.startswith('??')]
                }
                result["total_changes"] = len(status_lines)
            else:
                result["status"] = "error"
                result["error"] = proc.stderr
                
        elif operation == "log":
            # Get recent commit log
            limit = params if params.isdigit() else "10"
            proc = subprocess.run(
                ["git", "log", f"--max-count={limit}", "--oneline", "--decorate"],
                cwd=target_path,
                capture_output=True,
                text=True
            )
            
            if proc.returncode == 0:
                commits = []
                for line in proc.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(' ', 1)
                        commits.append({
                            "hash": parts[0],
                            "message": parts[1] if len(parts) > 1 else ""
                        })
                
                result["status"] = "success"
                result["commits"] = commits
                result["commit_count"] = len(commits)
            else:
                result["status"] = "error"
                result["error"] = proc.stderr
                
        elif operation == "branch":
            # Get branch information
            proc = subprocess.run(
                ["git", "branch", "-v"],
                cwd=target_path,
                capture_output=True,
                text=True
            )
            
            if proc.returncode == 0:
                branches = []
                current_branch = None
                
                for line in proc.stdout.strip().split('\n'):
                    if line:
                        is_current = line.startswith('*')
                        branch_info = line[2:] if is_current else line
                        parts = branch_info.split()
                        
                        if parts:
                            branch_data = {
                                "name": parts[0],
                                "hash": parts[1] if len(parts) > 1 else "",
                                "message": " ".join(parts[2:]) if len(parts) > 2 else ""
                            }
                            
                            if is_current:
                                current_branch = branch_data["name"]
                            
                            branches.append(branch_data)
                
                result["status"] = "success"
                result["branches"] = branches
                result["current_branch"] = current_branch
            else:
                result["status"] = "error"
                result["error"] = proc.stderr
        
        return result
        
    except Exception as e:
        return {"error": f"Git operation failed: {str(e)}"}

if __name__ == "__main__":
    mcp.run() 