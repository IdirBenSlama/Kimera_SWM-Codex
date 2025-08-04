#!/usr/bin/env python3
"""
Kimera Cognitive MCP Server
A specialized MCP server for Kimera's cognitive architecture needs.
"""

import os
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("kimera-cognitive")

# Kimera project paths
KIMERA_ROOT = Path("D:/DEV/Kimera_SWM_Alpha_Prototype V0.1 140625")
KIMERA_LOGS = KIMERA_ROOT / "logs"
KIMERA_REPORTS = KIMERA_ROOT / "reports"
KIMERA_BACKEND = KIMERA_ROOT / "backend"

@mcp.tool()
def analyze_kimera_logs(log_type: str = "all") -> Dict[str, Any]:
    """
    Analyze Kimera system logs for cognitive insights and performance metrics.
    
    Args:
        log_type: Type of logs to analyze ('all', 'cognitive', 'trading', 'errors')
    
    Returns:
        Dictionary containing log analysis results
    """
    try:
        if not KIMERA_LOGS.exists():
            return {"error": "Kimera logs directory not found"}
        
        log_files = list(KIMERA_LOGS.glob("*.log")) + list(KIMERA_LOGS.glob("*.json"))
        
        if not log_files:
            return {"warning": "No log files found"}
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "log_type": log_type,
            "files_analyzed": len(log_files),
            "insights": [],
            "performance_metrics": {},
            "recent_activities": []
        }
        
        # Analyze recent log files
        recent_files = sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]
        
        for log_file in recent_files:
            try:
                if log_file.suffix == '.json':
                    with open(log_file, 'r') as f:
                        data = json.load(f)
                        analysis["recent_activities"].append({
                            "file": log_file.name,
                            "type": "json",
                            "size": log_file.stat().st_size,
                            "modified": datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                        })
                else:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()[-50:]  # Last 50 lines
                        analysis["recent_activities"].append({
                            "file": log_file.name,
                            "type": "log",
                            "lines": len(lines),
                            "size": log_file.stat().st_size,
                            "modified": datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                        })
            except Exception as e:
                analysis["insights"].append(f"Could not read {log_file.name}: {str(e)}")
        
        return analysis
        
    except Exception as e:
        return {"error": f"Log analysis failed: {str(e)}"}

@mcp.tool()
def get_kimera_system_status() -> Dict[str, Any]:
    """
    Get comprehensive Kimera system status including components and health.
    
    Returns:
        Dictionary containing system status information
    """
    try:
        status = {
            "timestamp": datetime.now().isoformat(),
            "system_health": "checking",
            "components": {},
            "directories": {},
            "recent_reports": []
        }
        
        # Check core directories
        core_dirs = {
            "backend": KIMERA_BACKEND,
            "logs": KIMERA_LOGS,
            "reports": KIMERA_REPORTS,
            "root": KIMERA_ROOT
        }
        
        for name, path in core_dirs.items():
            status["directories"][name] = {
                "exists": path.exists(),
                "path": str(path),
                "files": len(list(path.glob("*"))) if path.exists() else 0
            }
        
        # Check for recent reports
        if KIMERA_REPORTS.exists():
            report_files = sorted(KIMERA_REPORTS.glob("*.json"), 
                                key=lambda x: x.stat().st_mtime, reverse=True)[:3]
            
            for report in report_files:
                status["recent_reports"].append({
                    "name": report.name,
                    "size": report.stat().st_size,
                    "modified": datetime.fromtimestamp(report.stat().st_mtime).isoformat()
                })
        
        # Overall health assessment
        healthy_dirs = sum(1 for d in status["directories"].values() if d["exists"])
        status["system_health"] = "healthy" if healthy_dirs >= 3 else "degraded"
        
        return status
        
    except Exception as e:
        return {"error": f"System status check failed: {str(e)}"}

@mcp.tool()
def search_kimera_codebase(query: str, file_type: str = "py") -> Dict[str, Any]:
    """
    Search through Kimera codebase for specific patterns or functions.
    
    Args:
        query: Search term or pattern
        file_type: File extension to search in ('py', 'md', 'json', 'all')
    
    Returns:
        Dictionary containing search results
    """
    try:
        if not KIMERA_ROOT.exists():
            return {"error": "Kimera root directory not found"}
        
        results = {
            "query": query,
            "file_type": file_type,
            "matches": [],
            "summary": {}
        }
        
        # Define search patterns
        if file_type == "all":
            patterns = ["*.py", "*.md", "*.json", "*.yml", "*.yaml"]
        else:
            patterns = [f"*.{file_type}"]
        
        match_count = 0
        file_count = 0
        
        for pattern in patterns:
            for file_path in KIMERA_ROOT.rglob(pattern):
                if file_path.is_file() and not any(skip in str(file_path) for skip in ['.git', '__pycache__', '.venv', 'node_modules']):
                    file_count += 1
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            if query.lower() in content.lower():
                                match_count += 1
                                # Find line numbers
                                lines = content.split('\n')
                                matching_lines = []
                                for i, line in enumerate(lines, 1):
                                    if query.lower() in line.lower():
                                        matching_lines.append({
                                            "line_number": i,
                                            "content": line.strip()[:100]  # Truncate long lines
                                        })
                                
                                results["matches"].append({
                                    "file": str(file_path.relative_to(KIMERA_ROOT)),
                                    "matches": len(matching_lines),
                                    "lines": matching_lines[:5]  # Limit to first 5 matches per file
                                })
                                
                                if len(results["matches"]) >= 20:  # Limit total results
                                    break
                    except Exception as e:
                        continue
        
        results["summary"] = {
            "files_searched": file_count,
            "files_with_matches": match_count,
            "total_matches": sum(m["matches"] for m in results["matches"])
        }
        
        return results
        
    except Exception as e:
        return {"error": f"Codebase search failed: {str(e)}"}

@mcp.tool()
def get_kimera_cognitive_metrics() -> Dict[str, Any]:
    """
    Extract and analyze cognitive performance metrics from Kimera system.
    
    Returns:
        Dictionary containing cognitive metrics and insights
    """
    try:
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "cognitive_state": "analyzing",
            "memory_usage": {},
            "processing_patterns": {},
            "recent_optimizations": []
        }
        
        # Check for optimization results
        opt_dir = KIMERA_ROOT / "comprehensive_optimization_results"
        if opt_dir.exists():
            opt_files = list(opt_dir.glob("*.json"))
            for opt_file in sorted(opt_files, key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
                try:
                    with open(opt_file, 'r') as f:
                        data = json.load(f)
                        metrics["recent_optimizations"].append({
                            "file": opt_file.name,
                            "timestamp": data.get("timestamp", "unknown"),
                            "type": data.get("optimization_type", "general"),
                            "status": data.get("status", "unknown")
                        })
                except Exception as e:
                    # Log this error, maybe? For now, just skip the file.
                    continue
        
        # Analyze test results for cognitive patterns
        test_dir = KIMERA_ROOT / "test_results"
        if test_dir.exists():
            test_files = list(test_dir.glob("*.json"))
            metrics["processing_patterns"]["test_runs"] = len(test_files)
            
            # Get latest test results
            if test_files:
                latest_test = sorted(test_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                try:
                    with open(latest_test, 'r') as f:
                        test_data = json.load(f)
                        metrics["processing_patterns"]["latest_test"] = {
                            "file": latest_test.name,
                            "performance": test_data.get("performance", {}),
                            "timestamp": test_data.get("timestamp", "unknown")
                        }
                except Exception as e:
                    # If the latest test result is corrupted, we can't do much.
                    pass
        
        metrics["cognitive_state"] = "operational"
        return metrics
        
    except Exception as e:
        return {"error": f"Cognitive metrics extraction failed: {str(e)}"}

@mcp.tool()
def manage_kimera_memory(action: str, key: str = "", value: str = "", memory_type: str = "general") -> Dict[str, Any]:
    """
    Manage Kimera's persistent memory system for cognitive continuity.
    
    Args:
        action: Action to perform ('store', 'retrieve', 'list', 'delete')
        key: Memory key identifier
        value: Value to store (for 'store' action)
        memory_type: Type of memory ('general', 'cognitive', 'learning', 'context')
    
    Returns:
        Dictionary containing memory operation results
    """
    try:
        # Use SQLite for persistent memory
        memory_db = Path("D:/DEV/MCP servers/kimera_memory.db")
        
        with sqlite3.connect(memory_db) as conn:
            cursor = conn.cursor()
            
            # Initialize memory table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS kimera_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE,
                    value TEXT,
                    memory_type TEXT,
                    timestamp TEXT,
                    access_count INTEGER DEFAULT 0
                )
            ''')
            
            result = {
                "action": action,
                "key": key,
                "memory_type": memory_type,
                "timestamp": datetime.now().isoformat()
            }
            
            if action == "store":
                cursor.execute('''
                    INSERT OR REPLACE INTO kimera_memory (key, value, memory_type, timestamp, access_count)
                    VALUES (?, ?, ?, ?, COALESCE((SELECT access_count FROM kimera_memory WHERE key = ?), 0))
                ''', (key, value, memory_type, result["timestamp"], key))
                result["status"] = "stored"
                result["message"] = f"Memory '{key}' stored successfully"
                
            elif action == "retrieve":
                cursor.execute('''
                    SELECT value, memory_type, timestamp, access_count FROM kimera_memory WHERE key = ?
                ''', (key,))
                row = cursor.fetchone()
                if row:
                    # Update access count
                    cursor.execute('UPDATE kimera_memory SET access_count = access_count + 1 WHERE key = ?', (key,))
                    result["status"] = "found"
                    result["value"] = row[0]
                    result["stored_type"] = row[1]
                    result["stored_timestamp"] = row[2]
                    result["access_count"] = row[3] + 1
                else:
                    result["status"] = "not_found"
                    result["message"] = f"Memory '{key}' not found"
                    
            elif action == "list":
                if memory_type == "all":
                    cursor.execute('SELECT key, memory_type, timestamp, access_count FROM kimera_memory ORDER BY timestamp DESC LIMIT 20')
                else:
                    cursor.execute('SELECT key, memory_type, timestamp, access_count FROM kimera_memory WHERE memory_type = ? ORDER BY timestamp DESC LIMIT 20', (memory_type,))
                
                rows = cursor.fetchall()
                result["status"] = "listed"
                result["memories"] = [
                    {
                        "key": row[0],
                        "type": row[1],
                        "timestamp": row[2],
                        "access_count": row[3]
                    } for row in rows
                ]
                result["count"] = len(rows)
                
            elif action == "delete":
                cursor.execute('DELETE FROM kimera_memory WHERE key = ?', (key,))
                if cursor.rowcount > 0:
                    result["status"] = "deleted"
                    result["message"] = f"Memory '{key}' deleted successfully"
                else:
                    result["status"] = "not_found"
                    result["message"] = f"Memory '{key}' not found"
            
            conn.commit()
            return result
            
    except Exception as e:
        return {"error": f"Memory management failed: {str(e)}"}

if __name__ == "__main__":
    mcp.run() 