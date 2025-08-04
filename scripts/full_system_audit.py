#!/usr/bin/env python3
"""
KIMERA SWM - FULL CORE SYSTEM AUDIT
===================================

Comprehensive production-grade audit of the entire Kimera SWM system.
Examines architecture, performance, security, data integrity, error handling,
and all system integrations for enterprise deployment readiness.

Audit Categories:
1. System Architecture Integrity
2. Component Health and Status  
3. Performance and Resource Usage
4. Security and Access Control
5. Data Flow and Persistence
6. Error Handling and Recovery
7. Configuration Management
8. Integration Points
9. Monitoring and Observability
10. Production Readiness
"""

import os
import sys
import time
import json
import sqlite3
import asyncio
import psutil
import logging
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
import traceback
import gc

# Setup paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AuditResult:
    """Individual audit result"""
    category: str
    component: str
    status: str  # PASS, WARN, FAIL, INFO
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    severity: str = "info"  # critical, high, medium, low, info
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class AuditSummary:
    """Overall audit summary"""
    total_checks: int = 0
    passed_checks: int = 0
    warning_checks: int = 0
    failed_checks: int = 0
    critical_issues: int = 0
    high_issues: int = 0
    medium_issues: int = 0
    low_issues: int = 0
    audit_duration: float = 0.0
    overall_score: float = 0.0
    readiness_status: str = "unknown"

class KimeraSystemAuditor:
    """Comprehensive Kimera SWM system auditor"""
    
    def __init__(self):
        self.project_root = project_root
        self.audit_results: List[AuditResult] = []
        self.audit_summary = AuditSummary()
        self.start_time = None
        self.system_info = {}
        
    def add_result(self, result: AuditResult):
        """Add audit result"""
        self.audit_results.append(result)
        
        # Update summary counters
        self.audit_summary.total_checks += 1
        
        if result.status == "PASS":
            self.audit_summary.passed_checks += 1
        elif result.status == "WARN":
            self.audit_summary.warning_checks += 1
        elif result.status == "FAIL":
            self.audit_summary.failed_checks += 1
            
        # Update severity counters
        if result.severity == "critical":
            self.audit_summary.critical_issues += 1
        elif result.severity == "high":
            self.audit_summary.high_issues += 1
        elif result.severity == "medium":
            self.audit_summary.medium_issues += 1
        elif result.severity == "low":
            self.audit_summary.low_issues += 1
    
    def audit_system_architecture(self) -> List[AuditResult]:
        """Audit 1: System Architecture Integrity"""
        logger.info("🏗️ Auditing System Architecture Integrity...")
        
        results = []
        
        # Core system singleton integrity
        try:
            from src.core.kimera_system import get_kimera_system
            system = get_kimera_system()
            
            # Check singleton pattern
            system2 = get_kimera_system()
            if system is system2:
                results.append(AuditResult(
                    category="Architecture",
                    component="Core System Singleton",
                    status="PASS",
                    message="Singleton pattern correctly implemented",
                    details={"memory_address": hex(id(system))}
                ))
            else:
                results.append(AuditResult(
                    category="Architecture", 
                    component="Core System Singleton",
                    status="FAIL",
                    message="Singleton pattern violated - multiple instances",
                    severity="critical",
                    recommendations=["Fix singleton implementation"]
                ))
                
            # Check system initialization
            try:
                system.initialize()
                state = system.get_system_state()
                
                if state.get('state') == 'RUNNING':
                    results.append(AuditResult(
                        category="Architecture",
                        component="System Initialization",
                        status="PASS",
                        message="System initializes correctly",
                        details=state
                    ))
                else:
                    results.append(AuditResult(
                        category="Architecture",
                        component="System Initialization", 
                        status="FAIL",
                        message=f"System state invalid: {state.get('state')}",
                        severity="high",
                        details=state
                    ))
                    
            except Exception as e:
                results.append(AuditResult(
                    category="Architecture",
                    component="System Initialization",
                    status="FAIL", 
                    message=f"System initialization failed: {e}",
                    severity="critical"
                ))
                
        except Exception as e:
            results.append(AuditResult(
                category="Architecture",
                component="Core System Import",
                status="FAIL",
                message=f"Cannot import core system: {e}",
                severity="critical"
            ))
        
        # GPU architecture integration
        try:
            from src.core.gpu.gpu_manager import get_gpu_manager, is_gpu_available
            
            if is_gpu_available():
                gpu_manager = get_gpu_manager()
                device_info = gpu_manager.get_device_info()
                
                results.append(AuditResult(
                    category="Architecture",
                    component="GPU Integration",
                    status="PASS",
                    message="GPU architecture fully integrated",
                    details=device_info
                ))
            else:
                results.append(AuditResult(
                    category="Architecture",
                    component="GPU Integration",
                    status="WARN",
                    message="GPU not available - CPU fallback active",
                    severity="medium"
                ))
                
        except Exception as e:
            results.append(AuditResult(
                category="Architecture",
                component="GPU Integration",
                status="FAIL",
                message=f"GPU architecture audit failed: {e}",
                severity="high"
            ))
            
        # Orchestration architecture
        try:
            from src.orchestration.kimera_orchestrator import EngineCoordinator
            coordinator = EngineCoordinator()
            
            engine_count = len(coordinator.engines)
            gpu_engines = len([name for name in coordinator.engines.keys() if 'gpu' in name])
            
            if engine_count >= 4:  # Minimum expected engines
                results.append(AuditResult(
                    category="Architecture",
                    component="Orchestration System",
                    status="PASS",
                    message=f"Orchestration properly configured",
                    details={
                        "total_engines": engine_count,
                        "gpu_engines": gpu_engines,
                        "gpu_available": coordinator.gpu_available
                    }
                ))
            else:
                results.append(AuditResult(
                    category="Architecture",
                    component="Orchestration System",
                    status="WARN",
                    message=f"Low engine count: {engine_count}",
                    severity="medium",
                    details={"engine_count": engine_count}
                ))
                
        except Exception as e:
            results.append(AuditResult(
                category="Architecture",
                component="Orchestration System",
                status="FAIL",
                message=f"Orchestration audit failed: {e}",
                severity="high"
            ))
        
        return results
    
    def audit_component_health(self) -> List[AuditResult]:
        """Audit 2: Component Health and Status"""
        logger.info("🔍 Auditing Component Health and Status...")
        
        results = []
        
        # Core components health
        critical_components = [
            "src.core.kimera_system",
            "src.core.data_structures.geoid_state", 
            "src.core.processing.geoid_processor",
            "src.orchestration.kimera_orchestrator",
            "src.vault.vault_manager"
        ]
        
        for component in critical_components:
            try:
                __import__(component)
                results.append(AuditResult(
                    category="Component Health",
                    component=component,
                    status="PASS",
                    message="Component imports successfully"
                ))
            except Exception as e:
                results.append(AuditResult(
                    category="Component Health",
                    component=component,
                    status="FAIL",
                    message=f"Component import failed: {e}",
                    severity="critical"
                ))
        
        # GPU components health
        gpu_components = [
            "src.core.gpu.gpu_manager",
            "src.core.gpu.gpu_integration",
            "src.engines.gpu.gpu_geoid_processor",
            "src.engines.gpu.gpu_thermodynamic_engine"
        ]
        
        gpu_healthy = 0
        for component in gpu_components:
            try:
                __import__(component)
                gpu_healthy += 1
                results.append(AuditResult(
                    category="Component Health",
                    component=component,
                    status="PASS",
                    message="GPU component healthy"
                ))
            except Exception as e:
                results.append(AuditResult(
                    category="Component Health",
                    component=component,
                    status="WARN",
                    message=f"GPU component issue: {e}",
                    severity="medium"
                ))
        
        # GPU health summary
        gpu_health_percentage = (gpu_healthy / len(gpu_components)) * 100
        if gpu_health_percentage >= 75:
            status = "PASS"
            severity = "info"
        elif gpu_health_percentage >= 50:
            status = "WARN" 
            severity = "medium"
        else:
            status = "FAIL"
            severity = "high"
            
        results.append(AuditResult(
            category="Component Health",
            component="GPU Components Overall",
            status=status,
            message=f"GPU component health: {gpu_health_percentage:.0f}%",
            severity=severity,
            details={"healthy_components": gpu_healthy, "total_components": len(gpu_components)}
        ))
        
        # Database components health
        try:
            from src.vault.vault_manager import VaultManager
            vault = VaultManager()
            
            vault_health = {
                "db_initialized": vault.db_initialized,
                "neo4j_available": vault.neo4j_available,
            }
            
            if vault.db_initialized:
                results.append(AuditResult(
                    category="Component Health",
                    component="Vault Database", 
                    status="PASS",
                    message="Vault database operational",
                    details=vault_health
                ))
            else:
                results.append(AuditResult(
                    category="Component Health",
                    component="Vault Database",
                    status="WARN",
                    message="Vault database limited functionality",
                    severity="medium",
                    details=vault_health
                ))
                
        except Exception as e:
            results.append(AuditResult(
                category="Component Health",
                component="Vault System",
                status="FAIL",
                message=f"Vault system health check failed: {e}",
                severity="high"
            ))
        
        return results
    
    def audit_performance_resources(self) -> List[AuditResult]:
        """Audit 3: Performance and Resource Usage"""
        logger.info("📊 Auditing Performance and Resource Usage...")
        
        results = []
        
        # System resource usage
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # CPU usage audit
        if cpu_percent < 80:
            results.append(AuditResult(
                category="Performance",
                component="CPU Usage",
                status="PASS",
                message=f"CPU usage healthy: {cpu_percent:.1f}%",
                details={"cpu_percent": cpu_percent}
            ))
        else:
            results.append(AuditResult(
                category="Performance", 
                component="CPU Usage",
                status="WARN",
                message=f"High CPU usage: {cpu_percent:.1f}%",
                severity="medium",
                details={"cpu_percent": cpu_percent}
            ))
        
        # Memory usage audit
        memory_percent = memory.percent
        if memory_percent < 85:
            results.append(AuditResult(
                category="Performance",
                component="Memory Usage",
                status="PASS",
                message=f"Memory usage healthy: {memory_percent:.1f}%",
                details={
                    "memory_percent": memory_percent,
                    "available_gb": memory.available / (1024**3),
                    "total_gb": memory.total / (1024**3)
                }
            ))
        else:
            results.append(AuditResult(
                category="Performance",
                component="Memory Usage", 
                status="WARN",
                message=f"High memory usage: {memory_percent:.1f}%",
                severity="medium",
                details={"memory_percent": memory_percent}
            ))
        
        # GPU performance audit
        try:
            import torch
            if torch.cuda.is_available():
                # GPU memory check
                gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                gpu_memory_cached = torch.cuda.memory_reserved(0) / (1024**3)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                gpu_usage_percent = (gpu_memory_allocated / gpu_memory_total) * 100
                
                # GPU performance test
                start_time = time.time()
                x = torch.randn(1000, 1000, device='cuda')
                y = torch.randn(1000, 1000, device='cuda')
                z = torch.matmul(x, y)
                torch.cuda.synchronize()
                gpu_time = time.time() - start_time
                
                gflops = (2 * 1000**3) / gpu_time / 1e9
                
                if gflops > 100:  # Good performance threshold
                    results.append(AuditResult(
                        category="Performance",
                        component="GPU Performance",
                        status="PASS", 
                        message=f"GPU performance excellent: {gflops:.0f} GFLOPS",
                        details={
                            "gflops": gflops,
                            "memory_allocated_gb": gpu_memory_allocated,
                            "memory_total_gb": gpu_memory_total,
                            "usage_percent": gpu_usage_percent
                        }
                    ))
                else:
                    results.append(AuditResult(
                        category="Performance",
                        component="GPU Performance",
                        status="WARN",
                        message=f"GPU performance below optimal: {gflops:.0f} GFLOPS",
                        severity="medium",
                        details={"gflops": gflops}
                    ))
                    
        except Exception as e:
            results.append(AuditResult(
                category="Performance",
                component="GPU Performance",
                status="FAIL",
                message=f"GPU performance audit failed: {e}",
                severity="high"
            ))
        
        # Startup performance audit
        try:
            startup_start = time.time()
            from src.core.kimera_system import get_kimera_system
            system = get_kimera_system()
            system.initialize()
            startup_time = time.time() - startup_start
            
            if startup_time < 10.0:  # Good startup time
                results.append(AuditResult(
                    category="Performance", 
                    component="Startup Time",
                    status="PASS",
                    message=f"Startup time excellent: {startup_time:.2f}s",
                    details={"startup_time": startup_time}
                ))
            elif startup_time < 30.0:  # Acceptable startup time
                results.append(AuditResult(
                    category="Performance",
                    component="Startup Time",
                    status="WARN",
                    message=f"Startup time acceptable: {startup_time:.2f}s",
                    severity="low",
                    details={"startup_time": startup_time}
                ))
            else:  # Poor startup time
                results.append(AuditResult(
                    category="Performance",
                    component="Startup Time", 
                    status="FAIL",
                    message=f"Startup time poor: {startup_time:.2f}s",
                    severity="medium",
                    details={"startup_time": startup_time},
                    recommendations=["Optimize initialization sequence", "Profile bottlenecks"]
                ))
                
        except Exception as e:
            results.append(AuditResult(
                category="Performance",
                component="Startup Performance",
                status="FAIL",
                message=f"Startup performance audit failed: {e}",
                severity="high"
            ))
            
        return results
    
    def audit_security_access(self) -> List[AuditResult]:
        """Audit 4: Security and Access Control"""
        logger.info("🔒 Auditing Security and Access Control...")
        
        results = []
        
        # File permissions audit
        critical_files = [
            "src/core/kimera_system.py",
            "src/core/gpu/gpu_manager.py", 
            "src/vault/vault_manager.py",
            "config/development.yaml"
        ]
        
        for file_path in critical_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                # Check file permissions
                stat_info = full_path.stat()
                mode = oct(stat_info.st_mode)[-3:]
                
                # Check if file is world-writable (security risk)
                if stat_info.st_mode & 0o002:
                    results.append(AuditResult(
                        category="Security",
                        component=f"File Permissions: {file_path}",
                        status="FAIL",
                        message="File is world-writable (security risk)",
                        severity="high",
                        details={"permissions": mode},
                        recommendations=["Fix file permissions"]
                    ))
                else:
                    results.append(AuditResult(
                        category="Security",
                        component=f"File Permissions: {file_path}",
                        status="PASS",
                        message="File permissions secure",
                        details={"permissions": mode}
                    ))
        
        # Configuration security audit
        try:
            config_file = self.project_root / "config/development.yaml"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    content = f.read()
                    
                # Check for potential secrets in config
                security_patterns = ['password', 'secret', 'key', 'token']
                found_patterns = []
                
                for pattern in security_patterns:
                    if pattern.lower() in content.lower():
                        found_patterns.append(pattern)
                
                if found_patterns:
                    results.append(AuditResult(
                        category="Security",
                        component="Configuration Security",
                        status="WARN",
                        message="Potential secrets found in configuration",
                        severity="medium",
                        details={"patterns_found": found_patterns},
                        recommendations=["Move secrets to environment variables", "Use secret management"]
                    ))
                else:
                    results.append(AuditResult(
                        category="Security",
                        component="Configuration Security",
                        status="PASS",
                        message="No obvious secrets in configuration"
                    ))
                    
        except Exception as e:
            results.append(AuditResult(
                category="Security",
                component="Configuration Security",
                status="FAIL", 
                message=f"Configuration security audit failed: {e}",
                severity="medium"
            ))
        
        # Database security audit
        try:
            db_path = self.project_root / "data/database/kimera_system.db"
            if db_path.exists():
                stat_info = db_path.stat()
                
                # Check database file permissions
                if stat_info.st_mode & 0o044:  # World or group readable
                    results.append(AuditResult(
                        category="Security",
                        component="Database Security",
                        status="WARN",
                        message="Database file has broad read permissions",
                        severity="medium",
                        recommendations=["Restrict database file permissions"]
                    ))
                else:
                    results.append(AuditResult(
                        category="Security", 
                        component="Database Security",
                        status="PASS",
                        message="Database file permissions secure"
                    ))
                    
        except Exception as e:
            results.append(AuditResult(
                category="Security",
                component="Database Security",
                status="FAIL",
                message=f"Database security audit failed: {e}",
                severity="medium"
            ))
        
        # Import security audit
        try:
            # Check for potentially dangerous imports
            dangerous_imports = []
            
            for py_file in self.project_root.rglob("*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Check for potentially dangerous patterns
                    if 'eval(' in content or 'exec(' in content:
                        dangerous_imports.append(str(py_file.relative_to(self.project_root)))
                        
                except Exception:
                    continue  # Skip files that can't be read
            
            if dangerous_imports:
                results.append(AuditResult(
                    category="Security",
                    component="Code Security",
                    status="WARN", 
                    message="Potentially dangerous code patterns found",
                    severity="medium",
                    details={"files_with_issues": dangerous_imports},
                    recommendations=["Review dynamic code execution", "Consider safer alternatives"]
                ))
            else:
                results.append(AuditResult(
                    category="Security",
                    component="Code Security", 
                    status="PASS",
                    message="No obvious dangerous code patterns found"
                ))
                
        except Exception as e:
            results.append(AuditResult(
                category="Security",
                component="Code Security",
                status="FAIL",
                message=f"Code security audit failed: {e}",
                severity="low"
            ))
            
        return results
    
    def audit_data_flow_persistence(self) -> List[AuditResult]:
        """Audit 5: Data Flow and Persistence"""
        logger.info("💾 Auditing Data Flow and Persistence...")
        
        results = []
        
        # Database connectivity and integrity
        try:
            db_path = self.project_root / "data/database/kimera_system.db"
            if db_path.exists():
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                
                # Check database integrity
                cursor.execute("PRAGMA integrity_check;")
                integrity_result = cursor.fetchone()[0]
                
                if integrity_result == "ok":
                    results.append(AuditResult(
                        category="Data Flow",
                        component="Database Integrity",
                        status="PASS",
                        message="Database integrity check passed"
                    ))
                else:
                    results.append(AuditResult(
                        category="Data Flow",
                        component="Database Integrity",
                        status="FAIL",
                        message=f"Database integrity check failed: {integrity_result}",
                        severity="high"
                    ))
                
                # Check table structure
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                
                expected_tables = ['geoid_states', 'cognitive_transitions', 'semantic_embeddings']
                missing_tables = [table for table in expected_tables if table not in tables]
                
                if not missing_tables:
                    results.append(AuditResult(
                        category="Data Flow",
                        component="Database Schema",
                        status="PASS",
                        message="All expected tables present",
                        details={"tables": tables}
                    ))
                else:
                    results.append(AuditResult(
                        category="Data Flow",
                        component="Database Schema",
                        status="WARN",
                        message=f"Missing tables: {missing_tables}",
                        severity="medium",
                        details={"missing_tables": missing_tables}
                    ))
                
                # Test data operations
                try:
                    # Test insert
                    test_id = "audit_test_" + str(int(time.time()))
                    cursor.execute(
                        "INSERT INTO geoid_states (id, state_vector, entropy, coherence_factor) VALUES (?, ?, ?, ?)",
                        (test_id, "test_vector", 1.0, 0.5)
                    )
                    
                    # Test select
                    cursor.execute("SELECT * FROM geoid_states WHERE id = ?", (test_id,))
                    result = cursor.fetchone()
                    
                    if result:
                        # Clean up test data
                        cursor.execute("DELETE FROM geoid_states WHERE id = ?", (test_id,))
                        conn.commit()
                        
                        results.append(AuditResult(
                            category="Data Flow",
                            component="Database Operations",
                            status="PASS",
                            message="Database read/write operations functional"
                        ))
                    else:
                        results.append(AuditResult(
                            category="Data Flow",
                            component="Database Operations",
                            status="FAIL",
                            message="Database write operation failed",
                            severity="high"
                        ))
                        
                except Exception as e:
                    results.append(AuditResult(
                        category="Data Flow",
                        component="Database Operations",
                        status="FAIL",
                        message=f"Database operation test failed: {e}",
                        severity="high"
                    ))
                
                conn.close()
                
            else:
                results.append(AuditResult(
                    category="Data Flow",
                    component="Database Availability",
                    status="FAIL",
                    message="Database file not found",
                    severity="high",
                    recommendations=["Initialize database", "Check database setup"]
                ))
                
        except Exception as e:
            results.append(AuditResult(
                category="Data Flow",
                component="Database Connectivity",
                status="FAIL",
                message=f"Database connectivity audit failed: {e}",
                severity="high"
            ))
        
        # Vault system data flow
        try:
            from src.vault.vault_manager import VaultManager
            vault = VaultManager()
            
            # Test vault operations would go here
            # For now, just check initialization
            vault_status = {
                "initialized": vault.db_initialized,
                "neo4j": vault.neo4j_available
            }
            
            results.append(AuditResult(
                category="Data Flow",
                component="Vault System",
                status="PASS" if vault.db_initialized else "WARN",
                message="Vault system connectivity checked",
                details=vault_status,
                severity="medium" if not vault.db_initialized else "info"
            ))
            
        except Exception as e:
            results.append(AuditResult(
                category="Data Flow",
                component="Vault System",
                status="FAIL",
                message=f"Vault system audit failed: {e}",
                severity="high"
            ))
        
        # File system permissions for data directories
        data_dirs = ['data', 'data/database', 'data/logs', 'data/exports']
        
        for dir_path in data_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists():
                if os.access(full_path, os.R_OK | os.W_OK):
                    results.append(AuditResult(
                        category="Data Flow",
                        component=f"Directory Access: {dir_path}",
                        status="PASS",
                        message="Directory accessible for read/write"
                    ))
                else:
                    results.append(AuditResult(
                        category="Data Flow",
                        component=f"Directory Access: {dir_path}",
                        status="FAIL",
                        message="Directory not accessible",
                        severity="high",
                        recommendations=["Fix directory permissions"]
                    ))
            else:
                results.append(AuditResult(
                    category="Data Flow",
                    component=f"Directory Existence: {dir_path}",
                    status="WARN",
                    message="Data directory missing",
                    severity="medium",
                    recommendations=["Create missing directory"]
                ))
        
        return results
    
    def audit_error_handling(self) -> List[AuditResult]:
        """Audit 6: Error Handling and Recovery"""
        logger.info("🛡️ Auditing Error Handling and Recovery...")
        
        results = []
        
        # Test system resilience to errors
        try:
            from src.core.kimera_system import get_kimera_system
            system = get_kimera_system()
            
            # Test invalid operations gracefully handled
            try:
                # This should not crash the system
                state = system.get_system_state()
                if 'error_handling' in state.get('components', {}):
                    results.append(AuditResult(
                        category="Error Handling", 
                        component="Error Recovery System",
                        status="PASS",
                        message="Error handling system detected"
                    ))
                
            except Exception as e:
                # System should handle this gracefully
                results.append(AuditResult(
                    category="Error Handling",
                    component="System Resilience",
                    status="WARN",
                    message=f"System not fully resilient to errors: {e}",
                    severity="medium"
                ))
                
        except Exception as e:
            results.append(AuditResult(
                category="Error Handling",
                component="Error Handling Audit",
                status="FAIL",
                message=f"Error handling audit failed: {e}",
                severity="high"
            ))
        
        # Check logging configuration
        try:
            # Check if logs directory exists and is writable
            logs_dir = self.project_root / "data/logs"
            if logs_dir.exists() and os.access(logs_dir, os.W_OK):
                results.append(AuditResult(
                    category="Error Handling",
                    component="Logging System",
                    status="PASS",
                    message="Logging directory accessible"
                ))
            else:
                results.append(AuditResult(
                    category="Error Handling", 
                    component="Logging System",
                    status="WARN",
                    message="Logging directory not accessible",
                    severity="low",
                    recommendations=["Ensure logs directory exists and is writable"]
                ))
                
        except Exception as e:
            results.append(AuditResult(
                category="Error Handling",
                component="Logging Configuration",
                status="FAIL",
                message=f"Logging audit failed: {e}",
                severity="medium"
            ))
        
        return results
    
    def audit_configuration_management(self) -> List[AuditResult]:
        """Audit 7: Configuration Management"""
        logger.info("⚙️ Auditing Configuration Management...")
        
        results = []
        
        # Configuration file integrity
        config_files = ['config/development.yaml', 'config/production.yaml']
        
        for config_file in config_files:
            config_path = self.project_root / config_file
            if config_path.exists():
                try:
                    import yaml
                    with open(config_path, 'r') as f:
                        config_data = yaml.safe_load(f)
                    
                    # Check for required configuration sections
                    required_sections = ['environment', 'gpu', 'monitoring']
                    missing_sections = [section for section in required_sections if section not in config_data]
                    
                    if not missing_sections:
                        results.append(AuditResult(
                            category="Configuration",
                            component=f"Config File: {config_file}",
                            status="PASS",
                            message="Configuration file complete",
                            details={"sections": list(config_data.keys())}
                        ))
                    else:
                        results.append(AuditResult(
                            category="Configuration",
                            component=f"Config File: {config_file}",
                            status="WARN",
                            message=f"Missing configuration sections: {missing_sections}",
                            severity="low",
                            details={"missing_sections": missing_sections}
                        ))
                        
                except Exception as e:
                    results.append(AuditResult(
                        category="Configuration",
                        component=f"Config File: {config_file}",
                        status="FAIL",
                        message=f"Configuration file parsing failed: {e}",
                        severity="medium"
                    ))
            else:
                severity = "high" if "production" in config_file else "medium"
                results.append(AuditResult(
                    category="Configuration",
                    component=f"Config File: {config_file}",
                    status="WARN" if "production" in config_file else "INFO",
                    message="Configuration file missing",
                    severity=severity,
                    recommendations=["Create missing configuration file"]
                ))
        
        # Environment variable audit
        important_env_vars = ['PYTHONPATH', 'CUDA_PATH']
        for var in important_env_vars:
            if os.environ.get(var):
                results.append(AuditResult(
                    category="Configuration",
                    component=f"Environment: {var}",
                    status="PASS",
                    message="Environment variable set",
                    details={"value_length": len(os.environ[var])}
                ))
            else:
                results.append(AuditResult(
                    category="Configuration",
                    component=f"Environment: {var}",
                    status="INFO",
                    message="Environment variable not set",
                    severity="info"
                ))
        
        return results
    
    def audit_integration_points(self) -> List[AuditResult]:
        """Audit 8: Integration Points"""
        logger.info("🔗 Auditing Integration Points...")
        
        results = []
        
        # API integration points
        try:
            # Test main API components
            from src.main import app
            
            # Check if FastAPI app is properly configured
            route_count = len(app.routes)
            
            if route_count > 0:
                results.append(AuditResult(
                    category="Integration",
                    component="FastAPI Application",
                    status="PASS",
                    message=f"FastAPI app configured with {route_count} routes"
                ))
            else:
                results.append(AuditResult(
                    category="Integration",
                    component="FastAPI Application",
                    status="WARN",
                    message="FastAPI app has no routes configured",
                    severity="medium"
                ))
                
        except Exception as e:
            results.append(AuditResult(
                category="Integration",
                component="FastAPI Application",
                status="FAIL",
                message=f"FastAPI audit failed: {e}",
                severity="high"
            ))
        
        # GPU integration points
        try:
            from src.core.kimera_system import get_kimera_system
            system = get_kimera_system()
            system.initialize()
            
            # Check GPU integration
            if system.is_gpu_acceleration_enabled():
                gpu_components = system.get_system_state().get('gpu_components', {})
                active_gpu_components = sum(1 for status in gpu_components.values() if status)
                
                results.append(AuditResult(
                    category="Integration",
                    component="GPU Integration Points",
                    status="PASS",
                    message=f"GPU integration active: {active_gpu_components} components",
                    details=gpu_components
                ))
            else:
                results.append(AuditResult(
                    category="Integration",
                    component="GPU Integration Points",
                    status="WARN",
                    message="GPU integration not active",
                    severity="medium"
                ))
                
        except Exception as e:
            results.append(AuditResult(
                category="Integration",
                component="GPU Integration Points",
                status="FAIL",
                message=f"GPU integration audit failed: {e}",
                severity="high"
            ))
        
        return results
    
    def audit_monitoring_observability(self) -> List[AuditResult]:
        """Audit 9: Monitoring and Observability"""
        logger.info("📈 Auditing Monitoring and Observability...")
        
        results = []
        
        # System monitoring capabilities
        try:
            from src.monitoring.system_monitor import SystemMonitor
            monitor = SystemMonitor()
            
            results.append(AuditResult(
                category="Monitoring",
                component="System Monitor",
                status="PASS",
                message="System monitoring available"
            ))
            
        except Exception as e:
            results.append(AuditResult(
                category="Monitoring",
                component="System Monitor",
                status="FAIL",
                message=f"System monitoring audit failed: {e}",
                severity="medium"
            ))
        
        # Performance monitoring
        try:
            # Check if performance tracking is available
            self.system_info = {
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total / (1024**3),
                'disk_usage': psutil.disk_usage('/').percent,
                'python_version': sys.version.split()[0],
                'platform': sys.platform
            }
            
            results.append(AuditResult(
                category="Monitoring",
                component="Performance Metrics",
                status="PASS",
                message="Performance metrics collection available",
                details=self.system_info
            ))
            
        except Exception as e:
            results.append(AuditResult(
                category="Monitoring",
                component="Performance Metrics",
                status="FAIL",
                message=f"Performance monitoring audit failed: {e}",
                severity="medium"
            ))
        
        return results
    
    def audit_production_readiness(self) -> List[AuditResult]:
        """Audit 10: Production Readiness"""
        logger.info("🚀 Auditing Production Readiness...")
        
        results = []
        
        # Deployment readiness
        required_files = [
            'kimera.py',
            'src/main.py', 
            'requirements/base.txt',
            'config/development.yaml'
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
        
        if not missing_files:
            results.append(AuditResult(
                category="Production Readiness",
                component="Deployment Files",
                status="PASS",
                message="All required deployment files present"
            ))
        else:
            results.append(AuditResult(
                category="Production Readiness",
                component="Deployment Files",
                status="FAIL",
                message=f"Missing deployment files: {missing_files}",
                severity="high",
                details={"missing_files": missing_files}
            ))
        
        # Scalability audit
        try:
            from src.core.kimera_system import get_kimera_system
            system = get_kimera_system()
            
            # Check if system can handle concurrent operations
            # This is a basic check - real scalability testing would be more complex
            concurrent_safe = hasattr(system, '_lock') or hasattr(system, '_async_lock')
            
            if concurrent_safe:
                results.append(AuditResult(
                    category="Production Readiness",
                    component="Concurrency Safety",
                    status="PASS",
                    message="System appears to have concurrency protections"
                ))
            else:
                results.append(AuditResult(
                    category="Production Readiness",
                    component="Concurrency Safety",
                    status="WARN",
                    message="System may not be fully thread-safe",
                    severity="medium",
                    recommendations=["Review threading safety", "Add proper locks"]
                ))
                
        except Exception as e:
            results.append(AuditResult(
                category="Production Readiness",
                component="Scalability Check",
                status="FAIL",
                message=f"Scalability audit failed: {e}",
                severity="medium"
            ))
        
        # Documentation audit
        doc_files = ['README.md', 'CONTRIBUTING.md', 'docs/']
        existing_docs = []
        for doc_path in doc_files:
            if (self.project_root / doc_path).exists():
                existing_docs.append(doc_path)
        
        if len(existing_docs) >= 2:
            results.append(AuditResult(
                category="Production Readiness",
                component="Documentation",
                status="PASS",
                message="Adequate documentation present",
                details={"existing_docs": existing_docs}
            ))
        else:
            results.append(AuditResult(
                category="Production Readiness",
                component="Documentation",
                status="WARN", 
                message="Limited documentation available",
                severity="low",
                details={"existing_docs": existing_docs}
            ))
        
        return results
    
    def run_full_audit(self) -> Dict[str, Any]:
        """Run complete system audit"""
        logger.info("🔍 Starting Full Kimera SWM Core System Audit")
        logger.info("=" * 80)
        
        self.start_time = time.time()
        
        # Run all audit categories
        audit_functions = [
            self.audit_system_architecture,
            self.audit_component_health,
            self.audit_performance_resources,
            self.audit_security_access,
            self.audit_data_flow_persistence,
            self.audit_error_handling,
            self.audit_configuration_management,
            self.audit_integration_points,
            self.audit_monitoring_observability,
            self.audit_production_readiness
        ]
        
        for audit_func in audit_functions:
            try:
                category_results = audit_func()
                for result in category_results:
                    self.add_result(result)
            except Exception as e:
                logger.error(f"Audit function {audit_func.__name__} failed: {e}")
                self.add_result(AuditResult(
                    category="Audit System",
                    component=audit_func.__name__,
                    status="FAIL",
                    message=f"Audit function crashed: {e}",
                    severity="high"
                ))
        
        # Calculate final scores and status
        self.audit_summary.audit_duration = time.time() - self.start_time
        
        # Calculate overall score
        if self.audit_summary.total_checks > 0:
            score = (self.audit_summary.passed_checks / self.audit_summary.total_checks) * 100
            
            # Deduct points for severity
            penalty = (
                self.audit_summary.critical_issues * 20 +
                self.audit_summary.high_issues * 10 +
                self.audit_summary.medium_issues * 5 +
                self.audit_summary.low_issues * 1
            )
            
            self.audit_summary.overall_score = max(0, score - penalty)
        
        # Determine readiness status
        if self.audit_summary.critical_issues > 0:
            self.audit_summary.readiness_status = "not_ready"
        elif self.audit_summary.overall_score >= 90:
            self.audit_summary.readiness_status = "production_ready"
        elif self.audit_summary.overall_score >= 80:
            self.audit_summary.readiness_status = "nearly_ready"
        elif self.audit_summary.overall_score >= 70:
            self.audit_summary.readiness_status = "needs_improvement"
        else:
            self.audit_summary.readiness_status = "significant_issues"
        
        return {
            'summary': self.audit_summary,
            'results': self.audit_results,
            'system_info': self.system_info
        }
    
    def generate_audit_report(self, audit_data: Dict[str, Any]) -> str:
        """Generate comprehensive audit report"""
        summary = audit_data['summary']
        results = audit_data['results']
        
        report = []
        report.append("# KIMERA SWM - FULL CORE SYSTEM AUDIT REPORT")
        report.append("=" * 90)
        report.append(f"**Audit Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Overall Score**: {summary.overall_score:.1f}/100")
        report.append(f"**Readiness Status**: {summary.readiness_status.upper().replace('_', ' ')}")
        report.append(f"**Audit Duration**: {summary.audit_duration:.2f}s")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append(f"- **Total Checks**: {summary.total_checks}")
        report.append(f"- **Passed**: {summary.passed_checks} ({summary.passed_checks/summary.total_checks*100:.1f}%)")
        report.append(f"- **Warnings**: {summary.warning_checks}")
        report.append(f"- **Failed**: {summary.failed_checks}")
        report.append("")
        
        # Severity Analysis
        report.append("## Issue Severity Breakdown")
        report.append(f"- **Critical Issues**: {summary.critical_issues}")
        report.append(f"- **High Severity**: {summary.high_issues}")
        report.append(f"- **Medium Severity**: {summary.medium_issues}")
        report.append(f"- **Low Severity**: {summary.low_issues}")
        report.append("")
        
        # Category Results
        categories = {}
        for result in results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)
        
        for category, category_results in categories.items():
            report.append(f"## {category}")
            
            passed = len([r for r in category_results if r.status == "PASS"])
            total = len(category_results)
            
            report.append(f"**Status**: {passed}/{total} checks passed ({passed/total*100:.1f}%)")
            report.append("")
            
            # Show failed and warning items
            issues = [r for r in category_results if r.status in ["FAIL", "WARN"]]
            if issues:
                for issue in issues:
                    status_icon = "❌" if issue.status == "FAIL" else "⚠️"
                    report.append(f"- {status_icon} **{issue.component}**: {issue.message}")
                    if issue.recommendations:
                        for rec in issue.recommendations:
                            report.append(f"  - *Recommendation*: {rec}")
                report.append("")
        
        # Critical Issues Section
        critical_issues = [r for r in results if r.severity == "critical"]
        if critical_issues:
            report.append("## 🚨 Critical Issues Requiring Immediate Attention")
            for issue in critical_issues:
                report.append(f"- **{issue.component}**: {issue.message}")
                if issue.recommendations:
                    for rec in issue.recommendations:
                        report.append(f"  - *Action Required*: {rec}")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        
        if summary.readiness_status == "production_ready":
            report.append("🎉 **SYSTEM IS PRODUCTION READY**")
            report.append("- All critical systems operational")
            report.append("- No blocking issues identified")
            report.append("- Continue monitoring and maintenance")
        elif summary.readiness_status == "nearly_ready":
            report.append("✅ **SYSTEM NEARLY READY FOR PRODUCTION**")
            report.append("- Address remaining warnings")
            report.append("- Implement recommended improvements")
            report.append("- Consider additional testing")
        elif summary.readiness_status == "needs_improvement":
            report.append("⚠️ **SYSTEM NEEDS IMPROVEMENT**")
            report.append("- Address high and medium priority issues")
            report.append("- Improve system reliability")
            report.append("- Conduct additional testing")
        else:
            report.append("❌ **SYSTEM NOT READY FOR PRODUCTION**")
            report.append("- Critical issues must be resolved")
            report.append("- Significant improvements required")
            report.append("- Full remediation needed")
        
        report.append("")
        report.append("---")
        report.append("*Full System Audit completed by Kimera SWM Auditor*")
        
        return "\n".join(report)

def main():
    """Main audit function"""
    try:
        auditor = KimeraSystemAuditor()
        audit_data = auditor.run_full_audit()
        
        # Generate report
        report = auditor.generate_audit_report(audit_data)
        
        # Save results
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Ensure reports directory exists
        reports_dir = auditor.project_root / "docs" / "reports" / "audit"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        json_path = reports_dir / f"{timestamp}_full_system_audit.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            # Convert dataclasses to dict for JSON serialization
            json_data = {
                'summary': {
                    'total_checks': audit_data['summary'].total_checks,
                    'passed_checks': audit_data['summary'].passed_checks,
                    'warning_checks': audit_data['summary'].warning_checks,
                    'failed_checks': audit_data['summary'].failed_checks,
                    'critical_issues': audit_data['summary'].critical_issues,
                    'high_issues': audit_data['summary'].high_issues,
                    'medium_issues': audit_data['summary'].medium_issues,
                    'low_issues': audit_data['summary'].low_issues,
                    'audit_duration': audit_data['summary'].audit_duration,
                    'overall_score': audit_data['summary'].overall_score,
                    'readiness_status': audit_data['summary'].readiness_status,
                },
                'results': [
                    {
                        'category': r.category,
                        'component': r.component,
                        'status': r.status,
                        'message': r.message,
                        'details': r.details,
                        'recommendations': r.recommendations,
                        'severity': r.severity,
                        'timestamp': r.timestamp
                    } for r in audit_data['results']
                ],
                'system_info': audit_data['system_info']
            }
            json.dump(json_data, f, indent=2, default=str)
        
        # Save markdown report
        md_path = reports_dir / f"{timestamp}_full_system_audit.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Print summary
        summary = audit_data['summary']
        logger.info("\n" + "=" * 90)
        logger.info("KIMERA SWM - FULL CORE SYSTEM AUDIT COMPLETE")
        logger.info("=" * 90)
        logger.info(f"Overall Score: {summary.overall_score:.1f}/100")
        logger.info(f"Readiness Status: {summary.readiness_status.upper().replace('_', ' ')}")
        logger.info(f"Total Checks: {summary.total_checks}")
        logger.info(f"Passed: {summary.passed_checks} | Warnings: {summary.warning_checks} | Failed: {summary.failed_checks}")
        logger.info(f"Critical Issues: {summary.critical_issues}")
        logger.info(f"Audit Duration: {summary.audit_duration:.2f}s")
        logger.info(f"Detailed Report: {md_path}")
        
        # Return appropriate exit code
        if summary.critical_issues > 0:
            logger.info("\n❌ CRITICAL ISSUES DETECTED - IMMEDIATE ATTENTION REQUIRED")
            return 2
        elif summary.readiness_status in ['production_ready', 'nearly_ready']:
            logger.info(f"\n🎉 SYSTEM AUDIT COMPLETE - {summary.readiness_status.upper().replace('_', ' ')}")
            return 0
        else:
            logger.info(f"\n⚠️ SYSTEM NEEDS IMPROVEMENT - {summary.readiness_status.upper().replace('_', ' ')}")
            return 1
            
    except Exception as e:
        logger.error(f"Full system audit failed: {e}")
        traceback.print_exc()
        return 3

if __name__ == "__main__":
    sys.exit(main()) 