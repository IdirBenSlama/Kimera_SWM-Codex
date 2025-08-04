#!/usr/bin/env python3
"""
Kimera Deep System Analysis & Optimization
==========================================
Comprehensive analysis of all Kimera subsystems with optimization recommendations.
"""

import os
import sys
import json
import time
import asyncio
import logging
import psutil
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KimeraDeepAnalyzer:
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        sys.path.insert(0, str(self.project_root))
        
        self.base_url = "http://localhost:8000"
        self.analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "system_health": {},
            "performance_metrics": {},
            "scientific_integrity": {},
            "optimization_opportunities": [],
            "resource_utilization": {},
            "api_responsiveness": {}
        }
        
    def check_system_health(self):
        """Check overall system health"""
        logger.info("\n=== System Health Check ===")
        
        try:
            # Check if API is running
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✓ API server is running")
                self.analysis_results["system_health"]["api_status"] = "healthy"
            else:
                logger.warning(f"⚠ API returned status {response.status_code}")
                self.analysis_results["system_health"]["api_status"] = "unhealthy"
                
        except requests.exceptions.RequestException as e:
            logger.error(f"✗ API server not accessible: {e}")
            self.analysis_results["system_health"]["api_status"] = "offline"
            
        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        self.analysis_results["resource_utilization"] = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free / (1024**3)
        }
        
        logger.info(f"CPU Usage: {cpu_percent}%")
        logger.info(f"Memory Usage: {memory.percent}% ({memory.available / (1024**3):.2f} GB available)")
        logger.info(f"Disk Usage: {disk.percent}% ({disk.free / (1024**3):.2f} GB free)")
        
        # Check GPU
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024**3)
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_utilization = (gpu_memory / gpu_memory_total) * 100
            
            self.analysis_results["resource_utilization"]["gpu_memory_gb"] = gpu_memory
            self.analysis_results["resource_utilization"]["gpu_memory_percent"] = gpu_utilization
            
            logger.info(f"GPU Memory: {gpu_memory:.2f}/{gpu_memory_total:.2f} GB ({gpu_utilization:.1f}%)")
            
    async def test_api_endpoints(self):
        """Test all API endpoints for responsiveness"""
        logger.info("\n=== API Endpoint Testing ===")
        
        endpoints = [
            ("GET", "/", "Root"),
            ("GET", "/health", "Health"),
            ("GET", "/kimera/status", "Kimera Status"),
            ("POST", "/kimera/geoid/create", "Create Geoid"),
            ("POST", "/kimera/scars", "Create SCAR"),
            ("GET", "/kimera/contradictions/detect", "Detect Contradictions"),
            ("POST", "/kimera/insights/generate", "Generate Insights"),
            ("POST", "/kimera/revolutionary", "Revolutionary Intelligence"),
            ("GET", "/kimera/metrics", "System Metrics")
        ]
        
        test_data = {
            "concept": "quantum consciousness emergence",
            "embedding": np.random.rand(1024).tolist()
        }
        
        for method, endpoint, name in endpoints:
            try:
                start = time.time()
                
                if method == "GET":
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                else:
                    response = requests.post(f"{self.base_url}{endpoint}", 
                                           json=test_data, timeout=10)
                
                latency = (time.time() - start) * 1000  # ms
                
                status = "✓" if response.status_code in [200, 201] else "✗"
                logger.info(f"{status} {name}: {response.status_code} ({latency:.2f}ms)")
                
                self.analysis_results["api_responsiveness"][endpoint] = {
                    "status_code": response.status_code,
                    "latency_ms": latency,
                    "healthy": response.status_code in [200, 201]
                }
                
            except Exception as e:
                logger.error(f"✗ {name}: Failed - {str(e)}")
                self.analysis_results["api_responsiveness"][endpoint] = {
                    "status_code": 0,
                    "latency_ms": 0,
                    "healthy": False,
                    "error": str(e)
                }
                
    def analyze_scientific_integrity(self):
        """Analyze scientific integrity of the system"""
        logger.info("\n=== Scientific Integrity Analysis ===")
        
        try:
            from src.engines.foundational_thermodynamic_engine import FoundationalThermodynamicEngine
            from src.engines.quantum_field_engine import QuantumFieldEngine
            from src.engines.spde_engine import SPDEEngine
            
            # Test thermodynamic consistency
            thermo = FoundationalThermodynamicEngine()
            
            # Generate test data with known properties
            hot_temp = 1000.0
            cold_temp = 300.0
            
            hot_fields = [np.random.rand(100) * (hot_temp/300) for _ in range(10)]
            cold_fields = [np.random.rand(100) * (cold_temp/300) for _ in range(10)]
            
            cycle = thermo.run_zetetic_carnot_engine(hot_fields, cold_fields)
            
            # Calculate theoretical maximum
            carnot_max = 1 - (cold_temp / hot_temp)
            efficiency_ratio = cycle.actual_efficiency / carnot_max if carnot_max > 0 else 0
            
            self.analysis_results["scientific_integrity"]["thermodynamics"] = {
                "carnot_efficiency": cycle.actual_efficiency,
                "theoretical_max": carnot_max,
                "efficiency_ratio": efficiency_ratio,
                "physics_compliant": cycle.physics_compliant,
                "violations": len(thermo.physics_violations)
            }
            
            logger.info(f"Thermodynamics: {efficiency_ratio:.2%} of Carnot limit")
            logger.info(f"Physics violations: {len(thermo.physics_violations)}")
            
            # Test quantum mechanics
            qfe = QuantumFieldEngine(dimension=8)
            
            # Test uncertainty principle
            position_op = qfe.operators.get('position', np.eye(8))
            momentum_op = qfe.operators.get('momentum', np.eye(8))
            
            # Calculate commutator [x,p]
            commutator = position_op @ momentum_op - momentum_op @ position_op
            uncertainty_product = np.linalg.norm(commutator)
            
            self.analysis_results["scientific_integrity"]["quantum"] = {
                "uncertainty_product": uncertainty_product,
                "heisenberg_satisfied": uncertainty_product > 0,
                "dimension": qfe.dimension
            }
            
            logger.info(f"Quantum uncertainty: {uncertainty_product:.4f}")
            
            # Test diffusion conservation
            spde = SPDEEngine()
            test_field = np.ones((32, 32))
            evolved = spde.evolve(test_field, dt=0.01, steps=100)
            
            conservation_error = abs(np.sum(evolved) - np.sum(test_field)) / np.sum(test_field)
            
            self.analysis_results["scientific_integrity"]["diffusion"] = {
                "conservation_error": conservation_error,
                "conservation_satisfied": conservation_error < 0.01
            }
            
            logger.info(f"Diffusion conservation: {conservation_error:.6f}")
            
        except Exception as e:
            logger.error(f"Scientific integrity analysis failed: {e}")
            self.analysis_results["scientific_integrity"]["error"] = str(e)
            
    def analyze_performance_bottlenecks(self):
        """Identify performance bottlenecks"""
        logger.info("\n=== Performance Analysis ===")
        
        try:
            # Test embedding performance
            from src.core.embedding_utils import encode_text, encode_batch
            
            # Single encoding
            test_text = "Analyzing quantum semantic entanglement patterns"
            
            times = []
            for _ in range(5):
                start = time.time()
                _ = encode_text(test_text)
                times.append(time.time() - start)
                
            avg_single_time = np.mean(times[1:])  # Skip first (warmup)
            
            # Batch encoding
            batch_texts = [
                f"Test concept {i}: {np.random.choice(['quantum', 'semantic', 'cognitive', 'emergent'])}"
                for i in range(20)
            ]
            
            start = time.time()
            _ = encode_batch(batch_texts)
            batch_time = time.time() - start
            
            self.analysis_results["performance_metrics"]["embedding"] = {
                "single_encoding_ms": avg_single_time * 1000,
                "batch_encoding_ms": batch_time * 1000,
                "batch_size": len(batch_texts),
                "throughput_per_sec": len(batch_texts) / batch_time
            }
            
            logger.info(f"Single encoding: {avg_single_time*1000:.2f}ms")
            logger.info(f"Batch throughput: {len(batch_texts)/batch_time:.2f} texts/sec")
            
            # Test database performance
            from src.vault.database import SessionLocal
            
            db_times = []
            with SessionLocal() as session:
                for _ in range(10):
                    start = time.time()
                    result = session.execute("SELECT 1")
                    _ = result.scalar()
                    db_times.append(time.time() - start)
                    
            avg_db_time = np.mean(db_times)
            
            self.analysis_results["performance_metrics"]["database"] = {
                "query_latency_ms": avg_db_time * 1000,
                "queries_per_sec": 1 / avg_db_time
            }
            
            logger.info(f"Database query: {avg_db_time*1000:.2f}ms")
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            self.analysis_results["performance_metrics"]["error"] = str(e)
            
    def identify_optimization_opportunities(self):
        """Identify specific optimization opportunities"""
        logger.info("\n=== Optimization Opportunities ===")
        
        opportunities = []
        
        # Check CPU usage
        if self.analysis_results["resource_utilization"].get("cpu_percent", 0) > 80:
            opportunities.append({
                "area": "CPU",
                "issue": "High CPU usage",
                "recommendation": "Consider load balancing or async processing",
                "priority": "high"
            })
            
        # Check memory usage
        if self.analysis_results["resource_utilization"].get("memory_percent", 0) > 80:
            opportunities.append({
                "area": "Memory",
                "issue": "High memory usage",
                "recommendation": "Implement memory pooling and garbage collection",
                "priority": "high"
            })
            
        # Check GPU utilization
        gpu_util = self.analysis_results["resource_utilization"].get("gpu_memory_percent", 0)
        if gpu_util < 50 and torch.cuda.is_available():
            opportunities.append({
                "area": "GPU",
                "issue": "Low GPU utilization",
                "recommendation": "Batch more operations for GPU processing",
                "priority": "medium"
            })
            
        # Check API latencies
        slow_endpoints = []
        for endpoint, metrics in self.analysis_results.get("api_responsiveness", {}).items():
            if metrics.get("latency_ms", 0) > 500:
                slow_endpoints.append(endpoint)
                
        if slow_endpoints:
            opportunities.append({
                "area": "API",
                "issue": f"Slow endpoints: {', '.join(slow_endpoints)}",
                "recommendation": "Add caching or optimize database queries",
                "priority": "high"
            })
            
        # Check embedding performance
        embed_metrics = self.analysis_results.get("performance_metrics", {}).get("embedding", {})
        if embed_metrics.get("single_encoding_ms", 0) > 100:
            opportunities.append({
                "area": "Embeddings",
                "issue": "Slow embedding generation",
                "recommendation": "Use ONNX or TensorRT optimization",
                "priority": "medium"
            })
            
        # Check scientific integrity
        sci_integrity = self.analysis_results.get("scientific_integrity", {})
        
        thermo = sci_integrity.get("thermodynamics", {})
        if thermo.get("violations", 0) > 0:
            opportunities.append({
                "area": "Thermodynamics",
                "issue": f"{thermo['violations']} physics violations detected",
                "recommendation": "Review and fix thermodynamic calculations",
                "priority": "critical"
            })
            
        conservation = sci_integrity.get("diffusion", {})
        if conservation.get("conservation_error", 1) > 0.01:
            opportunities.append({
                "area": "Diffusion",
                "issue": "Conservation law violations",
                "recommendation": "Implement symplectic integrators",
                "priority": "high"
            })
            
        self.analysis_results["optimization_opportunities"] = opportunities
        
        for opp in opportunities:
            logger.info(f"{opp['priority'].upper()}: {opp['area']} - {opp['issue']}")
            logger.info(f"  → {opp['recommendation']}")
            
    async def run_comprehensive_analysis(self):
        """Run complete system analysis"""
        logger.info("Starting Kimera Deep System Analysis...")
        
        # Basic health check
        self.check_system_health()
        
        # API testing
        await self.test_api_endpoints()
        
        # Scientific integrity
        self.analyze_scientific_integrity()
        
        # Performance analysis
        self.analyze_performance_bottlenecks()
        
        # Optimization opportunities
        self.identify_optimization_opportunities()
        
        # Generate report
        return self.generate_report()
        
    def generate_report(self):
        """Generate comprehensive analysis report"""
        
        # Calculate overall health score
        health_score = 0
        health_factors = 0
        
        # API health
        api_healthy = sum(1 for m in self.analysis_results.get("api_responsiveness", {}).values() 
                         if m.get("healthy", False))
        api_total = len(self.analysis_results.get("api_responsiveness", {}))
        if api_total > 0:
            health_score += (api_healthy / api_total) * 100
            health_factors += 1
            
        # Resource health
        cpu_health = 100 - self.analysis_results["resource_utilization"].get("cpu_percent", 100)
        mem_health = 100 - self.analysis_results["resource_utilization"].get("memory_percent", 100)
        health_score += (cpu_health + mem_health) / 2
        health_factors += 1
        
        # Scientific integrity
        sci_health = 0
        sci_factors = 0
        
        thermo = self.analysis_results.get("scientific_integrity", {}).get("thermodynamics", {})
        if thermo.get("physics_compliant", False):
            sci_health += 100
        sci_factors += 1
        
        quantum = self.analysis_results.get("scientific_integrity", {}).get("quantum", {})
        if quantum.get("heisenberg_satisfied", False):
            sci_health += 100
        sci_factors += 1
        
        diffusion = self.analysis_results.get("scientific_integrity", {}).get("diffusion", {})
        if diffusion.get("conservation_satisfied", False):
            sci_health += 100
        sci_factors += 1
        
        if sci_factors > 0:
            health_score += sci_health / sci_factors
            health_factors += 1
            
        overall_health = health_score / health_factors if health_factors > 0 else 0
        
        self.analysis_results["overall_health_score"] = overall_health
        
        # Save report
        report_file = f"kimera_deep_analysis_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
            
        # Print summary
        logger.info("\n" + "="*70)
        logger.info("KIMERA DEEP SYSTEM ANALYSIS SUMMARY")
        logger.info("="*70)
        logger.info(f"Overall Health Score: {overall_health:.1f}%")
        
        logger.info("\nResource Utilization:")
        res = self.analysis_results["resource_utilization"]
        logger.info(f"  CPU: {res.get('cpu_percent', 0):.1f}%")
        logger.info(f"  Memory: {res.get('memory_percent', 0):.1f}%")
        if "gpu_memory_percent" in res:
            logger.info(f"  GPU: {res.get('gpu_memory_percent', 0):.1f}%")
            
        logger.info("\nAPI Health:")
        logger.info(f"  Healthy endpoints: {api_healthy}/{api_total}")
        
        logger.info("\nScientific Integrity:")
        logger.info(f"  Thermodynamics: {'✓' if thermo.get('physics_compliant', False) else '✗'}")
        logger.info(f"  Quantum: {'✓' if quantum.get('heisenberg_satisfied', False) else '✗'}")
        logger.info(f"  Conservation: {'✓' if diffusion.get('conservation_satisfied', False) else '✗'}")
        
        logger.info(f"\nOptimization Opportunities: {len(self.analysis_results['optimization_opportunities'])}")
        
        critical = sum(1 for o in self.analysis_results['optimization_opportunities'] 
                      if o['priority'] == 'critical')
        high = sum(1 for o in self.analysis_results['optimization_opportunities'] 
                  if o['priority'] == 'high')
        
        if critical > 0:
            logger.info(f"  CRITICAL: {critical}")
        if high > 0:
            logger.info(f"  HIGH: {high}")
            
        logger.info("="*70)
        logger.info(f"\nDetailed report saved to: {report_file}")
        
        return self.analysis_results


if __name__ == "__main__":
    analyzer = KimeraDeepAnalyzer()
    
    # Wait a moment for the server to be fully ready
    time.sleep(2)
    
    # Run analysis
    results = asyncio.run(analyzer.run_comprehensive_analysis()) 