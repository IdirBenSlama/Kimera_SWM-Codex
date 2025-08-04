#!/usr/bin/env python3
"""
KIMERA SWM Comprehensive System Analysis
========================================

This script performs a deep, scientific analysis of the KIMERA system,
examining all components, pipelines, and potential issues.
"""

import requests
import json
import time
import asyncio
from typing import Dict, Any, List, Tuple
from datetime import datetime
import traceback
import logging
logger = logging.getLogger(__name__)

class KimeraSystemAnalyzer:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "endpoints": {},
            "issues": [],
            "recommendations": []
        }
    
    def analyze_endpoint(self, path: str, method: str = "GET", payload: Dict = None, timeout: int = 60) -> Dict[str, Any]:
        """Analyze a single endpoint"""
        url = f"{self.base_url}{path}"
        logger.info(f"\n🔍 Analyzing: {method} {path}")
        
        try:
            start_time = time.time()
            
            if method == "GET":
                response = requests.get(url, timeout=timeout)
            elif method == "POST":
                response = requests.post(url, json=payload, timeout=timeout)
            else:
                return {"error": f"Unsupported method: {method}"}
            
            elapsed_time = time.time() - start_time
            
            result = {
                "status_code": response.status_code,
                "response_time": elapsed_time,
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else response.text
            }
            
            if response.status_code == 200:
                logger.info(f"   ✅ Success ({elapsed_time:.2f}s)")
            else:
                logger.info(f"   ❌ Failed: {response.status_code}")
                self.results["issues"].append({
                    "endpoint": path,
                    "issue": f"HTTP {response.status_code}",
                    "details": response.text[:200]
                })
            
            return result
            
        except requests.exceptions.Timeout:
            logger.info(f"   ⏱️ Timeout after {timeout}s")
            self.results["issues"].append({
                "endpoint": path,
                "issue": "Timeout",
                "details": f"Request timed out after {timeout} seconds"
            })
            return {"error": "timeout", "timeout": timeout}
            
        except Exception as e:
            logger.info(f"   ❌ Error: {str(e)}")
            self.results["issues"].append({
                "endpoint": path,
                "issue": "Exception",
                "details": str(e)
            })
            return {"error": str(e)}
    
    def analyze_core_components(self):
        """Analyze core KIMERA components"""
        logger.info("\n" + "="*60)
        logger.info("🧠 ANALYZING CORE COMPONENTS")
        logger.info("="*60)
        
        # System Status
        status = self.analyze_endpoint("/kimera/status")
        if status.get("success"):
            self.results["components"]["system"] = status["data"]
        
        # GPU Foundation
        gpu_status = self.analyze_endpoint("/kimera/system/gpu_foundation")
        if gpu_status.get("success"):
            self.results["components"]["gpu_foundation"] = gpu_status["data"]
        
        # Contradiction Engine
        contradiction = self.analyze_endpoint("/kimera/contradiction_engine")
        if contradiction.get("success"):
            self.results["components"]["contradiction_engine"] = contradiction["data"]
        
        # Thermodynamic Engine
        thermo = self.analyze_endpoint("/kimera/thermodynamic_engine")
        if thermo.get("success"):
            self.results["components"]["thermodynamic_engine"] = thermo["data"]
        
        # Vault Manager
        vault_stats = self.analyze_endpoint("/kimera/stats")
        if vault_stats.get("success"):
            self.results["components"]["vault_manager"] = vault_stats["data"]
    
    def analyze_cognitive_engines(self):
        """Analyze cognitive processing engines"""
        logger.info("\n" + "="*60)
        logger.info("🔬 ANALYZING COGNITIVE ENGINES")
        logger.info("="*60)
        
        # Revolutionary Intelligence
        revolutionary = self.analyze_endpoint("/kimera/revolutionary/status/complete")
        if revolutionary.get("success"):
            self.results["components"]["revolutionary_intelligence"] = revolutionary["data"]
        
        # Cognitive Pharmaceutical
        pharma = self.analyze_endpoint("/kimera/cognitive-pharmaceutical/system/status")
        if pharma.get("success"):
            self.results["components"]["cognitive_pharmaceutical"] = pharma["data"]
        
        # Law Enforcement
        law = self.analyze_endpoint("/kimera/law_enforcement/system_status")
        if law.get("success"):
            self.results["components"]["law_enforcement"] = law["data"]
        
        # Monitoring System
        monitoring = self.analyze_endpoint("/kimera/monitoring/status")
        if monitoring.get("success"):
            self.results["components"]["monitoring"] = monitoring["data"]
    
    def test_chat_functionality(self):
        """Test the chat/diffusion model functionality"""
        logger.info("\n" + "="*60)
        logger.info("💬 TESTING CHAT & DIFFUSION MODEL")
        logger.info("="*60)
        
        # Test basic chat
        chat_payload = {
            "message": "Explain your cognitive architecture and diffusion model implementation.",
            "mode": "cognitive_enhanced",
            "session_id": "analysis_session",
            "cognitive_mode": "cognitive_enhanced"
        }
        
        logger.info("\n📝 Testing enhanced cognitive chat...")
        chat_result = self.analyze_endpoint("/kimera/chat/", "POST", chat_payload, timeout=120)
        
        if chat_result.get("success"):
            data = chat_result["data"]
            logger.info(f"\n📊 Chat Metrics:")
            logger.info(f"   - Response length: {len(data.get('response', ''))} chars")
            logger.info(f"   - Confidence: {data.get('confidence', 0):.3f}")
            logger.info(f"   - Semantic Coherence: {data.get('semantic_coherence', 0):.3f}")
            logger.info(f"   - Cognitive Resonance: {data.get('cognitive_resonance', 0):.3f}")
            logger.info(f"   - Generation Time: {data.get('generation_time', 0):.2f}s")
            
            self.results["components"]["chat_diffusion"] = {
                "operational": True,
                "metrics": {
                    "confidence": data.get('confidence', 0),
                    "semantic_coherence": data.get('semantic_coherence', 0),
                    "cognitive_resonance": data.get('cognitive_resonance', 0),
                    "generation_time": data.get('generation_time', 0)
                }
            }
        else:
            self.results["components"]["chat_diffusion"] = {
                "operational": False,
                "error": chat_result.get("error", "Unknown error")
            }
    
    def analyze_integration_status(self):
        """Analyze system integration"""
        logger.info("\n" + "="*60)
        logger.info("🌉 ANALYZING SYSTEM INTEGRATION")
        logger.info("="*60)
        
        # Chat Integration
        integration = self.analyze_endpoint("/kimera/chat/integration/status")
        if integration.get("success"):
            self.results["components"]["integration_bridge"] = integration["data"]
        
        # Translator Hub Status
        translator = self.analyze_endpoint("/translator/status")
        if translator.get("success"):
            self.results["components"]["translator_hub"] = translator["data"]
    
    def perform_stress_tests(self):
        """Perform basic stress tests"""
        logger.info("\n" + "="*60)
        logger.info("⚡ PERFORMING STRESS TESTS")
        logger.info("="*60)
        
        # Test rapid endpoint access
        endpoints = ["/health", "/kimera/status", "/kimera/metrics"]
        
        for endpoint in endpoints:
            logger.info(f"\n📊 Stress testing {endpoint}...")
            times = []
            errors = 0
            
            for i in range(10):
                start = time.time()
                try:
                    r = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                    if r.status_code == 200:
                        times.append(time.time() - start)
                    else:
                        errors += 1
                except Exception as e:
                    logger.error(f"Error in kimera_comprehensive_analysis.py: {e}", exc_info=True)
                    raise  # Re-raise for proper error handling
                    errors += 1
            
            if times:
                avg_time = sum(times) / len(times)
                logger.info(f"   Average response time: {avg_time:.3f}s")
                logger.info(f"   Success rate: {len(times)/10*100:.0f}%")
            else:
                logger.info(f"   ❌ All requests failed")
    
    def identify_issues_and_recommendations(self):
        """Analyze results and provide recommendations"""
        logger.info("\n" + "="*60)
        logger.info("🔧 ISSUES & RECOMMENDATIONS")
        logger.info("="*60)
        
        # Check for timeout issues
        timeout_issues = [i for i in self.results["issues"] if i["issue"] == "Timeout"]
        if timeout_issues:
            self.results["recommendations"].append({
                "category": "Performance",
                "recommendation": "Model loading is slow. Consider implementing model preloading or caching.",
                "affected_endpoints": [i["endpoint"] for i in timeout_issues]
            })
        
        # Check component status
        if "chat_diffusion" in self.results["components"]:
            if not self.results["components"]["chat_diffusion"].get("operational"):
                self.results["recommendations"].append({
                    "category": "Critical",
                    "recommendation": "Chat/Diffusion model is not operational. Check model loading and GPU availability.",
                    "details": self.results["components"]["chat_diffusion"].get("error")
                })
        
        # Check GPU status
        if "gpu_foundation" in self.results["components"]:
            gpu_data = self.results["components"]["gpu_foundation"]
            if not gpu_data.get("gpu_available"):
                self.results["recommendations"].append({
                    "category": "Performance",
                    "recommendation": "GPU not available. System running on CPU which will impact performance.",
                    "impact": "Significant slowdown in model inference and tensor operations"
                })
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        logger.info("\n" + "="*60)
        logger.info("📄 COMPREHENSIVE ANALYSIS REPORT")
        logger.info("="*60)
        
        # System Overview
        logger.info("\n🏗️ SYSTEM ARCHITECTURE:")
        logger.info(f"   - Base URL: {self.base_url}")
        logger.info(f"   - Analysis Time: {self.results['timestamp']}")
        logger.info(f"   - Components Analyzed: {len(self.results['components'])}")
        logger.info(f"   - Issues Found: {len(self.results['issues'])}")
        
        # Component Status
        logger.info("\n📊 COMPONENT STATUS:")
        for component, data in self.results["components"].items():
            if isinstance(data, dict) and "operational" in data:
                status = "✅ Operational" if data["operational"] else "❌ Not Operational"
                logger.info(f"   - {component}: {status}")
            else:
                logger.info(f"   - {component}: ✅ Active")
        
        # Critical Issues
        if self.results["issues"]:
            logger.info("\n⚠️ CRITICAL ISSUES:")
            for issue in self.results["issues"][:5]:  # Show top 5
                logger.info(f"   - {issue['endpoint']}: {issue['issue']}")
        
        # Recommendations
        if self.results["recommendations"]:
            logger.info("\n💡 RECOMMENDATIONS:")
            for rec in self.results["recommendations"]:
                logger.info(f"\n   [{rec['category']}] {rec['recommendation']}")
                if "details" in rec:
                    logger.info(f"   Details: {rec['details']}")
        
        # Save detailed report
        report_file = f"kimera_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"\n📁 Detailed report saved to: {report_file}")
    
    def run_complete_analysis(self):
        """Run the complete system analysis"""
        logger.info("""
╔════���═════════════════════════════════════════════════════════════╗
║           KIMERA SWM - COMPREHENSIVE SYSTEM ANALYSIS              ║
║                                                                  ║
║  Performing deep analysis of:                                    ║
║  • Core Components & Architecture                                ║
║  • Cognitive Engines & Pipelines                                 ║
║  • Chat/Diffusion Model Integration                              ║
║  • System Performance & Stress Tests                             ║
║  • Issue Identification & Recommendations                        ║
╚══════════════════════════════════════════════════════════════════╝
""")
        
        # Run all analyses
        self.analyze_core_components()
        self.analyze_cognitive_engines()
        self.test_chat_functionality()
        self.analyze_integration_status()
        self.perform_stress_tests()
        self.identify_issues_and_recommendations()
        self.generate_report()
        
        logger.info("\n\n✅ Analysis complete!")

def main():
    analyzer = KimeraSystemAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()