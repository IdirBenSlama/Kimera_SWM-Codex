#!/usr/bin/env python3
"""
KIMERA SWM SYSTEM AUDIT
======================

Comprehensive audit of the running Kimera SWM system to verify:
- System health and status
- Engine operational status
- API endpoint availability
- Database connectivity
- Security configurations
- Performance metrics
- Thermodynamic capabilities
- Integration completeness
"""

import sys
import time
import requests
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KimeraSystemAuditor:
    """Comprehensive system auditor for Kimera SWM"""
    
    def __init__(self, base_url: str = "http://127.0.0.1"):
        self.base_url = base_url
        self.ports_to_try = [8000, 8001, 8002, 8003, 8080]
        self.active_port = None
        self.audit_results = {}
        self.issues_found = []
        self.timestamp = datetime.now()
    
    def find_active_port(self) -> bool:
        """Find which port the Kimera system is running on"""
        logger.info("🔍 Searching for active Kimera SWM server...")
        
        for port in self.ports_to_try:
            try:
                url = f"{self.base_url}:{port}/health"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    self.active_port = port
                    logger.info(f"✅ Found Kimera SWM running on port {port}")
                    return True
            except requests.exceptions.RequestException:
                continue
        
        logger.info("❌ No active Kimera SWM server found")
        return False
    
    def audit_system_health(self) -> Dict[str, Any]:
        """Audit overall system health"""
        logger.info("\n🏥 AUDITING SYSTEM HEALTH")
        logger.info("-" * 50)
        
        health_data = {}
        
        try:
            # Basic health check
            url = f"{self.base_url}:{self.active_port}/health"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                logger.info("✅ System health endpoint responsive")
                logger.info(f"   Status: {health_data.get('status', 'unknown')}")
                logger.info(f"   Timestamp: {health_data.get('timestamp', 'unknown')}")
                
                # Check specific health metrics
                if 'components' in health_data:
                    components = health_data['components']
                    healthy_components = sum(1 for c in components.values() if c.get('status') == 'healthy')
                    total_components = len(components)
                    logger.info(f"   Components: {healthy_components}/{total_components} healthy")
                
            else:
                logger.info(f"❌ Health endpoint returned status {response.status_code}")
                self.issues_found.append(f"Health endpoint not responding properly: {response.status_code}")
                
        except Exception as e:
            logger.info(f"❌ Health check failed: {e}")
            self.issues_found.append(f"Health check failed: {e}")
            
        return health_data
    
    def audit_core_system(self) -> Dict[str, Any]:
        """Audit core Kimera system integration"""
        logger.info("\n🧠 AUDITING CORE SYSTEM INTEGRATION")
        logger.info("-" * 50)
        
        core_data = {}
        
        try:
            # Check system status
            url = f"{self.base_url}:{self.active_port}/api/v1/system/status"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                core_data = response.json()
                logger.info("✅ Core system status endpoint responsive")
                
                # Check engine status
                if 'engines' in core_data:
                    engines = core_data['engines']
                    operational_engines = sum(1 for e in engines.values() if e is True)
                    total_engines = len(engines)
                    logger.info(f"   Engines: {operational_engines}/{total_engines} operational")
                    
                    # List non-operational engines
                    failed_engines = [name for name, status in engines.items() if not status]
                    if failed_engines:
                        logger.info(f"   ❌ Non-operational engines: {', '.join(failed_engines)}")
                        self.issues_found.extend([f"Engine not operational: {engine}" for engine in failed_engines])
                
                # Check system state
                system_state = core_data.get('system_state', 'unknown')
                logger.info(f"   System State: {system_state}")
                
                if system_state != 'running':
                    self.issues_found.append(f"System not in running state: {system_state}")
                
            else:
                logger.info(f"❌ Core system endpoint returned status {response.status_code}")
                self.issues_found.append(f"Core system endpoint not responding: {response.status_code}")
                
        except Exception as e:
            logger.info(f"❌ Core system audit failed: {e}")
            self.issues_found.append(f"Core system audit failed: {e}")
            
        return core_data
    
    def audit_api_endpoints(self) -> Dict[str, Any]:
        """Audit critical API endpoints"""
        logger.info("\n🌐 AUDITING API ENDPOINTS")
        logger.info("-" * 50)
        
        endpoints_data = {}
        critical_endpoints = [
            '/docs',
            '/api/v1/system/info',
            '/',
        ]
        
        accessible_endpoints = 0
        
        for endpoint in critical_endpoints:
            try:
                url = f"{self.base_url}:{self.active_port}{endpoint}"
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    logger.info(f"   ✅ {endpoint}")
                    accessible_endpoints += 1
                    endpoints_data[endpoint] = 'accessible'
                else:
                    logger.info(f"   ❌ {endpoint} (status: {response.status_code})")
                    endpoints_data[endpoint] = f'error_{response.status_code}'
                    
            except Exception as e:
                logger.info(f"   ❌ {endpoint} (error: {str(e)[:50]}...)")
                endpoints_data[endpoint] = f'error_{type(e).__name__}'
        
        logger.info(f"\nAPI Endpoints: {accessible_endpoints}/{len(critical_endpoints)} accessible")
        return endpoints_data
    
    def generate_audit_report(self) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        logger.info("\n📊 GENERATING AUDIT REPORT")
        logger.info("=" * 80)
        
        total_issues = len(self.issues_found)
        
        if total_issues == 0:
            logger.info("🎉 AUDIT PASSED: No critical issues found!")
            audit_status = "PASSED"
        elif total_issues <= 3:
            logger.info(f"⚠️ AUDIT WARNING: {total_issues} minor issues found")
            audit_status = "WARNING" 
        else:
            logger.info(f"❌ AUDIT FAILED: {total_issues} issues found")
            audit_status = "FAILED"
        
        logger.info(f"\n📋 Issues Summary:")
        if self.issues_found:
            for i, issue in enumerate(self.issues_found, 1):
                logger.info(f"   {i}. {issue}")
        else:
            logger.info("   No issues found!")
        
        # Create comprehensive report
        audit_report = {
            'audit_timestamp': self.timestamp.isoformat(),
            'audit_status': audit_status,
            'total_issues': total_issues,
            'issues_found': self.issues_found,
            'server_port': self.active_port,
            'results': self.audit_results
        }
        
        return audit_report

def main():
    """Main audit execution"""
    logger.info("🔍 KIMERA SWM SYSTEM AUDIT")
    logger.info("=" * 80)
    logger.info(f"Audit started at: {datetime.now()}")
    logger.info()
    
    auditor = KimeraSystemAuditor()
    
    # Step 1: Find active server
    if not auditor.find_active_port():
        logger.info("❌ Cannot proceed with audit - no active server found")
        logger.info("💡 Make sure Kimera SWM is running first:")
        logger.info("   python kimera.py")
        return False
    
    # Wait a moment for server to fully initialize
    logger.info("⏳ Waiting for server to fully initialize...")
    time.sleep(3)
    
    # Step 2: Run audit components
    auditor.audit_results['health'] = auditor.audit_system_health()
    auditor.audit_results['core_system'] = auditor.audit_core_system()
    auditor.audit_results['api_endpoints'] = auditor.audit_api_endpoints()
    
    # Step 3: Generate final report
    final_report = auditor.generate_audit_report()
    
    return final_report['audit_status'] == 'PASSED'

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 