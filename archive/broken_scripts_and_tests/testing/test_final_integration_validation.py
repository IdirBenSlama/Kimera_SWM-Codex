#!/usr/bin/env python3
"""
Final Integration Validation Test for KIMERA GPU Foundation
Tests the complete integration with the live KIMERA system
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinalIntegrationValidator:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_system_status(self):
        """Test basic system status"""
        try:
            async with self.session.get(f"{self.base_url}/system/status") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ System Status: {data.get('status', 'Unknown')}")
                    return True
                else:
                    logger.error(f"❌ System status failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ System status error: {e}")
            return False
    
    async def test_gpu_foundation_endpoint(self):
        """Test GPU Foundation specific endpoint"""
        try:
            async with self.session.get(f"{self.base_url}/system/gpu_foundation") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("✅ GPU Foundation Status:")
                    logger.info(f"   Device: {data.get('device_name', 'Unknown')}")
                    logger.info(f"   Memory: {data.get('total_memory_gb', 0):.1f} GB")
                    logger.info(f"   CUDA Cores: {data.get('cuda_cores', 0)}")
                    
                    # Check performance metrics
                    perf = data.get('performance_metrics', {})
                    if perf:
                        logger.info("   Performance Metrics:")
                        for key, value in perf.items():
                            logger.info(f"     {key}: {value}")
                    
                    return True
                else:
                    logger.error(f"❌ GPU Foundation endpoint failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ GPU Foundation endpoint error: {e}")
            return False
    
    async def test_cognitive_processing(self):
        """Test cognitive processing with GPU acceleration"""
        test_cases = [
            {
                "name": "GPU-Accelerated Semantic Analysis",
                "echoform_text": "Analyze the quantum computational advantages of GPU-accelerated neural networks in cognitive processing systems"
            },
            {
                "name": "GPU Memory Optimization",
                "echoform_text": "Evaluate GPU memory management strategies for large-scale cognitive vault operations"
            }
        ]
        
        results = []
        for test_case in test_cases:
            try:
                start_time = time.time()
                async with self.session.post(
                    f"{self.base_url}/geoids",
                    json={"echoform_text": test_case["echoform_text"]}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        end_time = time.time()
                        
                        logger.info(f"✅ {test_case['name']}")
                        logger.info(f"   Response Time: {(end_time - start_time) * 1000:.0f}ms")
                        logger.info(f"   Geoid ID: {data.get('geoid_id', 'Unknown')}")
                        logger.info(f"   Response Length: {len(data.get('response', ''))} chars")
                        
                        results.append({
                            "test": test_case["name"],
                            "success": True,
                            "response_time_ms": (end_time - start_time) * 1000,
                            "geoid_id": data.get('geoid_id'),
                            "response_length": len(data.get('response', ''))
                        })
                    else:
                        logger.error(f"❌ {test_case['name']} failed: {response.status}")
                        results.append({
                            "test": test_case["name"],
                            "success": False,
                            "error": f"HTTP {response.status}"
                        })
                        
            except Exception as e:
                logger.error(f"❌ {test_case['name']} error: {e}")
                results.append({
                    "test": test_case["name"],
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    async def test_concurrent_gpu_load(self):
        """Test concurrent GPU processing load"""
        logger.info("🚀 Testing concurrent GPU load...")
        
        async def single_request():
            try:
                async with self.session.post(
                    f"{self.base_url}/geoids",
                    json={"echoform_text": "GPU concurrent processing test"}
                ) as response:
                    return response.status == 200
            except:
                return False
        
        # Launch 5 concurrent requests
        start_time = time.time()
        tasks = [single_request() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        success_rate = sum(results) / len(results) * 100
        logger.info(f"✅ Concurrent Load Test:")
        logger.info(f"   Success Rate: {success_rate:.1f}% ({sum(results)}/{len(results)})")
        logger.info(f"   Total Time: {(end_time - start_time) * 1000:.0f}ms")
        logger.info(f"   Avg Time per Request: {(end_time - start_time) * 1000 / len(results):.0f}ms")
        
        return success_rate >= 80.0
    
    async def run_full_validation(self):
        """Run complete validation suite"""
        logger.info("🔬 Starting Final Integration Validation...")
        logger.info("=" * 60)
        
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "system_status": False,
            "gpu_foundation": False,
            "cognitive_processing": [],
            "concurrent_load": False,
            "overall_success": False
        }
        
        # Test 1: System Status
        logger.info("📊 Testing System Status...")
        test_results["system_status"] = await self.test_system_status()
        
        # Test 2: GPU Foundation Endpoint
        logger.info("\n🖥️ Testing GPU Foundation Endpoint...")
        test_results["gpu_foundation"] = await self.test_gpu_foundation_endpoint()
        
        # Test 3: Cognitive Processing
        logger.info("\n🧠 Testing Cognitive Processing...")
        test_results["cognitive_processing"] = await self.test_cognitive_processing()
        
        # Test 4: Concurrent Load
        logger.info("\n⚡ Testing Concurrent GPU Load...")
        test_results["concurrent_load"] = await self.test_concurrent_gpu_load()
        
        # Calculate overall success
        cognitive_success_rate = sum(1 for r in test_results["cognitive_processing"] if r["success"]) / max(len(test_results["cognitive_processing"]), 1)
        
        test_results["overall_success"] = (
            test_results["system_status"] and
            test_results["gpu_foundation"] and
            cognitive_success_rate >= 0.8 and
            test_results["concurrent_load"]
        )
        
        # Final Report
        logger.info("\n" + "=" * 60)
        logger.info("📋 FINAL INTEGRATION VALIDATION REPORT")
        logger.info("=" * 60)
        logger.info(f"System Status: {'✅ PASS' if test_results['system_status'] else '❌ FAIL'}")
        logger.info(f"GPU Foundation: {'✅ PASS' if test_results['gpu_foundation'] else '❌ FAIL'}")
        logger.info(f"Cognitive Processing: {'✅ PASS' if cognitive_success_rate >= 0.8 else '❌ FAIL'} ({cognitive_success_rate*100:.1f}%)")
        logger.info(f"Concurrent Load: {'✅ PASS' if test_results['concurrent_load'] else '❌ FAIL'}")
        logger.info("=" * 60)
        
        if test_results["overall_success"]:
            logger.info("🎉 KIMERA GPU FOUNDATION INTEGRATION: ✅ FULLY OPERATIONAL")
            logger.info("🚀 System is ready for Phase 1, Week 2!")
        else:
            logger.info("⚠️ KIMERA GPU FOUNDATION INTEGRATION: ❌ ISSUES DETECTED")
            logger.info("🔧 Review failed tests and address issues before proceeding")
        
        logger.info("=" * 60)
        
        # Save detailed results
        with open(f"final_integration_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(test_results, f, indent=2)
        
        return test_results

async def main():
    """Main validation function"""
    async with FinalIntegrationValidator() as validator:
        results = await validator.run_full_validation()
        return results["overall_success"]

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1) 