#!/usr/bin/env python3
"""
KIMERA 7-DAY CHALLENGE INTEGRATION TEST
=====================================
Tests the 7-day marathon challenge against a live KIMERA instance
"""

import asyncio
import json
import logging
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KimeraIntegrationTester:
    """Test suite for 7-day challenge integration with live KIMERA"""
    
    def __init__(self, kimera_base_url: str = "http://localhost:8001"):
        self.kimera_base_url = kimera_base_url
        self.test_results = []
        
    async def test_kimera_connectivity(self) -> bool:
        """Test if KIMERA API is accessible"""
        try:
            response = requests.get(f"{self.kimera_base_url}/system/status", timeout=5)
            if response.status_code == 200:
                logger.info("✅ KIMERA API connectivity: SUCCESS")
                return True
            else:
                logger.error(f"❌ KIMERA API returned status: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ KIMERA API connectivity failed: {e}")
            return False
    
    async def test_contradiction_engine(self) -> bool:
        """Test KIMERA's contradiction detection engine"""
        try:
            # Create test market data with contradictions
            test_data = {
                "content": [
                    {
                        "type": "price_action",
                        "data": {"BTC-USD": {"price": 45000, "trend": "bullish", "volume": "high"}}
                    },
                    {
                        "type": "news_sentiment", 
                        "data": {"BTC": {"sentiment": "bearish", "confidence": 0.8, "source": "financial_news"}}
                    },
                    {
                        "type": "social_sentiment",
                        "data": {"BTC": {"sentiment": "neutral", "confidence": 0.6, "volume": "medium"}}
                    }
                ],
                "analysis_type": "contradiction_detection"
            }
            
            response = requests.post(
                f"{self.kimera_base_url}/process/contradictions",
                json=test_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if "contradictions" in result:
                    logger.info("✅ Contradiction Engine: FUNCTIONAL")
                    logger.info(f"   Detected {len(result.get('contradictions', []))} contradictions")
                    return True
                else:
                    logger.warning("⚠️ Contradiction Engine: Response format unexpected")
                    return False
            else:
                logger.error(f"❌ Contradiction Engine: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Contradiction Engine test failed: {e}")
            return False
    
    async def test_7day_challenge_initialization(self) -> bool:
        """Test 7-day challenge initialization"""
        try:
            # Import and test the challenge class
            from kimera_1dollar_challenge import KimeraOneDollarChallenge
            
            challenge = KimeraOneDollarChallenge()
            
            # Verify 7-day parameters
            expected_duration = timedelta(days=7)
            actual_duration = challenge.end_time - challenge.start_time
            
            if abs((actual_duration - expected_duration).total_seconds()) < 60:  # Allow 1 minute tolerance
                logger.info("✅ 7-Day Challenge Duration: CORRECT")
                logger.info(f"   Duration: {actual_duration}")
            else:
                logger.error(f"❌ 7-Day Challenge Duration: Expected {expected_duration}, got {actual_duration}")
                return False
            
            # Verify marathon parameters
            if challenge.max_position_size == 0.6:
                logger.info("✅ Position Size (60%): CORRECT")
            else:
                logger.error(f"❌ Position Size: Expected 0.6, got {challenge.max_position_size}")
                return False
            
            if challenge.scan_interval == 10:
                logger.info("✅ Scan Interval (10s): CORRECT")
            else:
                logger.error(f"❌ Scan Interval: Expected 10, got {challenge.scan_interval}")
                return False
            
            if challenge.max_concurrent_positions == 8:
                logger.info("✅ Max Positions (8): CORRECT")
            else:
                logger.error(f"❌ Max Positions: Expected 8, got {challenge.max_concurrent_positions}")
                return False
            
            logger.info("✅ 7-Day Challenge Initialization: SUCCESS")
            return True
            
        except Exception as e:
            logger.error(f"❌ 7-Day Challenge Initialization failed: {e}")
            return False
    
    async def run_full_integration_test(self) -> Dict[str, Any]:
        """Run complete integration test suite"""
        logger.info("🚀 STARTING KIMERA 7-DAY CHALLENGE INTEGRATION TEST")
        logger.info("=" * 70)
        
        test_suite = [
            ("7-Day Challenge Init", self.test_7day_challenge_initialization),
            ("KIMERA Connectivity", self.test_kimera_connectivity),
            ("Contradiction Engine", self.test_contradiction_engine),
        ]
        
        results = {}
        passed = 0
        total = len(test_suite)
        
        for test_name, test_func in test_suite:
            logger.info(f"\n🧪 Testing: {test_name}")
            logger.info("-" * 40)
            
            try:
                result = await test_func()
                results[test_name] = result
                if result:
                    passed += 1
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
            except Exception as e:
                logger.error(f"💥 {test_name}: CRASHED - {e}")
                results[test_name] = False
        
        # Final summary
        logger.info("\n" + "=" * 70)
        logger.info("🏁 INTEGRATION TEST SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Tests Passed: {passed}/{total}")
        logger.info(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if passed == total:
            logger.info("🎉 ALL TESTS PASSED - 7-DAY CHALLENGE READY!")
        elif passed >= total * 0.8:
            logger.info("⚠️ MOSTLY FUNCTIONAL - Minor issues detected")
        else:
            logger.error("❌ CRITICAL ISSUES - System not ready for 7-day challenge")
        
        return results

async def main():
    """Run the integration test"""
    tester = KimeraIntegrationTester()
    await tester.run_full_integration_test()

if __name__ == "__main__":
    asyncio.run(main()) 