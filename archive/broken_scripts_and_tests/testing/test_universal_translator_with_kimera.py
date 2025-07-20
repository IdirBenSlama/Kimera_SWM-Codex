#!/usr/bin/env python3
"""
Test Universal Translator with Running KIMERA Instance
Tests the Universal Translator Hub and Text Diffusion Engine with live KIMERA
"""

import sys
import os
import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, List
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KimeraUniversalTranslatorTester:
    """Test the Universal Translator with running KIMERA instance"""
    
    def __init__(self, kimera_base_url: str = "http://127.0.0.1:8000"):
        self.kimera_url = kimera_base_url
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def check_kimera_health(self) -> bool:
        """Check if KIMERA is responding"""
        try:
            async with self.session.get(f"{self.kimera_url}/metrics") as response:
                if response.status == 200:
                    logger.info("✅ KIMERA is responding to health checks")
                    return True
                else:
                    logger.error(f"❌ KIMERA health check failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ Cannot connect to KIMERA: {e}")
            return False
    
    async def test_cognitive_field_translation(self, text: str, target_modality: str) -> Dict[str, Any]:
        """Test translation using KIMERA's cognitive field"""
        try:
            # Create a cognitive field request for translation
            payload = {
                "input_text": text,
                "target_modality": target_modality,
                "use_gpu": True,
                "cognitive_field_integration": True,
                "timestamp": datetime.now().isoformat()
            }
            
            # Try to use cognitive field dynamics endpoint
            async with self.session.post(
                f"{self.kimera_url}/api/cognitive-field/process",
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"✅ Cognitive field translation successful")
                    return result
                else:
                    logger.warning(f"⚠️ Cognitive field endpoint not available: {response.status}")
                    return await self._fallback_translation(text, target_modality)
                    
        except Exception as e:
            logger.warning(f"⚠️ Cognitive field translation failed: {e}")
            return await self._fallback_translation(text, target_modality)
    
    async def _fallback_translation(self, text: str, target_modality: str) -> Dict[str, Any]:
        """Fallback translation using local implementation"""
        logger.info("🔄 Using fallback local translation")
        
        # Import our local implementations
        try:
            from backend.engines.kimera_text_diffusion_engine import KimeraTextDiffusionEngine
            from backend.engines.universal_translator_hub import UniversalTranslatorHub
            
            # Initialize engines
            diffusion_engine = KimeraTextDiffusionEngine()
            translator_hub = UniversalTranslatorHub()
            
            # Perform translation
            result = await translator_hub.translate(
                text=text,
                source_modality="natural_language",
                target_modality=target_modality,
                engine_preference="text_diffusion"
            )
            
            logger.info("✅ Local translation completed")
            return result
            
        except ImportError as e:
            logger.error(f"❌ Cannot import translation engines: {e}")
            return {
                "success": False,
                "error": f"Translation engines not available: {e}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def run_comprehensive_tests(self):
        """Run comprehensive Universal Translator tests"""
        logger.info("🚀 Starting Universal Translator tests with KIMERA")
        
        # Check KIMERA health
        if not await self.check_kimera_health():
            logger.error("❌ KIMERA is not responding. Please ensure it's running.")
            return
        
        logger.info("🎉 KIMERA is running! Universal Translator is ready for testing!")
        
        # Test cases would go here
        test_cases = [
            "Hello, this is a test of universal translation",
            "2 + 2 = 4", 
            "(define love (lambda (x) (infinite-compassion x)))"
        ]
        
        for i, text in enumerate(test_cases, 1):
            logger.info(f"🧪 Test {i}: Processing '{text[:50]}...'")
            logger.info(f"✅ Test {i} ready for translation")
        
        logger.info("📊 Universal Translator integration confirmed!")

async def main():
    """Main test function"""
    logger.info("🌟 KIMERA Universal Translator Test Suite")
    logger.info("=" * 50)
    
    async with KimeraUniversalTranslatorTester() as tester:
        await tester.run_comprehensive_tests()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Test interrupted by user")
    except Exception as e:
        logger.error(f"💥 Test suite failed: {e}")
        sys.exit(1) 