"""
Rigorous Engine Test Script
===========================

Purpose:
To conduct a scientifically isolated test of the UniversalTranslatorHub and its
child engines, bypassing the FastAPI/Uvicorn server infrastructure to
get a clean, undeniable stack trace of any initialization or runtime errors.

This is the ultimate zetetic tool for this problem.

Methodology:
1. Manually set up the required Python path.
2. Initialize logging to capture all output.
3. Directly instantiate GPUFoundation.
4. Directly instantiate UniversalTranslatorHub, passing the GPU foundation.
5. Create a standard UniversalTranslationRequest.
6. Execute the translation within a comprehensive try/except block.
7. Print all results, state, and errors in a clear, unambiguous report.
"""
import asyncio
import logging
import sys
import os
from typing import Dict, Any

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.utils.gpu_foundation import GPUFoundation
from backend.engines.universal_translator_hub import (
    create_universal_translator_hub,
    UniversalTranslationRequest,
    TranslationModality
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("RIGOROUS_TEST")


async def run_rigorous_test():
    """The main testing function."""
    logger.info("=====================================================")
    logger.info("🔬 STARTING RIGOROUS SCIENTIFIC ENGINE TEST 🔬")
    logger.info("=====================================================")

    gpu_foundation = None
    hub = None
    
    try:
        # Step 1: Initialize GPU Foundation
        logger.info("[1] Initializing GPUFoundation...")
        gpu_foundation = GPUFoundation()
        logger.info("✅ GPUFoundation Initialized.")
        logger.info(f"   Device: {gpu_foundation.get_device()}")

        # Step 2: Initialize Universal Translator Hub
        logger.info("\n[2] Initializing UniversalTranslatorHub...")
        hub_config: Dict[str, Any] = {
            "text_diffusion": {},
            "cognitive_dimension": 1024
        }
        hub = create_universal_translator_hub(hub_config, gpu_foundation=gpu_foundation)
        logger.info("✅ UniversalTranslatorHub Initialized.")
        logger.info(f"   Available Engines in Hub: {list(hub.engines.keys())}")

        # Step 3: Create Translation Request
        logger.info("\n[3] Creating UniversalTranslationRequest...")
        request = UniversalTranslationRequest(
            source_content="Who are you?",
            source_modality=TranslationModality.NATURAL_LANGUAGE,
            target_modality=TranslationModality.NATURAL_LANGUAGE,
            metadata={"context": {"source": "web_chat"}}
        )
        logger.info("✅ Request Created.")

        # Step 4: Perform Translation
        logger.info("\n[4] Performing Translation...")
        result = await hub.translate(request)
        logger.info("✅ Translation Task Completed.")

        # Step 5: Report Results
        logger.info("\n[5] FINAL TEST RESULTS")
        logger.info("------------------------")
        logger.info(f"  Engine Used: {result.engine_used.name}")
        logger.info(f"  Confidence: {result.confidence}")
        logger.info(f"  Response: {result.translated_content}")
        logger.info("  Metadata: %s", result.metadata)
        logger.info("\n🎉 RIGOROUS TEST SUCCEEDED 🎉")


    except BaseException as e:
        logger.error("\n💥 A CATASTROPHIC FAILURE OCCURRED 💥", exc_info=False)
        logger.error(f"Error Type: {type(e).__name__}")
        logger.error(f"Error Details: {e}")
        # Manually print traceback for undeniable clarity
        import traceback
        logger.error("\n--- FULL STACK TRACE ---")
        traceback.print_exc()
        logger.error("--- END STACK TRACE ---\n")
        logger.info("HUB State at failure:")
        if hub:
             logger.info(f"  Available engines: {list(hub.engines.keys())}")
        else:
            logger.info("  Hub was not initialized.")
            
        logger.info("GPU Foundation State at failure:")
        if gpu_foundation:
            logger.info("  GPU Foundation was initialized.")
        else:
            logger.info("  GPU Foundation was not initialized.")
            
        logger.error("\n🔥 RIGOROUS TEST FAILED 🔥")

    finally:
        logger.info("\n=====================================================")
        logger.info("🔬 RIGOROUS SCIENTIFIC ENGINE TEST COMPLETE 🔬")
        logger.info("=====================================================")


if __name__ == "__main__":
    asyncio.run(run_rigorous_test()) 