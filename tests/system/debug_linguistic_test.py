#!/usr/bin/env python3
"""
Debug linguistic intelligence cross-lingual issue
"""

import asyncio
import traceback


async def debug_linguistic_issue():
    """Debug the cross-lingual processing issue"""
    print("🔍 DEBUGGING LINGUISTIC INTELLIGENCE CROSS-LINGUAL ISSUE")
    print("=" * 60)

    try:
        from src.core.enhanced_capabilities.linguistic_intelligence_core import (
            LanguageProcessingMode,
            LinguisticIntelligenceCore,
        )

        linguistic_core = LinguisticIntelligenceCore(
            default_processing_mode=LanguageProcessingMode.SEMANTIC_ANALYSIS,
            supported_languages=["en", "es", "fr"],
            device="cpu",
        )
        print("✅ Linguistic Intelligence Core created")

        # Test cross-lingual analysis
        multilingual_texts = [
            ("Hello world", "en"),
            ("Hola mundo", "es"),
            ("Bonjour monde", "fr"),
        ]

        cross_lingual_results = []
        for text, expected_lang in multilingual_texts:
            print(f"   Testing: '{text}' (expected: {expected_lang})")
            result = await linguistic_core.analyze_linguistic_intelligence(
                text,
                processing_mode=LanguageProcessingMode.CROSS_LINGUAL,
                target_language="en",
                context={"cross_lingual_test": True},
            )
            print(f"   Detected: {result.detected_language}, Success: {result.success}")
            cross_lingual_results.append((result, expected_lang))

        # Check if languages were correctly detected
        correct_detections = sum(
            1
            for result, expected in cross_lingual_results
            if result.success and result.detected_language == expected
        )

        print(f"\n   Correct detections: {correct_detections}/3")

        if correct_detections >= 2:
            print("✅ Cross-lingual processing should pass")
        else:
            print("❌ Cross-lingual processing failed")

    except Exception as e:
        print(f"✅ Found the error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(debug_linguistic_issue())
