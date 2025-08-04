import sys

import torch

print("🧪 QUICK TEST")
print("PyTorch version:", torch.__version__)

try:
    from src.core.enhanced_capabilities.understanding_core import UnderstandingCore

    print("✅ Understanding Core imported")

    from src.core.enhanced_capabilities.consciousness_core import ConsciousnessCore

    print("✅ Consciousness Core imported")

    from src.core.enhanced_capabilities.linguistic_intelligence_core import (
        LinguisticIntelligenceCore,
    )

    print("✅ Linguistic Core imported")

    # Test language detection
    ling_core = LinguisticIntelligenceCore()
    en_lang = ling_core._detect_language("hello world")
    es_lang = ling_core._detect_language("hola mundo")
    print(f"Language detection: 'hello world' -> {en_lang}, 'hola mundo' -> {es_lang}")

    print("🎉 ALL CRITICAL IMPORTS WORKING!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
