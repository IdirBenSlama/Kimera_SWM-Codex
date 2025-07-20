#!/usr/bin/env python3
try:
    from backend.engines.kimera_full_integration_bridge import KimeraFullIntegrationBridge
    print("✅ Import successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc() 