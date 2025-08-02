#!/usr/bin/env python3
"""
Quick & Reliable Kimera Revolutionary Thermodynamic Server
=========================================================
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("🌟 KIMERA REVOLUTIONARY THERMODYNAMIC SYSTEM")
print("=" * 60)
print("🚀 Quick server startup for world's first physics-compliant AI")

try:
    print("📦 Loading revolutionary thermodynamic system...")
    from src.main import app
    print("✅ Revolutionary system loaded successfully!")
    
    print("🌐 Starting server on port 8003...")
    import uvicorn
    
    print("📡 Server URLs:")
    print("   • Documentation: http://localhost:8003/docs")
    print("   • Thermodynamic API: http://localhost:8003/kimera/unified-thermodynamic/status")
    print("   • Health Check: http://localhost:8003/health")
    print("")
    print("🔥 Starting now...")
    
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8003, 
        log_level="info",
        access_log=True
    )
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 