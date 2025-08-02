#!/usr/bin/env python3
"""
Simple Server Startup for Kimera Revolutionary Thermodynamic System
==================================================================
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def start_server():
    print("🌟 KIMERA REVOLUTIONARY THERMODYNAMIC SYSTEM")
    print("=" * 50)
    print("🚀 Starting the world's first physics-compliant AI server...")
    print("")
    
    try:
        # Import the FastAPI app
        from src.main import app
        print("✅ Revolutionary thermodynamic app loaded successfully!")
        
        # Start with uvicorn
        import uvicorn
        print("🌐 Starting API server...")
        print("📡 Server will be available at:")
        print("   • API Documentation: http://localhost:8001/docs")
        print("   • Revolutionary Endpoints: http://localhost:8001/kimera/unified-thermodynamic/")
        print("   • System Status: http://localhost:8001/kimera/unified-thermodynamic/status")
        print("")
        print("🔥 Starting revolutionary thermodynamic server...")
        
        # Configure and run server
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=8001,
            log_level="info",
            reload=False
        )
        
        server = uvicorn.Server(config)
        server.run()
        
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = start_server()
    sys.exit(exit_code) 