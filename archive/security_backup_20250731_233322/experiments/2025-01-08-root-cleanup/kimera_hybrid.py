#!/usr/bin/env python3
"""
Hybrid Kimera Startup Script
===========================
Starts Kimera with both high performance and debugging capabilities.

Environment Variables:
- KIMERA_DEBUG: Set to 'true' to enable debug mode (default: false)
- KIMERA_PERFORMANCE: Set to 'false' to disable performance optimizations (default: true)

Examples:
    # High performance mode (default)
    python kimera_hybrid.py
    
    # Debug mode with performance
    KIMERA_DEBUG=true python kimera_hybrid.py
    
    # Full debug mode (slower but comprehensive logging)
    KIMERA_DEBUG=true KIMERA_PERFORMANCE=false python kimera_hybrid.py
"""

import os
import sys
import uvicorn
from pathlib import Path
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import hybrid app
from src.api.main_hybrid import app


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Kimera Hybrid API - High performance with debugging capabilities"
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (can also use KIMERA_DEBUG=true)'
    )
    
    parser.add_argument(
        '--no-performance',
        action='store_true',
        help='Disable performance optimizations (can also use KIMERA_PERFORMANCE=false)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8001,
        help='Port to run the server on (default: 8001)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind the server to (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of worker processes (default: 1)'
    )
    
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development'
    )
    
    return parser.parse_args()


def configure_environment(args):
    """Configure environment based on arguments"""
    if args.debug:
        os.environ['KIMERA_DEBUG'] = 'true'
    
    if args.no_performance:
        os.environ['KIMERA_PERFORMANCE'] = 'false'
    
    # Print configuration
    debug_mode = os.getenv('KIMERA_DEBUG', 'false').lower() == 'true'
    performance_mode = os.getenv('KIMERA_PERFORMANCE', 'true').lower() == 'true'
    
    print("=" * 60)
    print("Kimera Hybrid API Configuration")
    print("=" * 60)
    print(f"Debug Mode:        {'ENABLED' if debug_mode else 'DISABLED'}")
    print(f"Performance Mode:  {'ENABLED' if performance_mode else 'DISABLED'}")
    print(f"Host:              {args.host}")
    print(f"Port:              {args.port}")
    print(f"Workers:           {args.workers}")
    print(f"Auto-reload:       {'ENABLED' if args.reload else 'DISABLED'}")
    print("=" * 60)
    
    if debug_mode and performance_mode:
        print("Mode: HIGH PERFORMANCE WITH DEBUG CAPABILITIES")
        print("- Sub-millisecond response times")
        print("- Request tracing enabled")
        print("- Debug API endpoints available")
        print("- Logs stored in ring buffer")
    elif debug_mode and not performance_mode:
        print("Mode: FULL DEBUG")
        print("- Comprehensive logging to console and file")
        print("- All debug features enabled")
        print("- Performance may be impacted")
    elif not debug_mode and performance_mode:
        print("Mode: PURE PERFORMANCE")
        print("- Maximum speed")
        print("- Minimal logging")
        print("- Debug endpoints still available")
    else:
        print("Mode: BASIC")
        print("- Standard logging")
        print("- No performance optimizations")
    
    print("=" * 60)
    print()
    
    # Show debug endpoints
    if debug_mode or True:  # Always show debug endpoints info
        print("Debug Endpoints:")
        print("- GET  /debug/info     - Comprehensive debug information")
        print("- POST /debug/mode     - Toggle debug mode at runtime")
        print("- POST /debug/profiling - Toggle performance profiling")
        print("- GET  /debug/logs     - View recent logs from ring buffer")
        print("- GET  /debug/traces   - View request traces")
        print("- POST /debug/log-level - Change log level dynamically")
        print("- GET  /performance/stats - Performance statistics")
        print()


if __name__ == "__main__":
    args = parse_arguments()
    configure_environment(args)
    
    # Determine log level based on mode
    debug_mode = os.getenv('KIMERA_DEBUG', 'false').lower() == 'true'
    performance_mode = os.getenv('KIMERA_PERFORMANCE', 'true').lower() == 'true'
    
    if debug_mode and not performance_mode:
        log_level = "debug"
    elif debug_mode:
        log_level = "info"
    else:
        log_level = "warning"
    
    # Configure uvicorn
    uvicorn_config = {
        "app": app,
        "host": args.host,
        "port": args.port,
        "workers": args.workers,
        "log_level": log_level,
        "reload": args.reload,
    }
    
    # Performance optimizations
    if performance_mode:
        uvicorn_config.update({
            "access_log": False,  # Disable access logs
            "limit_concurrency": 1000,  # High concurrency
            "limit_max_requests": 10000,  # Request limit before restart
            "timeout_keep_alive": 5,  # Keep-alive timeout
        })
    else:
        uvicorn_config["access_log"] = True
    
    # Run the server
    uvicorn.run(**uvicorn_config)