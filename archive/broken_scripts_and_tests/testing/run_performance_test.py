#!/usr/bin/env python3
"""
Performance Test Runner for KIMERA
Ensures server is running and executes comprehensive performance tests
"""

import subprocess
import time
import requests
import sys
import os
import signal
import threading
from pathlib import Path

class KimeraTestRunner:
    def __init__(self):
        self.server_process = None
        self.server_running = False
        
    def check_server_status(self):
        """Check if KIMERA server is running"""
        try:
            response = requests.get("http://127.0.0.1:8001/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def start_server(self):
        """Start KIMERA server if not running"""
        if self.check_server_status():
            print("✅ KIMERA server is already running")
            return True
        
        print("🚀 Starting KIMERA server...")
        
        # Start minimal server
        try:
            self.server_process = subprocess.Popen(
                [sys.executable, "minimal_server.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd()
            )
            
            # Wait for server to start
            for i in range(30):  # Wait up to 30 seconds
                time.sleep(1)
                if self.check_server_status():
                    print("✅ KIMERA server started successfully")
                    self.server_running = True
                    return True
                print(f"   Waiting for server... ({i+1}/30)")
            
            print("❌ Server failed to start within 30 seconds")
            return False
            
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False
    
    def stop_server(self):
        """Stop KIMERA server"""
        if self.server_process:
            print("🛑 Stopping KIMERA server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            self.server_running = False
            print("✅ Server stopped")
    
    def run_performance_test(self):
        """Run the comprehensive performance test"""
        print("\n" + "="*60)
        print("🎯 RUNNING COMPREHENSIVE PERFORMANCE TEST")
        print("="*60)
        
        try:
            # Run the performance test
            result = subprocess.run(
                [sys.executable, "comprehensive_performance_test.py"],
                cwd=os.getcwd(),
                capture_output=False,  # Show output in real-time
                text=True
            )
            
            if result.returncode == 0:
                print("\n✅ Performance test completed successfully")
                return True
            else:
                print(f"\n❌ Performance test failed with code: {result.returncode}")
                return False
                
        except Exception as e:
            print(f"\n❌ Error running performance test: {e}")
            return False
    
    def cleanup(self):
        """Cleanup resources"""
        if self.server_running:
            self.stop_server()

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Received interrupt signal, cleaning up...")
    runner.cleanup()
    sys.exit(0)

def main():
    global runner
    runner = KimeraTestRunner()
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🚀 KIMERA PERFORMANCE TEST RUNNER")
    print("="*50)
    
    try:
        # Step 1: Ensure server is running
        if not runner.start_server():
            print("❌ Cannot proceed without running server")
            return 1
        
        # Step 2: Run performance tests
        success = runner.run_performance_test()
        
        # Step 3: Cleanup
        runner.cleanup()
        
        if success:
            print("\n🎉 All tests completed successfully!")
            return 0
        else:
            print("\n❌ Some tests failed")
            return 1
            
    except KeyboardInterrupt:
        print("\n🛑 Test run interrupted by user")
        runner.cleanup()
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        runner.cleanup()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 