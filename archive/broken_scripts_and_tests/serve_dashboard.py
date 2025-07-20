#!/usr/bin/env python3
"""
Simple HTTP server for KIMERA SWM Dashboard
"""
import http.server
import socketserver
import os
import webbrowser
import threading
import time

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


def serve_dashboard(port=8080):
    """Serve the dashboard on the specified port"""
    os.chdir('dashboard')
    
    class Handler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            super().end_headers()
    
    with socketserver.TCPServer(("", port), Handler) as httpd:
        logger.info(f"🌐 KIMERA SWM Dashboard serving at http://localhost:{port}")
        logger.info(f"📊 Make sure the API server is running on port 8002")
        logger.info(f"🚀 Opening dashboard in browser...")
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(1)
            webbrowser.open(f'http://localhost:{port}')
        
        threading.Thread(target=open_browser, daemon=True).start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("\n🛑 Dashboard server stopped")

if __name__ == "__main__":
    serve_dashboard()
