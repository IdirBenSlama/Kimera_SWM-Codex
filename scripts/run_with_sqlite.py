#!/usr/bin/env python3
"""
Run Kimera with SQLite database
"""
import os
import sys

# Set environment variable to use SQLite
os.environ["DATABASE_URL"] = "sqlite:///./kimera_swm.db"

# Import and run the main kimera script
if __name__ == "__main__":
    # Add the project root to the Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Import uvicorn and run the app
    import uvicorn
    uvicorn.run(
        "backend.api.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )