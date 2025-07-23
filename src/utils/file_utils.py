"""
File utility functions for the Kimera project.
"""

import os
from pathlib import Path
from typing import Optional


def find_project_root(start_path: Optional[str] = None) -> Path:
    """
    Find the project root directory by looking for specific marker files.
    
    Args:
        start_path: Starting directory to search from. If None, uses current directory.
        
    Returns:
        Path to the project root directory.
    """
    if start_path is None:
        start_path = os.getcwd()
    
    current = Path(start_path).resolve()
    
    # Look for common project root indicators
    markers = ['.git', 'pyproject.toml', 'setup.py', 'requirements.txt', 'kimera.py']
    
    while current != current.parent:
        for marker in markers:
            if (current / marker).exists():
                return current
        current = current.parent
    
    # If no marker found, return the original start path
    return Path(start_path).resolve()


def ensure_dir(directory: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory.
        
    Returns:
        The directory path.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory