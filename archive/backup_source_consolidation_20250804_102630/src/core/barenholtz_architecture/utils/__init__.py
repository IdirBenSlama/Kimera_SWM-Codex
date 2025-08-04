"""
Utility Components for Barenholtz Architecture
==============================================
"""

from .memory_manager import WorkingMemoryManager
from .conflict_resolver import ConflictResolver

__all__ = [
    'WorkingMemoryManager',
    'ConflictResolver'
]
