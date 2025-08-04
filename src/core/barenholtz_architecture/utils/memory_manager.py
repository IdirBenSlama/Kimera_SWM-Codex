"""
Working Memory Manager
======================

Manages working memory for dual-system processing.
"""

from typing import Dict, List, Any, Optional
from collections import deque
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class WorkingMemoryManager:
    """Manages working memory with capacity constraints"""

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.access_count = {}

    def add(self, item: Dict[str, Any]):
        """Add item to working memory"""
        item_with_meta = {
            'content': item,
            'timestamp': datetime.now(),
            'id': id(item)
        }
        self.memory.append(item_with_meta)

    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get n most recent items"""
        items = list(self.memory)[-n:]

        # Track access
        for item in items:
            item_id = item['id']
            self.access_count[item_id] = self.access_count.get(item_id, 0) + 1

        return [item['content'] for item in items]

    def clear(self):
        """Clear working memory"""
        self.memory.clear()
        self.access_count.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            'size': len(self.memory),
            'capacity': self.capacity,
            'utilization': len(self.memory) / self.capacity,
            'total_accesses': sum(self.access_count.values())
        }
