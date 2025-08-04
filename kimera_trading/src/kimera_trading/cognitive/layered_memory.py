from collections import deque

class LayeredMemory:
    """A layered memory system that mimics human cognitive processes."""

    def __init__(self, short_term_size=100, medium_term_size=1000, long_term_size=10000):
        self.short_term = deque(maxlen=short_term_size)
        self.medium_term = deque(maxlen=medium_term_size)
        self.long_term = deque(maxlen=long_term_size)

    def add(self, event):
        """Adds an event to the memory system."""
        self.short_term.append(event)
        self.medium_term.append(event)
        self.long_term.append(event)

    def get_short_term_memory(self):
        return list(self.short_term)

    def get_medium_term_memory(self):
        return list(self.medium_term)

    def get_long_term_memory(self):
        return list(self.long_term)
