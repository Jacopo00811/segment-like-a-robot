"""Memory management for test-time adaptation."""

from collections import deque


class SceneMemory:
    """Class to manage scene memory for test-time adaptation"""
    
    def __init__(self, max_size=5):
        """
        Initialize a sliding window memory for scene indices
        
        Args:
            max_size: Maximum number of scene indices to keep in memory
        """
        self.max_size = max_size
        self.indices = deque(maxlen=max_size)
        
    def add_scene_index(self, index):
        """
        Add scene index to memory
        
        Args:
            index: The dataset index of the scene
        """
        if index not in self.indices:
            self.indices.append(index)
        
    def get_scene_indices(self):
        """
        Return all scene indices in memory
        
        Returns:
            List of scene indices
        """
        return list(self.indices)
    
    def is_empty(self):
        """Check if memory is empty"""
        return len(self.indices) == 0
    
    def size(self):
        """Return current memory size"""
        return len(self.indices)