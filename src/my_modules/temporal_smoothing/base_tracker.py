from abc import ABC, abstractmethod
import numpy as np

class Base3DTracker(ABC):
    """Abstract base class for all 3D object trackers."""
    
    @abstractmethod
    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        Update tracker with new detections.
        
        Args:
            detections: Numpy array of shape (N, 7+) containing 
                        [x, y, z, l, w, h, theta, score, ...] in KITTI format.
        
        Returns:
            Tracked objects with IDs and smoothed states.
            Format: [x, y, z, l, w, h, theta, track_id, score]
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset the tracker state for a new video sequence."""
        pass
