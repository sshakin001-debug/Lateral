"""
AB3DMOT Wrapper for 3D Multi-Object Tracking.

This module provides a wrapper for 3D tracking using Kalman filters.
AB3DMOT was removed - using SimpleKalman3DTracker instead for simpler
deployment and no dependency conflicts.
"""

import numpy as np
from .base_tracker import Base3DTracker
from .kalman_3d import SimpleKalman3DTracker


class AB3DMOTWrapper(Base3DTracker):
    """
    Wrapper for 3D multi-object tracking using Kalman filters.
    
    Originally designed for AB3DMOT, but now uses SimpleKalman3DTracker
    for simpler deployment without AB3DMOT dependencies.
    """
    
    def __init__(self, max_age=2, min_hits=3, dt=1.0):
        """
        Args:
            max_age: Maximum frames to keep a track without detection
            min_hits: Minimum detections needed to confirm a track
            dt: Time step between frames (seconds)
        """
        super().__init__()
        
        # Always use Kalman filter - no AB3DMOT dependency
        self.tracker = SimpleKalman3DTracker(
            max_age=max_age,
            min_hits=min_hits,
            dt=dt
        )
        self.available = True
        self.max_age = max_age
        self.min_hits = min_hits
    
    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        Update tracker with new detections.
        
        Args:
            detections: Numpy array of shape (N, 7+) containing 
                        [x, y, z, l, w, h, theta, score, ...]
        
        Returns:
            Tracked objects with IDs and smoothed states.
            Format: [x, y, z, l, w, h, theta, track_id, score]
        """
        return self.tracker.update(detections)
    
    def reset(self):
        """Reset the tracker state for a new video sequence."""
        self.tracker.reset()
