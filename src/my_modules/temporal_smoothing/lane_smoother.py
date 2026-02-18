import numpy as np
from collections import deque
from abc import ABC, abstractmethod

class BaseLaneSmoother(ABC):
    """Base class for temporal lane smoothing."""
    
    @abstractmethod
    def smooth(self, lane_data):
        """Smooth lane data from current frame."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset smoother state."""
        pass

class MovingAverageLaneSmoother(BaseLaneSmoother):
    """
    Moving average filter for lane coefficients.
    Works with your existing lane detection output format.
    """
    
    def __init__(self, window_size=5, lane_order=2):
        """
        Args:
            window_size: Number of frames to average
            lane_order: Order of polynomial (default 2 for quadratic)
        """
        self.window_size = window_size
        self.lane_order = lane_order
        
        # Based on your 1_lane.csv format:
        # - Left and right polynomial coefficients
        # - Left and right X/Y coordinates
        self.left_coeffs_history = deque(maxlen=window_size)
        self.right_coeffs_history = deque(maxlen=window_size)
        self.left_points_history = deque(maxlen=window_size)
        self.right_points_history = deque(maxlen=window_size)
    
    def smooth(self, lane_data):
        """
        Smooth lane data.
        
        Args:
            lane_data: Dictionary containing:
                - 'left_coeffs': Left lane polynomial coefficients
                - 'right_coeffs': Right lane polynomial coefficients
                - 'left_points': Left lane points (x, y)
                - 'right_points': Right lane points (x, y)
        
        Returns:
            Smoothed lane data in same format
        """
        smoothed = {}
        
        # Smooth coefficients
        if 'left_coeffs' in lane_data:
            self.left_coeffs_history.append(lane_data['left_coeffs'])
            if len(self.left_coeffs_history) >= 2:
                smoothed['left_coeffs'] = np.mean(self.left_coeffs_history, axis=0)
            else:
                smoothed['left_coeffs'] = lane_data['left_coeffs']
        
        if 'right_coeffs' in lane_data:
            self.right_coeffs_history.append(lane_data['right_coeffs'])
            if len(self.right_coeffs_history) >= 2:
                smoothed['right_coeffs'] = np.mean(self.right_coeffs_history, axis=0)
            else:
                smoothed['right_coeffs'] = lane_data['right_coeffs']
        
        # Smooth points (if needed)
        if 'left_points' in lane_data:
            self.left_points_history.append(lane_data['left_points'])
            if len(self.left_points_history) >= 2:
                # Average point-wise if same length, otherwise use most recent
                if all(len(p) == len(lane_data['left_points']) for p in self.left_points_history):
                    smoothed['left_points'] = np.mean(self.left_points_history, axis=0)
                else:
                    smoothed['left_points'] = lane_data['left_points']
            else:
                smoothed['left_points'] = lane_data['left_points']
        
        if 'right_points' in lane_data:
            self.right_points_history.append(lane_data['right_points'])
            if len(self.right_points_history) >= 2:
                if all(len(p) == len(lane_data['right_points']) for p in self.right_points_history):
                    smoothed['right_points'] = np.mean(self.right_points_history, axis=0)
                else:
                    smoothed['right_points'] = lane_data['right_points']
            else:
                smoothed['right_points'] = lane_data['right_points']
        
        return smoothed
    
    def reset(self):
        """Reset all histories."""
        self.left_coeffs_history.clear()
        self.right_coeffs_history.clear()
        self.left_points_history.clear()
        self.right_points_history.clear()

class ExponentialMovingAverageLaneSmoother(BaseLaneSmoother):
    """
    Exponential moving average for lane coefficients.
    Gives more weight to recent frames.
    """
    
    def __init__(self, alpha=0.3):
        """
        Args:
            alpha: Smoothing factor (0 < alpha <= 1)
                  Lower = smoother, higher = more responsive
        """
        self.alpha = alpha
        self.smoothed_left_coeffs = None
        self.smoothed_right_coeffs = None
        self.smoothed_left_points = None
        self.smoothed_right_points = None
    
    def smooth(self, lane_data):
        """Apply EMA smoothing to lane data."""
        smoothed = {}
        
        # Smooth coefficients
        if 'left_coeffs' in lane_data:
            if self.smoothed_left_coeffs is None:
                self.smoothed_left_coeffs = lane_data['left_coeffs'].copy()
            else:
                self.smoothed_left_coeffs = (self.alpha * lane_data['left_coeffs'] + 
                                           (1 - self.alpha) * self.smoothed_left_coeffs)
            smoothed['left_coeffs'] = self.smoothed_left_coeffs
        
        if 'right_coeffs' in lane_data:
            if self.smoothed_right_coeffs is None:
                self.smoothed_right_coeffs = lane_data['right_coeffs'].copy()
            else:
                self.smoothed_right_coeffs = (self.alpha * lane_data['right_coeffs'] + 
                                            (1 - self.alpha) * self.smoothed_right_coeffs)
            smoothed['right_coeffs'] = self.smoothed_right_coeffs
        
        # Smooth points (element-wise if same shape)
        if 'left_points' in lane_data:
            if self.smoothed_left_points is None:
                self.smoothed_left_points = lane_data['left_points'].copy()
            else:
                if self.smoothed_left_points.shape == lane_data['left_points'].shape:
                    self.smoothed_left_points = (self.alpha * lane_data['left_points'] + 
                                               (1 - self.alpha) * self.smoothed_left_points)
                else:
                    self.smoothed_left_points = lane_data['left_points'].copy()
            smoothed['left_points'] = self.smoothed_left_points
        
        if 'right_points' in lane_data:
            if self.smoothed_right_points is None:
                self.smoothed_right_points = lane_data['right_points'].copy()
            else:
                if self.smoothed_right_points.shape == lane_data['right_points'].shape:
                    self.smoothed_right_points = (self.alpha * lane_data['right_points'] + 
                                                (1 - self.alpha) * self.smoothed_right_points)
                else:
                    self.smoothed_right_points = lane_data['right_points'].copy()
            smoothed['right_points'] = self.smoothed_right_points
        
        return smoothed
    
    def reset(self):
        """Reset smoothed values."""
        self.smoothed_left_coeffs = None
        self.smoothed_right_coeffs = None
        self.smoothed_left_points = None
        self.smoothed_right_points = None
