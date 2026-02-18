import numpy as np
import os
import sys
from ..custom_pipeline import MyEnhancedPipeline
from .lane_smoother import MovingAverageLaneSmoother
from .kalman_3d import SimpleKalman3DTracker
from .ab3dmot_wrapper import AB3DMOTWrapper

class SmoothedVideoPipeline:
    """
    High-level wrapper that adds temporal smoothing to the existing pipeline.
    Designed to work seamlessly with MyEnhancedPipeline.
    Uses Kalman filter for 3D tracking - no AB3DMOT dependencies.
    """
    
    def __init__(self, 
                 weights_path=None,
                 use_3d_tracker=True,
                 use_lane_smoother=True,
                 tracker_max_age=2,
                 lane_smoother_window=5,
                 lane_smoother_alpha=0.3):
        """
        Initialize the smoothed pipeline.
        
        Args:
            weights_path: Path to model weights (passed to base pipeline)
            use_3d_tracker: Enable/disable 3D object tracking
            use_lane_smoother: Enable/disable lane smoothing
            tracker_max_age: Frames to keep track without detection
            lane_smoother_window: Frames for moving average
            lane_smoother_alpha: Smoothing factor for EMA
        """
        # Initialize the base pipeline (your existing class)
        self.base_pipeline = MyEnhancedPipeline(weights_path)
        
        # Initialize temporal modules
        self.use_3d_tracker = use_3d_tracker
        self.use_lane_smoother = use_lane_smoother
        
        # Set up 3D tracker (always uses Kalman filter)
        if use_3d_tracker:
            self.tracker = AB3DMOTWrapper(max_age=tracker_max_age)
        else:
            self.tracker = None
        
        # Set up lane smoother
        if use_lane_smoother:
            if lane_smoother_type == 'moving_average':
                self.lane_smoother = MovingAverageLaneSmoother(
                    window_size=lane_smoother_window
                )
            else:  # ema
                from .lane_smoother import ExponentialMovingAverageLaneSmoother
                self.lane_smoother = ExponentialMovingAverageLaneSmoother(
                    alpha=lane_smoother_alpha
                )
        else:
            self.lane_smoother = None
        
        # Frame counter
        self.frame_count = 0
        
        # Store last raw results for debugging
        self.last_raw_result = None
    
    def process_frame(self, image_path, image=None):
        """
        Process a single frame with temporal smoothing.
        
        Args:
            image_path: Path to the image file (for consistency with base pipeline)
            image: Optional pre-loaded image array
        
        Returns:
            dict: Contains all results from base pipeline plus:
                - 'frame_id': Current frame number
                - 'tracked_objects_3d': Tracked objects with IDs (if tracking enabled)
                - 'smoothed_lanes': Smoothed lane data (if smoothing enabled)
        """
        # Step 1: Get base pipeline results
        # Note: Your MyEnhancedPipeline.run() returns {'lanes': lanes, 'image_path': image_path}
        base_result = self.base_pipeline.run(image_path)
        self.last_raw_result = base_result
        
        result = base_result.copy()
        result['frame_id'] = self.frame_count
        
        # Step 2: Apply lane smoothing if enabled
        if self.use_lane_smoother and self.lane_smoother is not None:
            # Extract lane data from your detector's output format
            # This assumes your lane detector returns lane points
            lane_data = self._extract_lane_data(base_result)
            if lane_data:
                smoothed_lanes = self.lane_smoother.smooth(lane_data)
                result['smoothed_lanes'] = smoothed_lanes
                
                # Optionally replace original lanes with smoothed ones
                if 'left_points' in smoothed_lanes and 'right_points' in smoothed_lanes:
                    # You'll need to adapt this based on how your lanes are stored
                    result['lanes'] = self._format_smoothed_lanes(smoothed_lanes)
        
        # Step 3: Apply 3D tracking if enabled
        if self.use_3d_tracker and self.tracker is not None:
            # Extract 3D detections - you'll need to add this to your pipeline
            # For now, this is a placeholder that returns empty detections
            # You'll need to modify your MyEnhancedPipeline to return 3D detections
            detections_3d = self._extract_3d_detections(base_result)
            
            if len(detections_3d) > 0:
                tracked_objects = self.tracker.update(detections_3d)
                result['tracked_objects_3d'] = tracked_objects
            else:
                # Still update tracker to maintain tracks
                tracked_objects = self.tracker.update(np.empty((0, 7)))
                result['tracked_objects_3d'] = tracked_objects
        
        self.frame_count += 1
        return result
    
    def _extract_lane_data(self, base_result):
        """
        Extract lane data from base pipeline result.
        Adapt this based on your actual lane detection output format.
        """
        lane_data = {}
        
        # Check if we have lanes from the detector
        if 'lanes' in base_result:
            lanes = base_result['lanes']
            
            # This is a placeholder - you'll need to adapt this to your actual format
            # Based on your code, lanes might contain points and coefficients
            try:
                # Try to extract left and right lane points
                # This assumes lanes is a tuple/list of (left_points, right_points)
                if isinstance(lanes, (tuple, list)) and len(lanes) >= 2:
                    lane_data['left_points'] = np.array(lanes[0])
                    lane_data['right_points'] = np.array(lanes[1])
                
                # Try to extract coefficients if available
                # You might have coefficients stored in the detector's output
                if hasattr(self.base_pipeline.detector, 'last_coeffs'):
                    coeffs = self.base_pipeline.detector.last_coeffs
                    if 'left' in coeffs:
                        lane_data['left_coeffs'] = coeffs['left']
                    if 'right' in coeffs:
                        lane_data['right_coeffs'] = coeffs['right']
            except:
                pass
        
        return lane_data
    
    def _extract_3d_detections(self, base_result):
        """
        Extract 3D detections from base pipeline result.
        This is a placeholder - you'll need to implement actual 3D detection extraction.
        """
        # For now, return empty array
        # You'll need to modify MyEnhancedPipeline to include 3D detection results
        return np.empty((0, 7))
    
    def _format_smoothed_lanes(self, smoothed_lanes):
        """
        Format smoothed lanes back to the expected output format.
        Adapt this based on your actual lane format.
        """
        # This is a placeholder - return the smoothed points
        if 'left_points' in smoothed_lanes and 'right_points' in smoothed_lanes:
            return (smoothed_lanes['left_points'], smoothed_lanes['right_points'])
        return None
    
    def reset(self):
        """Reset the pipeline state for a new video."""
        self.frame_count = 0
        if self.tracker:
            self.tracker.reset()
        if self.lane_smoother:
            self.lane_smoother.reset()
        self.last_raw_result = None
    
    def process_video(self, image_paths, callback=None):
        """
        Process a sequence of frames (video).
        
        Args:
            image_paths: List of image paths in sequence order
            callback: Optional callback function called after each frame
            
        Returns:
            List of results for all frames
        """
        results = []
        self.reset()
        
        for i, img_path in enumerate(image_paths):
            print(f"Processing frame {i+1}/{len(image_paths)}: {img_path}")
            result = self.process_frame(img_path)
            results.append(result)
            
            if callback:
                callback(i, result)
        
        return results
