import numpy as np
from .base_tracker import Base3DTracker

class SimpleKalman3DTracker(Base3DTracker):
    """
    A simple 3D Kalman filter tracker for objects.
    This serves as a lightweight alternative to AB3DMOT.
    """
    
    def __init__(self, max_age=2, min_hits=3, dt=1.0):
        """
        Args:
            max_age: Maximum frames to keep a track without detection
            min_hits: Minimum detections needed to confirm a track
            dt: Time step between frames
        """
        super().__init__()
        self.max_age = max_age
        self.min_hits = min_hits
        self.dt = dt
        self.next_id = 0
        self.tracks = {}  # {id: {'kf': KalmanFilter, 'age': int, 'hits': int, 'last_detection': array}}
        
        # Import filterpy only when needed (optional dependency)
        try:
            from filterpy.kalman import KalmanFilter
            self.KalmanFilter = KalmanFilter
        except ImportError:
            print("Warning: filterpy not installed. Install with: pip install filterpy")
            self.KalmanFilter = None
    
    def _init_kalman_filter(self, detection):
        """Initialize a Kalman filter for a new track."""
        if self.KalmanFilter is None:
            return None
            
        kf = self.KalmanFilter(dim_x=10, dim_z=7)  # State: [x,y,z,vx,vy,vz,l,w,h,theta]
        
        # State transition matrix (constant velocity model)
        kf.F = np.array([[1, 0, 0, self.dt, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, self.dt, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, self.dt, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        
        # Measurement matrix (we measure position, size, orientation)
        kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        
        # Initial state
        kf.x[:7] = detection[:7]  # x,y,z,l,w,h,theta
        kf.x[7:10] = 0  # initial velocity
        
        # Covariance matrices
        kf.P *= 10  # Initial uncertainty
        kf.R = np.eye(7) * 0.1  # Measurement noise
        kf.Q = np.eye(10) * 0.01  # Process noise
        
        return kf
    
    def _predict_all(self):
        """Predict all tracks forward in time."""
        for track_id, track in self.tracks.items():
            if track['kf'] is not None:
                track['kf'].predict()
            track['age'] += 1
    
    def _update_track(self, track_id, detection):
        """Update a track with a new detection."""
        track = self.tracks[track_id]
        if track['kf'] is not None:
            track['kf'].update(detection[:7])
        track['last_detection'] = detection
        track['age'] = 0
        track['hits'] += 1
    
    def _create_track(self, detection):
        """Create a new track from a detection."""
        kf = self._init_kalman_filter(detection)
        self.tracks[self.next_id] = {
            'kf': kf,
            'age': 0,
            'hits': 1,
            'last_detection': detection
        }
        self.next_id += 1
    
    def _get_state(self, track):
        """Get the current state of a track."""
        if track['kf'] is not None:
            return track['kf'].x[:7]
        else:
            return track['last_detection'][:7]
    
    def _assign_detections(self, detections):
        """Simple nearest neighbor assignment."""
        assignments = {}
        if not self.tracks:
            return assignments
            
        # Simple greedy assignment (for production, use Hungarian algorithm)
        used_tracks = set()
        used_dets = set()
        
        for i, det in enumerate(detections):
            best_track = None
            best_dist = float('inf')
            
            for track_id, track in self.tracks.items():
                if track_id in used_tracks:
                    continue
                    
                track_state = self._get_state(track)
                dist = np.linalg.norm(det[:3] - track_state[:3])
                
                if dist < 2.0 and dist < best_dist:  # 2 meter threshold
                    best_dist = dist
                    best_track = track_id
            
            if best_track is not None:
                assignments[best_track] = i
                used_tracks.add(best_track)
                used_dets.add(i)
        
        return assignments
    
    def _remove_old_tracks(self):
        """Remove tracks that haven't been updated recently."""
        to_remove = []
        for track_id, track in self.tracks.items():
            if track['age'] > self.max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
    
    def _get_tracked_objects(self):
        """Get all confirmed tracks as an array."""
        tracked = []
        for track_id, track in self.tracks.items():
            if track['hits'] >= self.min_hits:
                state = self._get_state(track)
                score = track['last_detection'][7] if len(track['last_detection']) > 7 else 1.0
                tracked.append(np.concatenate([state, [track_id, score]]))
        
        return np.array(tracked) if tracked else np.empty((0, 9))
    
    def update(self, detections):
        """Update tracker with new detections."""
        if len(detections) == 0:
            self._predict_all()
            self._remove_old_tracks()
            return self._get_tracked_objects()
        
        self._predict_all()
        assignments = self._assign_detections(detections)
        
        # Update assigned tracks
        for track_id, det_idx in assignments.items():
            self._update_track(track_id, detections[det_idx])
        
        # Create new tracks for unassigned detections
        assigned_dets = set(assignments.values())
        for i, det in enumerate(detections):
            if i not in assigned_dets:
                self._create_track(det)
        
        self._remove_old_tracks()
        return self._get_tracked_objects()
    
    def reset(self):
        """Reset the tracker."""
        self.tracks = {}
        self.next_id = 0
