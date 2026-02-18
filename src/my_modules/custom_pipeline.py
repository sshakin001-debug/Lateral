"""
Custom pipeline for enhanced lane detection and 3D processing.
"""
import sys
from pathlib import Path

# Add src to path (if not using setup.py)
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import get_weights_path
from lateral_sota.ultrafastLaneDetector.ultrafastLaneDetector import UltrafastLaneDetector
from lateral_sota.ransacPlaneobject import ransacPlaneobject


class MyEnhancedPipeline:
    """
    Enhanced pipeline combining lane detection with 3D processing.
    """
    
    def __init__(self, weights_path=None):
        """
        Initialize the enhanced pipeline.
        
        Args:
            weights_path: Path to the lane detection weights file.
                         If None, uses default path from utils.paths.
        """
        if weights_path is None:
            weights_path = get_weights_path("lane_detection", "tusimple_res18.pth")
        
        # Convert to string if Path object
        weights_path = str(weights_path)
        
        self.detector = UltrafastLaneDetector(weights_path, model_type='TUSIMPLE')
    
    def run(self, image_path):
        """
        Run the enhanced pipeline on an image.
        
        Args:
            image_path: Path to the input image.
            
        Returns:
            dict: Detection results including lanes and 3D information.
        """
        # Your custom logic
        lanes = self.detector.detect_lanes(image_path)
        
        return {
            'lanes': lanes,
            'image_path': image_path
        }


if __name__ == "__main__":
    # Example usage
    pipeline = MyEnhancedPipeline()
    print("Enhanced pipeline initialized successfully!")
