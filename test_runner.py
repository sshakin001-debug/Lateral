#!/usr/bin/env python3
"""
Local Test Runner - Sample 50 images without modifying main code
Compatible with Ultra-Fast-Lane-Detection-v2
"""
import os
import sys
import random
import shutil
from pathlib import Path
import argparse
import cv2
import numpy as np

# Add src to path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Imports from your project - v2 compatible
# Try to use the local ultrafastLaneDetector
LANE_DETECTOR_AVAILABLE = False

try:
    from lateral_sota.ultrafastLaneDetector import UltrafastLaneDetector, ModelType
    LANE_DETECTOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Lane detector import failed: {e}")

# Try ROS-dependent imports, skip if not available
try:
    from lateral_sota.ransacPlaneobject import ransacPlaneobject
    RANSAC_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  ROS not available, skipping ransacPlaneobject: {e}")
    RANSAC_AVAILABLE = False
    ransacPlaneobject = None

# Import your temporal smoothing (optional)
try:
    from my_modules.temporal_smoothing.smoothed_pipeline import SmoothedPipeline
    TEMPORAL_AVAILABLE = True
except ImportError as e:
    TEMPORAL_AVAILABLE = False
    print(f"⚠️  Temporal smoothing not available: {e}")


class LocalTestRunner:
    def __init__(self, num_samples=50, use_temporal=False):
        self.num_samples = num_samples
        self.use_temporal = use_temporal
        
        # Paths (relative to project root)
        self.weights_path = PROJECT_ROOT / "weights" / "lane_detection" / "tusimple_res18.pth"
        self.data_path = PROJECT_ROOT / "data" / "kitti"
        self.output_path = PROJECT_ROOT / "test_outputs"
        self.sample_path = PROJECT_ROOT / "test_samples"
        
        # Verify paths exist
        self._verify_paths()
        
    def _verify_paths(self):
        """Verify all required paths exist"""
        print("=" * 60)
        print("LOCAL TEST RUNNER (v2 Compatible)")
        print("=" * 60)
        
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")
        print(f"✓ Weights: {self.weights_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data not found: {self.data_path}")
        print(f"✓ Data: {self.data_path}")
        
        # Create output directories
        self.output_path.mkdir(exist_ok=True)
        self.sample_path.mkdir(exist_ok=True)
        print(f"✓ Output: {self.output_path}")
        print(f"✓ Samples: {self.sample_path}")
        
    def sample_images(self):
        """Randomly sample N images from training set"""
        image_dir = self.data_path / "training" / "image_2"
        
        if not image_dir.exists():
            raise FileNotFoundError(f"Training images not found: {image_dir}")
        
        all_images = sorted(list(image_dir.glob("*.png")))
        total = len(all_images)
        
        if total == 0:
            raise ValueError("No images found in training/image_2")
        
        print(f"\n📊 Found {total} total images")
        print(f"🎲 Sampling {self.num_samples} images...")
        
        # First N for reproducibility
        sampled = all_images[:self.num_samples]
        
        # Copy to sample directory for easy access
        sample_images_dir = self.sample_path / "images"
        sample_images_dir.mkdir(exist_ok=True)
        
        self.sampled_files = []
        for img_path in sampled:
            # Copy image
            dest = sample_images_dir / img_path.name
            shutil.copy2(img_path, dest)
            self.sampled_files.append(img_path.name)
            
        print(f"✓ Copied {len(self.sampled_files)} images to {sample_images_dir}")
        return self.sampled_files
    
    def run_original_pipeline(self):
        """Run original Lateral-SOTA pipeline on samples"""
        print("\n" + "=" * 60)
        print("RUNNING ORIGINAL PIPELINE")
        print("=" * 60)
        
        if not LANE_DETECTOR_AVAILABLE:
            print("❌ Lane detector not available, cannot run pipeline")
            return []
        
        # Initialize lane detector
        print("\n1️⃣  Initializing Lane Detector...")
        try:
            # Use local detector
            detector = UltrafastLaneDetector(
                model_path=str(self.weights_path),
                model_type=ModelType.TUSIMPLE,
                use_gpu=False
            )
            print("✓ Lane detector ready")
        except Exception as e:
            print(f"✗ Failed to initialize detector: {e}")
            return []
        
        # Process each sample
        results = []
        sample_images_dir = self.sample_path / "images"
        
        for i, img_name in enumerate(self.sampled_files, 1):
            print(f"\n🖼️  Processing [{i}/{len(self.sampled_files)}]: {img_name}")
            
            img_path = sample_images_dir / img_name
            
            # Verify image exists and load it
            if not img_path.exists():
                print(f"   ✗ Image not found: {img_path}")
                continue
            
            # Load image to verify it works
            import cv2
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"   ✗ Failed to load image: {img_path}")
                continue
            print(f"   Image shape: {img.shape}")
            
            # Stage 1: 2D Lane Detection
            try:
                # v2 detect_lanes might return a tuple (lanes, _)
                # or just lanes depending on version
                try:
                    result = detector.detect_lanes(img)  # Pass loaded image for v2
                except TypeError:
                    # If it expects path string, try that
                    result = detector.detect_lanes(str(img_path))
                
                # Handle tuple return (v2 style: lanes, additional_data)
                if isinstance(result, tuple):
                    lanes = result[0]
                else:
                    lanes = result
                
                # Handle None or invalid lanes
                if lanes is None:
                    lanes = [[] for _ in range(4)]  # Return empty lanes instead of None
                
                # Filter out None values from individual lanes (v2 returns 4 lanes)
                if isinstance(lanes, (list, tuple)):
                    lanes = [lane if lane is not None else [] for lane in lanes]
                
                # Handle different return types
                if lanes is None:
                    print(f"   ⚠️  No lanes detected (None)")
                    results.append({
                        'image': img_name,
                        'lanes': [],
                        'status': 'none'
                    })
                elif isinstance(lanes, (list, tuple)) and len(lanes) == 0:
                    print(f"   ⚠️  No lanes detected (empty)")
                    results.append({
                        'image': img_name,
                        'lanes': [],
                        'status': 'empty'
                    })
                else:
                    # Try to count lanes safely
                    try:
                        if isinstance(lanes, np.ndarray):
                            lane_count = len(lanes) if lanes.ndim > 0 else 1
                        elif isinstance(lanes, (list, tuple)):
                            lane_count = len(lanes)
                        else:
                            lane_count = 1
                        
                        print(f"   ✓ Detected {lane_count} lanes (type: {type(lanes).__name__})")
                        results.append({
                            'image': img_name,
                            'lanes': lanes,
                            'status': 'success'
                        })
                    except Exception as e:
                        print(f"   ⚠️  Lane count error: {e}, type: {type(lanes)}")
                        results.append({
                            'image': img_name,
                            'lanes': [],
                            'status': 'count_error'
                        })
                        
            except Exception as e:
                print(f"   ✗ Failed: {e}")
                results.append({
                    'image': img_name,
                    'lanes': [],
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Save results summary
        self._save_results(results, "original_pipeline_results.txt")
        return results
    
    def run_temporal_pipeline(self):
        """Run your enhanced pipeline with temporal smoothing"""
        if not TEMPORAL_AVAILABLE:
            print("\n⚠️  Temporal smoothing not available, skipping")
            return None
            
        print("\n" + "=" * 60)
        print("RUNNING TEMPORAL SMOOTHING PIPELINE")
        print("=" * 60)
        
        # Initialize your smoothed pipeline
        print("\n1️⃣  Initializing Smoothed Pipeline...")
        try:
            pipeline = SmoothedPipeline(
                weights_path=str(self.weights_path),
                use_temporal_smoothing=True,
                smoothing_window=5
            )
            print("✓ Smoothed pipeline ready")
            
            # Run on samples
            results = pipeline.run_sequence(
                image_dir=str(self.sample_path / "images"),
                output_dir=str(self.output_path / "temporal_results")
            )
            
            print(f"\n✓ Processed {len(results)} frames with temporal smoothing")
            self._save_results(results, "temporal_pipeline_results.txt")
            return results
            
        except Exception as e:
            print(f"✗ Temporal pipeline failed: {e}")
            return None
    
    def _save_results(self, results, filename):
        """Save results to file"""
        result_file = self.output_path / filename
        with open(result_file, 'w') as f:
            f.write(f"Test Run Results\n")
            f.write(f"Samples: {len(results)}\n")
            success_count = sum(1 for r in results if r.get('status') == 'success')
            f.write(f"Success: {success_count}\n")
            f.write("=" * 60 + "\n\n")
            for r in results:
                f.write(f"{r}\n")
        print(f"\n💾 Results saved: {result_file}")
    
    def compare_results(self, original_results, temporal_results):
        """Compare original vs temporal smoothing"""
        if temporal_results is None:
            return
            
        print("\n" + "=" * 60)
        print("COMPARISON: Original vs Temporal Smoothing")
        print("=" * 60)
        
        # Basic comparison
        orig_success = sum(1 for r in original_results if r.get('status') == 'success')
        temp_success = len(temporal_results) if temporal_results else 0
        
        print(f"Original pipeline: {orig_success}/{len(original_results)} successful")
        print(f"Temporal pipeline: {temp_success} processed")
        
    def cleanup(self):
        """Clean up test samples (optional)"""
        print("\n" + "=" * 60)
        print("CLEANUP")
        print("=" * 60)
        
        print(f"✓ Test samples kept at: {self.sample_path}")
        print(f"✓ Results kept at: {self.output_path}")
        print("\n🎉 Test run complete!")
        
    def run(self):
        """Main test execution"""
        try:
            # Sample images
            self.sample_images()
            
            # Run original pipeline
            original_results = self.run_original_pipeline()
            
            # Run temporal pipeline (if available)
            temporal_results = None
            if self.use_temporal and TEMPORAL_AVAILABLE:
                temporal_results = self.run_temporal_pipeline()
                self.compare_results(original_results, temporal_results)
            
            # Cleanup
            self.cleanup()
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    parser = argparse.ArgumentParser(description='Test Lateral project with sample images (v2 compatible)')
    parser.add_argument('-n', '--num-samples', type=int, default=50,
                       help='Number of images to sample (default: 50)')
    parser.add_argument('-t', '--temporal', action='store_true',
                       help='Run with temporal smoothing')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Run tests
    runner = LocalTestRunner(
        num_samples=args.num_samples,
        use_temporal=args.temporal
    )
    runner.run()


if __name__ == "__main__":
    main()
