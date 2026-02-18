#!/usr/bin/env python3
"""
Path utilities for cross-platform compatibility.
Works on local Windows, Kaggle, Colab, etc.
"""
import os
from pathlib import Path

# Cache for project root to avoid repeated lookups
_project_root = None


def get_project_root():
    """
    Get project root directory (works anywhere).
    Cached after first call for performance.
    """
    global _project_root
    if _project_root is not None:
        return _project_root
    
    # Start from current file and go up to find project root
    current_file = Path(__file__).resolve()
    # Go up: paths.py -> utils/ -> src/ -> project_root/
    _project_root = current_file.parent.parent.parent
    return _project_root


def get_src_path(subfolder=None):
    """
    Get path to src directory.
    
    Args:
        subfolder: Optional subfolder within src (e.g., 'lateral_sota')
    
    Returns:
        Path: Path to src directory or subfolder
    """
    src_dir = get_project_root() / "src"
    if subfolder:
        return src_dir / subfolder
    return src_dir


def get_weights_path(subfolder=None, filename=None):
    """
    Get path to weights directory.
    
    Args:
        subfolder: Optional subfolder within weights (e.g., 'lane_detection')
        filename: Optional filename within the weights folder
    
    Returns:
        Path: Path to weights directory or specific file
    """
    weights_dir = get_project_root() / "weights"
    if subfolder:
        weights_dir = weights_dir / subfolder
    if filename:
        return weights_dir / filename
    return weights_dir


def get_data_path(subfolder=None):
    """
    Get path to data directory.
    
    Args:
        subfolder: Optional subfolder within data (e.g., 'kitti')
    
    Returns:
        Path: Path to data directory or subfolder
    """
    data_dir = get_project_root() / "data"
    if subfolder:
        return data_dir / subfolder
    return data_dir


def get_output_path(filename=None, create=True):
    """
    Get path to outputs directory.
    
    Args:
        filename: Optional filename within the outputs folder
        create: Whether to create the directory if it doesn't exist
    
    Returns:
        Path: Path to outputs directory or specific file
    """
    output_dir = get_project_root() / "outputs"
    if create:
        output_dir.mkdir(exist_ok=True)
    if filename:
        return output_dir / filename
    return output_dir


def get_config_path(filename=None):
    """
    Get path to configs directory.
    
    Args:
        filename: Optional filename within configs
    
    Returns:
        Path: Path to configs directory or specific file
    """
    config_dir = get_project_root() / "configs"
    if filename:
        return config_dir / filename
    return config_dir


# Common weight file paths
LANE_DETECTION_WEIGHTS = get_weights_path("lane_detection", "tusimple_res18.pth")
LANE_DETECTION_MODEL = get_weights_path("lane_detection", "tusimple_18.pth")
DETECTION_3D_WEIGHTS = get_weights_path("3d_detection")


# Common data paths
KITTI_DATA_PATH = get_data_path("kitti")


def is_kaggle_environment():
    """Check if running on Kaggle."""
    return os.path.exists('/kaggle/input')


def setup_kaggle_paths():
    """
    Setup paths for Kaggle environment.
    
    Returns:
        dict: Dictionary with Kaggle-specific paths, or None if not on Kaggle
    """
    kaggle_data = Path("/kaggle/input/kitti-3d-object-detection-dataset")
    kaggle_working = Path("/kaggle/working")
    
    if kaggle_data.exists():
        print("✓ Kaggle environment detected")
        return {
            'data': kaggle_data,
            'working': kaggle_working,
            'weights': kaggle_working / "weights",
            'output': kaggle_working / "outputs"
        }
    return None


def get_paths():
    """
    Auto-detect environment and return paths dict.
    
    Returns:
        dict: Dictionary with all project paths
    """
    # Try Kaggle first
    kaggle_paths = setup_kaggle_paths()
    if kaggle_paths:
        return kaggle_paths
    
    # Default: local project structure
    root = get_project_root()
    return {
        'root': root,
        'src': root / "src",
        'data': root / "data",
        'data_kitti': root / "data" / "kitti",
        'weights': root / "weights",
        'weights_lane': root / "weights" / "lane_detection",
        'weights_3d': root / "weights" / "3d_detection",
        'outputs': root / "outputs",
        'configs': root / "configs"
    }


# Convenience function for getting model path
def get_model_path(model_name="tusimple_18.pth", model_type="lane_detection"):
    """
    Get path to a model weight file.
    
    Args:
        model_name: Name of the model file
        model_type: Type of model ('lane_detection' or '3d_detection')
    
    Returns:
        Path: Full path to the model file
    """
    return get_weights_path(model_type, model_name)


if __name__ == "__main__":
    # Test the path utilities
    print("Project paths:")
    paths = get_paths()
    for key, value in paths.items():
        print(f"  {key}: {value}")
    
    print(f"\n✓ All paths resolved successfully!")
    print(f"  Project root: {get_project_root()}")
    print(f"  Lane detection weights: {LANE_DETECTION_WEIGHTS}")
    print(f"  Data path: {KITTI_DATA_PATH}")
