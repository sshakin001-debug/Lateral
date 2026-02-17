#!/usr/bin/env python3
"""
Setup KITTI dataset via KaggleHub
Run from project root: python scripts/setup_dataset.py
"""
import os
import kagglehub

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def main():
    """Download and setup the KITTI dataset."""
    print("Setting up KITTI dataset...")
    
    # Download via KaggleHub
    path = kagglehub.dataset_download("garymk/kitti-3d-object-detection-dataset")
    print(f"✓ Dataset cached at: {path}")
    
    # Create symlink at project root
    kitti_link = os.path.join(DATA_DIR, "kitti")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Remove existing symlink if present
    if os.path.islink(kitti_link):
        os.remove(kitti_link)
    
    # Link to actual dataset location
    if os.path.exists(os.path.join(path, "kitti")):
        target = os.path.join(path, "kitti")
    else:
        target = path
    
    # Create symlink (note: requires admin privileges on Windows)
    try:
        os.symlink(target, kitti_link)
        print(f"✓ Created symlink: {kitti_link} -> {target}")
    except OSError as e:
        print(f"⚠ Could not create symlink: {e}")
        print(f"  You may need to run as administrator on Windows.")
        print(f"  Alternatively, manually copy the dataset to: {kitti_link}")


if __name__ == "__main__":
    main()
