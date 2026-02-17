#!/usr/bin/env python3
"""
Download pretrained weights
Run from project root: python scripts/download_weights.py
"""
import os
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")


def download(url, filename):
    """
    Download a file from URL to the weights directory.
    
    Args:
        url: URL to download from.
        filename: Relative path within weights directory.
    """
    dest = os.path.join(WEIGHTS_DIR, filename)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    
    if os.path.exists(dest):
        print(f"✓ {filename} exists")
        return
    
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, dest)
    size = os.path.getsize(dest) / (1024*1024)
    print(f"✓ Saved ({size:.1f} MB)")


def main():
    """Download all required pretrained weights."""
    print("Downloading pretrained weights...")
    print(f"Destination: {WEIGHTS_DIR}")
    
    # Ultrafast-ResNet (TUSIMPLE)
    download(
        "https://github.com/cfzd/Ultra-Fast-Lane-Detection/releases/download/v1.0/tusimple_res18.pth",
        "lane_detection/tusimple_res18.pth"
    )
    
    # Optional: PV-RCNN
    # download(
    #     "https://download.openmmlab.com/mmdetection3d/v1.1.0_models/pv_rcnn/...",
    #     "3d_detection/pv_rcnn_kitti.pth"
    # )


if __name__ == "__main__":
    main()
