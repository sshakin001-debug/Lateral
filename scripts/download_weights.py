#!/usr/bin/env python3
"""
Download all pretrained weights (lane detection + 3D detection)
"""
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")

def download_with_gdown(file_id, filename, subfolder):
    """Download from Google Drive"""
    folder_path = os.path.join(WEIGHTS_DIR, subfolder)
    os.makedirs(folder_path, exist_ok=True)
    dest = os.path.join(folder_path, filename)
    
    if os.path.exists(dest):
        size = os.path.getsize(dest) / (1024*1024)
        print(f"✓ {filename} already exists ({size:.1f} MB)")
        return True
    
    print(f"\n📥 Downloading {filename}...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "gdown",
            f"https://drive.google.com/uc?id={file_id}",
            "-O", dest,
            "--no-cookies", "--fuzzy"
        ], capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(dest):
            size = os.path.getsize(dest) / (1024*1024)
            print(f"✓ Downloaded ({size:.1f} MB)")
            return True
        else:
            print(f"✗ gdown failed")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def download_with_curl(url, filename, subfolder):
    """Download using curl"""
    folder_path = os.path.join(WEIGHTS_DIR, subfolder)
    os.makedirs(folder_path, exist_ok=True)
    dest = os.path.join(folder_path, filename)
    
    if os.path.exists(dest):
        size = os.path.getsize(dest) / (1024*1024)
        print(f"✓ {filename} already exists ({size:.1f} MB)")
        return True
    
    print(f"\n📥 Downloading {filename}...")
    print(f"From: {url[:60]}...")
    
    try:
        result = subprocess.run([
            "curl", "-L", url, "-o", dest
        ], capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(dest):
            size = os.path.getsize(dest) / (1024*1024)
            print(f"✓ Downloaded ({size:.1f} MB)")
            return True
        else:
            print(f"✗ curl failed")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("Downloading Pretrained Weights")
    print("=" * 60)
    
    # 1. Lane Detection (TUSIMPLE) - v2 weights
    print("\n1. Lane Detection (Ultra-Fast-Lane-Detection-v2)")
    lane_success = download_with_gdown(
        "1Clnj9-dLz81S3wXiYtlkc4HVusCb978t",  # v2 weights
        "tusimple_res18.pth",
        "lane_detection"
    )
    
    # 2. 3D Object Detection (PV-RCNN)
    print("\n2. 3D Object Detection (PV-RCNN)")
    pvrcnn_url = "https://download.openmmlab.com/mmdetection3d/v1.1.0_models/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class/pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428-b384d22f.pth"
    pvrcnn_success = download_with_curl(pvrcnn_url, "pv_rcnn_kitti.pth", "3d_detection")
    
    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"Lane Detection: {'✅' if lane_success else '❌'}")
    print(f"PV-RCNN (3D):   {'✅' if pvrcnn_success else '❌'}")
    
    if not pvrcnn_success:
        print("\nManual download for PV-RCNN:")
        print("URL: https://download.openmmlab.com/mmdetection3d/v1.1.0_models/pv_rcnn/")
        print("File: pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428-b384d22f.pth")
        print(f"Save to: {WEIGHTS_DIR}\\3d_detection\\pv_rcnn_kitti.pth")

if __name__ == "__main__":
    main()