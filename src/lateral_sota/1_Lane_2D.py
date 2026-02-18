import cv2
import sys
from pathlib import Path

# Add src to path for imports
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.paths import get_weights_path, get_data_path, get_output_path
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType

# Use relative paths via utility functions
# Default to tusimple_18.pth if available, otherwise tusimple_res18.pth
model_path = get_weights_path("lane_detection", "tusimple_18.pth")
if not model_path.exists():
    model_path = get_weights_path("lane_detection", "tusimple_res18.pth")

model_type = ModelType.TUSIMPLE
use_gpu = False

# Get image and output paths
sn = int(sys.argv[1]) if len(sys.argv) > 1 else 7  # default 0-7517
name = '%06d' % sn  # 6 digit zeropadding

# Try to find KITTI dataset - check multiple possible locations
kitti_base = get_data_path()
possible_image_paths = [
    kitti_base / "training" / "image_2" / f"{name}.png",
    kitti_base / "kitti" / "training" / "image_2" / f"{name}.png",
    PROJECT_ROOT / "dataset" / "training" / "image_2" / f"{name}.png",
]

image_path = None
for p in possible_image_paths:
    if p.exists():
        image_path = str(p)
        break

if image_path is None:
    # Fallback: use the first path as default
    image_path = str(possible_image_paths[0])

# Output CSV path
csv_path = get_output_path("1_lane.csv")

# Initialize lane detection model
lane_detector = UltrafastLaneDetector(str(model_path), model_type, use_gpu)

# Read RGB images
img = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Detect the lanes
output_img = lane_detector.detect_lanes(img, draw_points=True, csv_path=str(csv_path))

# Draw estimated depth
cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)
cv2.imshow("Detected lanes", output_img)
cv2.waitKey(0)

# Optional: save output
# cv2.imwrite("output.jpg", output_img)
