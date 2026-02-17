import cv2
import sys

from ultrafastLaneDetector import UltrafastLaneDetector, ModelType

model_path = "../models/tusimple_18.pth"
model_type = ModelType.TUSIMPLE
use_gpu = False

#image_path = "input.jpg"
sn = int(sys.argv[1]) if len(sys.argv)>1 else 7 #default 0-7517
name = '%06d'%sn # 6 digit zeropadding
image_path = f'../../../../dataset/training/image_2/{name}.png'
csv_path = "../../CSV_Communication/1_lane.csv"

#print(model_type)
#print(model_type.griding_num)
#print(model_type.cls_num_per_lane)

# Initialize lane detection model
lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)

# Read RGB images
img = cv2.imread(image_path, cv2.IMREAD_COLOR)
#print(img.shape)

# Detect the lanes
output_img = lane_detector.detect_lanes(img, draw_points=True, csv_path= csv_path)

# Draw estimated depth
cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL) 
cv2.imshow("Detected lanes", output_img)
cv2.waitKey(0)

#cv2.imwrite("output.jpg",output_img)
