import cv2
import torch
import scipy.special
import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from enum import Enum
from scipy.spatial.distance import cdist
import csv
import math
from ultrafastLaneDetector.model import parsingNet

lane_colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]

tusimple_row_anchor = [ 64,  68,  72,  76,      80,  84,  88,  92,      96, 100, 104, 108, 112,
                                                116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
                                                168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
                                                220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
                                                272, 276, 280, 284]
culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]


class ModelType(Enum):
        TUSIMPLE = 0
        CULANE = 1

class ModelConfig():

        def __init__(self, model_type):
                if model_type == ModelType.TUSIMPLE:
                        self.init_tusimple_config()
                else:
                        self.init_culane_config()

        def init_tusimple_config(self):
                #self.img_w = 1280
                #self.img_h = 720
                self.img_w = 1242
                self.img_h = 375
                self.row_anchor = tusimple_row_anchor
                self.griding_num = 100
                self.cls_num_per_lane = 56

        def init_culane_config(self):
                self.img_w = 1640
                self.img_h = 590
                self.row_anchor = culane_row_anchor
                self.griding_num = 200
                self.cls_num_per_lane = 18

class UltrafastLaneDetector():

        def __init__(self, model_path, model_type=ModelType.TUSIMPLE, use_gpu=False):

                self.use_gpu = use_gpu

                # Load model configuration based on the model type
                self.cfg = ModelConfig(model_type)

                # Initialize model
                self.model = self.initialize_model(model_path, self.cfg, use_gpu)

                # Initialize image transformation
                self.img_transform = self.initialize_image_transform()
                self.leftPlotlist = []
                self.rightPlotlist = []

        @staticmethod
        def initialize_model(model_path, cfg, use_gpu):

                # Load the model architecture
                net = parsingNet(pretrained = False, backbone='18', cls_dim = (cfg.griding_num+1,cfg.cls_num_per_lane,4),
                                                                                use_aux=False) # we dont need auxiliary segmentation in testing


                # Load the weights from the downloaded model
                if use_gpu:
                                net = net.cuda()
                                state_dict = torch.load(model_path, map_location='cuda')['model'] # CUDA
                else:
                                state_dict = torch.load(model_path, map_location='cpu')['model'] # CPU

                compatible_state_dict = {}
                for k, v in state_dict.items():
                                if 'module.' in k:
                                                compatible_state_dict[k[7:]] = v
                                else:
                                                compatible_state_dict[k] = v

                # Load the weights into the model
                net.load_state_dict(compatible_state_dict, strict=False)
                net.eval()

                return net

        @staticmethod
        def initialize_image_transform():
                # Create transfom operation to resize and normalize the input images
                img_transforms = transforms.Compose([
                                transforms.Resize((288, 800)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])

                return img_transforms

        def detect_lanes(self, image, draw_points=True, csv_path=None):

                input_tensor = self.prepare_input(image)

                # Perform inference on the image
                output = self.inference(input_tensor)

                # Process output data
                self.lanes_points, self.lanes_detected, self.detectedLanelist = self.process_output(output, self.cfg)
                
                 
                self.write_lanes(self.detectedLanelist, csv_path)


                # Draw depth image
                visualization_img = self.draw_lanes(image, self.lanes_points, self.lanes_detected, self.cfg, draw_points, self.leftPlotlist, self.rightPlotlist)
                


                return visualization_img
        
        def write_lanes(self, lane_lists, csv_path):
                # lane_lists = [leftx_list, lefty_list, rightx_list, righty_list]
                
                f = open(csv_path, 'w', newline="")
                #print(lane_lists)
                if len(lane_lists[0]) != 0 and len(lane_lists[1]) != 0 and len(lane_lists[2]) != 0 and len(lane_lists[3]) != 0:
                        leftCoefficient = np.polyfit(lane_lists[1], lane_lists[0], 2) 
                        rightCoefficient = np.polyfit(lane_lists[3], lane_lists[2], 2)
                        wr = csv.writer(f)
                        wr.writerow(["Left_Lane_Coefficient", leftCoefficient[0], leftCoefficient[1], leftCoefficient[2]])
                        wr.writerow(["Right_Lane_Coefficient", rightCoefficient[0], rightCoefficient[1], rightCoefficient[2]])
                        
                        a = leftCoefficient[0] - rightCoefficient[0]
                        b = leftCoefficient[1] - rightCoefficient[1]
                        c = leftCoefficient[2] - rightCoefficient[2]
                        
                        # calculate the meeting points between left lanes and right lanes
                        if a != 0 and b * b - 4 * a * c >= 0:  # case for a != 0
                                        contact_y = (-b - math.sqrt(b * b - 4 * a * c)) / (2 * a)
                                        contact_y = int(contact_y)
                                        contact_y = contact_y + 20
                        else:  # case for a == 0. Just finding contact point of lines
                                        contact_y = -c / b
                                        contact_y = int(contact_y)
                                        contact_y = contact_y + 20
                        contact_y = abs(contact_y)
                        contact_x = leftCoefficient[0] * math.pow(contact_y, 2) + leftCoefficient[1] * contact_y + leftCoefficient[2]
                        #ploty = np.linspace(contact_y, frame.shape[0] - 1, frame.shape[0] - contact_y)  # y좌표에 대해서 상위 1/3 지점 부터 최하단 까지 그래프를 그리기위한 코드
                        #1280 720
                        ploty = np.linspace(contact_y, self.cfg.img_h - 1, self.cfg.img_h - contact_y)  # y좌표에 대해서 상위 1/3 지점 부터 최하단 까지 그래프를 그리기위한 코드
                        
                        left_fitx = leftCoefficient[0] * ploty ** 2 + leftCoefficient[1] * ploty + leftCoefficient[2]  # 방정식에 대한 x 값을 얻어오기 위한 코드
                        right_fitx = rightCoefficient[0] * ploty ** 2 + rightCoefficient[1] * ploty + rightCoefficient[2]
                        
                        # lists to save pixel value on equated lanes
                        leftEquationx = []
                        leftEquationy = []
                        rightEquationx = []
                        rightEquationy = []
                        for ii, x in enumerate(left_fitx):
                                q = (int(x), int(ii + contact_y))  # 상위 1/3 지점부터 그래프를 그리기 시작하므로 인덱스에 해당하는 ii 에 그만큼 추가해준다.
                                self.leftPlotlist.append(q)
                                if (x >= 0):
                                                #k.append(q)
                                                leftEquationx.append(q[0])
                                                leftEquationy.append(q[1])
                        for ii, x in enumerate(right_fitx):
                                q = (int(x), int(ii + contact_y))
                                self.rightPlotlist.append(q)
                                if (x >= 0):
                                                #l.append(q)
                                                rightEquationx.append(q[0])
                                                rightEquationy.append(q[1])
                wr.writerow(leftEquationx)
                wr.writerow(leftEquationy)
                wr.writerow(rightEquationx)
                wr.writerow(rightEquationy)
                f.close

        def prepare_input(self, img):
                # Transform the image for inference
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img)
                input_img = self.img_transform(img_pil)
                input_tensor = input_img[None, ...]

                if self.use_gpu:
                                input_tensor = input_tensor.cuda()

                return input_tensor

        def inference(self, input_tensor):
                with torch.no_grad():
                        output = self.model(input_tensor)

                return output

        @staticmethod
        def process_output(output, cfg):                        
                # Parse the output of the model
                processed_output = output[0].data.cpu().numpy()
                processed_output = processed_output[:, ::-1, :]
                prob = scipy.special.softmax(processed_output[:-1, :, :], axis=0)
                idx = np.arange(cfg.griding_num) + 1
                idx = idx.reshape(-1, 1, 1)
                loc = np.sum(prob * idx, axis=0)
                processed_output = np.argmax(processed_output, axis=0)
                loc[processed_output == cfg.griding_num] = 0
                processed_output = loc


                col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
                col_sample_w = col_sample[1] - col_sample[0]

                lanes_points = []
                lanes_detected = []
                leftx = []
                lefty = []
                rightx = []
                righty = []
                
                #print(f"col_sample_w: {col_sample_w} / img_w: {cfg.img_w} / cls_num_per_lane: {cfg.cls_num_per_lane}")

                max_lanes = processed_output.shape[1]
                #print(max_lanes)
                for lane_num in range(max_lanes):
                        lane_points = []
                        # Check if there are any points detected in the lane
                        if np.sum(processed_output[:, lane_num] != 0) > 2:

                                lanes_detected.append(True)

                                # Process each of the points for each lane
                                for point_num in range(processed_output.shape[0]):
                                        if processed_output[point_num, lane_num] > 0:
                                                lane_point = [int(processed_output[point_num, lane_num] * col_sample_w * cfg.img_w / 800) - 1, int(cfg.img_h * (cfg.row_anchor[cfg.cls_num_per_lane-1-point_num]/288)) - 1 ]
                                                lane_points.append(lane_point)
                                                # about left lane
                                                if lane_num == 1:
                                                        leftx.append(lane_point[0])
                                                        lefty.append(lane_point[1])
                                                if lane_num == 2:
                                                        rightx.append(lane_point[0])
                                                        righty.append(lane_point[1])
                                                                                        
                        else:
                                lanes_detected.append(False)

                        lanes_points.append(lane_points)
                return np.array(lanes_points), np.array(lanes_detected), [leftx, lefty, rightx, righty]

        @staticmethod
        def draw_lanes(input_img, lanes_points, lanes_detected, cfg, draw_points, leftplotlist, rightplotlist):
                # Write the detected line points in the image
                visualization_img = cv2.resize(input_img, (cfg.img_w, cfg.img_h), interpolation = cv2.INTER_AREA)

                # Draw a mask for the current lane
                #if(lanes_detected[1] and lanes_detected[2]):
                #        lane_segment_img = visualization_img.copy()
                #        cv2.fillPoly(lane_segment_img, pts = [np.vstack((lanes_points[1],np.flipud(lanes_points[2])))], color =(255,191,0))
                #        visualization_img = cv2.addWeighted(visualization_img, 0.7, lane_segment_img, 0.3, 0)

                #if(draw_points):
                #        for lane_num,lane_points in enumerate(lanes_points):
                #                for lane_point in lane_points:
                #                        cv2.circle(visualization_img, (lane_point[0],lane_point[1]), 3, lane_colors[lane_num], -1)
                # Draw a quadratic lines on the image
                leftplotlist = np.array(leftplotlist, np.int32)
                rightplotlist = np.array(rightplotlist, np.int32)
                
                #(B,G,R)
                #cv2.polylines(visualization_img, [leftplotlist], False, (0,255,0), thickness=7)
                cv2.polylines(visualization_img, [leftplotlist], False, (104,255,0), thickness=7)
                #cv2.polylines(visualization_img, [rightplotlist], False, (0,0,255), thickness=7)
                cv2.polylines(visualization_img, [rightplotlist], False, (0,244,239), thickness=7)

                return visualization_img


                







