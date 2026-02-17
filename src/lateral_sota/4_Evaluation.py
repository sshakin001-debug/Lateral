import rospy
import math
import torch, os, cv2
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
import shapely.geometry as geom
from shapely.geometry import Point
import matplotlib.pyplot as plt
from std_msgs.msg import ColorRGBA
import sys
import csv

sn = int(sys.argv[1]) if len(sys.argv)>1 else 7 #default 0-7517
name = '%06d'%sn # 6 digit zeropadding
label_dir = f'../../../../labeled_txt/{name}.txt'
txt_dir = f'../../CSV_Communication/evaluation.txt'

txt_file = open(txt_dir, 'a')

class NearestPoint(object):
    def __init__(self, left_line, right_line, ax):
        self.left_line = left_line
        self.right_line = right_line
        self.ax = ax
        ax.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, xy_point, side, point_on_lane, sign, direction):
        assert side == 'left' or side == 'right', 'Type among left or right'
        if side == 'left':
            distance = self.left_line.distance(xy_point)
        else:
            distance = self.right_line.distance(xy_point)
        print(f'Distance to line({side}): {sign}', distance, 'm')
        
        instant_resultList = [point_on_lane.y, -point_on_lane.x, distance]
        
        txt_file.write(f"{sn} {direction} {xy_point.y} {-xy_point.x} {point_on_lane.y} {-point_on_lane.x} {sign}{distance} ({side})\n")

        return instant_resultList
            
    def draw_segment(self, object_list):
        resultList = []
        # determine the points whether those are at the left side or right side.
        for i in object_list:
            if i[0] <= 0:
                left_point_list.append(Point(i[0], i[1]))
            else:
                right_point_list.append(Point(i[0], i[1]))
        # Draw the lateral lane about the left objects
        for point in left_point_list:
            point_on_line = self.left_line.interpolate(self.left_line.project(point))
            self.ax.plot([point.x, point_on_line.x], [point.y, point_on_line.y], 
                     color='red', marker='o', scalex=False, scaley=False)
            fig.canvas.draw()

            print(f"x: {point.x}, x:{point.y}")
            print(f"x: {point_on_line.x}, x:{point_on_line.y}")

            sign = ''
            direction = ''
            

            if float(i[-1]) >= 0:
                direction = "same"
            else:
                direction = "opposite"

            if -point.x > -point_on_line.x:
                sign = '+'
            else:
                sign = '-'
                direction = "same"

            # get the distance in meter unit
            resultList = resultList + [point.y, -point.x] + self.__call__(point, 'left', point_on_line, sign, direction)
        # Draw the lateral lane about the right objects
        for point in right_point_list:
            point_on_line = self.right_line.interpolate(self.right_line.project(point))
            self.ax.plot([point.x, point_on_line.x], [point.y, point_on_line.y], 
                     color='red', marker='o', scalex=False, scaley=False)
            fig.canvas.draw()

            sign = ''
            direction = ''
           
            if float(i[-1]) >= 0:
                direction = "same"
            else:
                direction = "opposite"

            if -point.x < -point_on_line.x:
                sign = '+'
            else:
                sign = '-'
                direction = "same"

            # get the distance in meter unit
            resultList = resultList + [point.y, -point.x] + self.__call__(point, 'right', point_on_line, sign, direction)
        
        # result list label: [object_x, object_y, lane_x, lane_y, distance_value]
        return resultList

def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]

class Point_Coordinate():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class Lanelabel():
    all = []
    def __init__(self, l_x, l_y, l_z, r_x, r_y, r_z):
        left_point = Point_Coordinate(l_x, l_y, l_z)

        right_point = Point_Coordinate(r_x, r_y, r_z)

        Lanelabel.all.append(self)
    
    def __repr__(self):
        return f"Lane('{self.x}', {self.y}, {self.z})"

if __name__ == "__main__":
    # this line is for getting line from the txt file
    label_file = '/home/kaai/dataset/training/label_2/000007.txt'
    #label_file = '/home/kaai/dataset/training/label_2/000192.txt'

    # initial starting location I might want to move to the param listesultList = resultList + [point.y, -point.x] + self.__call__(point, 'right', point_on_line)
    #h = rospy.get_param("height", 100)
    #w = rospy.get_param("width", 100)
    
    # make a list which contains labeled line
    label_list = []

    height = -1.50115
    
    l_coords = np.loadtxt('/home/kaai/chicago_ws/src/CSV_Communication/left_lane_Ransac.txt')
    r_coords = np.loadtxt('/home/kaai/chicago_ws/src/CSV_Communication/right_lane_Ransac.txt')
    o_coords = np.loadtxt('/home/kaai/chicago_ws/src/CSV_Communication/object.txt')
    # get the lane list
    left_lane_equat_point_x = []
    left_lane_equat_point_y = []
    right_lane_equat_point_x = []
    right_lane_equat_point_y = []
    object_list = []
    
    for i in l_coords:
        left_lane_equat_point_x.append(i[1])
        left_lane_equat_point_y.append(-i[0])
    for i in r_coords:
        right_lane_equat_point_x.append(i[1])
        right_lane_equat_point_y.append(-i[0])
    for i in o_coords:
        object_list.append([i[0],i[1]])

    # process the minimum distance calculation
    left_line = geom.LineString(l_coords)
    right_line = geom.LineString(r_coords)

    left_point_list = []
    right_point_list = []
    max_point_l = float(l_coords[l_coords.shape[0] - 1][1]) # indicate tesultList = resultList + [point.y, -point.x] + self.__call__(point, 'right', point_on_line)he maximum value from lane equation
    # remove the object point which has bigger value than the maximum esultList = resultList + [point.y, -point.x] + self.__call__(point, 'right', point_on_line)point
    max_point_r = float(r_coords[r_coords.shape[0]-1][1])
    max_point = max(max_point_l, max_point_r)
    for i in range(len(object_list)):
        if object_list[i][1] > max_point:
            del object_list[i:]
            break

    # Plotting Section
    fig, ax = plt.subplots()
    ax.plot(*l_coords.T)
    ax.plot(*r_coords.T)
    ax.axis('equal')
    # Set the frame
    ax.set_xlim(-10, 10)
    ax.set_ylim(-1, 85)
    
    # Make the class
    #distance_class = NearestPoint(left_line, right_line, ax)
    #result = distance_class.draw_segment(object_list)
    
    objectList = []

    with open(label_dir, 'r') as f:
        lines = f.readlines()
        for line in lines:
            label = line.strip().split(' ')
            src = line
            
            if len(label) > 1:
                cls_type = label[0]
                cls_id = cls_type_to_id(cls_type)
                truncation = float(label[1])
                occlusion = float(label[2])  # 0:fully visible 1:partly ocesultList = resultList + [point.y, -point.x] + self.__call__(point, 'right', point_on_line)cluded 2:largely occluded 3:unknown
                alpha = float(label[3])
                box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
                h = float(label[8])
                w = float(label[9])
                l = float(label[10])
                #objectLocation = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
                #dis_to_cam = np.linalg.norm(objectLocation)
                #ry = float(label[14])
                #score = float(label[15]) if label.__len__() == 16 else -1.0
                #level_str = None


                # KITTI Coordinate
                objectLocation = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
                # World Coordinate
                #objectLocation = np.array((float(label[13]), -float(label[11]), -float(label[12])), dtype=np.float32)
                
                objectList.append([objectLocation[0], objectLocation[2]-l*(1/2), cls_type, label[-2]])
    
    def sortSecond(val):
        return val[1]
    objectList.sort(key=sortSecond)
    #print(objectList)
    for i in range(len(objectList)):
        if objectList[i][1] > max_point:
            del objectList[i:]
            break
    distance_class = NearestPoint(left_line, right_line, ax)
    result = distance_class.draw_segment(objectList)
   
    plt.show()
 
