import rospy
import math
import torch, os, cv2
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker

def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]

class Point():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class Lanelabel():
    all = []
    def __init__(self, l_x, l_y, l_z, r_x, r_y, r_z):
       left_point = Point(l_x, l_y, l_z)
       right_point = Point(r_x, r_y, r_z)

       Lanelabel.all.append(self)
    
    def __repr__(self):
        return f"Lane('{self.x}', {self.y}, {self.z})"

if __name__ == "__main__":
    pub = rospy.Publisher('plotting', PointStamped, queue_size=10)
    markerPub = rospy.Publisher('robotMarker', Marker, queue_size=10)

    rospy.init_node('plotting_node', anonymous=True)

    rate = rospy.Rate(1)
    state = PointStamped()

    # initial starting location I might want to move to the param list
    h = rospy.get_param("height", 100)
    w = rospy.get_param("width", 100)
    #state.point.x = h
    #state.point.y = w
    #state.point.z = 0
    
    # make a list which contains labeled line
    label_list = []

    # this line is for getting line from the txt file
    #label_file = '/home/kaai/chicago_ws/src/kitti_visualizer/object/training/label_2/000007.txt'
    label_file = '/home/kaai/chicago_ws/src/kitti_visualizer/object/training/label_2/000192.txt'
    with open(label_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            label = line.strip().split(' ')
            src = line
            cls_type = label[0]
            cls_id = cls_type_to_id(cls_type)
            truncation = float(label[1])
            occlusion = float(label[2])  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
            alpha = float(label[3])
            box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
            h = float(label[8])
            w = float(label[9])
            l = float(label[10])
            loc = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
            dis_to_cam = np.linalg.norm(loc)
            ry = float(label[14])
            score = float(label[15]) if label.__len__() == 16 else -1.0
            level_str = None
            
            # Filter the line which has lane labels
            print(len(label))
            if len(label) != 15:
                label_list.append(label)
        
        #state.point.x = float(i[17])
        #state.point.y = -float(i[15])
        #state.point.z = float(i[16])
        robotMarker = Marker()
        robotMarker.header.frame_id = "livox_frame"
        robotMarker.header.stamp = rospy.Time.now()
        robotMarker.ns = "plotting"
        robotMarker.id = 0
        robotMarker.type = Marker.SPHERE_LIST
        robotMarker.action = Marker.ADD
        #robotMarker.pose.position = state.point
        #robotMarker.pose.position.z = 1  # shift sphere up

        #robotMarker.pose.orientation.x = 0
        #robotMarker.pose.orientation.y = 0
        #robotMarker.pose.orientation.z = 0
        #robotMarker.pose.orientation.w = 1.0
        robotMarker.scale.x = .5
        robotMarker.scale.y = .5
        robotMarker.scale.z = .5

        robotMarker.color.r = 1.0
        robotMarker.color.g = 0.2
        robotMarker.color.b = 1.0
        robotMarker.color.a = 1.0

        for i in label_list:
            left = Point(float(i[17]), -float(i[15]), float(i[16]))
            right = Point(float(i[20]), -float(i[18]), float(i[19]))
            #b = Point(float(i[17]), float(i[15]), float(i[16]))
            #a.append(b)
            #robotMarker.pose.position.append(Point(float(i[17]), -float(i[15]), float(i[16])))
            robotMarker.points.append(left)
            robotMarker.points.append(right)
        print(robotMarker.points[0].x)
        print(robotMarker.points[0].y)
        print(robotMarker.points[0].z)

        while not rospy.is_shutdown():
            markerPub.publish(robotMarker)
            #pub.publish(robotMarker)
            rate.sleep()

        '''
        # Check all the label data
        for i in (label_list):
            #dict1 = {i: label_list}
            state.point.x = float(i[17])
            state.point.y = -float(i[15])
            state.point.z = float(i[16])
            robotMarker = Marker()
            robotMarker.header.frame_id = "livox_frame"
            robotMarker.header.stamp = rospy.Time.now()
            robotMarker.ns = "plotting"
            robotMarker.id = 0
            robotMarker.type = Marker.SPHERE
            robotMarker.action = Marker.ADD
            robotMarker.pose.position = state.point
            print(robotMarker.pose.position)
            robotMarker.pose.position.z = 1  # shift sphere up

            robotMarker.pose.orientation.x = 0
            robotMarker.pose.orientation.y = 0
            robotMarker.pose.orientation.z = 0
            robotMarker.pose.orientation.w = 1.0
            robotMarker.scale.x = 3.0
            robotMarker.scale.y = 3.0
            robotMarker.scale.z = 3.0

            robotMarker.color.r = 1.0
            robotMarker.color.g = 1.0
            robotMarker.color.b = 1.0
            robotMarker.color.a = 1.0

            markerPub.publish(robotMarker)

            rate.sleep()
        
        '''



        '''
        lane_loc = np.array((float(label[15]), float(label[16]), float(label[17])), dtype=np.float32)
        state.point.x = lane_loc[2]
        state.point.y = -lane_loc[0]
        state.point.z = lane_loc[1]

        pub.publish(state)
        '''
        # markerPub.publish(self.robotMarker)
        '''
        robotMarker = Marker()
        robotMarker.header.frame_id = "livox_frame"
        robotMarker.header.stamp = rospy.Time.now()
        robotMarker.ns = "plotting"
        robotMarker.id = 0
        robotMarker.type = Marker.SPHERE
        robotMarker.action = Marker.ADD
        robotMarker.pose.position = state.point
        print(robotMarker.pose.position)
        robotMarker.pose.position.z = 1  # shift sphere up

        robotMarker.pose.orientation.x = 0
        robotMarker.pose.orientation.y = 0
        robotMarker.pose.orientation.z = 0
        robotMarker.pose.orientation.w = 1.0
        robotMarker.scale.x = 1.0
        robotMarker.scale.y = 1.0
        robotMarker.scale.z = 1.0

        robotMarker.color.r = 1.0
        robotMarker.color.g = 1.0
        robotMarker.color.b = 1.0
        robotMarker.color.a = 1.0

        markerPub.publish(robotMarker)
        print("sending marker", robotMarker.pose.position)
        rate.sleep()
        '''

