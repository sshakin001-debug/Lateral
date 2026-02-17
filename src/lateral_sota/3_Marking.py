import rospy
import math
import torch, os, cv2
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
import shapely.geometry as geom
from shapely.geometry import Point
import matplotlib.pyplot as plt
from std_msgs.msg import ColorRGBA
import sys

sn = int(sys.argv[1]) if len(sys.argv)>1 else 7 #default 7
name = '%06d'%sn # 6 digit zeropadding
label_path = f'../../../../dataset/training/label_2/{name}.txt'

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
            resultList = resultList + [[point.y, -point.x] + self.__call__(point, 'left', point_on_line, sign, direction)]
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
            resultList = resultList + [[point.y, -point.x] + self.__call__(point, 'right', point_on_line, sign, direction)]
        
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
    pub = rospy.Publisher('plotting', PointStamped, queue_size=10)
    markerPub_left = rospy.Publisher('left_eq', Marker, queue_size=10)
    markerPub_right = rospy.Publisher('right_eq', Marker, queue_size=10)
    markerPub_answer = rospy.Publisher('answer', Marker, queue_size=10)
    markerPub_text = rospy.Publisher('text', MarkerArray, queue_size=10)

    rospy.init_node('plotting_node', anonymous=True)

    rate = rospy.Rate(1)
    state = PointStamped()

    # initial starting location I might want to move to the param listesultList = resultList + [point.y, -point.x] + self.__call__(point, 'right', point_on_line)
    h = rospy.get_param("height", 100)
    w = rospy.get_param("width", 100)
    
    # make a list which contains labeled line
    label_list = []

    height = -1.50115
    
    l_coords = np.loadtxt('../../CSV_Communication/left_lane_Ransac.txt')
    r_coords = np.loadtxt('../../CSV_Communication/right_lane_Ransac.txt')
    o_coords = np.loadtxt('../../CSV_Communication/object.txt')
    # get the lane list
    left_lane_equat_point_x = []
    left_lane_equat_point_y = []
    right_lane_equat_point_x = []
    right_lane_equat_point_y = []
    #object_list = []
    
    for i in l_coords:
        left_lane_equat_point_x.append(i[1])
        left_lane_equat_point_y.append(-i[0])
    for i in r_coords:
        right_lane_equat_point_x.append(i[1])
        right_lane_equat_point_y.append(-i[0])
    #for i in o_coords:
    #    object_list.append([i[0],i[1]])

    # process the minimum distance calculation
    left_line = geom.LineString(l_coords)
    right_line = geom.LineString(r_coords)

    left_point_list = []
    right_point_list = []
    max_point = float(l_coords[l_coords.shape[0] - 1][1]) # indicate tesultList = resultList + [point.y, -point.x] + self.__call__(point, 'right', point_on_line)he maximum value from lane equation
    # remove the object point which has bigger value than the maximum esultList = resultList + [point.y, -point.x] + self.__call__(point, 'right', point_on_line)point
    #for i in range(len(object_list)):
    #    if object_list[i][1] > max_point:
    #        del object_list[i:]
    #        break

    # Plotting Section
    fig, ax = plt.subplots()
    ax.plot(*l_coords.T)
    ax.plot(*r_coords.T)
    ax.axis('equal')
    # Set the frame
    ax.set_xlim(-10, 10)
    #ax.set_ylim(-1, object_list[-1][1]+ 10)
    ax.set_ylim(-1, 70)
    
    # Make the class
    #distance_class = NearestPoint(left_line, right_line, ax)
    #result = distance_class.draw_segment(object_list)

    #print(object_list)
    #print(result)
    
    objectList = []

    # this line is for getting line from the txt file
    #label_file = '../../../../dataset/training/label_2/000007.txt'
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            label = line.strip().split(' ')
            src = line
            '''
            cls_type = label[0]
            cls_id = cls_type_to_id(cls_type)
            truncation = float(label[1])
            occlusion = float(label[2])  # 0:fully visible 1:partly ocesultList = resultList + [point.y, -point.x] + self.__call__(point, 'right', point_on_line)cluded 2:largely occluded 3:unknown
            alpha = float(label[3])
            box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
            h = float(label[8])
            w = float(label[9])
            l = float(label[10])
            #loc = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
            #dis_to_cam = np.linalg.norm(loc)
            #ry = float(label[14])
            #score = float(label[15]) if label.__len__() == 16 else -1.0
            #level_str = None
            
            
            # Filter the line which has lane labels
            #print(len(label))
            print(len(label))
            if len(label) != 15:
                label_list.append(label)
                print(label)
            '''
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
	
        # Define the marker of left lane
        left_points = left_line_strip = left_line_list = Marker()
        left_points.header.frame_id = left_line_list.header.frame_id = left_line_list.header.frame_id = "livox_frame"
        left_points.header.stamp = left_line_list.header.stamp = left_line_list.header.stamp = rospy.Time.now()
        
        left_points.ns= left_line_list.ns = left_line_list.ns = "plotting"
        left_points.action = left_line_strip.action = left_line_list.action = Marker.ADD
        left_points.pose.orientation.w = left_line_strip.pose.orientation.w = left_line_list.pose.orientation.w = 1.0

        left_points.id = 0
        left_line_list.id = 1
        left_line_strip.id = 2
        
        left_points.type = Marker.POINTS
        left_line_list.type = Marker.LINE_LIST
        left_line_strip.type = Marker.LINE_STRIP

        left_points.scale.x = 0.2
        left_points.scale.y = 0.2
        left_line_strip.scale.x = 0.1
        # set the lane's thickness
        left_line_list.scale.x = 0.3

        # Points are green
        left_points.color.r = 1.0
        left_points.color.g = 1.0
        left_points.color.b = 0.0
        left_points.color.a = 1.0

        # Line strip is blue
        left_line_strip.color.r = 0.0
        left_line_strip.color.g = 1.0
        left_line_strip.color.b = 0.0
        left_line_strip.color.a = 1.0
        
        # Line list is red
        left_line_list.color.r = 0.0
        left_line_list.color.g = 1.0
        left_line_list.color.b = 0.0
        left_line_list.color.a = 1.0
        
        #g = open("left_lane.txt", "w")
        for i in range(len(left_lane_equat_point_x)):
            left = Point(left_lane_equat_point_x[i], left_lane_equat_point_y[i], height)
            #a.append(b)
            #robotMarker.pose.position.append(Point(float(i[17]), -float(i[15]), float(i[16])))
            left_points.points.append(left)
            left_line_strip.points.append(left)
            left_line_list.points.append(left)
            #left = Point(left_lane_equat_point_x[i], left_lane_equat_point_y[i], height+1)
            #left_line_list.points.append(left)

            #word = str(-left_lane_equat_point_y[i]) + ' ' + str(left_point_on_line)lane_equat_point_x[i]) + '\n'
            #g.write(word)
        #g.close() 

        # Define the marker of right lane
        right_points = right_line_strip = right_line_list = Marker()
        right_points.header.frame_id = right_line_list.header.frame_id = right_line_list.header.frame_id = "livox_frame"
        right_points.header.stamp = right_line_list.header.stamp = right_line_list.header.stamp = rospy.Time.now()
        right_points.action = right_line_strip.action = right_line_list.action = Marker.ADD
        right_points.pose.orientation.w = right_line_strip.pose.orientation.w = right_line_list.pose.orientation.w = 1.0

        
        right_points.ns= right_line_list.ns = right_line_list.ns = "plotting"
        right_points.id = 0
        right_line_list.id = 1
        right_line_strip.id = 2
        
        right_points.type = Marker.POINTS
        right_line_list.type = Marker.LINE_LIST
        right_line_strip.type = Marker.LINE_STRIP

        right_points.scale.x = 0.2
        right_points.scale.y = 0.2
        right_line_strip.scale.x = 0.1
        # set the lane's thickness
        right_line_list.scale.x = 0.3

        # Points are green
        right_points.color.r = 1.0
        right_points.color.g = 1.0
        right_points.color.b = 0.0
        right_points.color.a = 1.0
        
        # right color is yellow
        right_line_strip.color.r = 1.0
        right_line_strip.color.g = 1.0
        right_line_strip.color.b = 0.0
        right_line_strip.color.a = 1.0
        
        # Line list is red
        right_line_list.color.r = 1.0
        right_line_list.color.g = 1.0
        right_line_list.color.b = 0.0
        right_line_list.color.a = 1.0


        #g = open("right_lane.txt", "w")
        for i in range(len(right_lane_equat_point_x)):
            right = Point(right_lane_equat_point_x[i], right_lane_equat_point_y[i], height)
            right_points.points.append(right)
            right_line_strip.points.append(right)

            #print(right)

            # The line list needs two points for each line
            right_line_list.points.append(right)
            #right = Point(right_lane_equat_point_x[i], right_lane_equat_point_y[i], height+1.0)
            #right_line_list.points.append(right)

            #word = str(-right_lane_equat_point_y[i]) + ' ' + str(right_lane_equat_point_x[i]) + '\n'
            #g.write(word)
        #g.close()

        # Define the marker of Answer
        answer_points = answer_line_strip = answer_line_list = Marker()
        answer_points.header.frame_id = answer_line_list.header.frame_id = answer_line_list.header.frame_id = "livox_frame"
        answer_points.header.stamp = answer_line_list.header.stamp = answer_line_list.header.stamp = rospy.Time.now()
        answer_points.action = answer_line_strip.action = answer_line_list.action = Marker.ADD
        answer_points.pose.orientation.w = answer_line_strip.pose.orientation.w = answer_line_list.pose.orientation.w = 1.0

        answer_points.ns= answer_line_list.ns = answer_line_list.ns = "plotting"
        answer_points.id = 0
        answer_line_list.id = 1
        answer_line_strip.id = 2
        
        answer_points.type = Marker.POINTS
        answer_line_list.type = Marker.LINE_LIST
        answer_line_strip.type = Marker.LINE_STRIP

        answer_points.scale.x = 1.6
        answer_points.scale.y = 1.6
        answer_line_strip.scale.x = 0.1
        # set the answer line's thickness
        answer_line_list.scale.x = 0.2

        # Points are green
        answer_points.color.g = 1.0
        answer_points.color.a = 1.0

        # Line strip is blue
        answer_line_strip.color.b = 1.0
        answer_line_strip.color.a = 1.0
        
        # Line list is red
        answer_line_list.color.r = 2.0
        answer_line_list.color.g = 0.0
        answer_line_list.color.a = 1.0
        

        # Marker for text
        rvizTextarray = MarkerArray()
        from geometry_msgs.msg import Point
        '''
        rviz_text = Marker()
        rviz_text.header.frame_id = "livox_frame"
        rviz_text.ns = "plotting"
        rviz_text.action = Marker.ADD
        rviz_text.type = Marker.TEXT_VIEW_FACING
        rviz_text.color = ColorRGBA(1,1,1,1)
        rviz_text.scale.z = 1.2
        #rviz_text.text = str(round(result[4],2)) + ' M'
        #from geometry_msgs.msg import Point
        #txt_location = Point(result[2], result[3], height+1.0)
        ##rviz_text.pose.position = Point(2, 1, 0)
        #rviz_text.pose.position = txt_location
        
        #rviz_text.pose.position = Point(result[2], result[3], height+2.0)
        '''
        # result list label: [object_x, object_y, lane_x, lane_y, distance_value]        
        #g = open("right_lane.txt", "w")
        print("YOUNGIL  ", result, "  YOUGNIL")
        for ii,i in enumerate (result):
            answer_lane = Point(i[2], i[3], height)
            answer_obj =  Point(i[0], i[1], height)
            distance_value = i[4]
            answer_points.points.append(answer_lane)
            answer_line_strip.points.append(answer_lane)
            answer_points.points.append(answer_obj)
            answer_line_strip.points.append(answer_obj)

            # The line list needs two points for each line
            answer_line_list.points.append(answer_lane)
            answer_lane = Point(i[2], i[3], height)
            answer_obj =  Point(i[0], i[1], height)
            answer_line_list.points.append(answer_lane)
            
            rviz_text = Marker()
            rviz_text.header.frame_id = "livox_frame"
            rviz_text.ns = "plotting"+ f'{ii}'
            rviz_text.action = Marker.ADD
            rviz_text.type = Marker.TEXT_VIEW_FACING
            rviz_text.color = ColorRGBA(1,1,1,1)
            rviz_text.scale.z = 1.2 
            text_location = Point(i[2], i[3], height + 2.0)

            rviz_text.pose.position= text_location
            rviz_text.text = str(round(distance_value, 2)) + ' M'
            rvizTextarray.markers.append(rviz_text)
        #markerPub_text.publish(rvizTextarray)

        while not rospy.is_shutdown():
            #markerPub_left.publish(left_points)
            #markerPub_left.publish(left_line_strip)
            markerPub_left.publish(left_line_list)
            markerPub_right.publish(right_points)
            markerPub_right.publish(right_line_strip)
            markerPub_right.publish(right_line_list)
            markerPub_answer.publish(answer_points)
            markerPub_answer.publish(answer_line_strip)
            markerPub_answer.publish(answer_line_list)
            #markerPub_text.publish(rviz_text)
            markerPub_text.publish(rvizTextarray)
            rate.sleep()
            # plt the image of distance estimation on image
            plt.show()
    



