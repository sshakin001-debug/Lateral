import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import csv
from ransacPlaneobject import *
import math
from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RANSACRegressor

class PolynomialRegression(object):
    def __init__(self, degree=2, coeffs=None):
        self.degree = degree
        self.coeffs = coeffs

    def fit(self, X, y):
        self.coeffs = np.polyfit(X.ravel(), y, self.degree)

    def get_params(self, deep=False):
        return {'coeffs': self.coeffs}

    def set_params(self, coeffs=None, random_state=None):
        self.coeffs = coeffs

    def predict(self, X):
        poly_eqn = np.poly1d(self.coeffs)
        y_hat = poly_eqn(X.ravel())
        return y_hat

    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))

def ransac_plot(xArray, yArray, y_hat, inlier_mask):
    line_width = 2
    plt.plot(xArray, yArray, 'bx', label='input samples')
    plt.plot(xArray[inlier_mask], yArray[inlier_mask], 'go', label='inliers')
    plt.plot(xArray, y_hat, 'r-', label='estimated curve')
    plt.legend(loc="lower right")
    plt.xlabel("Input")
    plt.ylabel("Response")
    plt.show()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

sn = int(sys.argv[1]) if len(sys.argv)>1 else 7 #default 7
name = '%06d'%sn # 6 digit zeropadding
img = f'../../../../dataset/training/image_2/{name}.png'
binary = f'../../../../dataset/training/velodyne1/{name}.bin'
pcd= f'../../../../dataset/training/velodyne/{name}.pcd'
with open(f'../../../../dataset/training/calib/{name}.txt','r') as f:
    calib = f.readlines()

# read the lane pixel from 1_Lane_2D.py
'''
[About csv file]
line 0: the coefficient of left lane equation
line 1st: the coefficient of right lane equation
line 2nd: leftlaneX
line 3rd: leftlaneY
line 4th: rightlaneX
line 5th: rightlaneY
'''
lane_csv = open("../../CSV_Communication/1_lane.csv")

# get the plane pointcloud data and object pointcloud data 
# planePCDarray: (x,y,z,intensity)
planePCDarray, objectPCDarray = ransacPlaneobject(pcd)

csvreader = csv.reader(lane_csv)
rows = []
for row in csvreader:
    rows.append(row)
# get the lane lists from csv
leftlaneX = rows[2]
leftlaneY = rows[3]
rightlaneX = rows[4]
rightlaneY = rows[5]

leftlane3D = []
leftlane3Dx = []
leftlane3Dy = []
leftlane3Dz = []
rightlane3D = []
rightlane3Dx = []
rightlane3Dy = []
rightlane3Dz = []

# P2 (3 x 4) for left eye
P2 = np.matrix([float(x) for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3,4)
R0_rect = np.matrix([float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3,3)
# Add a 1 in bottom-right, reshape to 4 x 4
R0_rect = np.insert(R0_rect,3,values=[0,0,0],axis=0)
R0_rect = np.insert(R0_rect,3,values=[0,0,0,1],axis=1)
Tr_velo_to_cam = np.matrix([float(x) for x in calib[5].strip('\n').split(' ')[1:]]).reshape(3,4)
Tr_velo_to_cam = np.insert(Tr_velo_to_cam,3,values=[0,0,0,1],axis=0)

# read raw data from binary
#scan = np.fromfile(binary, dtype=np.float32).reshape((-1,4))
scan = planePCDarray
points = scan[:, 0:3] # lidar xyz (front, left, up)
points_with_intensity = scan[:, 0:4]

# TODO: use fov filter? 
velo = np.insert(points,3,1,axis=1).T
velo_with_intensity = np.insert(points_with_intensity,4,1,axis=1).T

velo = np.delete(velo,np.where(velo[0,:]<0),axis=1)
cam = P2 * R0_rect * Tr_velo_to_cam * velo


#cam = np.delete(cam,np.where(cam[2,:]<0)[1],axis=1)
new_cam = np.transpose(cam)
new_velo = np.transpose(velo)
new_velo_with_intensity = np.transpose(velo_with_intensity)

new_veloList = list(np.squeeze(np.asarray(new_velo)))
new_veloList_with_intensity = list(np.squeeze(np.asarray(new_velo_with_intensity)))
new_camList = list(np.squeeze(np.asarray(new_cam)))

index = 0

# loop the lidar points
for i in range(len(new_veloList)):
    '''
    veloList[0]: 3D x value
    veloList[1]: 3D y value
    veloList[2]: 3D z value
    '''
    veloList = list(np.squeeze(np.asarray(new_velo[i]).reshape(-1)))
    
    onecamList = list(np.squeeze(np.asarray(new_camList[i]).reshape(-1)))
    
    #print(f"1st: {onecamList[0]} / 2nd: {onecamList[1]} / 3rd: {onecamList[2]}")

    project2Dx= onecamList[0]/onecamList[2]
    project2Dy = onecamList[1]/onecamList[2]

    #print(f"first: {project2Dx}/ second: {project2Dy}")

    index += 1
    
    # consider just the front view
    if veloList[0] >=0:
        # loop the left lane pixels
        for j in range(len(leftlaneX)):
            if (math.floor(project2Dx)==int(leftlaneX[j]) or math.ceil(project2Dx)==int(leftlaneX[j])) and (int(project2Dy)==int(leftlaneY[j]) or int(project2Dy)==int(leftlaneY[j])):
            #if int(project2Dx)==int(leftlaneX[j]) and int(project2Dy)==int(leftlaneY[j]):
                leftlane3D.append([veloList[0], veloList[1], veloList[2]])

        # loop the right lane pixels
        for j in range(len(rightlaneX)):
            if (math.floor(project2Dx)==int(rightlaneX[j]) or math.ceil(project2Dx)==int(rightlaneX[j])) and (int(project2Dy)==int(rightlaneY[j]) or int(project2Dy)==int(rightlaneY[j])):
            #if int(project2Dx)==int(rightlaneX[j]) and int(project2Dy)==int(rightlaneY[j]):
                rightlane3D.append([veloList[0], veloList[1], veloList[2]])

#print(f"overlapped(left): {len(leftlane3D)}")
#print(f"overlapped(right): {len(rightlane3D)}")

leftlane3D = sorted(leftlane3D)
rightlane3D = sorted(rightlane3D)

#make left lane txt file
with open('../../CSV_Communication/2_matched_left.txt', 'w') as left_lane_file:
    for i in range(len(leftlane3D)):
        left_lane_file.write(f"{leftlane3D[i][1]} {leftlane3D[i][0]}\n")

#make right lane txt file
with open('../../CSV_Communication/2_matched_right.txt', 'w') as right_lane_file:
    for i in range(len(rightlane3D)):
        right_lane_file.write(f"{rightlane3D[i][1]} {rightlane3D[i][0]}\n")


# <3D Lane Range Prediction>
# Get the points (leftX1, leftY1, leftX2, leftY2) / (rightX1, rightY1, rightX2, rightY2)
leftMinX = leftlane3D[0][0]
leftMinY = min(leftlane3D[0][1], leftlane3D[-1][1])
leftMaxX = leftlane3D[-1][0]
leftMaxY = max(leftlane3D[0][1], leftlane3D[-1][1])
rightMinX = rightlane3D[0][0]
rightMinY = min(rightlane3D[0][1], rightlane3D[-1][1])
rightMaxX = rightlane3D[-1][0]
rightMaxY = max(rightlane3D[0][1], rightlane3D[-1][1])

leftPointinRange = []
rightPointinRange = []

# instant variables
leftXXX = []
leftYYY = []
leftZZZ = []
rightXXX = []
rightYYY = []
rightZZZ = []

# set the extra range as 2 meters
extraRange = 0.5 

index = 0
# loop the lidar points
for i in range(len(new_veloList_with_intensity)):
    '''
    veloList[0]: 3D x value
    veloList[1]: 3D y value
    veloList[2]: 3D z value
    '''
    veloList = list(np.squeeze(np.asarray(new_velo_with_intensity[i]).reshape(-1)))
    
    #onecamList = list(np.squeeze(np.asarray(new_camList[i]).reshape(-1)))

    project2Dx= onecamList[0]/onecamList[2]
    project2Dy = onecamList[1]/onecamList[2]

    if veloList[0] >= -2:
        # left lane process
        if veloList[0] > leftMinX-extraRange and veloList[0] < leftMaxX+extraRange+20 and veloList[1] > leftMinY -extraRange and veloList[1] < leftMaxY + extraRange and veloList[3]>0.47:
            leftPointinRange.append([veloList[0], veloList[1], veloList[2]])
            leftXXX.append(veloList[0])
            leftYYY.append(veloList[1])
            leftZZZ.append(veloList[2])
        
        # right lane process
        if veloList[0] > rightMinX-extraRange and veloList[0] < rightMaxX+extraRange+20 and veloList[1] > rightMinY-extraRange and veloList[1] < rightMaxY+extraRange and veloList[3]>0.47:
            rightPointinRange.append([veloList[0], veloList[1], veloList[2]])
            rightXXX.append(veloList[0])
            rightYYY.append(veloList[1])
            rightZZZ.append(veloList[2])

# image matching process
left_img_point_cloud = []
right_img_point_cloud = []

# (1) generate point numpy
for i in range(len(leftXXX)):
    left_img_point_cloud.append([leftXXX[i], leftYYY[i], leftZZZ[i]])

for i in range(len(rightXXX)):
    right_img_point_cloud.append([rightXXX[i], rightYYY[i], rightZZZ[i]])

left_img_point_cloud = np.array(left_img_point_cloud)
right_img_point_cloud = np.array(right_img_point_cloud)

#print(left_img_point_cloud.shape)
#print(right_img_point_cloud.shape)

left_velo = np.insert(left_img_point_cloud, 3, 1, axis=1).T
right_velo = np.insert(right_img_point_cloud, 3, 1, axis=1).T

left_cam = P2 * R0_rect * Tr_velo_to_cam * left_velo
right_cam = P2 * R0_rect * Tr_velo_to_cam * right_velo

left_new_cam = np.transpose(left_cam)
left_new_velo = np.transpose(left_velo)

right_new_cam = np.transpose(right_cam)
right_new_velo = np.transpose(right_velo)

left_new_camList = list(np.squeeze(np.asarray(left_new_cam)))
left_new_veloList = list(np.squeeze(np.asarray(left_new_velo)))
right_new_camList = list(np.squeeze(np.asarray(right_new_cam)))
right_new_veloList = list(np.squeeze(np.asarray(right_new_velo)))

png = mpimg.imread(img)
gray = rgb2gray(png).transpose()

img_intensity_lx = []
img_intensity_ly = []
img_intensity_rx = []
img_intensity_ry = []

intensity_threadhold = 0.35
for i in range(len(left_new_veloList)):
    onecamList = list(np.squeeze(np.asarray(left_new_camList[i]).reshape(-1)))

    project2Dx= onecamList[0]/onecamList[2]
    project2Dy = onecamList[1]/onecamList[2]
    
    first_pixel = [math.floor(project2Dx), math.floor(project2Dy)]
    second_pixel = [math.ceil(project2Dx), math.ceil(project2Dy)]
    third_pixel = [math.floor(project2Dx), math.ceil(project2Dy)]
    fourth_pixel = [math.ceil(project2Dx), math.floor(project2Dy)]

    # [1] access to the points on the image array that are matched with inlier points
    if first_pixel[1]>375 or second_pixel[1]>375 or third_pixel[1]>375 or fourth_pixel[1]>375:
        continue
    # [2] get the pixel values of the above and compare it with threshold value
    if gray[first_pixel[0]][first_pixel[1]] >= intensity_threadhold or gray[second_pixel[0]][second_pixel[1]] >= intensity_threadhold or gray[third_pixel[0]][third_pixel[1]] >= intensity_threadhold or gray[fourth_pixel[0]][fourth_pixel[1]] >= intensity_threadhold:
        img_intensity_lx.append(project2Dx)
        img_intensity_ly.append(project2Dy)

for i in range(len(right_new_veloList)):
    onecamList = list(np.squeeze(np.asarray(right_new_camList[i]).reshape(-1)))

    project2Dx= onecamList[0]/onecamList[2]
    project2Dy = onecamList[1]/onecamList[2]
    
    first_pixel = [math.floor(project2Dx), math.floor(project2Dy)]
    second_pixel = [math.ceil(project2Dx), math.ceil(project2Dy)]
    third_pixel = [math.floor(project2Dx), math.ceil(project2Dy)]
    fourth_pixel = [math.ceil(project2Dx), math.floor(project2Dy)]

    # [1] access to the points on the image array that are matched with inlier points
    if first_pixel[1]>374 or second_pixel[1]>374 or third_pixel[1]>374 or fourth_pixel[1]>374 or first_pixel[0]>1240 or second_pixel[0]>1240 or third_pixel[0]>1240 or fourth_pixel[0]>1240:
        continue
    # [2] get the pixel values of the above and compare it with threshold value
    if gray[first_pixel[0]][first_pixel[1]] >= intensity_threadhold or gray[second_pixel[0]][second_pixel[1]] >= intensity_threadhold or gray[third_pixel[0]][third_pixel[1]] >= intensity_threadhold or gray[fourth_pixel[0]][fourth_pixel[1]] >= intensity_threadhold:
        img_intensity_rx.append(project2Dx)
        img_intensity_ry.append(project2Dy)

#--------

leftX_RansacData = np.array(leftXXX).reshape(-1,1)
leftY_RansacData = np.array(leftYYY).reshape(-1,1)
rightX_RansacData = np.array(rightXXX).reshape(-1,1)
rightY_RansacData = np.array(rightYYY).reshape(-1,1)

#leftXXX.insert(0, leftXXX[0]);leftXXX.insert(0,leftXXX[0]);leftXXX.insert(0,leftXXX[0]); leftYYY.insert(0,leftYYY[0]+0.01);leftYYY.insert(0, leftYYY[0]+0.015);leftYYY.insert(0,leftYYY[0]-0.001)
#rightXXX.insert(0, rightXXX[0]);rightXXX.insert(0,rightXXX[0]);rightXXX.insert(0,rightXXX[0]); rightYYY.insert(0,rightYYY[0]);rightYYY.insert(0, rightYYY[0]);rightYYY.insert(0,rightYYY[0]); rightXXX.insert(0, rightXXX[0]);rightXXX.insert(0,rightXXX[0]);rightXXX.insert(0,rightXXX[0]); rightYYY.insert(0,rightYYY[0]);rightYYY.insert(0, rightYYY[0]);rightYYY.insert(0,rightYYY[0]);rightXXX.insert(0, rightXXX[0]);rightXXX.insert(0,rightXXX[0]);rightXXX.insert(0,rightXXX[0]); rightYYY.insert(0,rightYYY[0]);rightYYY.insert(0, rightYYY[0]);rightYYY.insert(0,rightYYY[0]); rightXXX.insert(0, rightXXX[0]);rightXXX.insert(0,rightXXX[0]);rightXXX.insert(0,rightXXX[0]); rightYYY.insert(0,rightYYY[0]);rightYYY.insert(0, rightYYY[0]);rightYYY.insert(0,rightYYY[0])
#leftXXX.append(leftXXX[0]);leftXXX.append(leftXXX[0]);leftXXX.append(leftXXX[0]); leftYYY.append(leftYYY[0]+0.01);leftYYY.append(leftYYY[0]+0.015);leftYYY.append(leftYYY[0]-0.001)


# Ransac data preprocessing
test_leftX = np.array(leftXXX)
test_leftY = np.array(leftYYY)

test_rightX = np.array(rightXXX)
test_rightY = np.array(rightYYY)

# [Quadratic RANSAC Process] left lane
left_ransac = RANSACRegressor(PolynomialRegression(degree=2),
                         residual_threshold=1 * np.std(test_leftY),
                         random_state=0)
left_ransac.fit(np.expand_dims(test_leftX, axis=1), test_leftY)
left_inlier_mask = left_ransac.inlier_mask_
left_y_hat = left_ransac.predict(np.expand_dims(test_leftX, axis=1))
ransac_plot(test_leftX, test_leftY, left_y_hat, left_inlier_mask)

# [Quadratic RANSAC Process] left lane
right_ransac = RANSACRegressor(PolynomialRegression(degree=2),
                         residual_threshold= 1.3 * np.std(test_rightY),
                         random_state=0)
right_ransac.fit(np.expand_dims(test_rightX, axis=1), test_rightY)
right_inlier_mask = right_ransac.inlier_mask_
right_y_hat = right_ransac.predict(np.expand_dims(test_rightX, axis=1))
ransac_plot(test_rightX, test_rightY, right_y_hat, right_inlier_mask)

# get u,v,z
cam[:2] /= cam[2,:]
# do projection staff
plt.figure(figsize=(12,5),dpi=96,tight_layout=True)
png = mpimg.imread(img)
IMG_H,IMG_W,_ = png.shape

plt.plot(img_intensity_lx, img_intensity_ly, color="blue", linewidth=3)
plt.plot(img_intensity_rx, img_intensity_ry, color="blue", linewidth=3)

# restrict canvas in range
plt.axis([0,IMG_W,IMG_H,0])
plt.imshow(png)
# filter point out of canvas
u,v,z = cam
u_out = np.logical_or(u<0, u>IMG_W)
v_out = np.logical_or(v<0, v>IMG_H)
outlier = np.logical_or(u_out, v_out)
cam = np.delete(cam,np.where(outlier),axis=1)
# generate color map from depth
u,v,z = cam
plt.scatter([u],[v],c=[z],cmap='rainbow_r',alpha=0.5,s=2)
plt.title(name)
plt.savefig(f'../../../../dataset/projected_output/{name}.png',bbox_inches='tight')
plt.show()
