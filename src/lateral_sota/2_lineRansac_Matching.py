import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import csv
from ransacPlaneobject import *
import math

from sklearn import linear_model, datasets

def ransacPlot(xArray, yArray, side, maximum_distance = 50):
    # Robustly fit linear model with RANSAC algorithm
    ransac = linear_model.RANSACRegressor()
    ransac.fit(xArray, yArray)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # Predict data of estimated models: line_X is X coordinates / line_y_ransac is Y coordinates
    #line_X = np.arange(xArray.min(), xArray.max())[:, np.newaxis]
    line_X = np.arange(0, maximum_distance)[:, np.newaxis]
    line_y_ransac = ransac.predict(line_X)

    
    # prepare to write in text file
    if side == 'left':
        with open('../../CSV_Communication/left_lane_Ransac.txt', 'w') as left_lane_file:
            for ii, v in enumerate(line_X):
                left_lane_file.write(f"{-float(line_y_ransac[ii])} {ii}\n")


    else:
        with open('../../CSV_Communication/right_lane_Ransac.txt', 'w') as right_lane_file:
            for ii, v in enumerate(line_X):
                right_lane_file.write(f"{-float(line_y_ransac[ii])} {ii}\n")


    lw = 2
    plt.scatter(
        xArray[inlier_mask], yArray[inlier_mask], color="yellowgreen", marker=".", label="Inliers"
    )
    plt.scatter(
        xArray[outlier_mask], yArray[outlier_mask], color="gold", marker=".", label="Outliers"
    )
    #plt.plot(line_X, line_y, color="navy", linewidth=lw, label="Linear regressor")
    plt.plot(
        line_X,
        line_y_ransac,
        color="cornflowerblue",
        linewidth=lw,
        label="RANSAC regressor",
    )
    plt.legend(loc="lower right")
    plt.xlabel("Input")
    plt.ylabel("Response")
    plt.show()

        

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

print(points.shape)
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
print("YOUNGIL")

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
            if int(project2Dx)==int(leftlaneX[j]) and int(project2Dy)==int(leftlaneY[j]):
                leftlane3D.append([veloList[0], veloList[1], veloList[2]])

        # loop the right lane pixels
        for j in range(len(rightlaneX)):
             if int(project2Dx)==int(rightlaneX[j]) and int(project2Dy)==int(rightlaneY[j]):
                rightlane3D.append([veloList[0], veloList[1], veloList[2]])

    print(f"overlapped(left): {len(leftlane3D)}")
    print(f"overlapped(right): {len(rightlane3D)}")

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

print("----YOUGNIL TEXT")

print(leftMaxX)
print(rightMaxX)

print("YOUGNIL TEXT----")

leftPointinRange = []
rightPointinRange = []

# instant variables
leftXXX = []
leftYYY = []
rightXXX = []
rightYYY = []

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
        if veloList[0] > leftMinX-extraRange and veloList[0] < leftMaxX+extraRange+20 and veloList[1] > leftMinY -extraRange and veloList[1] < leftMaxY + extraRange and veloList[3]>0.4:
            leftPointinRange.append([veloList[0], veloList[1], veloList[2]])
            leftXXX.append(veloList[0])
            leftYYY.append(veloList[1])
        
        # right lane process
        if veloList[0] > rightMinX-extraRange and veloList[0] < rightMaxX+extraRange+20 and veloList[1] > rightMinY-extraRange and veloList[1] < rightMaxY+extraRange and veloList[3]>0.4:
            rightPointinRange.append([veloList[0], veloList[1], veloList[2]])
            rightXXX.append(veloList[0])
            rightYYY.append(veloList[1])



leftX_RansacData = np.array(leftXXX).reshape(-1,1)
leftY_RansacData = np.array(leftYYY).reshape(-1,1)
ransacPlot(leftX_RansacData, leftY_RansacData,'left', rightMaxX)

rightX_RansacData = np.array(rightXXX).reshape(-1,1)
rightY_RansacData = np.array(rightYYY).reshape(-1,1)
ransacPlot(rightX_RansacData, rightY_RansacData,'right', rightMaxX)
'''
# Robustly fit linear model with RANSAC algorithm
ransac = linear_model.RANSACRegressor()
ransac.fit(leftX_RansacData, leftY_RansacData)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# Predict data of estimated models
line_X = np.arange(leftX_RansacData.min(), leftX_RansacData.max())[:, np.newaxis]
line_y_ransac = ransac.predict(line_X)

lw = 2
plt.scatter(
    leftX_RansacData[inlier_mask], leftY_RansacData[inlier_mask], color="yellowgreen", marker=".", label="Inliers"
)
plt.scatter(
    leftX_RansacData[outlier_mask], leftY_RansacData[outlier_mask], color="gold", marker=".", label="Outliers"
)
#plt.plot(line_X, line_y, color="navy", linewidth=lw, label="Linear regressor")
plt.plot(
    line_X,
    line_y_ransac,
    color="cornflowerblue",
    linewidth=lw,
    label="RANSAC regressor",
)
plt.legend(loc="lower right")
plt.xlabel("Input")
plt.ylabel("Response")
plt.show()
'''

# get u,v,z
cam[:2] /= cam[2,:]
# do projection staff
plt.figure(figsize=(12,5),dpi=96,tight_layout=True)
png = mpimg.imread(img)
IMG_H,IMG_W,_ = png.shape

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
