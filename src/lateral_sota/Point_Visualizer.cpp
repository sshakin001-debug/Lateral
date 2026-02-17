# include <iostream>
# include <string>
#include <math.h>

#include <ros/ros.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/io/pcd_io.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui.hpp>

//#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

using namespace cv;
typedef pcl::PointCloud<pcl::PointXYZI> PointCloud;
using namespace std;

/*
projection material
https://github.com/azureology/kitti-velo2cam/blob/master/proj_velo2cam.py
*/

cv::Mat img(720,1280,CV_8UC3, cv::Scalar(255,255,255));


int main(int argc, char** argv)
{
  ros::init (argc, argv, "my_pcl_tutorial");
  ros::NodeHandle nh;
  double x_cloud; double y_cloud; double z_cloud;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr PCD_cloud_o(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr PCD_cloud_p(new pcl::PointCloud<pcl::PointXYZI>);
  
  ros::Publisher pub1 = nh.advertise<PointCloud> ("points_plane", 1);
  ros::Publisher pub2 = nh.advertise<PointCloud> ("points_obstacle", 1);
  ros::Publisher pub3 = nh.advertise<sensor_msgs::Image> ("projected_obstacle", 1);
  ros::Publisher pub4 = nh.advertise<sensor_msgs::Image> ("projected_plane", 1);

  ros::Publisher marker_pub_left = nh.advertise<visualization_msgs::Marker>("visualization_marker_left", 10);
  ros::Publisher marker_pub_right = nh.advertise<visualization_msgs::Marker>("visualization_marker_right", 10);

  //ros::Publisher pub1 = nh.advertise<sensor_msgs::ImagePtr> ("image", 1);
  //pcl::io::loadPCDFile<pcl::PointXYZI> ("/home/kaai/chicago_ws/src/first_pkg/src/Kitti_File/pcd/1039.pcd", *cloud);
  //pcl::io::loadPCDFile<pcl::PointXYZI> ("/home/kaai/chicago_ws/src/first_pkg/src/KITTI/pcd/0000000215.pcd", *cloud);
  pcl::io::loadPCDFile<pcl::PointXYZI> ("/home/kaai/dataset/training/velodyne/000007.pcd", *cloud);
  //pcl::io::loadPCDFile<pcl::PointXYZI> ("/home/kaai/chicago_ws/src/first_pkg/Lateral/outliers.pcd", *cloud);
  Mat color_img = imread("/home/kaai/dataset/training/image_2/000007.png",IMREAD_COLOR);

  //000021.png  2090899988.pcd

  //베스트 : 000030.png  2101599309.pcd
  //RANSAC Section
  pcl::SACSegmentation<pcl::PointXYZI> seg;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(100);
  seg.setDistanceThreshold(0.3);
  seg.setInputCloud(cloud);
  seg.segment(*inliers, *coefficients);

  if(inliers->indices.size() == 0)            // 0일때, 우리는 우리 데이터에 적합한 모델을 찾지 못했다는 것을 의미한다.
  {
      std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
  }

  for(int index : inliers->indices)       //inliers가 가지는 모든 인덱스 값에 해당하는 cloud값을 planeCloud에 넣어준다.
  {
      PCD_cloud_p->points.push_back(cloud->points[index]);
  }

  pcl::ExtractIndices<pcl::PointXYZI> extract;
  extract.setInputCloud(cloud);       //이 reference cloud 에서 inliers에 해당하는 모든 포인트가 사라져서 obstCloud 만 남게된다.
  extract.setIndices(inliers);
  extract.setNegative(true);
  extract.filter(*PCD_cloud_o);

  sensor_msgs::PointCloud2 ROS_cloud_o;
  sensor_msgs::PointCloud2 ROS_cloud_p;

/*
  //toPCL을 하여 fromPCL을 사용하기위한 선언
  pcl::PCLPointCloud2 * PCL_cloud_o(new pcl::PCLPointCloud2);
  pcl::PCLPointCloud2 * PCL_cloud_p(new pcl::PCLPointCloud2);

  pcl::toPCLPointCloud2(*PCD_cloud_o, *PCL_cloud_o);
  pcl::toPCLPointCloud2(*PCD_cloud_p, *PCL_cloud_p);
  pcl_conversions::fromPCL(*PCL_cloud_o, ROS_cloud_o);
  pcl_conversions::fromPCL(*PCL_cloud_p, ROS_cloud_p);
*/
  pcl::toROSMsg(*PCD_cloud_o, ROS_cloud_o);
  pcl::toROSMsg(*PCD_cloud_p, ROS_cloud_p);
  
  ROS_cloud_o.header.frame_id = "livox_frame";
  ROS_cloud_p.header.frame_id = "livox_frame";




  //img=cv::imread("/home/kaai/chicago_ws/src/first_pkg/src/Kitti_File/001013.png");

  //sensor_msgs::ImagePtr msg1 = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();

  //Mat color_img = imread("/home/kaai/chicago_ws/src/first_pkg/src/Kitti_File/image/001039.png",IMREAD_COLOR);
  

  

  cv::Mat p(3,4,cv::DataType<double>::type);
  cv::Mat r(3,3,cv::DataType<double>::type);
  cv::Mat tr(3,4,cv::DataType<double>::type);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// 카메라 projection matrix (3 X 4)
  p.at<double>(0,0) = 7.215377000000e+02;    p.at<double>(0,1) = 0.000000000000e+00;    p.at<double>(0,2) = 6.095593000000e+02;    p.at<double>(0,3) = 0.000000000000e+0;
  p.at<double>(1,0) = 0.00000000;    p.at<double>(1,1) = 7.215377000000e+02;    p.at<double>(1,2) = 1.728540000000e+02;    p.at<double>(1,3) = 0.000000000000e+0;  
  p.at<double>(2,0) = 0.00000000;    p.at<double>(2,1) = 0.00000000;    p.at<double>(2,2) = 1.00000000;    p.at<double>(2,3) = 0.000000000000e+0;  

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Rotation matrix (3 X 3)
  r.at<double>(0,0) = 9.999239000000e-01;    r.at<double>(0,1) = 9.837760000000e-03;    r.at<double>(0,2) = -7.445048000000e-03;
  r.at<double>(1,0) = -9.869795000000e-033;    r.at<double>(1,1) = 9.999421000000e-01;    r.at<double>(1,2) = -4.278459000000e-03;
  r.at<double>(2,0) = 7.402527000000e-03;    r.at<double>(2,1) = 4.351614000000e-03;    r.at<double>(2,2) = 9.999631000000e-01;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Tr_velo_to_cam (3 X 4)
  tr.at<double>(0,0) = 7.533745000000e-03;    tr.at<double>(0,1) = -9.999714000000e-01;    tr.at<double>(0,2) = -6.166020000000e-04;    tr.at<double>(0,3) = -4.069766000000e-03;  // x축 (좌측이동시 -)
  tr.at<double>(1,0) = 1.480249000000e-02;    tr.at<double>(1,1) = 7.280733000000e-04;    tr.at<double>(1,2) = -9.998902000000e-01;    tr.at<double>(1,3) = -7.631618000000e-02;  // y축 이동(위로 이동시 -)
  tr.at<double>(2,0) = 9.998621000000e-01;    tr.at<double>(2,1) = 7.523790000000e-03;    tr.at<double>(2,2) = 1.480755000000e-02;    tr.at<double>(2,3) = -2.717806000000e-01;  //확대 축소 (줌인(-))
  
  
  cv::Mat X(4,1,cv::DataType<double>::type);
  cv::Mat Y(3,1,cv::DataType<double>::type);
  cv::Mat Z(4,1,cv::DataType<double>::type);
  cv::Mat F(3,1,cv::DataType<double>::type);
  cv::Mat visImg_p = color_img.clone();
  cv::Mat visImg_o = color_img.clone();
  cv::Mat overlay_p = visImg_p.clone();
  cv::Mat overlay_o = visImg_o.clone();

  cv::Point *pt_Object = new cv::Point;
  cv::Point *pt_Lane = new cv::Point;
  cv::Point pt_o;
  cv::Point pt_p;
  float object_lidar_array[3]; // for getting 3D points for object
  float lane_lidar_array[3];   // for getting 3D points for lane

  int object1_x = 407;
  int object1_y = 297;
  int lane1_x = 494;
  int lane1_y = 297;

  // 단순히 x, y 픽셀 평행 이동 
  double offset_x = 0.00;
  double offset_y = 0.00;
  int number = 0;
  int left_lane_all_x[] = {586 ,586 ,585 ,584 ,583 ,583 ,582 ,581 ,580 ,580 ,579 ,578 ,577 ,577 ,576 ,575 ,574 ,574 ,573 ,572 ,571 ,571 ,570 ,569 ,568 ,568 ,567 ,566 ,565 ,565 ,564 ,563 ,562 ,562 ,561 ,560 ,559 ,558 ,558 ,557 ,556 ,555 ,555 ,554 ,553 ,552 ,551 ,551 ,550 ,549 ,548 ,547 ,547 ,546 ,545 ,544 ,543 ,543 ,542 ,541 ,540 ,539 ,539 ,538 ,537 ,536 ,535 ,535 ,534 ,533 ,532 ,531 ,530 ,530 ,529 ,528 ,527 ,526 ,526 ,525 ,524 ,523 ,522 ,521 ,521 ,520 ,519 ,518 ,517 ,516 ,515 ,515 ,514 ,513 ,512 ,511 ,510 ,510 ,509 ,508 ,507 ,506 ,505 ,504 ,503 ,503 ,502 ,501 ,500 ,499 ,498 ,497 ,497 ,496 ,495 ,494 ,493 ,492 ,491 ,490 ,489 ,489 ,488 ,487 ,486 ,485 ,484 ,483 ,482 ,481 ,481 ,480 ,479 ,478 ,477 ,476 ,475 ,474 ,473 ,472 ,472 ,471 ,470 ,469 ,468 ,467 ,466 ,465 ,464 ,463 ,462 ,461 ,461 ,460 ,459 ,458 ,457 ,456 ,455 ,454 ,453 ,452 ,451 ,450 ,449 ,448 ,447 ,446 ,446 ,445 ,444 ,443 ,442 ,441 ,440 ,439 ,438 ,437 ,436 ,435 ,434 ,433 ,432 ,431 ,431 };
  int left_lane_all_y[] = {191 ,192 ,193 ,194 ,195 ,196 ,197 ,198 ,199 ,200 ,201 ,202 ,203 ,204 ,205 ,206 ,207 ,208 ,209 ,210 ,211 ,212 ,213 ,214 ,215 ,216 ,217 ,218 ,219 ,220 ,221 ,222 ,223 ,224 ,225 ,226 ,227 ,228 ,229 ,230 ,231 ,232 ,233 ,234 ,235 ,236 ,237 ,238 ,239 ,240 ,241 ,242 ,243 ,244 ,245 ,246 ,247 ,248 ,249 ,250 ,251 ,252 ,253 ,254 ,255 ,256 ,257 ,258 ,259 ,260 ,261 ,262 ,263 ,264 ,265 ,266 ,267 ,268 ,269 ,270 ,271 ,272 ,273 ,274 ,275 ,276 ,277 ,278 ,279 ,280 ,281 ,282 ,283 ,284 ,285 ,286 ,287 ,288 ,289 ,290 ,291 ,292 ,293 ,294 ,295 ,296 ,297 ,298 ,299 ,300 ,301 ,302 ,303 ,304 ,305 ,306 ,307 ,308 ,309 ,310 ,311 ,312 ,313 ,314 ,315 ,316 ,317 ,318 ,319 ,320 ,321 ,322 ,323 ,324 ,325 ,326 ,327 ,328 ,329 ,330 ,331 ,332 ,333 ,334 ,335 ,336 ,337 ,338 ,339 ,340 ,341 ,342 ,343 ,344 ,345 ,346 ,347 ,348 ,349 ,350 ,351 ,352 ,353 ,354 ,355 ,356 ,357 ,358 ,359 ,360 ,361 ,362 ,363 ,364 ,365 ,366 ,367 ,368 ,369 ,370 ,371 ,372 ,373 ,374 ,374 };
  int right_lane_all_x[] = {616 ,617 ,618 ,620 ,621 ,622 ,623 ,625 ,626 ,627 ,628 ,630 ,631 ,632 ,633 ,635 ,636 ,637 ,639 ,640 ,641 ,642 ,644 ,645 ,646 ,647 ,649 ,650 ,651 ,652 ,654 ,655 ,656 ,658 ,659 ,660 ,661 ,663 ,664 ,665 ,666 ,668 ,669 ,670 ,671 ,673 ,674 ,675 ,677 ,678 ,679 ,680 ,682 ,683 ,684 ,685 ,687 ,688 ,689 ,691 ,692 ,693 ,694 ,696 ,697 ,698 ,699 ,701 ,702 ,703 ,705 ,706 ,707 ,708 ,710 ,711 ,712 ,714 ,715 ,716 ,717 ,719 ,720 ,721 ,722 ,724 ,725 ,726 ,728 ,729 ,730 ,731 ,733 ,734 ,735 ,737 ,738 ,739 ,740 ,742 ,743 ,744 ,746 ,747 ,748 ,749 ,751 ,752 ,753 ,755 ,756 ,757 ,758 ,760 ,761 ,762 ,764 ,765 ,766 ,767 ,769 ,770 ,771 ,773 ,774 ,775 ,777 ,778 ,779 ,780 ,782 ,783 ,784 ,786 ,787 ,788 ,789 ,791 ,792 ,793 ,795 ,796 ,797 ,799 ,800 ,801 ,802 ,804 ,805 ,806 ,808 ,809 ,810 ,811 ,813 ,814 ,815 ,817 ,818 ,819 ,821 ,822 ,823 ,824 ,826 ,827 ,828 ,830 ,831 ,832 ,834 ,835 ,836 ,837 ,839 ,840 ,841 ,843 ,844 ,845 ,847 ,848 ,849 ,851 ,851 };
  int right_lane_all_y[] = {191 ,192 ,193 ,194 ,195 ,196 ,197 ,198 ,199 ,200 ,201 ,202 ,203 ,204 ,205 ,206 ,207 ,208 ,209 ,210 ,211 ,212 ,213 ,214 ,215 ,216 ,217 ,218 ,219 ,220 ,221 ,222 ,223 ,224 ,225 ,226 ,227 ,228 ,229 ,230 ,231 ,232 ,233 ,234 ,235 ,236 ,237 ,238 ,239 ,240 ,241 ,242 ,243 ,244 ,245 ,246 ,247 ,248 ,249 ,250 ,251 ,252 ,253 ,254 ,255 ,256 ,257 ,258 ,259 ,260 ,261 ,262 ,263 ,264 ,265 ,266 ,267 ,268 ,269 ,270 ,271 ,272 ,273 ,274 ,275 ,276 ,277 ,278 ,279 ,280 ,281 ,282 ,283 ,284 ,285 ,286 ,287 ,288 ,289 ,290 ,291 ,292 ,293 ,294 ,295 ,296 ,297 ,298 ,299 ,300 ,301 ,302 ,303 ,304 ,305 ,306 ,307 ,308 ,309 ,310 ,311 ,312 ,313 ,314 ,315 ,316 ,317 ,318 ,319 ,320 ,321 ,322 ,323 ,324 ,325 ,326 ,327 ,328 ,329 ,330 ,331 ,332 ,333 ,334 ,335 ,336 ,337 ,338 ,339 ,340 ,341 ,342 ,343 ,344 ,345 ,346 ,347 ,348 ,349 ,350 ,351 ,352 ,353 ,354 ,355 ,356 ,357 ,358 ,359 ,360 ,361 ,362 ,363 ,364 ,365 ,366 ,367 ,368 ,369 ,370 ,371 ,372 ,373 ,374 ,374 };

  int len_lane_pixel = sizeof(left_lane_all_x)/ sizeof(left_lane_all_x[0]);
  cout << len_lane_pixel << endl;
  ROS_INFO("%d", len_lane_pixel);
  float * left_lane_point_x = new float[100];
  float * left_lane_point_y = new float[100];
  float * left_lane_point_z = new float[100]; 

  float * right_lane_point_x = new float[100];
  float * right_lane_point_y = new float[100];
  float * right_lane_point_z = new float[100];

  int count_left = 0;
  int count_right = 0;

  

  // [access point on plane] ------------------------------------------
  for(auto it=PCD_cloud_p->begin(); it!=PCD_cloud_p->end(); ++it) {
    if(it->x >=0){

    X.at<double>(0,0) = it->x; X.at<double>(1,0) = it->y; X.at<double>(2,0) = it->z; X.at<double>(3,0) = 1;
    Y = r * tr * X;
    Z.at<double>(0,0) = Y.at<double>(0,0); Z.at<double>(1,0) = Y.at<double>(1,0); Z.at<double>(2,0) = Y.at<double>(2,0); Z.at<double>(3,0) = 1;
    F = p * Z;
    
    pt_p.x = F.at<double>(0,0) / F.at<double>(0,2);
    pt_p.y = F.at<double>(1,0) / F.at<double>(0,2);
    //std::cout << it->z <<std::endl;

    float val = it->x;
    float intensity = it-> x;
    //ROS_INFO("intensity : %f", intensity);
    
    float maxval = 10;
    int green = std::min(255, (int)(255*abs((val - maxval)/maxval)));
    //std::cout<<red<<std::endl;
    int red = std::min(255, (int)(255*(1-abs((val - maxval)/maxval))));

    
    
    // 좌측 Lane 우측 Lane 에 대해서 오버랩 부분 처리하기
    for(int i = 0; i < len_lane_pixel; i+=1)
    {
      if((int(pt_o.x) >= left_lane_all_x[i])&& (int(pt_o.x) <= left_lane_all_x[i]) && (int(pt_o.y) <= left_lane_all_y[i]) && (int(pt_o.y) >= left_lane_all_y[i]))
      //if((int(pt_o.x) == left_lane_all_x[i]) && (int(pt_o.y) == left_lane_all_y[i]))
      {
        left_lane_point_x[count_left] = X.at<double>(0,0);
        left_lane_point_y[count_left] = X.at<double>(1,0);
        left_lane_point_z[count_left] = X.at<double>(2,0);
        count_left = count_left + 1;
        //ROS_INFO("left %d : %f, %f, %f", count_left-1, X.at<double>(0,0), X.at<double>(1,0), X.at<double>(2,0));
      }
    }
    for(int i = 0; i < len_lane_pixel; i+=1)
    {
      if((int(pt_o.x) == right_lane_all_x[i]) && (int(pt_o.y) == right_lane_all_y[i]))
      {
        right_lane_point_x[count_right] = X.at<double>(0,0);
        right_lane_point_y[count_right] = X.at<double>(1,0);
        right_lane_point_z[count_right] = X.at<double>(2,0);
        count_right = count_right + 1;
        //ROS_INFO("right %d : %f, %f, %f", count_right-1, X.at<double>(0,0), X.at<double>(1,0), X.at<double>(2,0));
      }
    }
    // [put the point on Image]
    cv::circle(overlay_p, pt_p, 1., cv::Scalar(it->intensity , red,green), -1);
    }
  }
 
  // [access point on plane] ------------------------------------------
  for(auto it=PCD_cloud_o->begin(); it!=PCD_cloud_o->end(); ++it) {
    if(it->x >=0){

    X.at<double>(0,0) = it->x; X.at<double>(1,0) = it->y; X.at<double>(2,0) = it->z; X.at<double>(3,0) = 1;
    Y = r * tr * X;
    Z.at<double>(0,0) = Y.at<double>(0,0); Z.at<double>(1,0) = Y.at<double>(1,0); Z.at<double>(2,0) = Y.at<double>(2,0); Z.at<double>(3,0) = 1;
    F = p * Z;
    
    pt_o.x = F.at<double>(0,0) / F.at<double>(0,2);
    pt_o.y = F.at<double>(1,0) / F.at<double>(0,2);
    //std::cout << it->z <<std::endl;

    float val = it->x;
    float intensity = it-> x;
    //ROS_INFO("intensity : %f", intensity);
    
    float maxval = 10;
    int green = std::min(255, (int)(255*abs((val - maxval)/maxval)));
    //std::cout<<red<<std::endl;
    int red = std::min(255, (int)(255*(1-abs((val - maxval)/maxval))));

    // [put the point on Image]
    cv::circle(overlay_o, pt_o, 1., cv::Scalar(it->intensity , red,green), -1);
    
    }
    }
  
  
  ROS_INFO("count_right : %d",count_right);
  // Left Point 정렬
  int minimum_L = 0;
  for(int i = 0; i < count_left -1 ; i++)
  {
    minimum_L = i;
    for(int j = i ; j < count_left ; j++)
    {
      if (left_lane_point_x[minimum_L] > left_lane_point_x[j]){
        minimum_L = j;
      }
    }
    float temp_x = left_lane_point_x[i];
    left_lane_point_x[i] = left_lane_point_x[minimum_L];
    left_lane_point_x[minimum_L] = temp_x;

    float temp_y = left_lane_point_y[i];
    left_lane_point_y[i] = left_lane_point_y[minimum_L];
    left_lane_point_y[minimum_L] = temp_y;

    float temp_z = left_lane_point_z[i];
    left_lane_point_z[i] = left_lane_point_z[minimum_L];
    left_lane_point_z[minimum_L] = temp_z;

    //ROS_INFO("%f", left_lane_point_x[i]);
  }
  // Right Point 정렬
  int minimum_R = 0;
  for(int i = 0; i < count_right -1 ; i++)
  {
    minimum_R = i;
    for(int j = i ; j < count_right ; j++)
    {
      if (right_lane_point_x[minimum_R] > right_lane_point_x[j]){
        minimum_R = j;
      }
    }
    float temp_x = right_lane_point_x[i];
    right_lane_point_x[i] = right_lane_point_x[minimum_R];
    right_lane_point_x[minimum_R] = temp_x;

    float temp_y = right_lane_point_y[i];
    right_lane_point_y[i] = right_lane_point_y[minimum_R];
    right_lane_point_y[minimum_R] = temp_y;

    float temp_z = right_lane_point_z[i];
    right_lane_point_z[i] = right_lane_point_z[minimum_R];
    right_lane_point_z[minimum_R] = temp_z;

    //ROS_INFO("%f", right_lane_point_x[i]);
  }
  // [다음 노드로 전달하기 위해서]
  // 최소 거리를 구하기 위한 leftx 좌표 찍기
  cout << "leftx = [";
  for (int i = 0 ; i < count_left; i++)
  {
    if (i == count_left-1)
    {
      cout << left_lane_point_x[i] << "]" << endl;
      break;
    }
    cout << left_lane_point_x[i] << ",";
  }
  // 최소 거리를 구하기 위한 lefty 좌표 찍기
  cout << "lefty = [";
  for (int i = 0 ; i < count_left; i++)
  {
    if (i == count_left-1)
    {
      cout << left_lane_point_y[i] << "]" << endl;
      break;
    }
    cout << left_lane_point_y[i] << ",";
  }
  // 최소 거리를 구하기 위한 rightx 좌표 찍기
  cout << "rightx = [";
  for (int i = 0 ; i < count_right; i++)
  {
    if (i == count_right-1)
    {
      cout << right_lane_point_x[i] << "]" << endl;
      break;
    }
    cout << right_lane_point_x[i] << ",";
  }
  // 최소 거리를 구하기 위한 righty 좌표 찍기
  cout << "righty = [";
  for (int i = 0 ; i < count_right; i++)
  {
    if (i == count_right-1)
    {
      cout << right_lane_point_y[i] << "]" << endl;
      break;
    }
    cout << right_lane_point_y[i] << ",";
  }
  // z 의 평균 값을 측정하는 코드
  float average_height = 0;
  float sum = 0;
  for (int i = 0 ; i < count_left; i++)
  {
    sum += right_lane_point_z[i];
  }
  for (int i = 0 ; i < count_right; i++)
  {
    sum += right_lane_point_z[i];
  }
  average_height = sum/(count_left+count_right);
  cout << "height = "<< average_height << ";" << "\n";

  
  float opacity = 0.7;
  cv::addWeighted(overlay_p, opacity, visImg_p, 1-opacity, 0, visImg_p);
  cv::addWeighted(overlay_o, opacity, visImg_o, 1-opacity, 0, visImg_o);

  float distance = fabs(object_lidar_array[1] - lane_lidar_array[1]);
  
  //ROS_INFO("Distance is : %f ", distance);

  //ROS_INFO("Object_lidar_array : %f, %f, %f", object_lidar_array[0], object_lidar_array[1], object_lidar_array[2]);
  //ROS_INFO("Lane_lidar_array : %f, %f, %f", lane_lidar_array[0], lane_lidar_array[1], lane_lidar_array[2]);

  //string string_distance = to_string(distance);
  //string_distance.append("M");
  //cv::Point Distance_Point;
  //cv::Mat img(720,1280,CV_8UC3, cv::Scalar(255,255,255));

  //Distance_Point.x = lane1_x - 40;
  //Distance_Point.y = lane1_y - 10;
  //cv::putText(visImg_p, string_distance, Distance_Point, 3, 0.6, CV_RGB(255, 255, 255));



  ros::Rate loop_rate(4);
  /*
  imshow("color_img_plane", visImg_p);
  imshow("color_img_obstacle", visImg_o);
  waitKey(0);
  */

  sensor_msgs::ImagePtr img_msg_obs = cv_bridge::CvImage(std_msgs::Header(), "bgr8", visImg_o).toImageMsg();
  sensor_msgs::ImagePtr img_msg_plane = cv_bridge::CvImage(std_msgs::Header(), "bgr8", visImg_p).toImageMsg();

  
  while (nh.ok())
  {
    //msg->header.stamp = ros::Time::now().toNSec();
    pub1.publish (ROS_cloud_p);
    pub2.publish (ROS_cloud_o);
    pub3.publish(img_msg_obs);
    pub4.publish(img_msg_plane);

    visualization_msgs::Marker left_points, left_line_strip, left_line_list;
    left_points.header.frame_id = left_line_strip.header.frame_id = left_line_list.header.frame_id = "livox_frame";
    left_points.header.stamp = left_line_strip.header.stamp = left_line_list.header.stamp = ros::Time::now();
    left_points.ns = left_line_strip.ns = left_line_list.ns = "points_and_lines";
    left_points.action = left_line_strip.action = left_line_list.action = visualization_msgs::Marker::ADD;
    left_points.pose.orientation.w = left_line_strip.pose.orientation.w = left_line_list.pose.orientation.w = 1.0;
    left_points.id = 0;
    left_line_strip.id = 1;
    left_line_list.id = 2;
    left_points.type = visualization_msgs::Marker::POINTS;  // POINTS 를 SPHERE 로 바꿔보자 나중에
    left_line_strip.type = visualization_msgs::Marker::LINE_STRIP;
    left_line_list.type = visualization_msgs::Marker::LINE_LIST;

    left_points.scale.x = 0.2;
    left_points.scale.y = 0.2;
    left_line_strip.scale.x = 0.1;
    left_line_list.scale.x = 0.1;

    // Points are green
    left_points.color.g = 1.0f;
    left_points.color.a = 1.0;

    // Line strip is blue
    left_line_strip.color.b = 1.0;
    left_line_strip.color.a = 1.0;
    
    // Line list is red
    left_line_list.color.r = 1.0;
    left_line_list.color.a = 1.0;
    // Create the vertices for the points and lines
    for (uint32_t i = 0; i < count_left; ++i)
    {
      geometry_msgs::Point p_l;
      p_l.x = left_lane_point_x[i];
      p_l.y = left_lane_point_y[i];
      p_l.z = left_lane_point_z[i];
      left_points.points.push_back(p_l);
      left_line_strip.points.push_back(p_l); 
      // The line list needs two points for each line
      left_line_list.points.push_back(p_l);
      p_l.z += 1.0;
      left_line_list.points.push_back(p_l);
  }

    visualization_msgs::Marker right_points, right_line_strip, right_line_list;
    right_points.header.frame_id = right_line_strip.header.frame_id = right_line_list.header.frame_id = "livox_frame";
    right_points.header.stamp = right_line_strip.header.stamp = right_line_list.header.stamp = ros::Time::now();
    right_points.ns = right_line_strip.ns = right_line_list.ns = "points_and_lines";
    right_points.action = right_line_strip.action = right_line_list.action = visualization_msgs::Marker::ADD;
    right_points.pose.orientation.w = right_line_strip.pose.orientation.w = right_line_list.pose.orientation.w = 1.0;
    right_points.id = 0;
    right_line_strip.id = 1;
    right_line_list.id = 2;
    right_points.type = visualization_msgs::Marker::POINTS;  // POINTS 를 SPHERE 로 바꿔보자 나중에
    right_line_strip.type = visualization_msgs::Marker::LINE_STRIP;
    right_line_list.type = visualization_msgs::Marker::LINE_LIST;

    right_points.scale.x = 0.2;
    right_points.scale.y = 0.2;
    right_line_strip.scale.x = 0.1;
    right_line_list.scale.x = 0.1;

    // Points are green
    right_points.color.g = 1.0f;
    right_points.color.a = 1.0;

    // Line strip is blue
    right_line_strip.color.b = 1.0;
    right_line_strip.color.a = 1.0;
    
    // Line list is red
    right_line_list.color.r = 1.0;
    right_line_list.color.a = 1.0;
    // Create the vertices for the points and lines
    for (uint32_t i = 0; i < count_right; ++i)
    {

      geometry_msgs::Point p_r;
      p_r.x = right_lane_point_x[i];
      p_r.y = right_lane_point_y[i];
      p_r.z = right_lane_point_z[i];
      right_points.points.push_back(p_r);
      right_line_strip.points.push_back(p_r); 
      // The line list needs two points for each line
      right_line_list.points.push_back(p_r);
      p_r.z += 1.0;
      right_line_list.points.push_back(p_r);
  }


    marker_pub_left.publish(left_points);
    marker_pub_left.publish(left_line_strip);
    marker_pub_left.publish(left_line_list);
    marker_pub_right.publish(right_points);
    marker_pub_right.publish(right_line_strip);
    marker_pub_right.publish(right_line_list);

    


    //pub1.publish(msg);

    ros::spinOnce ();
    loop_rate.sleep ();

    

  }
}
