/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science
 *and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <algorithm>
#include <list>
#include <numeric>
#include <vector>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/assert.h>
#include <ros/console.h>

#include "../utility/tic_toc.h"
#include "parameters.h"

// 该类存放一对左右目匹配特征点
class FeaturePerFrame {
public:
  /**
   * 添加左目特征点
   * @param _point 特征点信息
   * @param td IMU和cam的时间误差 cam_time + td = imu_time
   */
  FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td) {
    point.x() = _point(0);
    point.y() = _point(1);
    point.z() = _point(2);
    uv.x() = _point(3);
    uv.y() = _point(4);
    velocity.x() = _point(5);
    velocity.y() = _point(6);
    cur_td = td;
    is_stereo = false;
  }

  /**
   * @brief 添加对应帧的右目特征点
   * @param _point 特征点信息
   */
  void rightObservation(const Eigen::Matrix<double, 7, 1> &_point) {
    pointRight.x() = _point(0);
    pointRight.y() = _point(1);
    pointRight.z() = _point(2);
    uvRight.x() = _point(3);
    uvRight.y() = _point(4);
    velocityRight.x() = _point(5);
    velocityRight.y() = _point(6);
    is_stereo = true;
  }

  double cur_td;

  Vector3d point, pointRight; // 特征点在归一化相机坐标系下的坐标
  Vector2d uv, uvRight;       // 像素坐标
  Vector2d velocity, velocityRight; // 归一化相机平面速度
  bool is_stereo;                   // 是否是双目的特征点
};

class FeaturePerId {
public:
  const int feature_id; // 特征点ID

  int start_frame; // 该特征点第一次出现在滑动窗口的第几帧图像上

  vector<FeaturePerFrame> feature_per_frame; // 队列，存放该特征点出现过的全部帧
  int used_num;
  ， double estimated_depth; // 特征点深度，在首帧观测帧下的深度值 todo
  int solve_flag;            // 0 haven't solve yet; 1 solve succ; 2 solve fail;

  FeaturePerId(int _feature_id, int _start_frame)
      : feature_id(_feature_id), start_frame(_start_frame), used_num(0),
        estimated_depth(-1.0), solve_flag(0) {}

  int endFrame();
};

class FeatureManager {
public:
  FeatureManager(Matrix3d _Rs[]);

  void setRic(Matrix3d _ric[]);
  void clearState();
  int getFeatureCount();
  bool addFeatureCheckParallax(
      int frame_count,
      const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
      double td);
  vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l,
                                                    int frame_count_r);
  // void updateDepth(const VectorXd &x);
  void setDepth(const VectorXd &x);
  void removeFailures();
  void clearDepth();
  VectorXd getDepthVector();
  void triangulate(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[],
                   Matrix3d ric[]);
  void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0,
                        Eigen::Matrix<double, 3, 4> &Pose1,
                        Eigen::Vector2d &point0, Eigen::Vector2d &point1,
                        Eigen::Vector3d &point_3d);
  void initFramePoseByPnP(int frameCnt, Vector3d Ps[], Matrix3d Rs[],
                          Vector3d tic[], Matrix3d ric[]);
  bool solvePoseByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial,
                      vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D);
  void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P,
                            Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
  void removeBack();
  void removeFront(int frame_count);
  void removeOutlier(set<int> &outlierIndex);
  list<FeaturePerId> feature; // 存放之前添加的全部帧的特征点
  int last_track_num; // 当前帧中有多少特征点在之前已经出现过
  double last_average_parallax;
  int new_feature_num; // 当前帧中新出现的特征点
  int long_track_num; // 当前识别到的特征点已经出现了至少4次（包括本次）

private:
  double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
  const Matrix3d *Rs;
  // 相机的外参矩阵imu_R_cam，最多两个相机
  Matrix3d ric[2];
};

#endif