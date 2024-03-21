/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science
 *and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

#include <csignal>
#include <cstdio>
#include <eigen3/Eigen/Dense>
#include <execinfo.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <queue>

#include "../estimator/parameters.h"
#include "../utility/tic_toc.h"
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker {
public:
  FeatureTracker();
  map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>
  trackImage(double _cur_time, const cv::Mat &_img,
             const cv::Mat &_img1 = cv::Mat());
  void setMask();
  void readIntrinsicParameter(const vector<string> &calib_file);
  void showUndistortion(const string &name);
  void rejectWithF();
  void undistortedPoints();
  vector<cv::Point2f> undistortedPts(vector<cv::Point2f> &pts,
                                     camodocal::CameraPtr cam);
  vector<cv::Point2f> ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts,
                                  map<int, cv::Point2f> &cur_id_pts,
                                  map<int, cv::Point2f> &prev_id_pts);
  void showTwoImage(const cv::Mat &img1, const cv::Mat &img2,
                    vector<cv::Point2f> pts1, vector<cv::Point2f> pts2);
  void drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight,
                 vector<int> &curLeftIds, vector<cv::Point2f> &curLeftPts,
                 vector<cv::Point2f> &curRightPts,
                 map<int, cv::Point2f> &prevLeftPtsMap);
  void setPrediction(map<int, Eigen::Vector3d> &predictPts);
  double distance(cv::Point2f &pt1, cv::Point2f &pt2);
  void removeOutliers(set<int> &removePtsIds);
  cv::Mat getTrackImage();
  bool inBorder(const cv::Point2f &pt);

  int row, col;
  cv::Mat imTrack; // 标记了特征点识别结果的跟踪图片

  // 非极大值抑制用的mask
  cv::Mat mask;
  cv::Mat fisheye_mask;

  cv::Mat prev_img, cur_img; // cur_img为当前左目原图片

  vector<cv::Point2f> n_pts; // 当前帧新提取的特征点
  vector<cv::Point2f> predict_pts; // 使用匀速模型预测的下一帧特征点位置
  vector<cv::Point2f> predict_pts_debug;

  // 上一帧左目的特征点、当前帧左目的特征点、当前帧右目的特征点（像素坐标）
  vector<cv::Point2f> prev_pts, cur_pts, cur_right_pts;

  // 上一帧左目、当前帧左目、当前帧右目中识别到的特征点 转换到
  // 归一化相机坐标系的坐标（归一化相机x,y坐标，去畸变后的）
  vector<cv::Point2f> prev_un_pts, cur_un_pts, cur_un_right_pts;

  // pts_velocity：当前帧特征点中是上一帧跟踪得到的特征点的速度；right_pts_velocity：右目的特征点速度
  // （在归一化平面x,y方向上的移动速度）
  vector<cv::Point2f> pts_velocity, right_pts_velocity;

  // ids：当前帧识别到的特征点ID号；ids_right：当前帧右目识别到的特征点ID号
  vector<int> ids, ids_right;

  // 当前帧识别到的特征点的连续帧跟踪次数（比如跟踪次数为2表示该特征点在当前帧、上一帧中都被识别，所以里面的跟踪次数至少为1，也就是当前帧被识别到）
  vector<int> track_cnt;

  // 字典，cur_un_pts_map为当前帧左目的（ids，cur_un_pts
  // 归一化相机x,y坐标），prev_un_pts_map为上一帧的（ids，prev_un_pts
  // 归一化相机x,y坐标）
  map<int, cv::Point2f> cur_un_pts_map,
      prev_un_pts_map; // index: 当前帧左目特征点id, value:
                       // 对应去畸变的在归一化平面上的坐标
  map<int, cv::Point2f> cur_un_right_pts_map, prev_un_right_pts_map;

  // 字典，上一帧左目的（ids，cur_pts 像素坐标）
  map<int, cv::Point2f> prevLeftPtsMap;

  vector<camodocal::CameraPtr> m_camera; // 相机实例化对象，在读取相机内参时创建
  double cur_time;                       // 当前时间戳
  double prev_time;                      // 上一帧的时间戳
  bool stereo_cam;                       // 是否为双目相机
  int n_id;
  bool hasPrediction; // 是否开启预测
};
