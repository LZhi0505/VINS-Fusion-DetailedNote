/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science
 *and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include "../utility/utility.h"
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <vector>

using namespace std;

const double FOCAL_LENGTH = 460.0; // 焦距
const int WINDOW_SIZE = 10;        // 滑动窗口大小
const int NUM_OF_F = 1000;         // 滑窗中feature数量上限

//#define UNIT_SPHERE_ERROR //在单位球面（而非成像平面）上计算重投影误差，详见论文；作者关闭了这个宏

extern double INIT_DEPTH;
extern double MIN_PARALLAX;
extern int ESTIMATE_EXTRINSIC; //是否在线估计外参

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;

extern std::vector<Eigen::Matrix3d> RIC; // 左右目到IMU的外参旋转
extern std::vector<Eigen::Vector3d> TIC; // 外参平移
extern Eigen::Vector3d G;                // 重力常量（3*1矢量）

extern double BIAS_ACC_THRESHOLD;
extern double BIAS_GYR_THRESHOLD;
extern double SOLVER_TIME;
extern int NUM_ITERATIONS;
extern std::string EX_CALIB_RESULT_PATH;
extern std::string VINS_RESULT_PATH;
extern std::string OUTPUT_FOLDER;
extern std::string IMU_TOPIC;
extern double TD;
extern int ESTIMATE_TD; //是否在线估计这个时延
extern int ROLLING_SHUTTER;
extern int ROW, COL;
extern int NUM_OF_CAM;
extern int STEREO;
extern int USE_IMU;
extern int MULTIPLE_THREAD;
// pts_gt for debug purpose;
extern map<int, Eigen::Vector3d> pts_gt;

extern std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
extern std::string FISHEYE_MASK;
extern std::vector<std::string> CAM_NAMES;
extern int MAX_CNT;        //允许最多追踪多少个特征点，通常是150或200
extern int MIN_DIST;       //控制特征点密度的像素距离参数，使特征尽可能均匀分布
extern double F_THRESHOLD; //用基础矩阵剔除outlier时的阈值（实际未启用）
extern int SHOW_TRACK;     //是否在独立opencv窗口中显示图像，用于debug
extern int FLOW_BACK;      //是否启用反向追踪检验，通常是启用的

void readParameters(std::string config_file);

enum SIZE_PARAMETERIZATION { SIZE_POSE = 7, SIZE_SPEEDBIAS = 9, SIZE_FEATURE = 1 };

enum StateOrder { O_P = 0, O_R = 3, O_V = 6, O_BA = 9, O_BG = 12 };

enum NoiseOrder { O_AN = 0, O_GN = 3, O_AW = 6, O_GW = 9 };
