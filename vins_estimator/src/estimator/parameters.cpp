/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science
 *and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "parameters.h"

double INIT_DEPTH;   // 特征点深度的初始估计值
double MIN_PARALLAX; // 判断是否是关键帧的视差阈值，真实世界最小视差
double ACC_N, ACC_W; // 加速度计accelerometer的白噪声和游走零偏
double GYR_N, GYR_W; // 陀螺仪gyroscope的白噪声和游走零偏

std::vector<Eigen::Matrix3d> RIC; // 队列，存储body_R_cam0矩阵，后续不断优化该参数
std::vector<Eigen::Vector3d> TIC; // 队列，存储body_t_cam0矩阵，后续不断优化该参数

Eigen::Vector3d G{0.0, 0.0, 9.8}; // 重力加速度g

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;     // 求解器迭代的最长时间 ，单位ms
int NUM_ITERATIONS;     // 求解器的最大迭代次数
int ESTIMATE_EXTRINSIC; // 0表示传感器外参固定为配置文件里面参数；1表示以配置文件参数为初始值，vins会进一步优化传感器参数；2表示无配置文件的先验参数，完全靠vins的优化得到参数
int ESTIMATE_TD;        // 是否在线标定相机和IMU之间的时间偏差，0为固定，1为在线标定
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string OUTPUT_FOLDER;

std::string IMU_TOPIC; // imu话题
int ROW, COL;          // 图像的高和宽
double TD;             // 相机和IMU之间的时间偏差的初始值
int NUM_OF_CAM;        // 相机数目
int STEREO;            // 是否为双目相机
int USE_IMU;           // 是否使用imu，bool
int MULTIPLE_THREAD;   // 是否使用多线程
map<int, Eigen::Vector3d> pts_gt;

std::string IMAGE0_TOPIC, IMAGE1_TOPIC; // 图像话题名称
std::string FISHEYE_MASK;
std::vector<std::string> CAM_NAMES; // 队列，存储全部相机的内参文件路径
int MAX_CNT;                        // 一帧图片最多允许筛选的特征点数目
int MIN_DIST;                       // MASK非极大值抑制的圆半径（图片的一个圆区域中只保留最好的特征点）
double F_THRESHOLD;                 // cv::findFundamentalMat求解基础矩阵的系数
int SHOW_TRACK;                     // 设置为1表示允许在当前帧图像上标记出特征点并发布出去，0则不发布
int FLOW_BACK;                      // 设置为1表示开启反向光流，提高特征点跟踪精度

/**
 * ROS参数读取
 */
template <typename T> T readParam(ros::NodeHandle &n, std::string name) {
    T ans;
    if (n.getParam(name, ans)) {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    } else {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

/**
 * @brief：读取配置文件
 * @param config_file：配置文件路径
 */
void readParameters(std::string config_file) {
    // 1.首先判断配置文件是否存在
    FILE *fh = fopen(config_file.c_str(), "r");
    if (fh == NULL) {
        ROS_WARN("config_file dosen't exist; wrong config_file path");
        ROS_BREAK();
        return;
    }
    fclose(fh);

    // 2.改用opencv读取yaml文件
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    // FileStorage读取yaml文件的两种方法 (字符串类型不支持使用=读取)
    fsSettings["image0_topic"] >> IMAGE0_TOPIC; // "/cam0/image_raw"
    fsSettings["image1_topic"] >> IMAGE1_TOPIC; // "/cam1/image_raw"
    MAX_CNT = fsSettings["max_cnt"];            // 数值类型可以使用=也可以使用>>
    MIN_DIST = fsSettings["min_dist"];
    F_THRESHOLD = fsSettings["F_threshold"]; // ransac threshold (pixel)
    SHOW_TRACK = fsSettings["show_track"];
    FLOW_BACK = fsSettings["flow_back"];

    MULTIPLE_THREAD = fsSettings["multiple_thread"]; // 使用udoxiancheng

    USE_IMU = fsSettings["imu"];
    printf("USE_IMU: %d\n", USE_IMU);

    // 如果使用imu，还需要读取imu的参数
    if (USE_IMU) {
        fsSettings["imu_topic"] >> IMU_TOPIC; // "/imu0"
        printf("IMU_TOPIC: %s\n", IMU_TOPIC.c_str());
        ACC_N = fsSettings["acc_n"];
        ACC_W = fsSettings["acc_w"];
        GYR_N = fsSettings["gyr_n"];
        GYR_W = fsSettings["gyr_w"];
        G.z() = fsSettings["g_norm"];
    }

    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];

    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    // 真实世界最小视差 = 最小视差(像素) / 焦距f
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    // 创建输出文件
    fsSettings["output_path"] >> OUTPUT_FOLDER;
    VINS_RESULT_PATH = OUTPUT_FOLDER + "/vio.csv";
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    // 先获取左目到IMU的外参 到 RIC[0] TIC[0]
    ESTIMATE_EXTRINSIC =
        fsSettings["estimate_extrinsic"]; // 0：有确定的Tbc外参；1：有初始估计的外参，之后还要优化；2：没有外参，需要标定，Ric初始化为单位阵，tic初始化为零向量
    // 不使用配置文件里面的传感器参数，实际参数完全靠vins的优化得到
    if (ESTIMATE_EXTRINSIC == 2) {
        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC.push_back(Eigen::Matrix3d::Identity());
        TIC.push_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
    } else {
        // 以配置文件的传感器参数为初始值，后续优化该传感器参数
        if (ESTIMATE_EXTRINSIC == 1) {
            ROS_WARN(" Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
        }
        // 直接把配置文件的参数当做真值，不续也不优化该参数了
        if (ESTIMATE_EXTRINSIC == 0)
            ROS_WARN(" fix extrinsic param ");

        cv::Mat cv_T;
        fsSettings["body_T_cam0"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    }

    NUM_OF_CAM = fsSettings["num_of_cam"];
    printf("camera number %d\n", NUM_OF_CAM);

    // 相机个数必须为1或2
    if (NUM_OF_CAM != 1 && NUM_OF_CAM != 2) {
        printf("num_of_cam should be 1 or 2\n");
        assert(0);
    }

    // 3.读取相同文件夹下的两个相机内参文件
    // 得到config_file文件夹路径
    int pn = config_file.find_last_of('/');
    std::string configPath = config_file.substr(0, pn);

    std::string cam0Calib;
    fsSettings["cam0_calib"] >> cam0Calib;
    std::string cam0Path = configPath + "/" + cam0Calib;
    // 先存入左目的内参文件路径
    CAM_NAMES.push_back(cam0Path);

    // 再存入右目的内参文件路径
    if (NUM_OF_CAM == 2) {
        STEREO = 1;
        std::string cam1Calib;
        fsSettings["cam1_calib"] >> cam1Calib;
        std::string cam1Path = configPath + "/" + cam1Calib;
        // printf("%s cam1 path\n", cam1Path.c_str() );
        CAM_NAMES.push_back(cam1Path);

        // 再获取右目到IMU的外参 到 RIC[1] TIC[1]
        cv::Mat cv_T;
        fsSettings["body_T_cam1"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    }

    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    TD = fsSettings["td"];                   // 初始时间偏移(s)
    ESTIMATE_TD = fsSettings["estimate_td"]; // 是否在线获取相机与IMU的时间漂移，1表示两个传感器未同步，0表示两个传感器已同步
    if (ESTIMATE_TD)
        ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << TD);
    else
        ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    ROS_INFO("ROW: %d COL: %d ", ROW, COL);

    if (!USE_IMU) {
        ESTIMATE_EXTRINSIC = 0;
        ESTIMATE_TD = 0;
        printf("no imu, fix extrinsic param; no time offset calibration\n");
    }

    fsSettings.release();
}
