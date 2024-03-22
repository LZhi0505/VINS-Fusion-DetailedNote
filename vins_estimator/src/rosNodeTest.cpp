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

#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <stdio.h>

#include <map>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>

#include "estimator/estimator.h"
#include "estimator/parameters.h"
#include "utility/visualization.h"

// 创建位姿估计器。实际程序的开始处，而不是main()
Estimator estimator;

queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::ImageConstPtr> img0_buf; // 队列，存储接收到的img0的msg
queue<sensor_msgs::ImageConstPtr> img1_buf; // 队列，存储接受到的img1的msg
std::mutex m_buf;                           // 用于更新buf时的锁

/**
 * @brief 相机0的回调函数，保存相机0的msg
 * @param img_msg
 */
void img0_callback(const sensor_msgs::ImageConstPtr &img_msg) {
    m_buf.lock();

    img0_buf.push(img_msg);

    m_buf.unlock();
}

/**
 * @brief 相机1的回调函数，保存相机1的msg
 * @param img_msg
 */
void img1_callback(const sensor_msgs::ImageConstPtr &img_msg) {
    m_buf.lock();

    img1_buf.push(img_msg);

    m_buf.unlock();
}

/**
 * ROS图像转换成CV格式 (从msg中获取图片)
 * @param img_msg  当前图像msg的指针
 * return   cv::Mat格式的图片
 */
cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg) {
    cv_bridge::CvImageConstPtr ptr;

    // 灰度图片需要额外处理
    if (img_msg->encoding == "8UC1") {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    // 彩色图片可以直接转换
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat img = ptr->image.clone();
    return img;
}

/**
 * 获取时间同步(时间戳差值<0.003s)后的 左右目照片。传入到estimator中，调用inputImage进行处理
 */
void sync_process() {
    while (1) {
        if (STEREO) {
            cv::Mat image0, image1;
            std_msgs::Header header;
            double time = 0;
            m_buf.lock();

            // 如果两个img buf里面都有未处理的msg
            if (!img0_buf.empty() && !img1_buf.empty()) {
                double time0 = img0_buf.front()->header.stamp.toSec();
                double time1 = img1_buf.front()->header.stamp.toSec();

                // 当双目图片的时间戳大于0.003，则丢弃里面最早的图片msg，直到两个图片时间戳小于0.003
                // 不了解这里不同步后只丢弃一张图片，难道还指望未被丢弃的这张图片会和下一帧图片会时间戳对齐？
                if (time0 < time1 - 0.003) {
                    img0_buf.pop();
                    printf("throw img0\n");
                } else if (time0 > time1 + 0.003) {
                    img1_buf.pop();
                    printf("throw img1\n");
                }
                // 时间戳小于0.003则认为没有时间差，取出缓存队列中最早的一帧
                // 左右目图像，并从队列中删除
                else {
                    time = img0_buf.front()->header.stamp.toSec();
                    header = img0_buf.front()->header;

                    image0 = getImageFromMsg(img0_buf.front());
                    img0_buf.pop();

                    image1 = getImageFromMsg(img1_buf.front());
                    img1_buf.pop();
                }
            }
            m_buf.unlock();

            // 将取出的cv格式图片 传入 estimator 中
            if (!image0.empty())
                estimator.inputImage(time, image0, image1);
        }
        // 单目
        else {
            cv::Mat image;
            std_msgs::Header header;
            double time = 0;
            m_buf.lock();

            // 单目不考虑时间戳同步问题
            if (!img0_buf.empty()) {
                time = img0_buf.front()->header.stamp.toSec();
                header = img0_buf.front()->header;

                image = getImageFromMsg(img0_buf.front());
                img0_buf.pop();
            }
            m_buf.unlock();

            // 传入入estimator中
            if (!image.empty())
                estimator.inputImage(time, image);
        }

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

/**
 * 获取新一时刻的IMU数据，并输入到 estimator 的 accBuf 和 gyrBuf中，进行 IMU预积分，获取当前帧状态
 */
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg) {
    double t = imu_msg->header.stamp.toSec();
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Vector3d acc(dx, dy, dz);
    Vector3d gyr(rx, ry, rz);

    // 输入到 estimator 的 accBuf 和 gyrBuf 中，并进行IMU预积分 与 广播
    estimator.inputIMU(t, acc, gyr);

    return;
}

/**
 * 从接收到的 特征点的点云信息，提取其featureID、cameraID、3D坐标、像素坐标、速度。输入到 estimator 的 featureBuf
 * 但在VINS-Fusion中没有发布该节点，该函数没有用到
 * @param feature_msg 特征点话题msg
 */
void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg) {
    // feature_id, (camera_id, x, y, z, p_u, p_v, velocity_x, velocity_y)
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    // 遍历每个特征点i
    for (unsigned int i = 0; i < feature_msg->points.size(); i++) {
        int feature_id = feature_msg->channels[0].values[i];
        int camera_id = feature_msg->channels[1].values[i];
        double x = feature_msg->points[i].x; // 三维坐标
        double y = feature_msg->points[i].y;
        double z = feature_msg->points[i].z;
        double p_u = feature_msg->channels[2].values[i]; // 像素坐标
        double p_v = feature_msg->channels[3].values[i];
        double velocity_x = feature_msg->channels[4].values[i]; // 速度
        double velocity_y = feature_msg->channels[5].values[i];
        if (feature_msg->channels.size() > 5) {
            double gx = feature_msg->channels[6].values[i]; // 重力？
            double gy = feature_msg->channels[7].values[i];
            double gz = feature_msg->channels[8].values[i];
            pts_gt[feature_id] = Eigen::Vector3d(gx, gy, gz);
            // printf("receive pts gt %d %f %f %f\n", feature_id, gx, gy, gz);
        }
        ROS_ASSERT(z == 1);
        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
    }
    double t = feature_msg->header.stamp.toSec();

    estimator.inputFeature(t, featureFrame);

    return;
}

/**
 * 订阅，重启节点
 * 是否重启estimator，并重新设置参数
 */
void restart_callback(const std_msgs::BoolConstPtr &restart_msg) {
    if (restart_msg->data == true) {
        ROS_WARN("restart the estimator!");
        estimator.clearState();
        estimator.setParameter();
    }
    return;
}

/**
 * 订阅，双目，是否使用IMU开关
 */
void imu_switch_callback(const std_msgs::BoolConstPtr &switch_msg) {
    if (switch_msg->data == true) {
        // ROS_WARN("use IMU!");
        estimator.changeSensorType(1, STEREO);
    } else {
        // ROS_WARN("disable IMU!");
        estimator.changeSensorType(0, STEREO);
    }
    return;
}

/**
 * 订阅，单双目切换
 */
void cam_switch_callback(const std_msgs::BoolConstPtr &switch_msg) {
    if (switch_msg->data == true) {
        // ROS_WARN("use stereo!");
        estimator.changeSensorType(USE_IMU, 1);
    } else {
        // ROS_WARN("use mono camera (left)!");
        estimator.changeSensorType(USE_IMU, 0);
    }
    return;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    // 改变日志显示输出的级别设置
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    if (argc != 2) {
        printf("please intput: rosrun vins vins_node [config file] \n"
               "for example: rosrun vins vins_node "
               "~/catkin_ws/src/VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml "
               "\n");
        return 1;
    }

    string config_file = argv[1];
    printf("config_file: %s\n", argv[1]); // 一般C++项目里面比起cout更喜欢printf，因为printf打印效率更高，输出更快

    // 读取YAML配置文件内容
    readParameters(config_file);

    // 设置参数到位姿估计器中，当setParameter()时候，就开启了一个Estimator类内的新线程：processMeasurements()
    estimator.setParameter();

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

    ROS_WARN("waiting for image and imu...");

    // 创建话题发布者对象vins_estimator，在此节点下发布话题
    registerPub(n);

    /*
    ros::Subscriber subscribe (const std::string &topic, uint32_t queue_size, void(*fp)(M), const TransportHints &transport_hints=TransportHints())
    参数1：订阅话题的名称；
    参数2：订阅队列的长度；（如果收到的消息都没来得及处理，那么新消息入队，旧消息就会出队）；
    参数3：回调函数的指针，指向回调函数来处理接收到的消息！
    参数4：似乎与延迟有关系，暂时不关心。（该成员函数有13重载）
    */

    // 如果程序写了相关的消息订阅函数，那么程序在执行过程中，除了主程序以外，ROS还会自动在后台按照你规定的格式，接受订阅的消息，但是所接到的消息并不是立刻就被处理，
    // 而是必须要等到ros::spin()或ros::spinOnce()执行的时候才被调用，这就是消息回调函数的原理

    // 创建订阅者对象
    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay()); // 订阅IMU消息
    ros::Subscriber sub_feature = n.subscribe("/feature_tracker/feature", 2000, feature_callback); // 订阅feature_tracker所提供的 跟踪光流点
    ros::Subscriber sub_img0 = n.subscribe(IMAGE0_TOPIC, 100, img0_callback);                      // 订阅左目图像
    ros::Subscriber sub_img1 = n.subscribe(IMAGE1_TOPIC, 100, img1_callback);                      // 订阅右目图像

    ros::Subscriber sub_restart = n.subscribe("/vins_restart", 100, restart_callback);          // 订阅，重启节点
    ros::Subscriber sub_imu_switch = n.subscribe("/vins_imu_switch", 100, imu_switch_callback); // 订阅，双目下，IMU开关
    ros::Subscriber sub_cam_switch = n.subscribe("/vins_cam_switch", 100, cam_switch_callback); // 订阅，单双目切换

    //! 创建线程 sync_process：如果图像buffer里面有数据的话，读入数据并且添加到estimator中
    // 为什么不在左右目相机图像的回调函数就input？对于双目的话，能够检测同步问题，能够将同样时间戳的两帧图片同时放入estimator中。所以对于IMU以及feature直接在回调函数中进行添加
    std::thread sync_thread{sync_process};
    ros::spin(); // 用于触发topic, service的响应队列

    return 0;
}
