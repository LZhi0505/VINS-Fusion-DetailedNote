/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "utility.h"
#include <iostream>

/**
 * 计算 IMU的Z轴 对齐到 重力加速度方向 所需的旋转矩阵
 * @param[in] g 第一帧t0-t1间的 平均加速度
 * @return w_R0_g
 */
Eigen::Matrix3d Utility::g2R(const Eigen::Vector3d &g) {
    Eigen::Matrix3d R0;

    Eigen::Vector3d ng1 = g.normalized(); // IMU加速度的单位向量
    Eigen::Vector3d ng2{0, 0, 1.0};       // ENU世界坐标系下的z方向（重力加速度方向）

    // R0 * ng1 = ng2：求解 ng1 --> ng2 的旋转矩阵
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix(); // ng2_R0_ng1

    // 测试代码
    // static bool done = false;
    // if(!done)
    // {
    //     std::cout << "ng1:" << ng1.transpose() << ",ng2:" << ng2.transpose() << std::endl;
    //     std::cout << "R0:" << std::endl << R0 << std::endl;
    //     std::cout << "R0*ng1:" << R0 * ng1 << std::endl;
    //     std::cout << "R0*ng2:" << R0 * ng2 << std::endl;
    // }

    // 确保最终的旋转矩阵在偏航角上与原始R0相反，从而确保IMU的Z轴与重力加速度方向完全对齐
    // 取出yaw角（绕z轴旋转）
    double yaw = Utility::R2ypr(R0).x(); // 旋转矩阵 ==> 偏航-俯仰-滚动角，单位为°
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    // R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;

    // 测试代码
    // if(!done)
    // {
    //     done = true;
    //     std::cout << "yaw:" << yaw << std::endl;
    //     std::cout << "R0*ng1:" << R0 * ng1 << std::endl;
    //     std::cout << "R0*ng2:" << R0 * ng2 << std::endl;
    // }

    return R0;
}
