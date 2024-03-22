/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science
 *and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "estimator.h"

#include "../utility/visualization.h"

Estimator::Estimator() : f_manager{Rs} {
    ROS_INFO("init begins");

    initThreadFlag = false;

    clearState();
}

Estimator::~Estimator() {
    if (MULTIPLE_THREAD) {
        // 等待processThread线程释放
        processThread.join();
        printf("join thread \n");
    }
}

/**
 * 清理状态、缓存数据、变量、滑动窗口数据、位姿等
 * 系统重启或者滑窗优化失败都会调用
 */
void Estimator::clearState() {
    mProcess.lock();
    // 清除缓存数据
    while (!accBuf.empty())
        accBuf.pop();
    while (!gyrBuf.empty())
        gyrBuf.pop();
    while (!featureBuf.empty())
        featureBuf.pop();

    prevTime = -1;
    curTime = 0;
    openExEstimation = 0;
    initP = Eigen::Vector3d(0, 0, 0);
    initR = Eigen::Matrix3d::Identity();
    inputImageCnt = 0;
    initFirstPoseFlag = false;

    // 清除滑动窗口内数据
    for (int i = 0; i < WINDOW_SIZE + 1; i++) {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr) {
            delete pre_integrations[i];
        }
        pre_integrations[i] = nullptr;
    }

    // 清除相机的外参
    for (int i = 0; i < NUM_OF_CAM; i++) {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    first_imu = false; // false表示未处理过第一帧IMU数据
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();

    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState(); // 清除 滑动窗口内所有特征点的管理器 的状态

    failure_occur = 0;

    mProcess.unlock();
}

/**
 * 把读取到的配置文件参数设置到位姿估计器中
 */
void Estimator::setParameter() {
    mProcess.lock();
    // 从配置文件获取的左右目外参 设置到 ric,tic 中，body_T_cam
    for (int i = 0; i < NUM_OF_CAM; i++) {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
        cout << " exitrinsic cam " << i << endl << ric[i] << endl << tic[i].transpose() << endl;
    }
    f_manager.setRic(ric);

    // sqrt_info是信息矩阵
    ProjectionTwoFrameOneCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTwoFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionOneFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();

    // 相机和IMU之间的时间偏差的初始值，硬件传输等因素
    td = TD;
    g = G;
    cout << "set g " << g.transpose() << endl;

    // 特征点跟踪器，读取左右目相机内参，并实例化相机
    featureTracker.readIntrinsicParameter(CAM_NAMES);

    std::cout << "MULTIPLE_THREAD is " << MULTIPLE_THREAD << '\n';

    //! 如果是多线程，就会创建新线程 processMeasurements
    // 如果是多线程模式，就是一个线程做光流，一个线程做后端优化，否则，就是一个做完光流之后在做线程优化,串行处理
    if (MULTIPLE_THREAD && !initThreadFlag) {

        initThreadFlag = true;

        processThread = std::thread(&Estimator::processMeasurements, this);
    }
    mProcess.unlock();
}

/**
 * 双目，双目+IMU，单目+IMU，如果重新使用IMU，需要重启节点
 */
void Estimator::changeSensorType(int use_imu, int use_stereo) {
    bool restart = false;
    mProcess.lock();
    if (!use_imu && !use_stereo)
        printf("at least use two sensors! \n");
    else {
        if (USE_IMU != use_imu) {
            USE_IMU = use_imu;
            if (USE_IMU) {
                // reuse imu; restart system
                restart = true;
            } else {
                if (last_marginalization_info != nullptr)
                    delete last_marginalization_info;

                tmp_pre_integration = nullptr;
                last_marginalization_info = nullptr;
                last_marginalization_parameter_blocks.clear();
            }
        }

        STEREO = use_stereo;
        printf("use imu %d use stereo %d\n", USE_IMU, STEREO);
    }
    mProcess.unlock();
    if (restart) {
        clearState();
        setParameter();
    }
}

/**
 * 输入一帧的左右目图片(或单目)，会执行VO优化得到位姿
 * 1、featureTracker，提取当前帧特征点
 * 2、添加一帧特征点，processMeasurements处理
 * @param t 当前帧时间戳
 * @param _img  左目图像
 * @param _img1 右目图像
 */
void Estimator::inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1) {
    inputImageCnt++;
    // featureFrame: feature_id，[camera_id (0为左目，1为右目), x, y, z (去畸变的归一化相机平面坐标), pu, pv (像素坐标), vx, vy(归一化相机平面移动速度)]
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    TicToc featureTrackerTime;

    // 1. featureTracker::trackImage() 提取特征点，跟踪当前帧
    // 更新特征点跟踪次数；保存当前帧特征点数据（归一化相机平面坐标，像素坐标，相对于前一帧的归一化相机平面移动速度）
    if (_img1.empty())
        featureFrame = featureTracker.trackImage(t, _img);
    else
        featureFrame = featureTracker.trackImage(t, _img, _img1);
    // printf("featureTracker time: %f\n", featureTrackerTime.toc());

    // 2.
    // 发布加上特征点后的图片（用蓝点和红点标注跟踪次数不同的特征点，红色少，蓝色多，画箭头指向前一帧特征点位置；如果是双目，右图画个绿色点）
    if (SHOW_TRACK) {
        cv::Mat imgTrack = featureTracker.getTrackImage();
        pubTrackImage(imgTrack, t);
    }

    // 3. 添加一帧特征点到featureBuf，处理一帧的特征点。
    // processMeasurements线程开启，后端非线性优化IMU和图片数据，得到位姿
    // 多线程
    if (MULTIPLE_THREAD) {
        if (inputImageCnt % 2 == 0) {
            mBuf.lock();
            // featureBuf: 时间戳, {feature_id，[camera_id (0为左目，1为右目), x, y, z
            // (去畸变的归一化相机平面坐标), pu, pv (像素坐标), vx, vy
            // (归一化相机平面移动速度)]}
            featureBuf.push(make_pair(t, featureFrame));
            // 这里没有调用函数processMeasurements是因为多线程在前面estimator.setParameter()中已经开启了该函数的线程
            mBuf.unlock();
        }
    }
    // 单线程
    else {
        mBuf.lock();
        featureBuf.push(make_pair(t, featureFrame));
        mBuf.unlock();

        TicToc processTime;
        // 处理一帧
        processMeasurements();

        printf("process time: %f\n", processTime.toc());
    }
}

/**
 * 缓存新一时刻的IMU数据，供后端的非线性优化（每订阅到新的IMU话题数据就会执行该函数）
 * @param t                     新一时刻的 时间戳
 * @param linearAcceleration    新一时刻的 线加速度
 * @param angularVelocity       新一时刻的 角速度
 *
 * 当IMU初始化完成后:
 * 1. 会调用fastPredictIMU: 更新位置P、旋转Q、速度V
 * 2. 发布到话题imu_propagate中，话题更新的频率和imu频率相同
 */
void Estimator::inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity) {
    mBuf.lock();
    accBuf.push(make_pair(t, linearAcceleration));
    gyrBuf.push(make_pair(t, angularVelocity));
    // printf("input imu with time %f \n", t);
    mBuf.unlock();

    // 位姿估计器estimator的状态为 非线性优化，表示IMU已经完成初始化
    if (solver_flag == NON_LINEAR) {
        mPropagate.lock();
        // IMU预积分：使用上一时刻的状态进行快速预积分 来预测当前帧位姿：旋转矩阵latest_Q、位置latest_P、速度latest_V
        // 这个信息根据processIMU的最新数据Ps[frame_count]、Rs[frame_count]、Vs[frame_count]、Bas[frame_count]、Bgs[frame_count]来进行预积分
        fastPredictIMU(t, linearAcceleration, angularVelocity);
        // 发布 预测的当前时刻的位姿
        pubLatestOdometry(latest_P, latest_Q, latest_V, t);
        mPropagate.unlock();
    }
}

/**
 * 输入一帧特征点，processMeasurements 进行处理
 * @param t             当前帧的 时间戳
 * @param featureFrame  当前帧的特征数据
 */
void Estimator::inputFeature(double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame) {
    mBuf.lock();
    // 存入帧特征缓存队列中
    featureBuf.push(make_pair(t, featureFrame));
    mBuf.unlock();

    // 单线程，则进入processMeasurements处理
    if (!MULTIPLE_THREAD)
        processMeasurements();
}

/**
 * 从IMU数据队列 accBuf、gyrBuf 中，提取(t0, t1)时间段的数据
 * @param t0    前一帧时间戳
 * @param t1    当前帧时间戳
 * @param accVector 输出的加速度数据
 * @param gyrVector 输出的陀螺仪数据
 * @return 找到返回true，否则返回false
 */
bool Estimator::getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector, vector<pair<double, Eigen::Vector3d>> &gyrVector) {
    if (accBuf.empty()) {
        printf("not receive imu\n");
        return false;
    }
    // printf("get imu from %f %f\n", t0, t1);
    // printf("imu fornt time %f   imu end time %f\n", accBuf.front().first,
    // accBuf.back().first);

    if (t1 <= accBuf.back().first) {
        // 弹出时间戳 <= t0的数据
        while (accBuf.front().first <= t0) {
            accBuf.pop();
            gyrBuf.pop();
        }
        // 暂存时间戳 < t1的数据
        while (accBuf.front().first < t1) {
            accVector.push_back(accBuf.front());
            accBuf.pop();
            gyrVector.push_back(gyrBuf.front());
            gyrBuf.pop();
        }
        accVector.push_back(accBuf.front());
        gyrVector.push_back(gyrBuf.front());
    }
    // t1 > 最新的加速度时间，即数据还未存够，需等待
    else {
        printf("wait for imu\n");
        return false;
    }
    return true;
}

/**
 * 判断t时刻的IMU数据是否可用
 * @param t 补偿后的当前帧时间
 * @return
 */
bool Estimator::IMUAvailable(double t) {
    // 加速度队列有有数据 且 当前帧时间<=加速度队列中最后一个数据的时间，说明存完了前一帧到当前帧的IMU数据
    if (!accBuf.empty() && t <= accBuf.back().first)
        return true;
    else
        return false;
}

/**
 * @brief 处理 IMU数据 和 图像特征点数据，进行非线性优化，得到相机位姿：
 * 1.IMU积分更新位姿
 * 2.利用当前帧图像信息，进行后续的关键帧判断、初始化、非线性优化、边缘化、滑动窗口移动等操作
 * 3.将结果用ROS话题发布出去
 */
void Estimator::processMeasurements() {

    while (1) {
        // printf("process measurments\n");

        // 存储一帧图像的特征点信息，key：时间戳，value：特征点数据
        pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>> feature;

        // 两帧图像之间的IMU数据，key是时间戳，value是IMU数据值 (加速度计/陀螺仪数据)
        vector<pair<double, Eigen::Vector3d>> accVector, gyrVector;

        // featureBuf 中有东西，开始进行以下处理
        // 有图像数据后，程序才发给跟踪器叫他产生feature，因此当featureBuf不等于空，所有的buffer，包括imu,图像，都不为空
        if (!featureBuf.empty()) {
            // 1. 取出当前帧识别的特征点信息: key: 时间戳, value: feature_id，[camera_id (0为左目，1为右目), x, y, z (去畸变的归一化相机平面坐标), pu, pv
            // 像素坐标), vx, vy (归一化相机平面移动速度)]
            feature = featureBuf.front();

            // 由于触发器等各种原因，IMU和图像帧之间存在时间延迟，因此需要进行补偿。秦通博士处理：将td认为是一个常值（在极短时间内是不变化的）
            // cam_time + td = imu_time
            curTime = feature.first + td;

            // 2. 等待合适的IMU数据
            while (1) {
                // 非IMU模式 或 当前帧时刻的IMU数据可用，则进行后面的
                if ((!USE_IMU || IMUAvailable(feature.first + td)))
                    break;
                // IMU模式 且 IMU数据不可用，则直接返回，认为处理完毕
                else {
                    printf("wait for imu ... \n"); // 有时候程序一直报wait for imu...，可能是因为配置文件中的td设置太大了
                    if (!MULTIPLE_THREAD)
                        return;
                    std::chrono::milliseconds dura(5);
                    std::this_thread::sleep_for(dura);
                }
            }

            mBuf.lock();

            // 3. 获取前一帧与当前帧图像之间的IMU数据：时间戳，加速度计/陀螺仪数据提取出来，存入accVector、gyrVector中
            if (USE_IMU)
                getIMUInterval(prevTime, curTime, accVector, gyrVector);

            // 从featureBuf中弹出当前帧，意为已操作完，可以从队列中清除了
            featureBuf.pop();
            mBuf.unlock();

            // 4. IMU积分更新位姿
            if (USE_IMU) {
                // 如果第一帧的IMU位姿未初始化，则初始化
                if (!initFirstPoseFlag) {
                    // 因为IMU不是水平放置，所以Z轴和{0,0, 1.0}对齐，通过对齐获得Rs[0]的初始位姿将初始时刻加速度方向对齐重力加速度方向，得到一个旋转矩阵，使得初始IMU的z轴指向重力加速度方向
                    initFirstIMUPose(accVector);
                }

                // 第一帧的IMU位姿已初始化，则进行 IMU积分
                // 用前一图像帧位姿，前一图像帧与当前图像帧之间的IMU数据，积分计算得到当前图像帧位姿 Rs，Ps，Vs

                // 遍历两帧间的IMU数据，对相邻时刻的IMU数据，进行积分
                for (size_t i = 0; i < accVector.size(); i++) {
                    // 前一时刻 到 当前时刻的 dt
                    double dt;
                    if (i == 0) {
                        dt = accVector[i].first - prevTime;
                    } else if (i == accVector.size() - 1) {
                        dt = curTime - accVector[i - 1].first;
                    } else {
                        dt = accVector[i].first - accVector[i - 1].first;
                    }
                    //! IMU预积分，更新位姿
                    processIMU(accVector[i].first, dt, accVector[i].second, gyrVector[i].second);
                }
            }
            mProcess.lock();

            //! 5. 利用当前帧图像信息，进行后续的关键帧判断、初始化、非线性优化、边缘化、滑动窗口移动等操作
            processImage(feature.second, feature.first);

            prevTime = curTime;

            printStatistics(*this, 0);

            // 6. 发布ROS话题
            std_msgs::Header header;
            header.frame_id = "world";
            header.stamp = ros::Time(feature.first);
            // 发布
            pubOdometry(*this, header);
            pubKeyPoses(*this, header);
            pubCameraPose(*this, header);
            pubPointCloud(*this, header);
            pubKeyframe(*this);
            pubTF(*this, header);
            mProcess.unlock();
        }

        // 如果是单线程，就会退出这个循环
        // 也就是每次Estimator::inputImage输入图片，然后执行该函数，结束该函数，然后等待下一次Estimator::inputImage，成为串行的结构
        if (!MULTIPLE_THREAD)
            break;

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

/**
 * 初始第一帧的 IMU姿态初始化
 * 用初始时刻加速度方向对齐重力加速度方向，得到一个旋转矩阵，使得初始IMU的z轴指向重力加速度方向
 * @param accVector     第一帧t0-t1时间段的加速度数据
 */
void Estimator::initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector) {
    printf("init first imu pose\n");
    initFirstPoseFlag = true;
    // return;

    //! step1: 计算这段时间的平均加速度
    Eigen::Vector3d averAcc(0, 0, 0); // 平均加速度初始化为0
    int n = (int)accVector.size();    // 这段时间的加速度数据 个数
    for (size_t i = 0; i < accVector.size(); i++) {
        averAcc = averAcc + accVector[i].second;
    }
    averAcc = averAcc / n;
    printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());

    //! step2: 计算初始IMU的z轴对齐到重力加速度方向所需的旋转矩阵
    // 后面每时刻的位姿都是在当前初始IMU坐标系下的，乘上R0就是世界坐标系了
    // 如果初始时刻IMU是绝对水平放置，那么z轴是对应重力加速度方向的，但如果倾斜了，那么就是需要这个旋转让它竖直
    Matrix3d R0 = Utility::g2R(averAcc);

    // 获取这个旋转矩阵的yaw角（绕z轴转），单位为°
    double yaw = Utility::R2ypr(R0).x(); // 已经是0了，后面似乎多余
    // cout << "init R0 before " << endl << R0 << endl;
    // cout << "yaw:" << yaw << endl;
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    Rs[0] = R0;
    cout << "init R0 " << endl << Rs[0] << endl;
    // Vs[0] = Vector3d(5, 0, 0);
}

void Estimator::initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r) {
    Ps[0] = p;
    Rs[0] = r;
    initP = p;
    initR = r;
}

/**
 * IMU预积分，积分结果会作为后端非线性优化的初始值，包括RPV和delta_RPV
 * 用前一图像帧位姿，前一图像帧与当前图像帧之间的IMU数据，积分计算得到当前图像帧位姿 Rs，Ps，Vs
 * @param t                     两帧间IMU数据 某时刻
 * @param dt                    与前一IMU时刻的 间隔
 * @param linear_acceleration   当前时刻IMU加速度
 * @param angular_velocity      当前时刻IMU角速度
 */
void Estimator::processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity) {
    // 未处理过第一帧的IMU数据
    if (!first_imu) {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    // 滑动窗口内的第frame_count帧，即当前帧，未创建预积分器
    if (!pre_integrations[frame_count]) {
        // 则新建一个
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }

    if (frame_count != 0) {
        // 当前帧的预积分器，添加前一图像帧与当前图像帧之间的 每个时刻的IMU数据
        // push_back重载的时候就已经进行了预积分
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);

        // if(solver_flag != NON_LINEAR)
        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        // 缓存IMU数据
        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        // Rs Ps Vs是frame_count这一个图像帧开始的预积分值,是在绝对坐标系下的
        int j = frame_count;
        // 前一时刻加速度 （去除了零偏）
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        // 中值积分，用前一时刻与当前时刻角速度平均值，对时间积分
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        // 当前时刻姿态Q
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        // 当前时刻加速度
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        // 中值积分，用前一时刻与当前时刻加速度平均值，对时间积分
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        // 当前时刻位置P
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        // 当前时刻速度V
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

/**
 * 对图像特征点和IMU预积分结果进行后端非线性优化：
 * 1、提取前一帧与当前帧的匹配点
 * 2、在线标定外参旋转
 *     利用两帧之间的Camera旋转和IMU积分旋转，构建最小二乘问题，SVD求解外参旋转
 *     1)
 * Camera系，两帧匹配点计算本质矩阵E，分解得到四个解，根据三角化成功点比例确定最终正确解
 * R、t，得到两帧之间的旋转R 2) IMU系，积分计算两帧之间的旋转 3)
 * 根据旋转构建最小二乘问题，SVD求解外参旋转 3、系统初始化 4、3d-2d
 * Pnp求解当前帧位姿 5、三角化当前帧特征点
 * 6、滑窗执行Ceres优化，边缘化，更新滑窗内图像帧的状态（位姿、速度、偏置、外参、逆深度、相机与IMU时差）
 * 7、剔除outlier点
 * 8、用当前帧与前一帧位姿变换，估计下一帧位姿，初始化下一帧特征点的位置
 * 9、移动滑窗，更新特征点的观测帧集合、观测帧索引（在滑窗中的位置）、首帧观测帧和深度值，删除没有观测帧的特征点
 * 10、删除优化后深度值为负的特征点
 * @param image  当前帧特征  feature_id，[camera_id (0为左目，1为右目), x, y, z
 * (去畸变的归一化相机平面坐标), pu, pv (像素坐标), vx, vy
 * (归一化相机平面移动速度)]
 * @param header 当前帧时间戳
 */
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const double header) {
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());
    // VINS为了减少优化的计算量，只优化滑动窗口内的帧，因此保证滑动窗口内帧的质量很关键，每来新的一帧是一定会加入到滑动窗口中的，
    // 但是要挤出去的是上一帧还是窗口最旧帧
    // 是依据新的一帧是否为关键帧决定，保证了滑动窗口中处理的最新帧可能不是关键帧，其他帧都会是关键帧

    // 1. 判断当前帧是否为关键帧，同时完成特征点和帧之间关系的建立
    if (f_manager.addFeatureCheckParallax(frame_count, image, td)) {
        marginalization_flag = MARGIN_OLD; // 如果当前帧为关键帧，则边缘化marg滑动窗口中最旧的帧
                                           // printf("keyframe\n");
    } else {
        marginalization_flag =
            MARGIN_SECOND_NEW; // 如果不是关键帧，边缘化marg滑动窗口的上一帧（为什么不是关键帧要把此新帧踢出去，因为是否关键帧是判断其与前面几帧的视差大不大）
                               // printf("non-keyframe\n");
    }

    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;

    ImageFrame imageframe(image, header);

    imageframe.pre_integration = tmp_pre_integration; // 当前帧预积分（前一帧与当前帧之间的IMU预积分）

    // 把输入图像插入到 all_image_frame 中
    all_image_frame.insert(make_pair(header, imageframe));
    // 重置预积分器
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    // 2.
    // 进行camera到IMU(body)外参的标定。如果成功则把ESTIMATE_EXTRINSIC置1,输出ric和RIC
    // 2表示没有外参，需要标定，Ric初始化为单位阵，tic初始化为零向量。0:
    // 有确定的Tbc外参；1: 有初始估计的外参，之后还要优化；
    if (ESTIMATE_EXTRINSIC == 2) {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0) {
            // 找到frame_count - 1帧和frame_count帧的匹配特征点对
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);

            // 在线标定一个imu_T_cam外参作为初始值
            Matrix3d calib_ric;
            /**
             * 当外参完全不知道的时候，可以在线对其进行初步估计,然后在后续优化时，会在optimize函数中再次优化
             * 利用两帧之间的Camera旋转和IMU积分旋转，构建最小二乘问题，SVD求解外参旋转
             * 1、Camera系，两帧匹配点计算本质矩阵E，分解得到四个解，根据三角化成功点比例确定最终正确解R、t，得到两帧之间的旋转R
             * 2、IMU系，积分计算两帧之间的旋转
             * 3、根据旋转构建最小二乘问题，SVD求解外参旋转
             */
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric)) {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                // 然后标记变为：提供外参初始值，但需优化
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    // 3. 系统未初始化，则进行初始化。有三种模式：单目+IMU、双目+IMU、双目
    if (solver_flag == INITIAL) {
        // 3.1 单目+IMU初始化
        if (!STEREO && USE_IMU) {
            // 要求滑窗满，个数=10，可能是因为相比较双目，单目的尺度不确定性，需要多帧及特征点恢复
            // frame_count此时还没有更新，所以当前帧的ID是frame_count+1，也就是说现在一共有WINDOW_SIZE+1帧
            if (frame_count == WINDOW_SIZE) {
                bool result = false;
                // 如果上次初始化没有成功，要求间隔0.1s，才会进行新的初始化
                if (ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1) {
                    /**
                     * todo
                     * 系统初始化
                     * 1、计算滑窗内IMU加速度的标准差，用于判断移动快慢
                     * 2、在滑窗中找到与当前帧具有足够大的视差，同时匹配较为准确的一帧，计算相对位姿变换
                     *   1)
                     * 提取滑窗中每帧与当前帧之间的匹配点（要求点在两帧之间一直被跟踪到，属于稳定共视点），超过20个则计算视差
                     *   2) 两帧匹配点计算本质矩阵E，恢复R、t
                     *   3) 视差超过30像素，匹配内点数超过12个，则认为符合要求，返回当前帧
                     * 3、以上面找到的这一帧为参考系，Pnp计算滑窗每帧位姿，然后三角化所有特征点，构建BA（最小化点三角化前后误差）优化每帧位姿
                     *   1) 3d-2d Pnp求解每帧位姿
                     *   2) 对每帧与l帧、当前帧三角化
                     *   3) 构建BA，最小化点三角化前后误差，优化每帧位姿
                     *   4) 保存三角化点
                     * 4、对滑窗中所有帧执行Pnp优化位姿
                     * 5、Camera与IMU初始化，零偏、尺度、重力方向
                     */
                    result = initialStructure();
                    initial_timestamp = header;
                }
                // 如果初始化成功，就执行后端非线性优化
                if (result) {
                    // 先进行一次滑动窗口的Ceres非线性优化，边缘化，更新滑窗内图像帧的状态（位姿、速度、偏置、外参、逆深度、相机与IMU时差）
                    // 得到当前帧与第一帧的位姿
                    optimization();
                    // 用优化后的当前帧位姿更新IMU积分的基础位姿，用于展示IMU轨迹
                    updateLatestStates();
                    solver_flag = NON_LINEAR;
                    // 移动滑窗，更新特征点的观测帧集合、观测帧索引（在滑窗中的位置）、首帧观测帧和深度值，删除没有观测帧的特征点
                    slideWindow();
                    ROS_INFO("Initialization finish!");
                } else { // 初始化失败，滑掉这一窗
                    slideWindow();
                }
            }
        }

        // 3.2 双目+IMU初始化
        if (STEREO && USE_IMU) {
            // PnP求解当前帧的位姿：w_T_imu
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            // 双目三角化，恢复所有帧的特征点深度，结果放入 feature 的 estimated_depth
            // 中
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);

            // 如果滑动窗口第一次满了
            if (frame_count == WINDOW_SIZE) {
                map<double, ImageFrame>::iterator frame_it;
                int i = 0;
                for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++) {
                    frame_it->second.R = Rs[i];
                    frame_it->second.T = Ps[i];
                    i++;
                }

                // // 陀螺仪零偏校正，并根据更新后的bg进行IMU积分更新
                solveGyroscopeBias(all_image_frame, Bgs);
                // 依据新的IMU的加速度和角速度偏置值，重新IMU预积分（预积分的好处在于当得到新的Bgs，不需要重新再积分一遍，可以通过Bgs对位姿，速度的一阶导数，进行线性近似，得到新的Bgs求解出IMU的最终结果）
                for (int i = 0; i <= WINDOW_SIZE; i++) {
                    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
                }

                // 初始化成功了，就执行后端非线性优化
                // 滑窗执行Ceres优化，边缘化，更新滑窗内图像帧的状态（位姿、速度、偏置、外参、逆深度、相机与IMU时差）
                optimization();
                // 用优化后的当前帧位姿更新IMU积分的基础位姿，用于展示IMU轨迹
                updateLatestStates();
                solver_flag = NON_LINEAR;
                // 移动滑窗，更新特征点的观测帧集合、观测帧索引（在滑窗中的位置）、首帧观测帧和深度值，删除没有观测帧的特征点
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
        }

        // stereo only initilization
        // 3.3 双目初始化
        if (STEREO && !USE_IMU) {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);

            optimization(); // 优化

            // 如果滑动窗口满了，就执行后端的非线性优化
            if (frame_count == WINDOW_SIZE) {
                // 滑窗执行Ceres优化，边缘化，更新滑窗内图像帧的状态（位姿、速度、偏置、外参、逆深度、相机与IMU时差）
                optimization();
                // 用优化后的当前帧位姿更新IMU积分的基础位姿，用于展示IMU轨迹
                updateLatestStates();
                solver_flag = NON_LINEAR;
                // 移动滑窗，更新特征点的观测帧集合、观测帧索引（在滑窗中的位置）、首帧观测帧和深度值，删除没有观测帧的特征点
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
        }

        // 如果滑动窗口还没有满就添加到滑动窗口中
        if (frame_count < WINDOW_SIZE) {
            frame_count++;
            // 注意，这里frame_count已经是下一帧的索引了，这里就是把当前帧估计的位姿
            // 当作 下一帧的初始值
            int prev_frame = frame_count - 1;
            Ps[frame_count] = Ps[prev_frame];
            Vs[frame_count] = Vs[prev_frame];
            Rs[frame_count] = Rs[prev_frame];
            Bas[frame_count] = Bas[prev_frame];
            Bgs[frame_count] = Bgs[prev_frame];
        }

    }
    // 4. 如果完成了初始化，就进行后端优化
    // 在完成初始化后就只进行后端非线性优化了，还是需要将滑窗中的特征点尽可能多地恢复出对应的3D点，获取多帧之间更多的约束，
    // 进而得到更多的优化观测量, 使得优化结果更加鲁棒
    else {
        TicToc t_solve;
        // 纯视觉，3d-2d PnP求解当前帧位姿
        if (!USE_IMU) {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
        }
        // 三角化当前帧特征点
        f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
        // 滑窗执行Ceres优化，边缘化，更新滑窗内图像帧的状态（位姿、速度、偏置、外参、逆深度、相机与IMU时差）
        optimization();
        set<int> removeIndex;
        /**
         * 剔除outlier点
         * 遍历特征点，计算观测帧与首帧观测帧之间的重投影误差，计算误差均值，超过3个像素则被剔除
         */
        outliersRejection(removeIndex);
        // 实际调用剔除
        f_manager.removeOutlier(removeIndex);
        if (!MULTIPLE_THREAD) {
            featureTracker.removeOutliers(removeIndex);
            // 用当前帧与前一帧位姿变换，估计下一帧位姿，初始化下一帧特征点的位置
            predictPtsInNextFrame();
        }

        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        // 失败检测，失败了重置系统
        if (failureDetection()) {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }

        // 移动滑窗，更新特征点的观测帧集合、观测帧索引（在滑窗中的位置）、首帧观测帧和深度值，删除没有观测帧的特征点
        slideWindow();
        // 删除优化后深度值为负的特征点
        f_manager.removeFailures();
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
        // 用优化后的当前帧位姿更新IMU积分的基础位姿，用于展示IMU轨迹
        updateLatestStates();
    }
}

/**
 * todo
 * 系统初始化
 * 1、计算滑窗内IMU加速度的标准差，用于判断移动快慢
 * 2、在滑窗中找到与当前帧具有足够大的视差，同时匹配较为准确的一帧，计算相对位姿变换
 *   1)
 * 提取滑窗中每帧与当前帧之间的匹配点（要求点在两帧之间一直被跟踪到，属于稳定共视点），超过20个则计算视差
 *   2) 两帧匹配点计算本质矩阵E，恢复R、t
 *   3) 视差超过30像素，匹配内点数超过12个，则认为符合要求，返回当前帧
 * 3、以上面找到的这一帧为参考系，Pnp计算滑窗每帧位姿，然后三角化所有特征点，构建BA（最小化点三角化前后误差）优化每帧位姿
 *   1) 3d-2d Pnp求解每帧位姿
 *   2) 对每帧与l帧、当前帧三角化
 *   3) 构建BA，最小化点三角化前后误差，优化每帧位姿
 *   4) 保存三角化点
 * 4、对滑窗中所有帧执行Pnp优化位姿
 * 5、Camera与IMU初始化，零偏、尺度、重力方向
 */
bool Estimator::initialStructure() {
    TicToc t_sfm;
    // check imu observibility
    // 计算滑窗内IMU加速度的标准差，用于判断移动快慢，为什么不用旋转或者平移量判断
    // todo
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        // 从第2帧开始累加每帧加速度
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++) {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        // 加速度均值
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++) {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            // cout << "frame g " << tmp_g.transpose() << endl;
        }
        // 加速度标准差
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        // ROS_WARN("IMU variation %f!", var);
        if (var < 0.25) {
            ROS_INFO("IMU excitation not enouth!");
            // return false;
        }
    }
    // global sfm
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    // 遍历当前帧特征点
    for (auto &it_per_id : f_manager.feature) {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        // 遍历特征点出现的帧
        for (auto &it_per_frame : it_per_id.feature_per_frame) {
            imu_j++;
            // 特征点归一化相机平面点
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    }
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    /**
     * 在滑窗中找到与当前帧具有足够大的视差，同时匹配较为准确的一帧，计算相对位姿变换
     * 1、提取滑窗中每帧与当前帧之间的匹配点（要求点在两帧之间一直被跟踪到，属于稳定共视点），超过20个则计算视差
     * 2、两帧匹配点计算本质矩阵E，恢复R、t
     * 3、视差超过30像素，匹配内点数超过12个，则认为符合要求，返回当前帧
     */
    if (!relativePose(relative_R, relative_T, l)) {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    /**
     * 以第l帧为参考系，Pnp计算滑窗每帧位姿，然后三角化所有特征点，构建BA（最小化点三角化前后误差）优化每帧位姿
     * 1、3d-2d Pnp求解每帧位姿
     * 2、对每帧与l帧、当前帧三角化
     * 3、构建BA，最小化点三角化前后误差，优化每帧位姿
     * 4、保存三角化点
     */
    GlobalSFM sfm;
    if (!sfm.construct(frame_count + 1, Q, T, l, relative_R, relative_T, sfm_f, sfm_tracked_points)) {
        ROS_DEBUG("global SFM failed!");
        // 如果SFM三角化失败，marg最早的一帧
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    // solve pnp for all frame
    // 对滑窗中所有帧执行Pnp位姿优化 todo
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin();
    for (int i = 0; frame_it != all_image_frame.end(); frame_it++) {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if ((frame_it->first) == Headers[i]) {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if ((frame_it->first) > Headers[i]) {
            i++;
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = -R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        // 遍历当前帧特征点
        for (auto &id_pts : frame_it->second.points) {
            int feature_id = id_pts.first;
            // 遍历特征点的观测帧，提取像素坐标
            for (auto &i_p : id_pts.second) {
                it = sfm_tracked_points.find(feature_id);
                if (it != sfm_tracked_points.end()) {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    // 特征点3d坐标，sfm三角化后的点
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    // 特征点像素坐标
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        if (pts_3_vector.size() < 6) {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        // 3d-2d Pnp求解位姿
        if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1)) {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp, tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }
    // Camera与IMU初始化，零偏、尺度、重力方向
    if (visualInitialAlign())
        return true;
    else {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }
}

/**
 * 视觉和惯性的对齐, 对应
 * https://mp.weixin.qq.com/s/9twYJMOE8oydAzqND0UmFw中的visualInitialAlign
 * 分为5步:
 * 1、估计旋转外参
 * 2、估计陀螺仪bias
 * 3、估计中立方向,速度.尺度初始值
 * 4、对重力加速度进一步优化
 * 5、将轨迹对其到世界坐标系
 */
bool Estimator::visualInitialAlign() {
    TicToc t_g;
    VectorXd x;
    // solve scale
    // Camera与IMU初始化，零偏、尺度、重力方向
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if (!result) {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state
    for (int i = 0; i <= frame_count; i++) {
        Matrix3d Ri = all_image_frame[Headers[i]].R;
        Vector3d Pi = all_image_frame[Headers[i]].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i]].is_key_frame = true;
    }

    double s = (x.tail<1>())(0);
    for (int i = 0; i <= WINDOW_SIZE; i++) {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++) {
        if (frame_i->second.is_key_frame) {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }

    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    // Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++) {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());

    f_manager.clearDepth();
    f_manager.triangulate(frame_count, Ps, Rs, tic, ric);

    return true;
}

/**
 * 在滑窗中找到与当前帧具有足够大的视差，同时匹配较为准确的一帧，计算相对位姿变换
 * 1、提取滑窗中每帧与当前帧之间的匹配点（要求点在两帧之间一直被跟踪到，属于稳定共视点），超过20个则计算视差
 * 2、两帧匹配点计算本质矩阵E，恢复R、t
 * 3、视差超过30像素，匹配内点数超过12个，则认为符合要求，返回当前帧
 */
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l) {
    // find previous frame which contains enough correspondance and parallex with
    // newest frame 遍历滑窗
    for (int i = 0; i < WINDOW_SIZE; i++) {
        // 提取滑窗中每帧与当前帧之间的匹配点（要求点在两帧之间一直被跟踪到，属于稳定共视点）
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        // 匹配点超过20个
        if (corres.size() > 20) {
            // 计算匹配点之间的累积、平均视差（归一化相机坐标系下），作为当前两帧之间的视差
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++) {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;
            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            // 视差超过30个像素点；两帧匹配点计算本质矩阵E，恢复R、t，内点数超过12个
            if (average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T)) {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate "
                          "the whole structure",
                          average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

/**
 * 滑窗中的帧位姿、速度、偏置、外参、特征点逆深度等参数，转换成数组
 * ceres参数需要是数组形式
 */
void Estimator::vector2double() {
    for (int i = 0; i <= WINDOW_SIZE; i++) {
        // 平移、旋转
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        if (USE_IMU) {
            // 速度、加速度偏置、角速度偏置
            para_SpeedBias[i][0] = Vs[i].x();
            para_SpeedBias[i][1] = Vs[i].y();
            para_SpeedBias[i][2] = Vs[i].z();

            para_SpeedBias[i][3] = Bas[i].x();
            para_SpeedBias[i][4] = Bas[i].y();
            para_SpeedBias[i][5] = Bas[i].z();

            para_SpeedBias[i][6] = Bgs[i].x();
            para_SpeedBias[i][7] = Bgs[i].y();
            para_SpeedBias[i][8] = Bgs[i].z();
        }
    }

    for (int i = 0; i < NUM_OF_CAM; i++) {
        // 相机与IMU外参
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    // 当前帧特征点逆深度，限观测帧数量大于4个的特征点
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);

    // 相机与IMU时差
    para_Td[0][0] = td;
}

/**
 * 更新优化后的参数，包括位姿、速度、偏置、外参、特征点逆深度、相机与IMU时差
 */
void Estimator::double2vector() {
    // 第一帧优化前位姿
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    // 如果上一次优化失败了，Rs、Ps都会被清空，用备份的last_R0、last_P0
    if (failure_occur) {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }

    // 使用IMU时，第一帧没有固定，会有姿态变化
    if (USE_IMU) {
        // 本次优化后第一帧位姿
        Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6], para_Pose[0][3], para_Pose[0][4], para_Pose[0][5]).toRotationMatrix());
        // yaw角差量
        double y_diff = origin_R0.x() - origin_R00.x();
        // yaw角差量对应旋转矩阵
        Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));

        // pitch角接近90°，todo
        if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0) {
            ROS_DEBUG("euler singular point!");
            // 计算旋转位姿变换
            rot_diff = Rs[0] * Quaterniond(para_Pose[0][6], para_Pose[0][3], para_Pose[0][4], para_Pose[0][5]).toRotationMatrix().transpose();
        }

        // 遍历滑窗，位姿、速度全部施加优化前后第一帧的位姿变换（只有旋转） todo
        for (int i = 0; i <= WINDOW_SIZE; i++) {
            Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

            Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0], para_Pose[i][1] - para_Pose[0][1], para_Pose[i][2] - para_Pose[0][2]) + origin_P0;

            Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0], para_SpeedBias[i][1], para_SpeedBias[i][2]);

            Bas[i] = Vector3d(para_SpeedBias[i][3], para_SpeedBias[i][4], para_SpeedBias[i][5]);

            Bgs[i] = Vector3d(para_SpeedBias[i][6], para_SpeedBias[i][7], para_SpeedBias[i][8]);
        }
    }
    // 不使用IMU时，第一帧固定，后面的位姿直接赋值
    else {
        for (int i = 0; i <= WINDOW_SIZE; i++) {
            Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

            Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
        }
    }

    // 更新外参
    if (USE_IMU) {
        for (int i = 0; i < NUM_OF_CAM; i++) {
            tic[i] = Vector3d(para_Ex_Pose[i][0], para_Ex_Pose[i][1], para_Ex_Pose[i][2]);
            ric[i] = Quaterniond(para_Ex_Pose[i][6], para_Ex_Pose[i][3], para_Ex_Pose[i][4], para_Ex_Pose[i][5]).normalized().toRotationMatrix();
        }
    }

    // 更新逆深度
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);

    // 更新相机与IMU时差
    if (USE_IMU)
        td = para_Td[0][0];
}

/**
 * 失败检测
 */
bool Estimator::failureDetection() {
    return false;
    // 上一帧老特征点几乎全丢
    if (f_manager.last_track_num < 2) {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        // return true;
    }
    // 偏置太大
    if (Bas[WINDOW_SIZE].norm() > 2.5) {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    // 偏置太大
    if (Bgs[WINDOW_SIZE].norm() > 1.0) {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    // 优化前后当前帧位置相差过大
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5) {
        // ROS_INFO(" big translation");
        // return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1) {
        // ROS_INFO(" big z translation");
        // return true;
    }
    // 当前帧位姿优化前后角度相差过大
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50) {
        ROS_INFO(" big delta_angle ");
        // return true;
    }
    return false;
}

/**
 * 滑动窗口执行Ceres非线性优化，边缘化，更新滑窗内图像帧的状态（位姿、速度、偏置、外参、逆深度、相机与IMU时差）
 */
void Estimator::optimization() {

    TicToc t_whole, t_prepare;
    // 滑窗中的帧位姿、速度、偏置、外参、特征点逆深度等参数，转换成数组
    vector2double();

    ceres::Problem problem; // 优化问题对象

    ceres::LossFunction *loss_function; // 指定鲁邦核函数
    // loss_function = NULL;
    loss_function = new ceres::HuberLoss(1.0);
    // loss_function = new ceres::CauchyLoss(1.0 / FOCAL_LENGTH);
    // ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);

    /**
     * Step1. 调用AddParameterBlock，显式添加待优化变量（类似于g2o中添加顶点），需要固定的顶点固定一下
     */
    // ############################## 开始添加各种参数块，也即优化变量 ##############################

    // 遍历滑窗，添加位姿、速度、偏置参数
    // 首先是滑窗帧pose和bias到优化变量
    for (int i = 0; i < frame_count + 1; i++) {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        if (USE_IMU)
            problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }

    // 如果未启用IMU，固定第一帧位姿，IMU下第一帧不固定
    if (!USE_IMU)
        problem.SetParameterBlockConstant(para_Pose[0]);

    // 添加外参到优化变量，若未启用外参估计或激励不足，则设为常量
    for (int i = 0; i < NUM_OF_CAM; i++) {
        // 添加相机与IMU外参
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        // 估计外参
        if ((ESTIMATE_EXTRINSIC && frame_count == WINDOW_SIZE && Vs[0].norm() > 0.2) || openExEstimation) {
            // ROS_INFO("estimate extinsic param");
            openExEstimation = 1;
        } else {
            // ROS_INFO("fix extinsic param");
            // 不估计外参的时候，固定外参
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
    }

    // 添加相机与IMU时差
    // 添加sensor间时延到优化变量，若未启用或激励不足，则设为常量
    problem.AddParameterBlock(para_Td[0], 1);

    if (!ESTIMATE_TD || Vs[0].norm() < 0.2)
        problem.SetParameterBlockConstant(para_Td[0]);

    /**
     * Step2. 调用AddResidualBlock，添加各种残差数据（类似于g2o中的边）
     */
    // ############################## 开始添加各种残差块，也即约束 ##############################

    // (1) 添加先验残差，通过Marg的舒尔补操作，将被Marg部分的信息叠加到了保留变量的信息上
    // 添加边缘化约束，若存在的话
    if (last_marginalization_info && last_marginalization_info->valid) {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL, last_marginalization_parameter_blocks);
    }

    // (2) 添加IMU残差
    // 添加imu预积分约束，若存在的话
    if (USE_IMU) {
        for (int i = 0; i < frame_count; i++) {
            int j = i + 1;
            if (pre_integrations[j]->sum_dt > 10.0)
                continue;
            // 前后帧之间建立IMU残差
            IMUFactor *imu_factor = new IMUFactor(pre_integrations[j]);
            // 后面四个参数为变量初始值，优化过程中会更新
            problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
        }
    }

    // (3) 添加视觉重投影残差
    // 添加特征点的重投影约束【重点】
    int f_m_cnt = 0;
    int feature_index = -1;
    // 遍历特征点
    for (auto &it_per_id : f_manager.feature) {
        it_per_id.used_num = it_per_id.feature_per_frame.size();

        if (it_per_id.used_num < 4)
            continue;
        /** 这里应该值得强调一下，为什么只允许稳定（>=4次）的feature才能进入优化问题？
         * 因为涉及到BA求解策略？（在ceres中似乎不至于）
         * 优化结果的稳定性呗？
         */

        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        // 首帧归一化相机平面点
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        // 遍历特征点的观测帧
        for (auto &it_per_frame : it_per_id.feature_per_frame) {
            imu_j++;
            // 非首帧观测帧
            /*添加一次（左目的）帧间重投影误差*/
            if (imu_i != imu_j) {
                // 当前观测帧归一化相机平面点
                Vector3d pts_j = it_per_frame.point;
                // 首帧与当前观测帧建立重投影误差
                ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(
                    pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity, it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                // 优化变量：首帧位姿，当前帧位姿，外参（左目），特征点逆深度，相机与IMU时差
                problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
            }

            // 双目，重投影误差
            /*添加一次（双目的）帧间重投影误差*/
            if (STEREO && it_per_frame.is_stereo) {
                Vector3d pts_j_right = it_per_frame.pointRight;
                if (imu_i != imu_j) {
                    // 首帧与当前观测帧右目建立重投影误差
                    ProjectionTwoFrameTwoCamFactor *f =
                        new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                           it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    // 优化变量：首帧位姿，当前帧位姿，外参（左目），外参（右目），特征点逆深度，相机与IMU时差
                    problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1],
                                             para_Feature[feature_index], para_Td[0]);
                }
                /*添加一次同一帧（左右目）之间的重投影误差*/
                else {
                    // 首帧左右目建立重投影误差
                    ProjectionOneFrameTwoCamFactor *f =
                        new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                           it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    // 优化变量：外参（左目），外参（右目），特征点逆深度，相机与IMU时差
                    problem.AddResidualBlock(f, loss_function, para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
            }
            f_m_cnt++;
        }
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    // printf("prepare for ceres: %f \n", t_prepare.toc());

    /**
     * Step3. 设置优化器参数，执行优化
     */
    // ############################## 设置求解参数 ##############################

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    // options.use_explicit_schur_complement = true;
    // options.minimizer_progress_to_stdout = true;
    // options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;

    // ############################## 求解优化问题 ##############################

    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    // printf("solver costs: %f \n", t_solver.toc());

    // 更新优化后的参数，包括位姿、速度、偏置、外参、特征点逆深度、相机与IMU时差
    double2vector();
    // printf("frame_count: %d \n", frame_count);

    // ############################## 执行边缘化，计算新的边缘化因子 ##############################

    if (frame_count < WINDOW_SIZE)
        return;

    /**
     * Step4. 边缘化操作
     */

    // 以下是边缘化操作
    TicToc t_whole_marginalization;
    // Marg最早帧
    if (marginalization_flag == MARGIN_OLD) {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        // 滑窗中的帧位姿、速度、偏置、外参、特征点逆深度等参数，转换成数组
        vector2double();

        // 先验残差
        if (last_marginalization_info && last_marginalization_info->valid) {
            vector<int> drop_set;
            // 上一次Marg剩下的参数块
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++) {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] || last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL, last_marginalization_parameter_blocks, drop_set);
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        // 滑窗首帧与后一帧之间的IMU残差
        if (USE_IMU) {
            if (pre_integrations[1]->sum_dt < 10.0) {
                IMUFactor *imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
                    imu_factor, NULL, vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]}, vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        // 滑窗首帧与其他帧之间的视觉重投影残差
        {
            // 遍历特征点
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature) {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (it_per_id.used_num < 4)
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                // 首帧观测帧归一化相机平面点
                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                // 遍历观测帧
                for (auto &it_per_frame : it_per_id.feature_per_frame) {
                    imu_j++;
                    // 非首个观测帧
                    if (imu_i != imu_j) {
                        Vector3d pts_j = it_per_frame.point;
                        ProjectionTwoFrameOneCamFactor *f_td =
                            new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                               it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
                            f_td, loss_function, vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                            vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    if (STEREO && it_per_frame.is_stereo) {
                        Vector3d pts_j_right = it_per_frame.pointRight;
                        if (imu_i != imu_j) {
                            ProjectionTwoFrameTwoCamFactor *f =
                                new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                   it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
                                f, loss_function,
                                vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                vector<int>{0, 4});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        } else {
                            ProjectionOneFrameTwoCamFactor *f =
                                new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                   it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
                                f, loss_function, vector<double *>{para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]}, vector<int>{2});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                    }
                }
            }
        }

        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());

        // 执行marg边缘化
        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        // marg首帧之后，将参数数组中每个位置的值设为前面元素的值，记录到addr_shift里面
        // [<p1,p0>,<p2,p1>,...]
        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++) {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            printf("pose %ld,%ld\n", reinterpret_cast<long>(para_Pose[i]), reinterpret_cast<long>(para_Pose[i - 1]));
            if (USE_IMU) {
                addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                printf("speedBias %ld,%ld\n", reinterpret_cast<long>(para_SpeedBias[i]), reinterpret_cast<long>(para_SpeedBias[i - 1]));
            }
        }
        for (int i = 0; i < NUM_OF_CAM; i++) {
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            printf("exPose %ld,%ld\n", reinterpret_cast<long>(para_Ex_Pose[i]), reinterpret_cast<long>(para_Ex_Pose[i]));
        }

        addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        printf("td %ld,%ld\n", reinterpret_cast<long>(para_Td[0]), reinterpret_cast<long>(para_Td[0]));

        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        // 保存marg信息
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;

    }
    // Marg新帧
    else {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1])) {
            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info && last_marginalization_info->valid) {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++) {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL, last_marginalization_parameter_blocks, drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());

            // [<p0,p0>, <p1,p1>,...,<pn-1,pn-1>,<pn,pn-1>]
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++) {
                // 被Marg帧
                if (i == WINDOW_SIZE - 1)
                    continue;
                // 当前帧
                else if (i == WINDOW_SIZE) {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    if (USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                } else {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    if (USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
        }
    }
    // printf("whole marginalization costs: %f \n",
    // t_whole_marginalization.toc()); printf("whole time for ceres: %f \n",
    // t_whole.toc());
}

/**
 * 移动滑窗，更新特征点的观测帧集合、观测帧索引（在滑窗中的位置）、首帧观测帧和深度值，删除没有观测帧的特征点
 */
void Estimator::slideWindow() {
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD) {
        double t_0 = Headers[0];
        // 备份
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE) {
            // 删除第一帧，全部数据往前移
            for (int i = 0; i < WINDOW_SIZE; i++) {
                Headers[i] = Headers[i + 1];
                Rs[i].swap(Rs[i + 1]);
                Ps[i].swap(Ps[i + 1]);
                if (USE_IMU) {
                    std::swap(pre_integrations[i], pre_integrations[i + 1]);

                    dt_buf[i].swap(dt_buf[i + 1]);
                    linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                    angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                    Vs[i].swap(Vs[i + 1]);
                    Bas[i].swap(Bas[i + 1]);
                    Bgs[i].swap(Bgs[i + 1]);
                }
            }
            // 新增末尾帧，用当前帧数据初始化
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

            if (USE_IMU) {
                Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
                Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
                Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }
            // 删除首帧image
            if (true || solver_flag == INITIAL) {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                all_image_frame.erase(all_image_frame.begin(), it_0);
            }
            // 移动滑窗，从特征点观测帧集合中删除该帧，计算新首帧深度值
            slideWindowOld();
        }
    } else {
        if (frame_count == WINDOW_SIZE) {
            // 当前帧的前一帧用当前帧数据代替，当前帧还留下了
            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];

            if (USE_IMU) {
                // 当前帧的imu数据
                for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++) {
                    double tmp_dt = dt_buf[frame_count][i];
                    Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                    Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                    pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                    dt_buf[frame_count - 1].push_back(tmp_dt);
                    linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                    angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
                }

                Vs[frame_count - 1] = Vs[frame_count];
                Bas[frame_count - 1] = Bas[frame_count];
                Bgs[frame_count - 1] = Bgs[frame_count];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }
            // 边缘化当前帧前面一帧后，从特征点的观测帧集合中删除该帧，如果特征点没有观测帧了，删除这个特征点
            slideWindowNew();
        }
    }
}

/**
 * 边缘化当前帧前面一帧后，从特征点的观测帧集合中删除该帧，如果特征点没有观测帧了，删除这个特征点
 */
void Estimator::slideWindowNew() {
    sum_of_front++;
    f_manager.removeFront(frame_count);
}

/**
 * 移动滑窗，从特征点观测帧集合中删除该帧，计算新首帧深度值
 */
void Estimator::slideWindowOld() {
    sum_of_back++;

    // NON_LINEAR表示已经初始化过了
    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth) {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        // marg帧的位姿 Rwc
        R0 = back_R0 * ric[0];
        // 后面一帧的位姿 Rwc
        R1 = Rs[0] * ric[0];
        // marg帧的位置
        P0 = back_P0 + back_R0 * tic[0];
        // 后面一帧的位置
        P1 = Ps[0] + Rs[0] * tic[0];
        /**
         * 边缘化第一帧后，从特征点的观测帧集合中删除该帧，观测帧的索引相应全部-1，如果特征点没有观测帧少于2帧，删除这个特征点
         * 与首帧绑定的estimated_depth深度值，重新计算
         */
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    } else
        // 边缘化第一帧后，从特征点的观测帧集合中删除该帧，观测帧的索引相应全部-1，如果特征点没有观测帧了，删除这个特征点
        f_manager.removeBack();
}

/**
 * 当前帧位姿 Twi
 */
void Estimator::getPoseInWorldFrame(Eigen::Matrix4d &T) {
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[frame_count];
    T.block<3, 1>(0, 3) = Ps[frame_count];
}

/**
 * 位姿 Twi
 */
void Estimator::getPoseInWorldFrame(int index, Eigen::Matrix4d &T) {
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[index];
    T.block<3, 1>(0, 3) = Ps[index];
}

/**
 * 用当前帧与前一帧位姿变换，估计下一帧位姿，初始化下一帧特征点的位置
 */
void Estimator::predictPtsInNextFrame() {
    // printf("predict pts in next frame\n");
    if (frame_count < 2)
        return;
    // predict next pose. Assume constant velocity motion
    // 当前帧位姿Twc、前一帧位姿Twl
    Eigen::Matrix4d curT, prevT, nextT;
    getPoseInWorldFrame(curT);
    getPoseInWorldFrame(frame_count - 1, prevT);
    // 用前一帧位姿与当前帧位姿的变换，预测下一帧位姿Twn
    nextT = curT * (prevT.inverse() * curT); // Twc * (Tlw * Twc = Tlc = Tcn) = Twn
    map<int, Eigen::Vector3d> predictPts;

    // 遍历特征点
    for (auto &it_per_id : f_manager.feature) {
        if (it_per_id.estimated_depth > 0) {
            int firstIndex = it_per_id.start_frame;
            int lastIndex = it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1;
            // printf("cur frame index  %d last frame index %d\n", frame_count,
            // lastIndex);
            if ((int)it_per_id.feature_per_frame.size() >= 2 && lastIndex == frame_count) {
                double depth = it_per_id.estimated_depth;
                // 特征点在首帧观测帧中的IMU系坐标
                Vector3d pts_j = ric[0] * (depth * it_per_id.feature_per_frame[0].point) + tic[0];
                // 世界坐标
                Vector3d pts_w = Rs[firstIndex] * pts_j + Ps[firstIndex];
                // 转换到下一帧IMU坐标系
                Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() * (pts_w - nextT.block<3, 1>(0, 3));
                // 转到相机系
                Vector3d pts_cam = ric[0].transpose() * (pts_local - tic[0]);
                int ptsIndex = it_per_id.feature_id;
                predictPts[ptsIndex] = pts_cam;
            }
        }
    }
    // 设置下一帧跟踪点初始位置
    featureTracker.setPrediction(predictPts);
    // printf("estimator output %d predict pts\n",(int)predictPts.size());
}

/**
 * 计算重投影误差
 */
double Estimator::reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici, Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj,
                                    double depth, Vector3d &uvi, Vector3d &uvj) {
    // i点对应世界坐标
    Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    // 转换到j相机坐标系
    Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    // 在j归一化相机坐标系下计算误差
    Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}

/**
 * 剔除outlier点
 * 遍历特征点，计算观测帧与首帧观测帧之间的重投影误差，计算误差均值，超过3个像素则被剔除
 */
void Estimator::outliersRejection(set<int> &removeIndex) {
    // return;
    int feature_index = -1;
    // 遍历特征点
    for (auto &it_per_id : f_manager.feature) {
        double err = 0;
        int errCnt = 0;
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;

        feature_index++;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        // 首帧观测帧的归一化相机坐标点
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;
        // 深度值
        double depth = it_per_id.estimated_depth;
        // 遍历观测帧，计算与首帧观测帧之间的特征点重投影误差
        for (auto &it_per_frame : it_per_id.feature_per_frame) {
            imu_j++;
            // 非首帧观测帧
            if (imu_i != imu_j) {
                // 计算重投影误差
                Vector3d pts_j = it_per_frame.point;
                double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], Rs[imu_j], Ps[imu_j], ric[0], tic[0], depth, pts_i, pts_j);
                err += tmp_error;
                errCnt++;
                // printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
            }
            // need to rewrite projecton factor.........
            // 双目，右目同样计算一次
            if (STEREO && it_per_frame.is_stereo) {
                Vector3d pts_j_right = it_per_frame.pointRight;
                if (imu_i != imu_j) {
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], Rs[imu_j], Ps[imu_j], ric[1], tic[1], depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    // printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                } else {
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], Rs[imu_j], Ps[imu_j], ric[1], tic[1], depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    // printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
            }
        }
        // 重投影误差均值（归一化相机坐标系）
        double ave_err = err / errCnt;
        // 误差超过3个像素，就不要了
        if (ave_err * FOCAL_LENGTH > 3)
            removeIndex.insert(it_per_id.feature_id);
    }
}

/**
 * IMU快速预积分: 预测当前时刻的状态：旋转矩阵Q，位置P，速度V，IMU观测值
 * @param t     新一时刻的 时间戳
 * @param linear_acceleration   新一时刻的 加速度
 * @param angular_velocity      新一时刻的 角速度
 */
void Estimator::fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity) {
    double dt = t - latest_time; // 与上一时刻的时间差
    latest_time = t;             // 更新上一时刻的时间为 新时刻

    // 1. 计算上一时刻的 加速度真值（世界系）( 公式15-14得: a_实际 = R_w_b * (a_观测 - 零偏 - 噪声) + g ) TODO 确定这里是+g还是-g
    Eigen::Vector3d un_acc_0 = latest_Q * (latest_acc_0 - latest_Ba) - g;

    // 2. 计算当前时刻的 角速度真值（世界系）( (上一时刻w_观测 + 当前时刻w_观测) / 2 - 零偏)
    Eigen::Vector3d un_gyr = 0.5 * (latest_gyr_0 + angular_velocity) - latest_Bg;

    // 3. 更新为 当前时刻的 旋转矩阵Q：使用新角速度
    latest_Q = latest_Q * Utility::deltaQ(un_gyr * dt);

    // 4. 计算当前时刻的 加速度真值：使用新旋转矩阵 和 当前时刻加速度观测值
    Eigen::Vector3d un_acc_1 = latest_Q * (linear_acceleration - latest_Ba) - g;
    // 5. 计算平均两个时刻后的 加速度真值
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    // 6. 更新为 当前时刻的 位置P: p0 + v0 t + 1/2 a t^2
    latest_P = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;
    // 更新为 当前时刻的 速度V: v0 + a t
    latest_V = latest_V + dt * un_acc;

    // 上一时刻测得的IMU数据 更新为 当前时刻的值
    latest_acc_0 = linear_acceleration;
    latest_gyr_0 = angular_velocity;
}

/**
 * 用优化后的当前帧位姿更新IMU积分的基础位姿，用于展示IMU轨迹
 */
void Estimator::updateLatestStates() {
    mPropagate.lock();
    // 更新当前帧状态，是接下来IMU积分的基础
    latest_time = Headers[frame_count] + td;
    latest_P = Ps[frame_count];
    latest_Q = Rs[frame_count];
    latest_V = Vs[frame_count];
    latest_Ba = Bas[frame_count];
    latest_Bg = Bgs[frame_count];
    latest_acc_0 = acc_0;
    latest_gyr_0 = gyr_0;
    mBuf.lock();
    queue<pair<double, Eigen::Vector3d>> tmp_accBuf = accBuf;
    queue<pair<double, Eigen::Vector3d>> tmp_gyrBuf = gyrBuf;
    mBuf.unlock();
    while (!tmp_accBuf.empty()) {
        // 当前帧之后的一部分imu数据，用当前帧位姿预测，更新到最新，这个只用于IMU路径的展示
        double t = tmp_accBuf.front().first;
        Eigen::Vector3d acc = tmp_accBuf.front().second;
        Eigen::Vector3d gyr = tmp_gyrBuf.front().second;
        fastPredictIMU(t, acc, gyr);
        tmp_accBuf.pop();
        tmp_gyrBuf.pop();
    }
    mPropagate.unlock();
}
