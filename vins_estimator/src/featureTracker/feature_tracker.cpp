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

/*******************************************************
该文件定义了特征点识别的函数
 *******************************************************/

#include "feature_tracker.h"

/**
 * @brief 判断特征点是否在图片边界内
 * @param pt 特征点
 * @return
 */
bool FeatureTracker::inBorder(const cv::Point2f &pt) {
    const int BORDER_SIZE = 1; // 图片边框大小
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}

/**
 * @brief 计算两个特征点的欧氏距离
 * @param pt1 特征点1
 * @param pt2 特征点2
 * @return 两特征点的欧氏距离
 */
double distance(cv::Point2f pt1, cv::Point2f pt2) {
    // printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

/**
 * 删除集合中status为0的点
 */
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status) {
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

/**
 * @brief
 * 剔除特征点向量中的外点（status为0），只保留特征点向量中跟踪成功的特征点
 * @param v 特征点队列，该函数会剔除队列中跟踪失败的点
 * @param status 队列，存储特征点是否跟踪成功
 */
void reduceVector(vector<int> &v, vector<uchar> status) {
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

FeatureTracker::FeatureTracker() {
    stereo_cam = 0;
    n_id = 0;
    hasPrediction = false;
}

/**
 * @brief 非极大值抑制筛选特征点，在一个圆区域中只保留跟踪次数最多的特征点，使特征点均匀化
 */
void FeatureTracker::setMask() {
    // 创建一个单通道的mask，所有元素初始化为255（白色）
    mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));

    // prefer to keep features that are tracked for long time
    // key：特征点跟踪次数，value：(当前帧跟踪到的特征点，当前帧识别的id)
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;
    for (unsigned int i = 0; i < cur_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(cur_pts[i], ids[i])));

    // 当前帧特征点，按照跟踪次数track_cnt，从大到小排序
    sort(cnt_pts_id.begin(), cnt_pts_id.end(),
         [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b) { return a.first > b.first; });

    // 先清空
    cur_pts.clear();
    ids.clear();
    track_cnt.clear();

    // 非极大值抑制，相当于在一个圆区域中只保留跟踪次数最多的特征点
    // 按当前帧特征点的跟踪次数遍历
    for (auto &it : cnt_pts_id) {
        // 如果对应位置上mask的值为255（白色），即说明该特征点符合要求，则再存入其信息，保留该特征点；
        // 并将它在mask对应位置周围半径MIN_DIST范围内的值置为0，表示删除其半径为30个像素内的特征点
        if (mask.at<uchar>(it.second.first) == 255) {
            // 上面清空了，再重新加进来，并已按跟踪次数排序
            cur_pts.push_back(it.second.first); // 再次存入该特征点到 cur_pts中
            ids.push_back(it.second.second);    // id也存入
            track_cnt.push_back(it.first);      // 特征点跟踪次数也存入
            // 在mask对应位置画个黑圈
            cv::circle(mask,            // 画圈的图像
                       it.second.first, // 圆心位置
                       MIN_DIST,        // 半径为30个像素
                       0,               // 颜色（黑色）
                       -1);             // 实心圆
        }
    }
}

/**
 * @brief 计算两个特征点之间的欧氏距离
 * @param pt1 特征点1
 * @param pt2 特征点2
 * @return 特征点之间的欧氏距离
 */
double FeatureTracker::distance(cv::Point2f &pt1, cv::Point2f &pt2) {
    // printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

/**
 * 提取当前帧特征点
 * @param _cur_time 当前帧时间戳
 * @param _img  左目图像
 * @param _img1 右目图像
 * 1、用前一帧运动 估计 特征点在当前帧中的位置
 * 2、LK光流跟踪前一帧的特征点，正反向，删除跟丢的点；如果是双目，进行左右目的匹配，只删右目跟丢的特征点
 * 3、对于前后帧用LK光流跟踪到的匹配特征点，计算基础矩阵，用极线约束进一步剔除outlier点（代码注释掉了）
 * 4、如果特征点不够，剩余的用角点来凑；更新特征点跟踪次数
 * 5、计算特征点归一化相机平面坐标，并计算相对于前一帧的移动速度
 * 6、保存当前帧特征点数据（camera_id,
 * 归一化相机平面坐标，像素坐标，归一化相机平面移动速度）
 * 7、展示，左图特征点用颜色区分跟踪次数（红色少，蓝色多），画个箭头指向前一帧特征点位置，如果是双目，右图画个绿色点
 */
map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> FeatureTracker::trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1) {
    TicToc t_r;
    cur_time = _cur_time;
    cur_img = _img;
    row = cur_img.rows;
    col = cur_img.cols;
    cv::Mat rightImg = _img1;
    // 可添加直方图均衡化
    {
        // 创建一个clahe对象，用于执行直方图均衡加，3.0表示对比度阈值，cv::Size(8, 8)表示块大小
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(cur_img, cur_img);
        if(!rightImg.empty())
            clahe->apply(rightImg, rightImg);
    }
    cur_pts.clear();

    // 上一帧有左目的特征点（不是第一帧），则直接进行LK光流追踪当前帧的特征点
    if (prev_pts.size() > 0) {
        TicToc t_o;

        vector<uchar> status; // 存储光流法跟踪的结果，1为成功，0为失败
        vector<float> err;

        // 1.1 正向光流跟踪

        // 是否开启预测（初值为false）
        if (hasPrediction) {
            cur_pts = predict_pts; // 用预测模块预测的特征点位置作为光流法迭代初始值

            // 使用光流法跟踪上一帧的特征点，得到当前帧的特征点位置（金字塔为1层，总共2层）
            cv::calcOpticalFlowPyrLK(
                prev_img, cur_img, // 上一帧图片，当前帧图片
                prev_pts, cur_pts, // 上一帧左目的特征点，输出的当前帧左目的特征点
                status, err,       // 跟踪结果、误差
                cv::Size(21, 21),  // 每个金字塔层的搜索窗口大小
                1,                 // 金字塔层数, 图像层级总数为该值＋1
                cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), // 迭代搜索算法的终止条件，最大迭代次数与搜索窗口移动距离
                cv::OPTFLOW_USE_INITIAL_FLOW); // 操作标志：OPTFLOW_USE_INITIAL_FLOW使用初始估计，存储在nextPts中;
                                               // 如果未设置标志，则将prevPts复制到nextPts并将其视为初始估计
                                               //          OPTFLOW_LK_GET_MIN_EIGENVALS使用最小特征值作为误差测量;如果没有设置标志，则将原稿周围的色块和移动点之间的L1距离除以窗口中的像素数，用作误差测量

            // 统计跟踪到的特征点数量
            int succ_num = 0;
            for (size_t i = 0; i < status.size(); i++) {
                if (status[i])
                    succ_num++;
            }
            // 跟踪的特征点太少，金字塔调整为3层，总共为4层，再跟踪一次
            if (succ_num < 10)
                cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);
        }
        // 不使用预测模块
        else
            // LK光流跟踪两帧图像特征点
            cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);

        // 1.2 反向光流法，即在上一帧图像中跟踪 得到的当前帧特征点，提高光流法跟踪到的特征点匹配正确率
        if (FLOW_BACK) {
            vector<uchar> reverse_status;
            vector<cv::Point2f> reverse_pts = prev_pts; // 反向匹配的点

            cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 1,
                                     cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
            // cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 3);

            // 遍历正向匹配中有效的点
            for (size_t i = 0; i < status.size(); i++) {
                // 只有正向、反向都匹配成功，且 反向匹配的点与上一帧的原始特征点的距离<=0.5个像素，才最终认为跟踪成功
                if (status[i] && reverse_status[i] && distance(prev_pts[i], reverse_pts[i]) <= 0.5) {
                    status[i] = 1;
                } else
                    status[i] = 0;
            }
        }

        // 1.3 剔除跟踪失败的 且 超出图像边界的点
        for (int i = 0; i < int(cur_pts.size()); i++) {
            if (status[i] && !inBorder(cur_pts[i]))
                status[i] = 0;
        }
        // 删除跟踪丢失的特征点
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);

        // ids和track_cnt都只是记录当前帧成功跟踪到的特征点！！！
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
        // printf("track cnt %d\n", (int)ids.size());
    }

    // 当前帧与上一帧跟踪成功的特征点，其跟踪计数都+1
    for (auto &n : track_cnt)
        n++;

    // 2. 除了跟踪成功的旧特征点外，在当前帧中继续识别新的特征点
    if (1) {
        // 2.1 对于跟踪到的匹配特征点，计算基础矩阵F，进一步剔除外点
        // rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;

        // 2.2 非极大值抑制筛选特征点，在一个圆区域中只保留跟踪次数最多的特征点，使特征点均匀化
        // 特征点集合 按 跟踪次数 从大到小重排序；并且将已跟踪到特征点，对应的 mask 位置周围 画半径MIN_DIST(30)为 值为0(黑色)的实心圆
        setMask();

        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        // 2.3 提取新的特征点
        // 如果当前帧跟踪到的特征点<150，则提取角点，补上（最开始的一帧直接使用角点检测）
        int n_max_cnt = MAX_CNT - static_cast<int>(cur_pts.size());
        if (n_max_cnt > 0) {
            if (mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            // 在mask中不为0的区域检测新的特征点（Harris角点）
            cv::goodFeaturesToTrack(cur_img, // 输入图像
                                    n_pts,   // 保存检测出的角点vector
                                    MAX_CNT - cur_pts.size(), // 检测到的角点的最大个数，如果实际检测的角点超过此值，则只返回前maxCorners个强角点
                                    0.01,                     // 检测到的角点的质量水平（小于1.0的正数，一般在0.01-0.1之间）
                                    MIN_DIST, // 30个像素区分相邻两个角点的最小距离，小于此距离的点将进行合并，选更强的角点
                                    mask);    // 维度须和输入图像一致，且在mask值为0处不进行角点检测
        }
        // 跟踪的点 >= 阈值，则不再提取角点
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %f ms", t_t.toc());

        // 将新提取的角点 保存到 cur_pts、ids、track_cnt中
        for (auto &p : n_pts) {
            cur_pts.push_back(p);
            ids.push_back(n_id++);
            track_cnt.push_back(1);
        }
        // printf("feature cnt after add %d\n", (int)ids.size());
    }

    // 3. 对当前帧左目识别到的特征点(像素坐标)去畸变，并转换到归一化相机平面（只包含x,y）
    cur_un_pts = undistortedPts(cur_pts, m_camera[0]);

    // 4. 计算当前帧左目归一化相机平面上的特征点 在x、y方向上的移动速度
    pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);

    // 5.如果是双目，则对右目做相同操作，光流跟踪左目特征点 得到右目特征点，且 不提取新的特征点
    if (!_img1.empty() && stereo_cam) {
        ids_right.clear();
        cur_right_pts.clear();
        cur_un_right_pts.clear();
        right_pts_velocity.clear();
        cur_un_right_pts_map.clear(); // index: 当前帧右目特征点id, value: 对应去畸变的归一化相机平面坐标x,y

        // 左目存在特征点，则提取右目的
        if (!cur_pts.empty()) {
            // printf("stereo image; track feature on right image\n");
            vector<cv::Point2f> reverseLeftPts;    // 存储 在左目中跟踪 得到的右目特征点
            vector<uchar> status, statusRightLeft; // 存储 在左目中跟踪 得到的右目特征点 的状态
            vector<float> err;
            // cur left ---- cur right
            // 当前帧左目-右目之间 进行特征点匹配：在右目上追踪左目的特征点
            cv::calcOpticalFlowPyrLK(cur_img, rightImg, cur_pts, cur_right_pts, status, err, cv::Size(21, 21), 3);
            // reverse check cur right ---- cur left
            // 反向光流法检测
            if (FLOW_BACK) {
                cv::calcOpticalFlowPyrLK(rightImg, cur_img, cur_right_pts, reverseLeftPts, statusRightLeft, err, cv::Size(21, 21), 3);
                // 遍历正向匹配的点
                for (size_t i = 0; i < status.size(); i++) {
                    // 正反向都匹配到 且  右目的点在图像内 且 反向得到的点与左目原始特征点的误差<0.5个像素
                    if (status[i] && statusRightLeft[i] && inBorder(cur_right_pts[i]) && distance(cur_pts[i], reverseLeftPts[i]) <= 0.5) {
                        status[i] = 1;
                        // 保存左右目的匹配关系，用于划线
                        curLeftPtsTrackRight[i] = cur_pts[i]; // key: 右目特征点ID；value：左目特征点
                    } else {
                        status[i] = 0;
                    }
                }
            }

            ids_right = ids;
            reduceVector(cur_right_pts, status);
            reduceVector(ids_right, status);

            // 如果左目的特征点右目没有，也没关系，不会影响左目已经提出来的特征点
            // only keep left-right pts
            /*
            reduceVector(cur_pts, status);
            reduceVector(ids, status);
            reduceVector(track_cnt, status);
            reduceVector(cur_un_pts, status);
            reduceVector(pts_velocity, status);
            */

            // 当前帧右目特征点的 像素坐标 转为 归一化相机平面坐标（只包含x,y, 且去畸变）
            cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);
            // 计算右目特征点在归一化相机平面上的移动速度
            right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map);
        }
        prev_un_right_pts_map = cur_un_right_pts_map;
    }

    // 6. 展示，左图特征点用颜色区分跟踪次数（红色少，蓝色多），画个箭头指向前一帧特征点位置，如果是双目，右图画个绿色点
    if (SHOW_TRACK)
        drawTrack(cur_img, rightImg, ids, cur_pts, cur_right_pts, prevLeftPtsMap);

    // 7. 将当前帧的状态进行转移和保存，保存为函数输出结果
    prev_img = cur_img;
    prev_pts = cur_pts;               // 当前帧左目特征点的 像素坐标
    prev_un_pts = cur_un_pts;         // 当前帧左目特征点 归一化相机平面x,y坐标
    prev_un_pts_map = cur_un_pts_map; // index: 当前帧左目特征点id, value: 对应去畸变的归一化相机平面坐标x,y
    prev_time = cur_time;
    hasPrediction = false;

    prevLeftPtsMap.clear(); // index: 当前帧左目的特征点id, value: 对应特征点的像素坐标
    for (size_t i = 0; i < cur_pts.size(); i++)
        prevLeftPtsMap[ids[i]] = cur_pts[i];

    // 存储当前帧左右目跟踪到的特征点信息
    // featureFrame: feature_id，[camera_id (0为左目，1为右目), x, y, z(去畸变的归一化相机平面坐标), pu, pv (像素坐标), vx, vy(归一化相机平面移动速度)]
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;

    // 添加左目，camera_id = 0
    for (size_t i = 0; i < ids.size(); i++) {
        int feature_id = ids[i];

        double x, y, z; // 归一化平面3D坐标
        x = cur_un_pts[i].x;
        y = cur_un_pts[i].y;
        z = 1;

        double p_u, p_v; // 像素坐标
        p_u = cur_pts[i].x;
        p_v = cur_pts[i].y;

        int camera_id = 0;

        double velocity_x, velocity_y; // 归一化相机平面移动速度
        velocity_x = pts_velocity[i].x;
        velocity_y = pts_velocity[i].y;

        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
    }

    // 双目的右目 也存入到 featureFrame 中, camera_id = 1
    if (!_img1.empty() && stereo_cam) {
        for (size_t i = 0; i < ids_right.size(); i++) {
            int feature_id = ids_right[i];
            double x, y, z;
            x = cur_un_right_pts[i].x;
            y = cur_un_right_pts[i].y;
            z = 1;
            double p_u, p_v;
            p_u = cur_right_pts[i].x;
            p_v = cur_right_pts[i].y;
            int camera_id = 1;
            double velocity_x, velocity_y;
            velocity_x = right_pts_velocity[i].x;
            velocity_y = right_pts_velocity[i].y;

            Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
            featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
        }
    }

    // printf("feature track whole time %f\n", t_r.toc());
    return featureFrame;
}

/**
 * 对于前后帧用LK光流跟踪到的匹配特征点，计算基础矩阵，进一步剔除outlier点
 */
void FeatureTracker::rejectWithF() {
    if (cur_pts.size() >= 8) {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_prev_pts(prev_pts.size());
        // 特征点先转到归一下相机平面下，畸变校正，再转回来
        for (unsigned int i = 0; i < cur_pts.size(); i++) {
            Eigen::Vector3d tmp_p;
            m_camera[0]->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera[0]->liftProjective(Eigen::Vector2d(prev_pts[i].x, prev_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        // 两帧特征点匹配，算一个最佳的基础矩阵，剔除掉在该基础矩阵变换下匹配较差的匹配点对
        cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, cur_pts.size(), 1.0 * cur_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

/**
 * 读取左右目相机内参，并构建相机模型 camera
 * @param calib_file 存储 左右目相机的内参文件的 路径
 */
void FeatureTracker::readIntrinsicParameter(const vector<string> &calib_file) {

    for (size_t i = 0; i < calib_file.size(); i++) {
        ROS_INFO("reading paramerter of camera %s", calib_file[i].c_str());

        camodocal::CameraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);

        m_camera.push_back(camera);
    }
    if (calib_file.size() == 2)
        stereo_cam = 1;
}

/**
 * 对当前帧图像进行畸变校正，展示
 */
void FeatureTracker::showUndistortion(const string &name) {
    cv::Mat undistortedImg(row + 600, col + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < col; i++)
        for (int j = 0; j < row; j++) {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera[0]->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            // printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    // 计算每个像素点校正之后的像素坐标，可能会变成负数或者超出图像边界，所以扩大图像
    for (int i = 0; i < int(undistortedp.size()); i++) {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + col / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + row / 2;
        pp.at<float>(2, 0) = 1.0;
        // cout << trackerData[0].K << endl;
        // printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        // printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < row + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < col + 600) {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        } else {
            // ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x,
            // pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    // turn the following code on if you need
    // cv::imshow(name, undistortedImg);
    // cv::waitKey(0);
}

/**
 * @brief 对特征点2D像素坐标去畸变，并转为 相机平面坐标（只包含x,y）
 * @param pts 当前帧特征点的 2D像素坐标
 * @param cam 相机模型
 * @return 返回处理后的无畸变的归一化相机坐标
 */
vector<cv::Point2f> FeatureTracker::undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam) {
    vector<cv::Point2f> un_pts;
    for (unsigned int i = 0; i < pts.size(); i++) {
        Eigen::Vector2d a(pts[i].x, pts[i].y); // 像素坐标
        Eigen::Vector3d b;
        // 像素点计算归一化相机平面点，带畸变校正
        cam->liftProjective(a, b); // liftProjective函数不是得到2D特征点对应的3D地图点，而是其投影射线点
        // 归一化相机平面点
        un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    }
    return un_pts;
}

/**
 * 计算当前帧归一化相机平面特征点在x、y方向上的移动速度
 * @param ids   当前帧特征点id
 * @param pts   当前帧特征点的 归一化相机平面坐标 (只包含x,y, 已去畸变)
 * @param cur_id_pts    index: 当前帧特征点id, value: 对应归一化相机平面坐标x,y
 * @param prev_id_pts   index: 前一帧特征点id, value: 对应归一化相机平面坐标x,y
 * @return pts中全部特征点在两帧中的XY速度 = ((特征点1的X速度，特征点1的Y速度)，(特征点2的X速度，特征点2的Y速度)，...)
 */
vector<cv::Point2f> FeatureTracker::ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, map<int, cv::Point2f> &cur_id_pts,
                                                map<int, cv::Point2f> &prev_id_pts) {
    vector<cv::Point2f> pts_velocity;
    // 通过当前归一化相机坐标（仅x, y）和角点ID重新赋值cur_id_pts
    cur_id_pts.clear(); // index: 当前帧特征点id, value: 归一化相机平面坐标x,y
    for (unsigned int i = 0; i < ids.size(); i++) {
        cur_id_pts.insert(make_pair(ids[i], pts[i]));
    }

    // 计算上一帧和当前帧之间点的运动速度（或称为光流速度）

    // 前一帧有特征点
    if (!prev_id_pts.empty()) {
        double dt = cur_time - prev_time;

        // 遍历当前帧的每个 归一化相机平面特征点
        for (unsigned int i = 0; i < pts.size(); i++) {
            std::map<int, cv::Point2f>::iterator it;
            it = prev_id_pts.find(ids[i]); // 找到两帧中同一特征点
            // 匹配到了，则计算该点在归一化相机平面上x、y方向的移动速度
            if (it != prev_id_pts.end()) {
                // 计算点在归一化相机平面上x、y方向的移动速度
                double v_x = (pts[i].x - it->second.x) / dt;
                double v_y = (pts[i].y - it->second.y) / dt;
                pts_velocity.push_back(cv::Point2f(v_x, v_y));
            }
            // 找不到，则速度设为0
            else
                pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    // 前一帧没有特征点，则都设为0
    else {
        for (unsigned int i = 0; i < cur_pts.size(); i++) {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    return pts_velocity;
}

/**
 * 绘制跟踪图片，也就是添加了特征点识别结果的图片
 * 对于左目特征点：跟踪次数<20的，标记为蓝色点；跟踪次数>20的，标记为红色点
 * 对于右目特征点：统一标记为绿色
 * 对于上一帧左目特征点中跟踪成功的特征点：使用绿色箭头绘制出来
 * @param imLeft            当前帧左目图片
 * @param imRight           当前帧右目图片
 * @param curLeftIds        当前帧左图特征点ID
 * @param curLeftPts        当前帧左目识别到的特征点
 * @param curRightPts       当前帧右目识别到的特征点
 * @param prevLeftPtsMap    上一帧图像左目的特征点和ID
 */
void FeatureTracker::drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, vector<int> &curLeftIds, vector<cv::Point2f> &curLeftPts,
                               vector<cv::Point2f> &curRightPts, map<int, cv::Point2f> &prevLeftPtsMap) {
    // int rows = imLeft.rows;
    int cols = imLeft.cols;
    // 单目用左图，双目左右目图偏水平拼接
    if (!imRight.empty() && stereo_cam)
        cv::hconcat(imLeft, imRight, imTrack);
    else
        imTrack = imLeft.clone();
    cv::cvtColor(imTrack, imTrack, cv::COLOR_GRAY2RGB);

    // 绘制左目特征点。蓝色：跟踪次数<20；红色：跟踪次数>20
    for (size_t j = 0; j < curLeftPts.size(); j++) {
        double len = std::min(1.0, 1.0 * track_cnt[j] / 20);  // <20,=0; >20,=1
        cv::circle(imTrack,                                   // 绘制图像
                   curLeftPts[j],                             // 圆心位置
                   2,                                         // 半径
                   cv::Scalar(255 * (1 - len), 0, 255 * len), // 颜色BGR对应通道的值
                   2);                                        // <0，实心圆；>0空心圆的线宽
    }

    // 绘制右目特征点
    if (!imRight.empty() && stereo_cam) {
        for (size_t i = 0; i < curRightPts.size(); i++) {
            cv::Point2f rightPt = curRightPts[i];
            rightPt.x += cols;
            cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2); // 绿色
            // 连左右目的匹配线
            cv::Point2f leftPt = curLeftPtsTrackRight[i];
            cv::line(imTrack,
                     leftPt,                // 直线起点
                     rightPt,               // 直线终点
                     cv::Scalar(0, 255, 0), // 线的颜色：BGR
                     1,                     // 线条粗细
                     8,                     // 线条类型：8邻接连接线
                     0);                    // 点坐标的小数位数
        }
    }

    // 绘制上一帧左目特征点中 在当前帧跟踪成功的点 (当前帧 指向 上一帧)
    map<int, cv::Point2f>::iterator mapIt;
    for (size_t i = 0; i < curLeftIds.size(); i++) {
        int id = curLeftIds[i];
        mapIt = prevLeftPtsMap.find(id);
        // 当前帧特征点ID 在 上一帧特征点ID集合中
        if (mapIt != prevLeftPtsMap.end()) {
            cv::arrowedLine(imTrack,
                            curLeftPts[i],         // 箭头起点
                            mapIt->second,         // 箭头终点
                            cv::Scalar(0, 255, 0), // 箭头的颜色：BGR
                            1,                     // 线条粗细
                            8, // 线条类型：1就是FILLED（填满），4是LINE_4（4连通域），8是LINE_8（8连通域），LINE_AA（抗锯齿线）
                            0, // 点坐标的小数位数，默认为 0
                            0.2); // 箭头部分长度与线段长度的比例，默认为 0.1
        }
    }

    // draw prediction
    /*
    for(size_t i = 0; i < predict_pts_debug.size(); i++)
    {
        cv::circle(imTrack, predict_pts_debug[i], 2, cv::Scalar(0, 170, 255), 2);
    }
    */
    // printf("predict pts size %d \n", (int)predict_pts_debug.size());

    // cv::Mat imCur2Compress;
    // cv::resize(imCur2, imCur2Compress, cv::Size(cols, rows / 2));
}

/**
 * 用前一帧运动估计特征点在当前帧中的位置
 */
void FeatureTracker::setPrediction(map<int, Eigen::Vector3d> &predictPts) {
    hasPrediction = true;
    predict_pts.clear();
    predict_pts_debug.clear();
    map<int, Eigen::Vector3d>::iterator itPredict;
    for (size_t i = 0; i < ids.size(); i++) {
        // printf("prevLeftId size %d prevLeftPts size
        // %d\n",(int)prevLeftIds.size(), (int)prevLeftPts.size());
        int id = ids[i];
        itPredict = predictPts.find(id);
        if (itPredict != predictPts.end()) {
            Eigen::Vector2d tmp_uv;
            m_camera[0]->spaceToPlane(itPredict->second, tmp_uv);
            predict_pts.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
            predict_pts_debug.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
        } else
            predict_pts.push_back(prev_pts[i]);
    }
}

/**
 * 指定outlier点，并删除
 */
void FeatureTracker::removeOutliers(set<int> &removePtsIds) {
    std::set<int>::iterator itSet;
    vector<uchar> status;
    for (size_t i = 0; i < ids.size(); i++) {
        itSet = removePtsIds.find(ids[i]);
        if (itSet != removePtsIds.end())
            status.push_back(0);
        else
            status.push_back(1);
    }

    reduceVector(prev_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
}

cv::Mat FeatureTracker::getTrackImage() { return imTrack; }