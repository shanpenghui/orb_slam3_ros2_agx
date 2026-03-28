#include "stereo-inertial-node.hpp"

#include <opencv2/core/core.hpp>
#include <cstdlib>

using std::placeholders::_1;

StereoInertialNode::StereoInertialNode(ORB_SLAM3::System *SLAM, const string &strSettingsFile, const string &strDoRectify, const string &strDoEqual) :
    Node("ORB_SLAM3_ROS2"),
    SLAM_(SLAM),
    has_prev_pose_(false),
    prev_pos_(Eigen::Vector3f::Zero()),
    prev_q_(Eigen::Quaternionf::Identity()),
    prev_t_(0.0),
    running_(true)
{
    stringstream ss_rec(strDoRectify);
    ss_rec >> boolalpha >> doRectify_;

    stringstream ss_eq(strDoEqual);
    ss_eq >> boolalpha >> doEqual_;

    bClahe_ = doEqual_;
    std::cout << "Rectify: " << doRectify_ << std::endl;
    std::cout << "Equal: " << doEqual_ << std::endl;

    odom_frame_id_ = this->declare_parameter<std::string>("odom_frame_id", "orbslam3_odom");
    base_frame_id_ = this->declare_parameter<std::string>("base_frame_id", "d435i_link");
    path_max_length_ = static_cast<size_t>(this->declare_parameter<int>("path_max_length", 2000));

    if (doRectify_)
    {
        // Load settings related to stereo calibration
        cv::FileStorage fsSettings(strSettingsFile, cv::FileStorage::READ);
        if (!fsSettings.isOpened())
        {
            cerr << "ERROR: Wrong path to settings" << endl;
            assert(0);
        }

        cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
        fsSettings["LEFT.K"] >> K_l;
        fsSettings["RIGHT.K"] >> K_r;

        fsSettings["LEFT.P"] >> P_l;
        fsSettings["RIGHT.P"] >> P_r;

        fsSettings["LEFT.R"] >> R_l;
        fsSettings["RIGHT.R"] >> R_r;

        fsSettings["LEFT.D"] >> D_l;
        fsSettings["RIGHT.D"] >> D_r;

        int rows_l = fsSettings["LEFT.height"];
        int cols_l = fsSettings["LEFT.width"];
        int rows_r = fsSettings["RIGHT.height"];
        int cols_r = fsSettings["RIGHT.width"];

        if (K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
            rows_l == 0 || rows_r == 0 || cols_l == 0 || cols_r == 0)
        {
            cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
            assert(0);
        }

        cv::initUndistortRectifyMap(K_l, D_l, R_l, P_l.rowRange(0, 3).colRange(0, 3), cv::Size(cols_l, rows_l), CV_32F, M1l_, M2l_);
        cv::initUndistortRectifyMap(K_r, D_r, R_r, P_r.rowRange(0, 3).colRange(0, 3), cv::Size(cols_r, rows_r), CV_32F, M1r_, M2r_);
    }

    // Use sensor-data QoS to match RealSense IMU (best-effort) and image streams.
    const auto sensor_qos = rclcpp::SensorDataQoS();
    subImu_ = this->create_subscription<ImuMsg>("imu", sensor_qos, std::bind(&StereoInertialNode::GrabImu, this, _1));
    subImgLeft_ = this->create_subscription<ImageMsg>("camera/left", sensor_qos, std::bind(&StereoInertialNode::GrabImageLeft, this, _1));
    subImgRight_ = this->create_subscription<ImageMsg>("camera/right", sensor_qos, std::bind(&StereoInertialNode::GrabImageRight, this, _1));

    odomPub_ = this->create_publisher<OdomMsg>("tracking/odometry", 20);
    pathPub_ = this->create_publisher<PathMsg>("tracking/path", 10);
    statePub_ = this->create_publisher<Int32Msg>("tracking/state", 20);

    path_msg_.header.frame_id = odom_frame_id_;

    syncThread_ = new std::thread(&StereoInertialNode::SyncWithImu, this);
}

StereoInertialNode::~StereoInertialNode()
{
    running_.store(false);

    // Delete sync thread
    syncThread_->join();
    delete syncThread_;

    // Stop all threads
    SLAM_->Shutdown();

    // Save camera trajectory
    SLAM_->SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
}

void StereoInertialNode::GrabImu(const ImuMsg::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(bufMutex_);
    imuBuf_.push(msg);
}

void StereoInertialNode::GrabImageLeft(const ImageMsg::SharedPtr msgLeft)
{
    std::lock_guard<std::mutex> lock(bufMutexLeft_);

    if (!imgLeftBuf_.empty())
        imgLeftBuf_.pop();
    imgLeftBuf_.push(msgLeft);
}

void StereoInertialNode::GrabImageRight(const ImageMsg::SharedPtr msgRight)
{
    std::lock_guard<std::mutex> lock(bufMutexRight_);

    if (!imgRightBuf_.empty())
        imgRightBuf_.pop();
    imgRightBuf_.push(msgRight);
}

cv::Mat StereoInertialNode::GetImage(const ImageMsg::SharedPtr msg)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;

    try
    {
        cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception &e)
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return cv::Mat();
    }

    return cv_ptr->image.clone();
}

void StereoInertialNode::PublishTracking(
    const builtin_interfaces::msg::Time &stamp,
    double stamp_sec,
    const Sophus::SE3f &Tcw,
    int tracking_state)
{
    // ORB-SLAM gives Tcw. We publish Twc as camera/base pose in odom frame.
    const Sophus::SE3f Twc = Tcw.inverse();
    const Eigen::Vector3f pos = Twc.translation();
    const Eigen::Quaternionf q(Twc.unit_quaternion());

    OdomMsg odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = odom_frame_id_;
    odom.child_frame_id = base_frame_id_;

    odom.pose.pose.position.x = pos.x();
    odom.pose.pose.position.y = pos.y();
    odom.pose.pose.position.z = pos.z();
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();

    if (has_prev_pose_ && stamp_sec > prev_t_)
    {
        const double dt = stamp_sec - prev_t_;
        const Eigen::Vector3f dp = pos - prev_pos_;
        odom.twist.twist.linear.x = dp.x() / dt;
        odom.twist.twist.linear.y = dp.y() / dt;
        odom.twist.twist.linear.z = dp.z() / dt;

        Eigen::Quaternionf dq = q * prev_q_.inverse();
        if (dq.w() < 0.0f) {
            dq.coeffs() *= -1.0f;
        }
        Eigen::AngleAxisf aa(dq);
        Eigen::Vector3f omega = aa.axis() * aa.angle() / static_cast<float>(dt);
        if (!std::isfinite(omega.x()) || !std::isfinite(omega.y()) || !std::isfinite(omega.z())) {
            omega.setZero();
        }
        odom.twist.twist.angular.x = omega.x();
        odom.twist.twist.angular.y = omega.y();
        odom.twist.twist.angular.z = omega.z();
    }

    odomPub_->publish(odom);

    PoseStampedMsg ps;
    ps.header = odom.header;
    ps.pose = odom.pose.pose;
    path_msg_.header.stamp = stamp;
    path_msg_.poses.push_back(ps);
    if (path_msg_.poses.size() > path_max_length_) {
        path_msg_.poses.erase(path_msg_.poses.begin(), path_msg_.poses.begin() + (path_msg_.poses.size() - path_max_length_));
    }
    pathPub_->publish(path_msg_);

    Int32Msg s;
    s.data = tracking_state;
    statePub_->publish(s);

    prev_pos_ = pos;
    prev_q_ = q;
    prev_t_ = stamp_sec;
    has_prev_pose_ = true;
}

void StereoInertialNode::SyncWithImu()
{
    // Allow tuning for camera pair timestamp skew; default 20ms for RealSense D435i.
    double maxTimeDiff = 0.02;
    if (const char * env = std::getenv("ORBSLAM3_MAX_TIME_DIFF")) {
        try {
            maxTimeDiff = std::stod(env);
        } catch (...) {
            // keep default
        }
    }

    while (running_.load())
    {
        cv::Mat imLeft, imRight;
        double tImLeft = 0, tImRight = 0;
        bool ready = false;

        {
            std::lock_guard<std::mutex> lockL(bufMutexLeft_);
            std::lock_guard<std::mutex> lockR(bufMutexRight_);
            std::lock_guard<std::mutex> lockI(bufMutex_);
            ready = (!imgLeftBuf_.empty() && !imgRightBuf_.empty() && !imuBuf_.empty());
        }

        if (!ready)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        tImLeft = Utility::StampToSec(imgLeftBuf_.front()->header.stamp);
        tImRight = Utility::StampToSec(imgRightBuf_.front()->header.stamp);

        {
            std::lock_guard<std::mutex> lockR(bufMutexRight_);
            while ((tImLeft - tImRight) > maxTimeDiff && imgRightBuf_.size() > 1)
            {
                imgRightBuf_.pop();
                tImRight = Utility::StampToSec(imgRightBuf_.front()->header.stamp);
            }
        }

        {
            std::lock_guard<std::mutex> lockL(bufMutexLeft_);
            while ((tImRight - tImLeft) > maxTimeDiff && imgLeftBuf_.size() > 1)
            {
                imgLeftBuf_.pop();
                tImLeft = Utility::StampToSec(imgLeftBuf_.front()->header.stamp);
            }
        }

        if ((tImLeft - tImRight) > maxTimeDiff || (tImRight - tImLeft) > maxTimeDiff)
        {
            std::cout << "big time difference" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        {
            std::lock_guard<std::mutex> lockI(bufMutex_);
            if (tImLeft > Utility::StampToSec(imuBuf_.back()->header.stamp)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
        }

        {
            std::lock_guard<std::mutex> lockL(bufMutexLeft_);
            imLeft = GetImage(imgLeftBuf_.front());
            imgLeftBuf_.pop();
        }

        {
            std::lock_guard<std::mutex> lockR(bufMutexRight_);
            imRight = GetImage(imgRightBuf_.front());
            imgRightBuf_.pop();
        }

        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        {
            std::lock_guard<std::mutex> lockI(bufMutex_);
            while (!imuBuf_.empty() && Utility::StampToSec(imuBuf_.front()->header.stamp) <= tImLeft)
            {
                double t = Utility::StampToSec(imuBuf_.front()->header.stamp);
                cv::Point3f acc(
                    imuBuf_.front()->linear_acceleration.x,
                    imuBuf_.front()->linear_acceleration.y,
                    imuBuf_.front()->linear_acceleration.z);
                cv::Point3f gyr(
                    imuBuf_.front()->angular_velocity.x,
                    imuBuf_.front()->angular_velocity.y,
                    imuBuf_.front()->angular_velocity.z);
                vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc, gyr, t));
                imuBuf_.pop();
            }
        }

        if (imLeft.empty() || imRight.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        if (bClahe_)
        {
            clahe_->apply(imLeft, imLeft);
            clahe_->apply(imRight, imRight);
        }

        if (doRectify_)
        {
            cv::remap(imLeft, imLeft, M1l_, M2l_, cv::INTER_LINEAR);
            cv::remap(imRight, imRight, M1r_, M2r_, cv::INTER_LINEAR);
        }

        const Sophus::SE3f Tcw = SLAM_->TrackStereo(imLeft, imRight, tImLeft, vImuMeas);
        const int tr_state = SLAM_->GetTrackingState();

        if (tr_state == ORB_SLAM3::Tracking::OK || tr_state == ORB_SLAM3::Tracking::OK_KLT || tr_state == ORB_SLAM3::Tracking::RECENTLY_LOST)
        {
            builtin_interfaces::msg::Time stamp_msg;
            stamp_msg.sec = static_cast<int32_t>(tImLeft);
            stamp_msg.nanosec = static_cast<uint32_t>((tImLeft - static_cast<double>(stamp_msg.sec)) * 1e9);
            PublishTracking(stamp_msg, tImLeft, Tcw, tr_state);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
