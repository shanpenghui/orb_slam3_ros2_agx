#ifndef __STEREO_INERTIAL_NODE_HPP__
#define __STEREO_INERTIAL_NODE_HPP__

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "std_msgs/msg/int32.hpp"

#include <cv_bridge/cv_bridge.h>
#include <atomic>
#include <queue>
#include <mutex>
#include <thread>
#include <string>
#include <vector>
#include <sstream>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "Frame.h"
#include "Map.h"
#include "Tracking.h"

#include "utility.hpp"

using ImuMsg = sensor_msgs::msg::Imu;
using ImageMsg = sensor_msgs::msg::Image;
using OdomMsg = nav_msgs::msg::Odometry;
using PathMsg = nav_msgs::msg::Path;
using PoseStampedMsg = geometry_msgs::msg::PoseStamped;
using Int32Msg = std_msgs::msg::Int32;

class StereoInertialNode : public rclcpp::Node
{
public:
    StereoInertialNode(ORB_SLAM3::System* pSLAM, const string &strSettingsFile, const string &strDoRectify, const string &strDoEqual);
    ~StereoInertialNode();

private:
    void GrabImu(const ImuMsg::SharedPtr msg);
    void GrabImageLeft(const ImageMsg::SharedPtr msgLeft);
    void GrabImageRight(const ImageMsg::SharedPtr msgRight);
    cv::Mat GetImage(const ImageMsg::SharedPtr msg);
    void SyncWithImu();

    void PublishTracking(const builtin_interfaces::msg::Time &stamp, double stamp_sec,
                        const Sophus::SE3f &Tcw, int tracking_state);

    rclcpp::Subscription<ImuMsg>::SharedPtr   subImu_;
    rclcpp::Subscription<ImageMsg>::SharedPtr subImgLeft_;
    rclcpp::Subscription<ImageMsg>::SharedPtr subImgRight_;

    rclcpp::Publisher<OdomMsg>::SharedPtr odomPub_;
    rclcpp::Publisher<PathMsg>::SharedPtr pathPub_;
    rclcpp::Publisher<Int32Msg>::SharedPtr statePub_;

    ORB_SLAM3::System *SLAM_;
    std::thread *syncThread_;

    // IMU
    queue<ImuMsg::SharedPtr> imuBuf_;
    std::mutex bufMutex_;

    // Image
    queue<ImageMsg::SharedPtr> imgLeftBuf_, imgRightBuf_;
    std::mutex bufMutexLeft_, bufMutexRight_;

    bool doRectify_;
    bool doEqual_;
    cv::Mat M1l_, M2l_, M1r_, M2r_;

    bool bClahe_;
    cv::Ptr<cv::CLAHE> clahe_ = cv::createCLAHE(3.0, cv::Size(8, 8));

    std::string odom_frame_id_;
    std::string base_frame_id_;
    size_t path_max_length_;
    PathMsg path_msg_;

    bool has_prev_pose_;
    Eigen::Vector3f prev_pos_;
    Eigen::Quaternionf prev_q_;
    double prev_t_;

    std::atomic<bool> running_;
};

#endif
