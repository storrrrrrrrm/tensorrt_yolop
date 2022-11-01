
#ifndef TENSORRT_YOLOP_NODELET_HPP_
#define TENSORRT_YOLOp_NODELET_HPP_

#include <rclcpp/rclcpp.hpp>
#include <yolop.hpp>

#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>


class TensorrtYolopNode : public rclcpp::Node
{
public:
    explicit TensorrtYolopNode(const rclcpp::NodeOptions & options);
    void detect_test_on_dir();
private:
    std::unique_ptr<YolopNet> net_ptr_;

    std::unique_ptr<float[]> out_objs_ = nullptr;
    std::unique_ptr<float[]> out_drive_area_ = nullptr;
    std::unique_ptr<float[]> out_lane_ = nullptr;

    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;

    void callback(const sensor_msgs::msg::Image::ConstSharedPtr in_image_msg);
};

#endif
