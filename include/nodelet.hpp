
#ifndef TENSORRT_YOLOP_NODELET_HPP_
#define TENSORRT_YOLOp_NODELET_HPP_

#include <rclcpp/rclcpp.hpp>
#include <yolop.hpp>

class TensorrtYolopNode : public rclcpp::Node
{
public:
    explicit TensorrtYolopNode(const rclcpp::NodeOptions & options);
    void detect();
private:
    std::unique_ptr<YolopNet> net_ptr_;

    std::unique_ptr<float[]> out_objs_ = nullptr;
    std::unique_ptr<float[]> out_drive_area_ = nullptr;
    std::unique_ptr<float[]> out_lane_ = nullptr;
};

#endif
