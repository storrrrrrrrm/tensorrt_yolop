
#ifndef TENSORRT_YOLOP_NODELET_HPP_
#define TENSORRT_YOLOp_NODELET_HPP_

#include "yolop.hpp"

class TensorrtYolopNode
{
public:
    explicit TensorrtYolopNode(const std::string& engine_file,const std::string& onnx_file);
    void detect_on_img(const cv::Mat& img,const std::string& result_img_save_path);
private:
    std::unique_ptr<YolopNet> net_ptr_;

    std::unique_ptr<float[]> out_objs_ = nullptr;
    std::unique_ptr<float[]> out_drive_area_ = nullptr;
    std::unique_ptr<float[]> out_lane_ = nullptr;
};

#endif
