#include "nodelet.hpp"

TensorrtYolopNode::TensorrtYolopNode(const rclcpp::NodeOptions & options)
: Node("tensorrt_yolop", options)
{
    using std::placeholders::_1;

    std::string engine_file = declare_parameter("engine_file", "");
    
    std::ifstream f(engine_file.c_str());
    if(f.good())
    {
        std::cout<<"load engine_file: "<<engine_file<<std::endl;
        net_ptr_.reset(new YolopNet(engine_file));

        detect();
    }
    else
    {
        std::string onnx_file = declare_parameter("onnx_file", "");
        std::cout<<"load onnx_file: "<<onnx_file<<std::endl;
        net_ptr_.reset(new YolopNet(onnx_file, "FP16", 1,engine_file,false));
    }
}

void TensorrtYolopNode::detect()
{
    std::string imgpath = "/home/tensorrt_yolop/test.jpg";
    cv::Mat srcimg = cv::imread(imgpath);

    net_ptr_->detect(srcimg);
}