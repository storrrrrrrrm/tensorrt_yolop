#include "nodelet.hpp"
#include <fstream>

TensorrtYolopNode::TensorrtYolopNode(const std::string& engine_file,const std::string& onnx_file)
{    
    std::ifstream f(engine_file.c_str());
    if(f.good())
    {
        std::cout<<"load engine_file: "<<engine_file<<std::endl;
        net_ptr_.reset(new YolopNet(engine_file));
    }
    else
    {
        std::cout<<"load onnx_file: "<<onnx_file<<std::endl;
        net_ptr_.reset(new YolopNet(onnx_file, "FP16", 1,engine_file,false));
    }
}


void TensorrtYolopNode::detect_on_img(const cv::Mat & img,const std::string& result_img_save_path)
{
    net_ptr_->detect(img,result_img_save_path);
}

