#include "nodelet.hpp"
#include <fstream>

int main(int argc,char** argv)
{
    std::string onnx_path = "../data/yolop-640-640.onnx";
    std::string engine_path = "../data/yolop.engine";
    std::shared_ptr<TensorrtYolopNode> node = std::make_shared<TensorrtYolopNode>(engine_path,onnx_path);
    
    cv::Mat img = cv::imread("../test.jpg");
    for(int i = 0;i<50;i++)
    {
        node->detect_on_img(img,"../test_result.jpg");
    }

    return 0;
}