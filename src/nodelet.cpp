#include "nodelet.hpp"
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
// #include <cv_bridge/cv_bridge.h>
#include <ament_index_cpp/get_package_share_directory.hpp>

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
    auto string_ends_with = [](std::string const & fullString, std::string const & ending) -> bool 
    {
        if (fullString.length() >= ending.length()) 
        {
            return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
        } 
        else 
        {
            return false;
        }
    };

    
    boost::filesystem::directory_iterator end;
    for (boost::filesystem::directory_iterator it("/home/tensorrt_yolop/test_imgs"); it != end; ++it) 
    {
        std::string img_path = it->path().string();   
        if(!boost::filesystem::is_regular_file(img_path))
        {
            continue;
        }

        if (string_ends_with(img_path,"jpg"))
        {
            std::cout<<"img_path:"<<img_path<<std::endl;
            cv::Mat img = cv::imread(img_path);
            std::vector<std::string> splited_path;
            boost::algorithm::split(splited_path, img_path, boost::is_any_of("/"));
            std::string result_img_name = "/home/tensorrt_yolop/result_imgs/"+splited_path.back();
            net_ptr_->detect(img,result_img_name);
        }
    }

    // std::string imgpath = "/home/tensorrt_yolop/test.jpg";
    // cv::Mat srcimg = cv::imread(imgpath);

    // net_ptr_->detect(srcimg);
}