#include "nodelet.hpp"
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <cv_bridge/cv_bridge.h>
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

        // detect();
    }
    else
    {
        std::string onnx_file = declare_parameter("onnx_file", "");
        std::cout<<"load onnx_file: "<<onnx_file<<std::endl;
        net_ptr_.reset(new YolopNet(onnx_file, "FP16", 1,engine_file,false));
    }

    image_sub_ = image_transport::create_subscription(
      this, "in/image", std::bind(&TensorrtYolopNode::callback, this, _1), "raw",
      rmw_qos_profile_sensor_data);

    image_pub_ = image_transport::create_publisher(this, "out/image");
}

void TensorrtYolopNode::detect_test_on_dir()
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

    //遍历无法保证顺序是按照名称顺序排序的
    // boost::filesystem::directory_iterator end;
    // for (boost::filesystem::directory_iterator it("/home/tensorrt_yolop/test_imgs"); it != end; ++it) 
    // {
    //     std::string img_path = it->path().string();   
    //     if(!boost::filesystem::is_regular_file(img_path))
    //     {
    //         continue;
    //     }

    //     if (string_ends_with(img_path,"jpg"))
    //     {
    //         std::cout<<"img_path:"<<img_path<<std::endl;
    //         cv::Mat img = cv::imread(img_path);
    //         std::vector<std::string> splited_path;
    //         boost::algorithm::split(splited_path, img_path, boost::is_any_of("/"));
    //         std::string result_img_name = "/home/tensorrt_yolop/result_imgs/"+splited_path.back();
    //         net_ptr_->detect(img,result_img_name);

    //         auto header = std_msgs::msg::Header();
    //         auto msg_time_stamp = rclcpp::Clock().now();
    //         header.stamp = msg_time_stamp;
    //         header.frame_id = "front_camera";
    //         auto imgPtr = std::make_shared<cv_bridge::CvImage> (header,"bgr8",img);
    //         auto msg = *(imgPtr->toImageMsg());

    //         image_pub_.publish(msg);
    //     }
    // }

    std::string dir = "/home/tensorrt_yolop/test_imgs";
    auto num2str = [](int i) -> std::string
    {
        char ss[10];
        sprintf(ss,"%04d",i);
        return ss;
    };
    for(int i = 0; i < 7059;i++)
    {
        if(i%10 != 0)
        {
            continue;
        }
        std::string img_path = dir + "/frame" + num2str(i) + ".jpg"; 
        std::cout<<"img_path:"<<img_path<<std::endl;
        cv::Mat img = cv::imread(img_path);
        std::vector<std::string> splited_path;
        boost::algorithm::split(splited_path, img_path, boost::is_any_of("/"));
        std::string result_img_name = "/home/tensorrt_yolop/result_imgs/"+splited_path.back();
        net_ptr_->detect(img,result_img_name);

        auto header = std_msgs::msg::Header();
        auto msg_time_stamp = rclcpp::Clock().now();
        header.stamp = msg_time_stamp;
        header.frame_id = "front_camera";
        auto imgPtr = std::make_shared<cv_bridge::CvImage> (header,"bgr8",img);
        auto msg = *(imgPtr->toImageMsg());

        image_pub_.publish(msg);
    }
}

void TensorrtYolopNode::callback(const sensor_msgs::msg::Image::ConstSharedPtr in_image_msg)
{
    std::cout<<"TensorrtYolopNode::callback"<<std::endl;
    cv_bridge::CvImagePtr in_image_ptr;
    try 
    {
        in_image_ptr = cv_bridge::toCvCopy(in_image_msg, sensor_msgs::image_encodings::BGR8);
    } 
    catch (cv_bridge::Exception & e) 
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    net_ptr_->detect(in_image_ptr->image,"");

    image_pub_.publish(in_image_ptr->toImageMsg());
}
