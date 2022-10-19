#include "nodelet.hpp"
#include <fstream>

int main(int argc,char** argv)
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    
    rclcpp::spin(std::make_shared<TensorrtYolopNode>(options));
    rclcpp::shutdown();
    return 0;
}