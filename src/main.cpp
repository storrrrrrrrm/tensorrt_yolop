#include "nodelet.hpp"
#include <fstream>

int main(int argc,char** argv)
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    
    std::shared_ptr<TensorrtYolopNode> node = std::make_shared<TensorrtYolopNode>(options);
    node->detect_test_on_dir();

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}