使用tensorrt推理yolop.模型来自https://github.com/hustvl/YOLOP/tree/main/weights

x86环境. docker内测试. 
- 显卡gtx1050
- tensorrt版本:8.2.2-1+cuda11.4
- ros版本:ros2 galactic

fps(preprocess+infer):25    
fps(preprocess+infer+postprocess):20

读取test_imgs目录下的图片,推理结果图片存储于result_imgs目录. 路径写死了,修改的话去改detect_test_on_dir函数.