#ifndef YOLOP_HPP_
#define YOLOP_HPP_

#include "cuda_utils.hpp"

#include <string>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

#include <NvInfer.h>

class Detection
{
public:
    //center_x center_y w h
    float bbox[4];
    float conf;  // bbox_conf * cls_conf
    float class_id;

    void print()
    {
        std::cout<<"conf:"<<conf<<std::endl;
        std::cout<<"c_x:"<<bbox[0]
            <<",c_y:"<<bbox[1]
            <<",w:"<<bbox[2]
            <<",h:"<<bbox[3]<<std::endl;
    }

    bool operator <(const Detection & det) const 
    {
		  return conf < det.conf;
    }
};

struct Deleter
{
  template <typename T>
  void operator()(T * obj) const
  {
    if (obj) {
      delete obj;
    }
  }
};

template <typename T>
using unique_ptr = std::unique_ptr<T, Deleter>;


class Logger : public nvinfer1::ILogger
{
public:
  explicit Logger(bool verbose) : verbose_(verbose) {}

  void log(Severity severity, const char * msg) noexcept override
  {
    if (verbose_ || ((severity != Severity::kINFO) && (severity != Severity::kVERBOSE))) {
      std::cout << msg << std::endl;
    }
  }

private:
  bool verbose_{false};
};

class YolopNet
{
public:
  // Create engine from engine path
  explicit YolopNet(const std::string & engine_path, bool verbose = false);

  // Create engine from serialized onnx model
  YolopNet(
    const std::string & onnx_file_path, 
    const std::string & precision, const int max_batch_size,
    const std::string engine_save_path,
    bool verbose = false,
    size_t workspace_size = (1ULL << 30));

  ~YolopNet();

  std::vector<float> preprocess(const cv::Mat & img,const int c, const int w, const int h,bool keep_ratio=true);
  bool detect(const cv::Mat & in_img,const std::string result_img_save_path);
  void load(const std::string & path);
  void save(const std::string & path);
  void post_process(const cv::Mat img,float* out_objs,float* out_drive_area,float* out_lane,const std::string result_img_save_path);
  void nms(std::vector<Detection>& res, float* out_objs, float conf_thresh, float nms_thresh = 0.5);

  void post_process_detection(cv::Mat & img,float* out_objs);
private:
  unique_ptr<nvinfer1::IRuntime> runtime_ = nullptr;
  unique_ptr<nvinfer1::IHostMemory> plan_ = nullptr;
  unique_ptr<nvinfer1::ICudaEngine> engine_ = nullptr;
  
  unique_ptr<nvinfer1::IExecutionContext> context_ = nullptr;
  cudaStream_t stream_ = nullptr;

  //addr on device(gpu)
  cuda::unique_ptr<float[]> input_d_ = nullptr;
  cuda::unique_ptr<float[]> out_objs_d_ = nullptr;
  cuda::unique_ptr<float[]> out_drive_area_d_ = nullptr;
  cuda::unique_ptr<float[]> out_lane_d_ = nullptr;

  //addr on cpu
  std::unique_ptr<float[]> out_objs_ = nullptr;
  std::unique_ptr<float[]> out_drive_area_ = nullptr;
  std::unique_ptr<float[]> out_lane_ = nullptr;

  int model_input_c_ = 0;
  int model_input_h_ = 0;
  int model_input_w_ = 0;

  // float mean_[3] = { 0.485, 0.456, 0.406 }; //bgr
	// float std_[3] = { 0.229, 0.224, 0.225 }; //bgr

  const float mean_[3] = { 0.406,0.456, 0.485 }; //rgb
	const float std_[3] = { 0.225, 0.224, 0.229 }; //rgb

  const float anchors_[3][6] = { {3,9,5,11,4,20}, {7,18,6,39,12,31},{19,50,38,81,68,157} };
	const float stride_[3] = { 8.0, 16.0, 32.0 };
  int class_num_ = 1;

  float obj_score_thres_ = 0.5;
  float cls_score_thres_ = 0.25;
  float nms_thres_ = 0.5;

  int padw_=0;
  int padh_=0;
};

#endif