#include "yolop.hpp"

#include <NvOnnxParser.h>

#include <memory>
#include <numeric>

#define DEBUG_LINE std::cout << __FILE__ << "," << __LINE__ << std::endl;

static int MAX_BOX_COUNT = 25200;

// struct alignas(float) Detection {
//     //center_x center_y w h
//     float bbox[4];
//     float conf;  // bbox_conf * cls_conf
//     float class_id;

//     void print()
//     {
//         std::cout<<"conf:"<<conf<<std::endl;
//         std::cout<<"c_x:"<<bbox[0]
//             <<",c_y:"<<bbox[1]
//             <<",w:"<<bbox[2]
//             <<",h:"<<bbox[3]<<std::endl;
//     }
// };

YolopNet::YolopNet(const std::string &engine_path, bool verbose)
{
    Logger logger(verbose);
    runtime_ = unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    // load model to  engine_
    load(engine_path);

    // check model
    int max_batch_size = engine_->getProfileDimensions(0, 0, nvinfer1::OptProfileSelector::kMAX).d[0];
    std::cout << "max_batch_size:" << max_batch_size << std::endl;

    int nbBindings = engine_->getNbBindings();
    std::cout << "nbBindings:" << nbBindings << std::endl;
    for (int i = 0; i < nbBindings; i++)
    {
        auto dims = engine_->getBindingDimensions(i);
        int MAX_DIMS = 8;
        for (int j = 0; j < MAX_DIMS; j++)
        {
            std::cout << dims.d[j] << ",";
        }
        std::cout << std::endl;
    }

    context_ = unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (!context_)
    {
        std::cout << "failed to build engine" << std::endl;
        return;
    }

    // malloc addr on gpu
    max_batch_size = engine_->getProfileDimensions(0, 0, nvinfer1::OptProfileSelector::kMAX).d[0];
    auto dims = engine_->getBindingDimensions(0);
    int input_size = dims.d[1] * dims.d[2] * dims.d[3];
    input_d_ = cuda::make_unique<float[]>(max_batch_size * input_size);
    out_objs_d_ = cuda::make_unique<float[]>(max_batch_size * MAX_BOX_COUNT * 6);
    out_drive_area_d_ = cuda::make_unique<float[]>(max_batch_size * 2 * 640 * 640);
    out_lane_d_ = cuda::make_unique<float[]>(max_batch_size * 2 * 640 * 640);

    // malloc addr on cpu
    out_objs_ = std::make_unique<float[]>(max_batch_size * MAX_BOX_COUNT * 6);
    out_drive_area_ = std::make_unique<float[]>(max_batch_size * 2 * 640 * 640);
    out_lane_ = std::make_unique<float[]>(max_batch_size * 2 * 640 * 640);

    cudaStreamCreate(&stream_);
}

YolopNet::YolopNet(
    const std::string &onnx_file_path,
    const std::string &precision, const int max_batch_size,
    const std::string engine_save_path,
    bool verbose,
    size_t workspace_size)
{
    Logger logger(verbose);
    runtime_ = unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (!runtime_)
    {
        std::cout << "Fail to create runtime" << std::endl;
        return;
    }

    // Create builder
    auto builder = unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder)
    {
        std::cout << "Fail to create builder" << std::endl;
        return;
    }

    // create config
    auto config = unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        std::cout << "Fail to create builder config" << std::endl;
        return;
    }
#if (NV_TENSORRT_MAJOR * 1000) + (NV_TENSORRT_MINOR * 100) + NV_TENSOR_PATCH >= 8400
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspace_size);
#else
    config->setMaxWorkspaceSize(workspace_size);
#endif

    // create network
    const auto flag =
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flag));
    if (!network)
    {
        std::cout << "Fail to create network" << std::endl;
        return;
    }

    // parse onnx model
    auto parser = unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (!parser)
    {
        std::cout << "Fail to create parser" << std::endl;
        return;
    }
    parser->parseFromFile(onnx_file_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kERROR));
    const auto input = network->getInput(0);
    const auto input_dims = input->getDimensions();
    const auto input_channel = input_dims.d[1];
    const auto input_height = input_dims.d[2];
    const auto input_width = input_dims.d[3];
    std::cout << "input_channel:" << input_channel << std::endl;
    std::cout << "input_height:" << input_height << std::endl;
    std::cout << "input_width:" << input_width << std::endl;

    // check network output
    int nbOutputs = network->getNbOutputs();
    for (int i = 0; i < nbOutputs; i++)
    {
        std::cout << "output index:" << i << std::endl;
        auto output = network->getOutput(i);
        const auto output_dims = output->getDimensions();
        for (int j = 0; j < 4; j++)
        {
            std::cout << output_dims.d[j] << ",";
        }
        std::cout << std::endl;
    }

    // create profile
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions(
        network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4{max_batch_size, input_channel, input_height, input_width});
    profile->setDimensions(
        network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4{max_batch_size, input_channel, input_height, input_width});
    profile->setDimensions(
        network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4{max_batch_size, input_channel, input_height, input_width});
    config->addOptimizationProfile(profile);

    // Build engine
    std::cout << "Applying optimizations and building TRT CUDA engine..." << std::endl;
    plan_ = unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!plan_)
    {
        std::cout << "Fail to create serialized network" << std::endl;
        return;
    }
    engine_ = unique_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(plan_->data(), plan_->size()));

    // save engine file
    save(engine_save_path);
}

YolopNet::~YolopNet()
{
    if (stream_)
    {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
    }
}

std::vector<float> YolopNet::preprocess(const cv::Mat &in_img, const int c, const int w, const int h, bool keep_ratio)
{
    cv::Mat resized_image;
    int srch = in_img.rows, srcw = in_img.cols;
    float hw_scale = (float)srch / srcw;
    if (keep_ratio)
    {
        if (hw_scale > 1)
        {
            int newh = h;
            int neww = int(w / hw_scale);
            cv::resize(in_img, resized_image, cv::Size(neww, newh), cv::INTER_AREA);
            padw_ = int((w - neww) * 0.5); //更新padw_
            std::cout << "w:" << w << ",neww:" << neww << ",padw:" << padw_ << std::endl;
            copyMakeBorder(resized_image, resized_image, 0, 0, padw_, w - neww - padw_, cv::BORDER_CONSTANT, 0);
        }
        else
        {
            int newh = h * hw_scale;
            int neww = w;
            cv::resize(in_img, resized_image, cv::Size(neww, newh), cv::INTER_AREA);
            padh_ = (int)(h - newh) * 0.5; //更新padh_
            // std::cout << "h:" << h << ",newh:" << newh << ",padh:" << padh_ << std::endl;
            copyMakeBorder(resized_image, resized_image, padh_, h - newh - padh_, 0, 0, cv::BORDER_CONSTANT, 0);
        }
    }
    else
    {
        cv::resize(in_img, resized_image, cv::Size(w, h), cv::INTER_AREA);
    }

    // cv::imwrite("/home/tensorrt_yolop/test_resized.jpg", resized_image);

    // int x = 227, y = 306;
    // python version
    //  bgr order,img[306,227,:]=[68 41 32]
    //  after normalize,img[306,227,:]=[-0.95342064 -1.317927   -1.2467101 ]
    cv::Mat rgb;
    cv::cvtColor(resized_image, rgb, cv::COLOR_BGR2RGB);

    // int r = int(rgb.at<cv::Vec3b>(y, x)[0]);
    // int g = int(rgb.at<cv::Vec3b>(y, x)[1]);
    // int b = int(rgb.at<cv::Vec3b>(y, x)[2]);

    // std::cout << "r:" << r << ",g:" << g << ",b:" << b << std::endl;

    cv::Mat img_float;
    rgb.convertTo(img_float, CV_32FC3, 1 / 255.0);
    for (int i = 0; i < img_float.rows; i++)
    {
        float *pdata = (float *)(img_float.data + i * img_float.step);
        for (int j = 0; j < img_float.cols; j++)
        {
            pdata[0] = (pdata[0] - mean_[0]) / std_[0];
            pdata[1] = (pdata[1] - mean_[1]) / std_[1];
            pdata[2] = (pdata[2] - mean_[2]) / std_[2];
            pdata += 3;
        }
    }

    // float* pix_addr = (float*)(img_float.data + (y * img_float.cols * 3  + x * 3));
    // cv::Vec3f bgrPixel = img_float.at<cv::Vec3f>(y, x);
    // std::cout << "normalize:" << bgrPixel << std::endl;

    // HWC TO CHW
    std::vector<cv::Mat> input_channels(c);
    cv::split(img_float, input_channels);

    std::vector<float> result(h * w * c);
    auto data = result.data();
    int channel_length = h * w;
    for (int i = 0; i < c; ++i)
    {
        memcpy(data, input_channels[i].data, channel_length * sizeof(float));
        data += channel_length;
    }

    // for (int i = 0; i < 3; i++)
    // {
    //     DEBUG_LINE
    //     int index = y * 640 + x + i * 640 * 640;
    //     //   std::cout<<"index:"<<index<<std::endl;
    //     // std::cout << "value is:" << result[index] << std::endl;
    // }

    return result;
}

bool YolopNet::detect(const cv::Mat &in_img,const std::string result_img_save_path)
{
    auto start =  std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    auto dims = engine_->getBindingDimensions(0);
    model_input_c_ = dims.d[1];
    model_input_h_ = dims.d[2];
    model_input_w_ = dims.d[3];
    std::vector<float> input = preprocess(in_img, model_input_c_, model_input_w_, model_input_h_);
    // std::cout << "input size:" << input.size() << std::endl;
    CHECK_CUDA_ERROR(
        cudaMemcpy(input_d_.get(), input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<void *> buffers = {
        input_d_.get(), out_objs_d_.get(), out_drive_area_d_.get(), out_lane_d_.get()};
    try
    {
        if (!context_)
        {
            throw std::runtime_error("Fail to create context");
        }
        auto input_dims = engine_->getBindingDimensions(0);
        int batch_size = 1;
        context_->setBindingDimensions(0, nvinfer1::Dims4(batch_size, input_dims.d[1], input_dims.d[2], input_dims.d[3]));
        context_->enqueueV2(buffers.data(), stream_, nullptr);
        cudaStreamSynchronize(stream_);
    }
    catch (const std::runtime_error &e)
    {
        DEBUG_LINE
        return false;
    }

    // copy results to host
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
        out_objs_.get(), out_objs_d_.get(), sizeof(float) * MAX_BOX_COUNT * 6, cudaMemcpyDeviceToHost,
        stream_));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
        out_drive_area_.get(), out_drive_area_d_.get(), sizeof(float) * 2 * 640 * 640, cudaMemcpyDeviceToHost,
        stream_));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
        out_lane_.get(), out_lane_d_.get(), sizeof(float) * 2 * 640 * 640, cudaMemcpyDeviceToHost,
        stream_));
    cudaStreamSynchronize(stream_);

    end = std::chrono::high_resolution_clock::now();
    int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout<<"inference fps:"<<(1000/int_ms.count())<<std::endl;

    post_process(in_img, out_objs_.get(), out_drive_area_.get(), out_lane_.get(),result_img_save_path);
    
    end = std::chrono::high_resolution_clock::now();
    int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout<<"inference+postprocess fps:"<<(1000/int_ms.count())<<std::endl;
    
    return true;
}

void YolopNet::post_process(const cv::Mat img, float *out_objs, float *out_drive_area, float *out_lane,const std::string result_img_save_path)
{
    // cv::Mat outimg = img.clone();
    cv::Mat outimg = img;//浅拷贝 节省时间
    float ratioh = 1.0 * model_input_h_ / img.rows;
    float ratiow = 1.0 * model_input_w_ / img.cols;
    float r = ratioh; //判断前处理在缩放成640*640时缩放的比例
    if(ratiow < ratioh)
    {
        r = ratiow;
    }
    // std::cout << "ratioh=" << ratioh << ",ratiow=" << ratiow << std::endl;
    int area = model_input_h_ * model_input_w_;

    // debug
    // int x = 30;
    // int y = 400;
    // int index1 = y * 640 + x;
    // int index2 = 640 * 640 + y * 640 + x;
    // float p1 = out_drive_area[index1];
    // float p2 = out_drive_area[index2];
    // std::cout << "p1:" << p1 << std::endl;
    // std::cout << "p2:" << p2 << std::endl;

    //可行驶区域及车道线
    for (int y = 0; y < outimg.rows; y++)
    {
        for (int x = 0; x < outimg.cols; x++)
        {
            //模型输出中对应的x,y
            // int x_in_ouput = static_cast<int>(x / ratiow);
            // int y_in_output = static_cast<int>(y / ratioh);
            //原始图像上的x,y对应到模型输出上的x,y
            int x_in_ouput = int(x * r) + padw_;
            int y_in_output = int(y * r) + padh_;

            //输出相对输入尺寸未进行下采样 640x640输入,输出依然是640x640
            float p1 = out_drive_area[y_in_output * model_input_w_ + x_in_ouput];
            float p2 = out_drive_area[area + y_in_output * model_input_w_ + x_in_ouput];
            if (p1 < p2)
            {
                outimg.at<cv::Vec3b>(y, x)[0] = 0;
                outimg.at<cv::Vec3b>(y, x)[1] = 255;
                outimg.at<cv::Vec3b>(y, x)[2] = 0;
            }

            //绘制车道线
            float lane_c1 = out_lane[y_in_output * model_input_w_ + x_in_ouput];
            float lane_c2 = out_lane[area + y_in_output * model_input_w_ + x_in_ouput];
            if (lane_c1 < lane_c2)
            {
                outimg.at<cv::Vec3b>(y, x)[0] = 0;
                outimg.at<cv::Vec3b>(y, x)[1] = 0;
                outimg.at<cv::Vec3b>(y, x)[2] = 255;
            }
        }
    }

    //处理box
    post_process_detection(outimg, out_objs);

    if(!result_img_save_path.empty())
    {
        // std::cout<<"save to "<<result_img_save_path<<std::endl;
        cv::imwrite(result_img_save_path, outimg);
    }
}

void YolopNet::post_process_detection(cv::Mat &img, float *out_objs)
{
    std::vector<Detection> detections;
    nms(detections, out_objs, obj_score_thres_, nms_thres_);

    float ratioh = 1.0 * model_input_h_ / img.rows; // 640/1000
    float ratiow = 1.0 * model_input_w_ / img.cols; // 640/2000
    float r = std::min(ratioh, ratiow);
    for (auto det:detections)
    { 
        int cx = int(det.bbox[0]);
        int cy = int(det.bbox[1]);
        int w = int(det.bbox[2]);
        int h = int(det.bbox[3]);

        //缩放到原图
        int left = int((cx - w / 2 - padw_) / r);
        int right = int((cx + w / 2 - padw_) / r);
        int top = int((cy - h / 2 - padh_) / r);
        int bottom = int((cy + h / 2 - padh_) / r);

        // std::cout << "cx:" << cx << ",w:" << w << ",padw_:" << padw_ << ",r:" << r << std::endl;
        // std::cout << "left:" << left << ",top:" << top << ",right:" << right << ",bottom:" << bottom << std::endl;
        cv::rectangle(img, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 2);
        
    }
}

void YolopNet::nms(std::vector<Detection> &res, float *out_objs, float conf_thresh, float nms_thresh)
{
    // int det_size = sizeof(Detection) / sizeof(float);
    std::map<float, std::vector<Detection>> m;

    // 25200 x 6 [cx,cy,w,h,conf,cls]
    for (int i = 0; i < 25200; i++)
    {
        float obj_score = out_objs[6 * i + 4];
        if (obj_score > obj_score_thres_)
        {    
            Detection det;
            det.class_id = int(out_objs[6 * i + 5]);
            det.conf = obj_score;
            det.bbox[0] = out_objs[6 * i + 0];
            det.bbox[1] = out_objs[6 * i + 1];
            det.bbox[2] = out_objs[6 * i + 2];
            det.bbox[3] = out_objs[6 * i + 3];

            if (m.count(det.class_id) == 0)
                m.emplace(det.class_id, std::vector<Detection>());
            m[det.class_id].push_back(det);
        }
    }
    
    auto iou = [](float lbox[4], float rbox[4]) -> float
    {
        float interBox[] = {
            (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
            (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
            (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
            (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
        };

        if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
            return 0.0f;

        float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
        return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
    };

    for (auto it = m.begin(); it != m.end(); it++)
    {
        // std::cout << it->second[0].class_id << " --- " << std::endl;
        auto &dets = it->second;
        std::sort(dets.begin(), dets.end()); //同个类型的目标按照conf大小从高到低排序
        for (size_t m = 0; m < dets.size(); ++m)
        {
            auto &item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n)
            {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh)
                {
                    dets.erase(dets.begin() + n); //移除同一个目标的框
                    --n;                          //移除以后每个在这之后的元素的下标都要前移1.
                }
            }
        }
    }
}

void YolopNet::load(const std::string &path)
{
    std::ifstream file(path, std::ios::in | std::ios::binary);
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    char *buffer = new char[size];
    file.read(buffer, size);
    file.close();
    if (runtime_)
    {
        engine_ = unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(buffer, size));
    }
    delete[] buffer;
}

void YolopNet::save(const std::string &path)
{
    std::cout << "Writing to " << path << "..." << std::endl;
    std::ofstream file(path, std::ios::out | std::ios::binary);
    file.write(reinterpret_cast<const char *>(plan_->data()), plan_->size());
}