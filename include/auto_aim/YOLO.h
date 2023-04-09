//
// Created by hongbin on 3/6/23.
//

#ifndef CV_TENSORRT_YOLO_H
#define CV_TENSORRT_YOLO_H

#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "model.h"

#include <iostream>
#include <chrono>
#include <cmath>

#include <opencv2/opencv.hpp>

using namespace nvinfer1;

static Logger gLogger;
const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

class YOLO {
public:
    YOLO();
    ~YOLO();


    bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, bool& is_p6, float& gd, float& gw, std::string& img_dir);
    void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_output_buffer);
    void infer(IExecutionContext& context, cudaStream_t& stream, void** gpu_buffers, float* output, int batchsize);
    void serialize_engine(unsigned int max_batchsize, bool& is_p6, float& gd, float& gw, std::string& wts_name, std::string& engine_name);
    void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context);

private:
//    bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, bool& is_p6, float& gd, float& gw, std::string& img_dir);
//    void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_output_buffer);
//    void infer(IExecutionContext& context, cudaStream_t& stream, void** gpu_buffers, float* output, int batchsize);
//    void serialize_engine(unsigned int max_batchsize, bool& is_p6, float& gd, float& gw, std::string& wts_name, std::string& engine_name);
//    void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context);

};

#endif //CV_TENSORRT_YOLO_H