#include "../include/auto_aim/YOLO.h"
#include "../include/auto_aim/Camera.h"

static const int BLUE = 1;

inline float euclid_distance(float x1, float y1, float x2, float y2){
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}


inline cv::Point2f get_center(float width, float height) {return cv::Point2f(width / 2, height / 2); }

int main(int argc, char** argv) {
    cudaSetDevice(kGpuId);
    std::string wts_name = "";
    std::string engine_name = "";
    bool is_p6 = false;
    float gd = 0.0f, gw = 0.0f;
    std::string img_dir;

    YOLO detector;
//    Camera camera;
//    camera.init();

    if (!detector.parse_args(argc, argv, wts_name, engine_name, is_p6, gd, gw, img_dir)) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov5_det -s [.wts] [.engine] [n/s/m/l/x/n6/s6/m6/l6/x6 or c/c6 gd gw]  // serialize model to plan file" << std::endl;
        std::cerr << "./yolov5_det -d [.engine] ../images  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

//    // Create a model using the API directly and serialize it to a file
    if (!wts_name.empty()) {
        detector.serialize_engine(kBatchSize, is_p6, gd, gw, wts_name, engine_name);
        return 0;
    }

    // Deserialize the engine from file
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    detector.deserialize_engine(engine_name, &runtime, &engine, &context);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Init CUDA preprocessing
    cuda_preprocess_init(kMaxInputImageSize);

    // Prepare cpu and gpu buffers
    float* gpu_buffers[2];
    float* cpu_output_buffer = nullptr;
    detector.prepare_buffers(engine, &gpu_buffers[0], &gpu_buffers[1], &cpu_output_buffer);

    cv::VideoCapture cap("/dev/video0");
    std::string window_name = "Yolov5 USB Camera";

    // if not success, exit program
    if (cap.isOpened() == false)
    {
        std::cout << "Cannot open the video camera" << std::endl;
        std::cin.get(); //wait for any key press
        return -1;
    }

    float dWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
    float dHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video

    std::cout << "Resolution of the video : " << dWidth << " x " << dHeight << std::endl;


    while (true) {
        cv::Mat img;
        std::vector<cv::Mat> img_batch;
//        camera.getImage(img);
        bool bSuccess = cap.read(img); // read a new img from video

        //Breaking the while loop if the frames cannot be captured
        if (bSuccess == false) {
            std::cout << "Video camera is disconnected" << std::endl;
            std::cin.get(); //Wait for any key press
            break;
        }
        if (img.empty()) continue;

        img_batch.push_back(img);
        cuda_batch_preprocess(img_batch, gpu_buffers[0], kInputW, kInputH, stream);

        // Run inference
        auto start = std::chrono::system_clock::now();
        detector.infer(*context, stream, (void**)gpu_buffers, cpu_output_buffer, kBatchSize);
        auto end = std::chrono::system_clock::now();
//        std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        double t = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // NMS
        std::vector<std::vector<Detection>> res_batch;
        batch_nms(res_batch, cpu_output_buffer, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);

//        for (size_t i = 0; i < res_batch.size(); i++)  {
//            for (size_t j = 0; j < res_batch[i].size(); j++) {
//                if (res_batch[i][j].class_id == BLUE) {
//                    std::cout << res_batch[i][j].bbox[0]<< std::endl;
//                }
//            }
//        }

        Detection best_det = {{0.0f}, -1,-1,{0.0f} };
        float smallest_dist = std::numeric_limits<float>::max();



        for (size_t i = 0; i < res_batch.size(); i++)  {
            for (size_t j = 0; j < res_batch[i].size(); j++) {


                cv::Point2f target_center = get_center(res_batch[i][j].bbox[0] , res_batch[i][j].bbox[1]);
                cv::Point2f frame_center = get_center(dWidth, dHeight);

                float dist = euclid_distance(target_center.x, target_center.y, frame_center.x, frame_center.y);
                std::cout << dist << std::endl;
                if (dist < smallest_dist) {
                    best_det = res_batch[i][j];
                }
                std::cout << euclid_distance(target_center.x, target_center.y, frame_center.x, frame_center.y) << std::endl;

            }
        }
        draw_bbox(img, best_det);

        std::string label = cv::format("Inference time : %.2f ms", t);
        cv::putText(img,label, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,255));
        cv::imshow(window_name , img);


        if (cv::waitKey(1) == 113)
        {
            std::cout << "Esc key is pressed by user. Stopping the video" << std::endl;
            break;
        }
    }


//    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(gpu_buffers[0]));
    CUDA_CHECK(cudaFree(gpu_buffers[1]));
    delete[] cpu_output_buffer;
    cuda_preprocess_destroy();

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    std::cout << "Memory Freed";


    return 0;
}

