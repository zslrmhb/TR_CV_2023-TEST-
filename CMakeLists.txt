# adapted the CUDA and tensorrt part from
# https://github.com/wang-xinyu/tensorrtx/blob/master/yolov5/CMakeLists.txt

cmake_minimum_required(VERSION 3.24)
project(cv_tensorrt)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# TODO(Call for PR): make cmake compatible with Windows
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
# TODO(Call for PR): make TRT path configurable from command line
include_directories(/home/nvidia/TensorRT-8.2.5.1/include/)
link_directories(/home/nvidia/TensorRT-8.2.5.1/lib/)

include_directories(${PROJECT_SOURCE_DIR}/yolov5/src/)
include_directories(${PROJECT_SOURCE_DIR}/yolov5/plugin/)
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/yolov5/src/*.cpp ${PROJECT_SOURCE_DIR}/yolov5/src/*.cu)
file(GLOB_RECURSE PLUGIN_SRCS ${PROJECT_SOURCE_DIR}/yolov5/plugin/*.cu)

add_library(myplugins SHARED ${PLUGIN_SRCS})
target_link_libraries(myplugins nvinfer cudart)

# Opencv
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

# Huaray SDK
find_library( CAMERA_LIB NAMES MVSDK HINTS ./camera_lib/x86 REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp src/Camera.cpp src/YOLO.cpp ${SRCS} )

target_link_libraries(${PROJECT_NAME} nvinfer)
target_link_libraries(${PROJECT_NAME} cudart)
target_link_libraries(${PROJECT_NAME}  myplugins)
target_link_libraries(${PROJECT_NAME} ${CAMERA_LIB} ${OpenCV_LIBS} rt pthread)