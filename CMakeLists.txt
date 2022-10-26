cmake_minimum_required(VERSION 3.14)

project(tensorrt_yolop)

find_package(Boost REQUIRED)
find_package(Boost REQUIRED COMPONENTS
  serialization
  thread
  filesystem
  regex
)


find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)
add_definitions("-Wall -g")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
add_definitions(-O2 -pthread)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(CUDA REQUIRED)
option(CUDA_AVAIL "CUDA available" OFF)
find_package(CUDA)
if(CUDA_FOUND)
  find_library(CUBLAS_LIBRARIES cublas HINTS
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${CUDA_TOOLKIT_ROOT_DIR}/lib
  )
  if(CUDA_VERBOSE)
    message(STATUS "CUDA is available!")
    message(STATUS "CUDA Libs: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA Headers: ${CUDA_INCLUDE_DIRS}")
  endif()
  set(CUDA_AVAIL ON)
else()
  message("CUDA NOT FOUND")
  set(CUDA_AVAIL OFF)
endif()

# set flags for TensorRT availability
option(TRT_AVAIL "TensorRT available" OFF)
# try to find the tensorRT modules
find_library(NVINFER NAMES nvinfer)
find_library(NVONNXPARSER nvonnxparser)
find_library(NVINFER_PLUGIN NAMES nvinfer_plugin)
if(NVINFER AND NVONNXPARSER AND NVINFER_PLUGIN)
  if(CUDA_VERBOSE)
    message(STATUS "TensorRT is available!")
    message(STATUS "NVINFER: ${NVINFER}")
    message(STATUS "NVPARSERS: ${NVPARSERS}")
    message(STATUS "NVINFER_PLUGIN: ${NVINFER_PLUGIN}")
    message(STATUS "NVONNXPARSER: ${NVONNXPARSER}")
  endif()
  set(TRT_AVAIL ON)
else()
  message("TensorRT is NOT Available")
  set(TRT_AVAIL OFF)
endif()

# set flags for CUDNN availability
option(CUDNN_AVAIL "CUDNN available" OFF)
# try to find the CUDNN module
find_library(CUDNN_LIBRARY
NAMES libcudnn.so${__cudnn_ver_suffix} libcudnn${__cudnn_ver_suffix}.dylib ${__cudnn_lib_win_name}
PATHS $ENV{LD_LIBRARY_PATH} ${__libpath_cudart} ${CUDNN_ROOT_DIR} ${PC_CUDNN_LIBRARY_DIRS} ${CMAKE_INSTALL_PREFIX}
PATH_SUFFIXES lib lib64 bin
DOC "CUDNN library."
)
if(CUDNN_LIBRARY)
  if(CUDA_VERBOSE)
    message(STATUS "CUDNN is available!")
    message(STATUS "CUDNN_LIBRARY: ${CUDNN_LIBRARY}")
  endif()
  set(CUDNN_AVAIL ON)
else()
  message("CUDNN is NOT Available")
  set(CUDNN_AVAIL OFF)
endif()


if(TRT_AVAIL AND CUDA_AVAIL AND CUDNN_AVAIL)
  include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
  )

  set(SOURCE_FILE 
    ${PROJECT_SOURCE_DIR}/src/yolop.cpp
    ${PROJECT_SOURCE_DIR}/src/nodelet.cpp
    ${PROJECT_SOURCE_DIR}/src/main.cpp
  )

  ament_auto_add_executable(tensorrt_yolop ${SOURCE_FILE})
  target_link_libraries(tensorrt_yolop 
        ${NVINFER}
        ${NVONNXPARSER}
        ${NVINFER_PLUGIN}
        ${CUDA_LIBRARIES}
        ${CUBLAS_LIBRARIES}
        ${CUDNN_LIBRARY}
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES})

  ament_auto_package(INSTALL_TO_SHARE
    launch
    data
  )
  
  install(
    TARGETS
      tensorrt_yolop
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
  )

else()
  message("TensorrtYolo won't be built, CUDA and/or TensorRT were not found.")
  ament_auto_package(INSTALL_TO_SHARE
    config
    data
    launch
  )
endif()



# include_directories(${PROJECT_SOURCE_DIR}/include)
# # include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# # cuda
# include_directories(/usr/local/cuda/include)
# link_directories(/usr/local/cuda/lib64)
# # tensorrt
# include_directories(/usr/include/x86_64-linux-gnu/)
# link_directories(/usr/lib/x86_64-linux-gnu/)











