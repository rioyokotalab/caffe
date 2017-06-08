# Config file for the Caffe package.
#
# Note:
#   Caffe and this config file depends on opencv,
#   so put `find_package(OpenCV)` before searching Caffe
#   via `find_package(Caffe)`. All other lib/includes
#   dependencies are hard coded in the file
#
# After successful configuration the following variables
# will be defined:
#
#   Caffe_INCLUDE_DIRS - Caffe include directories
#   Caffe_LIBRARIES    - libraries to link against
#   Caffe_DEFINITIONS  - a list of definitions to pass to compiler
#
#   Caffe_HAVE_CUDA    - signals about CUDA support
#   Caffe_HAVE_CUDNN   - signals about cuDNN support
# 
#
#
# OpenCV dependency

if(ON)
  if(NOT OpenCV_FOUND)
    set(Caffe_OpenCV_CONFIG_PATH "/home/hiroki11/env/local/opencv_2.4.13/share/OpenCV")
    if(Caffe_OpenCV_CONFIG_PATH)
      get_filename_component(Caffe_OpenCV_CONFIG_PATH ${Caffe_OpenCV_CONFIG_PATH} ABSOLUTE)

      if(EXISTS ${Caffe_OpenCV_CONFIG_PATH} AND NOT TARGET opencv_core)
        message(STATUS "Caffe: using OpenCV config from ${Caffe_OpenCV_CONFIG_PATH}")
        include(${Caffe_OpenCV_CONFIG_PATH}/OpenCVModules.cmake)
      endif()

    else()
      find_package(OpenCV REQUIRED)
    endif()
    unset(Caffe_OpenCV_CONFIG_PATH)
  endif()
endif()

# Compute paths
get_filename_component(Caffe_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
set(Caffe_INCLUDE_DIRS "/home/hiroki11/env/local/cuda/include;/home/hiroki11/env/local/boost_1.63.0/include;/home/hiroki11/env/local/glog_0.3.4/include;/usr/include;/home/hiroki11/env/local/hdf5_1.10.0/include;/home/hiroki11/env/local/lmdb_0.9.18/include;/home/hiroki11/env/local/leveldb/include;/home/hiroki11/env/local/snappy_1.1.4/include;/usr/local/cuda-8.0/include;/home/hiroki11/env/local/opencv_2.4.13/include/opencv;/home/hiroki11/env/local/opencv_2.4.13/include;/home/hiroki11/env/local/atlas_3.10.3/include;/home/hiroki11/env/local/nccl_1.3.4/include")

get_filename_component(__caffe_include "${Caffe_CMAKE_DIR}/../../include" ABSOLUTE)
list(APPEND Caffe_INCLUDE_DIRS ${__caffe_include})
unset(__caffe_include)


# Our library dependencies
if(NOT TARGET caffe AND NOT caffe_BINARY_DIR)
  include("${Caffe_CMAKE_DIR}/CaffeTargets.cmake")
endif()

# List of IMPORTED libs created by CaffeTargets.cmake
set(Caffe_LIBRARIES caffe)

# Definitions
set(Caffe_DEFINITIONS "-DUSE_OPENCV;-DUSE_LMDB;-DUSE_LEVELDB;-DUSE_NCCL")

# Cuda support variables
set(Caffe_CPU_ONLY OFF)
set(Caffe_HAVE_CUDA TRUE)
set(Caffe_HAVE_CUDNN TRUE)

