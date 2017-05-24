#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "caffe" for configuration "Release"
set_property(TARGET caffe APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(caffe PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "proto;proto;/home/hiroki11/env/local/boost_1.63.0/lib/libboost_system.so;/home/hiroki11/env/local/boost_1.63.0/lib/libboost_thread.so;/home/hiroki11/env/local/boost_1.63.0/lib/libboost_filesystem.so;-lpthread;/home/hiroki11/env/local/glog_0.3.4/lib/libglog.so;/usr/lib64/libgflags.so;/usr/lib64/libprotobuf.so;-lpthread;/home/hiroki11/env/local/hdf5_1.10.0/lib/libhdf5_hl.so;/home/hiroki11/env/local/hdf5_1.10.0/lib/libhdf5.so;/home/hiroki11/env/local/hdf5_1.10.0/lib/libhdf5_hl.so;/home/hiroki11/env/local/hdf5_1.10.0/lib/libhdf5.so;/home/hiroki11/env/local/lmdb_0.9.18/lib/liblmdb.so;/home/hiroki11/env/local/leveldb/lib/libleveldb.so;/home/hiroki11/env/local/snappy_1.1.4/lib/libsnappy.so;/usr/local/cuda-8.0/lib64/libcudart.so;/usr/local/cuda-8.0/lib64/libcurand.so;/usr/local/cuda-8.0/lib64/libcublas.so;/home/hiroki11/env/local/cuda/lib64/libcudnn.so;opencv_core;opencv_highgui;opencv_imgproc;/home/hiroki11/env/local/atlas_3.10.3/lib/liblapack.a;/home/hiroki11/env/local/atlas_3.10.3/lib/libptcblas.a;/home/hiroki11/env/local/atlas_3.10.3/lib/libatlas.a;/usr/lib64/libpython2.7.so;/home/hiroki11/env/local/boost_1.63.0/lib/libboost_python.so;/home/hiroki11/env/local/nccl_1.3.4/lib/libnccl.so;/usr/lib64/nvidia/libnvidia-ml.so"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libcaffe-nv.so.0.16.1"
  IMPORTED_SONAME_RELEASE "libcaffe-nv.so.0.16"
  )

list(APPEND _IMPORT_CHECK_TARGETS caffe )
list(APPEND _IMPORT_CHECK_FILES_FOR_caffe "${_IMPORT_PREFIX}/lib/libcaffe-nv.so.0.16.1" )

# Import target "proto" for configuration "Release"
set_property(TARGET proto APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(proto PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libproto.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS proto )
list(APPEND _IMPORT_CHECK_FILES_FOR_proto "${_IMPORT_PREFIX}/lib/libproto.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
