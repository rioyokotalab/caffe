
# Installation

See http://caffe.berkeleyvision.org/installation.html for the latest
installation instructions.

Check the users group in case you need help:
https://groups.google.com/forum/#!forum/caffe-users


1.

```bash
$ git clone https://github.com/NVIDIA/caffe
```


2.

Edit ~/.bashrc


```diff
+ export LIBRARY_PATH=${GFLAGS_DIR}/lib:${LIBRARY_PATH}
+ export LD_LIBRARY_PATH=${GFLAGS_DIR}/lib:${LD_LIBRARY_PATH}
```

3.

Edit some Cmake Files

```diff
--- a/cmake/Cuda.cmake
+++ b/cmake/Cuda.cmake
@@ -251,6 +251,9 @@ if(USE_CUDNN)
   endif()
 endif()
 
+list(APPEND CUDA_NVCC_FLAGS "--pre-include $ENV{CUDNN6_HOME}/include/cudnn.h")
+include_directories(BEFORE "$ENV{CUDNN6_HOME}/include")
+
 if(UNIX OR APPLE)
   list(APPEND CUDA_NVCC_FLAGS -std=c++11;-Xcompiler;-fPIC)
 endif()
```

```diff
diff --git a/cmake/Modules/FindAtlas.cmake b/cmake/Modules/FindAtlas.cmake
index 6e15643..e83fbeb 100644
--- a/cmake/Modules/FindAtlas.cmake
+++ b/cmake/Modules/FindAtlas.cmake
@@ -28,7 +28,7 @@ find_path(Atlas_CLAPACK_INCLUDE_DIR NAMES clapack.h PATHS ${Atlas_INCLUDE_SEARCH
 
 find_library(Atlas_CBLAS_LIBRARY NAMES  ptcblas_r ptcblas cblas_r cblas PATHS ${Atlas_LIB_SEARCH_PATHS})
 find_library(Atlas_BLAS_LIBRARY NAMES   atlas_r   atlas                 PATHS ${Atlas_LIB_SEARCH_PATHS})
-find_library(Atlas_LAPACK_LIBRARY NAMES alapack_r alapack lapack_atlas  PATHS ${Atlas_LIB_SEARCH_PATHS})
+find_library(Atlas_LAPACK_LIBRARY NAMES alapack_r lapack lapack_atlas  PATHS ${Atlas_LIB_SEARCH_PATHS})
 
 set(LOOKED_FOR
   Atlas_CBLAS_INCLUDE_DIR
diff --git a/cmake/Modules/FindNCCL.cmake b/cmake/Modules/FindNCCL.cmake
index 1f7d97c..73f816c 100644
--- a/cmake/Modules/FindNCCL.cmake
+++ b/cmake/Modules/FindNCCL.cmake
@@ -20,6 +20,7 @@ find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIR NCCL_LIBRARY
 
 if(NCCL_FOUND)
   message(STATUS "Found NCCL (include: ${NCCL_INCLUDE_DIR}, library: ${NCCL_LIBRARY})")
+  include_directories(${NCCL_INCLUDE_DIR})
   mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARY)
 endif()
``` 

```diff
diff --git a/cmake/Modules/FindNVML.cmake b/cmake/Modules/FindNVML.cmake
index 8747ab3..2a8b3e7 100644
--- a/cmake/Modules/FindNVML.cmake
+++ b/cmake/Modules/FindNVML.cmake
@@ -14,8 +14,16 @@ find_path(NVML_INCLUDE_DIR NAMES nvml.h
     PATHS  ${CUDA_INCLUDE_DIRS} ${NVML_ROOT_DIR}/include
     )
 
+set(MLPATH "/usr/lib64/nvidia")
+
 find_library(NVML_LIBRARY nvidia-ml PATHS ${MLPATH} ${NVML_ROOT_DIR}/lib ${NVML_ROOT_DIR}/lib64)
 
+
+message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
+message(STATUS "NVML_INCLUDE_DIR: ${NVML_INCLUDE_DIR}")
+message(STATUS "NVML_LIBRARY: ${NVML_LIBRARY}")
+
+
 include(FindPackageHandleStandardArgs)
 find_package_handle_standard_args(NVML DEFAULT_MSG NVML_INCLUDE_DIR NVML_LIBRARY)
```

4. 
Then you can do CMAKE

```
cmake -DCMAKE_INSTALL_PREFIX=${CAFFE_0.16_HOME} -DCUDA_TOOLKIT_ROOT_DIR=${CUDA8_HOME}  ..
```

console output seems like 


```
-- Found gflags  (include: /usr/include, library: /usr/lib64/libgflags.so)
```

after that ,you should do make command

```
$ make all -j 64
$ make install -j 64
```

However, if you check library dependency by using ldd command

```
$ldd ${CAFFE_HOME}/build/tools/caffe
libgflags.so.2.2 => ${GFLAGS_DIR}libs//libgflags.so.2.2 (0x0000XXXXXXXXXX)
```
