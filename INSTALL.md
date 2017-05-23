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

```bash
$ cd caffe
$ mkdir build
```

modify cmake/Cuda.cmake
modify Makefile
modify cmake/Modules/FindAtlas.cmake

```bash
$ cd build
$ CMAKE_PREFIX_PATH=$NV_CAFFE_HOME/share \
  cmake \
  -DCMAKE_INSTALL_PREFIX=$NV_CAFFE_HOME \
  -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME\ #required >=8.0
  -DBLAS=$Atlas_LAPACK_LIBRARY \
  -DCUDNN_LIBRARY=$CUDNN_LIBRARY \ #required >=6.0
  -DCUDNN_INCLUDE=$CUDNN_INCLUDE \ #required >=6.0
  ..

$ make all -j 8
$
