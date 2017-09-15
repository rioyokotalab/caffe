mkdir build
cd build
LOCAL=/lustre/gi75/i75012/env/local

PYTHON_INCLUDE=/usr/include/python2.7:/lustre/gi75/i75012/env/src/pyenv/versions/2.7.10/lib/python2.7/site-packages/numpy/core/include
PYTHON_LIB=/usr/lib

declare -a PACKAGES=(\
'ATLAS' \
'boost_1.63.0' \
'gflags-2.2.0' \
'glog-0.3.4' \
'hdf5-1.10.0-patch1' \
'lmdb-LMDB_0.9.18/libraries/liblmdb' \
'protobuf-3.3.0' \
'cuda' \
'snappy-1.1.4' \
'opencv-2.4.13' \
'nccl-1.3.4-1' \
'ATLAS' \
)

PREFIX_PATH="$HOME/local/cudnn:/lustre/app/acc/cuda/8.0"

for s in "${PACKAGES[@]}"; do
PREFIX_PATH=$LOCAL/$s:${PREFIX_PATH}
done


CMAKE_PREFIX_PATH=$PYTHON_INCLUDE:$PYTHON_LIB:$HDF5_HL_LIBRARIES:$PREFIX_PATH cmake \
-DCMAKE_INSTALL_PREFIX=$LOCAL/caffe \
-DCUDNN_ROOT=$CUDNN_ROOT \
-DUSE_NCCL=ON \
-DTEST_FP16=ON \
-DUSE_LEVELDB=OFF \
.. | tee configure.log

make -j 248 all && make -j 248 test && make install

#-DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME \
