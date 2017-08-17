
mkdir build
cd build
LOCAL=/home/hiroki11x/env/local

PYTHON_INCLUDE=/usr/include/python2.7:/usr/lib/python2.7/dist-packages/numpy/core/include
PYTHON_LIB=/usr/lib

declare -a PACKAGES=(\
'boost_1_63_0' \
'gflags-2.2.0' \
'glog-0.3.4' \
'hdf5-1.10.0-patch1' \
'lmdb-LMDB_0.9.18' \
'protobuf-3.3.0' \
'cudnn7/cuda' \
'snappy-1.1.4' \
'opencv-2.4.13' \
'nccl-1.3.4-1' \
'ATLAS' \
)

PREFIX_PATH="$HOME/local/cudnn:/usr/local/cuda"

for s in "${PACKAGES[@]}"; do
PREFIX_PATH=$PREFIX_PATH:$LOCAL/$s
done


for file in `git grep -l '"/home/hiroki11x/env/local/cuda/include/cudnn.h"'`; do
sed -i -e \
's@"/home/hiroki11x/env/local/cuda/include/cudnn.h"@<cudnn.h>@g' \
"$file"
done

CMAKE_PREFIX_PATH=$PYTHON_INCLUDE:$PYTHON_LIB:$HDF5_HL_LIBRARIES:$PREFIX_PATH cmake \
-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
-DCMAKE_INSTALL_PREFIX=/home/hiroki11x/dl/nvcaffe/local \
-DUSE_NCCL=ON \
-DAtlas_LAPACK_LIBRARY=/home/hiroki11x/env/local/ATLAS \
-DUSE_LEVELDB=OFF \
.. | tee configure.log

make all -j 248 && make test -j 248 && make install
