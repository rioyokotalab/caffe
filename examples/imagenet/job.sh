#!/bin/sh
#PBS -q h-small
#PBS -l select=1:mpiprocs=1:ompthreads=1 
#PBS -W group_list=gi75
#PBS -l walltime=24:00:00
cd $PBS_O_WORKDIR
. /etc/profile.d/modules.sh
module unload anaconda2/4.3.0
module load boost/1.61

declare -a PACKAGES=(\
'ATLAS' \
'boost_1.63.0' \
'glog-0.3.4' \
'gflags-2.2.0' \
'hdf5-1.10.0-patch1' \
'leveldb' \
'lmdb-LMDB_0.9.18' \
'snappy-1.1.4' \
'autoconf-2.68' \
'libtool-2.4.6' \
'gnuplot-5.0.1' \
'opencv-2.4.13' \
'nccl-1.3.4-1' \
'protobuf-3.3.0' \
'redis-2.8.2' \
'opencv-2.4.13' \
'cudnn7/cuda' \
'hiredis \'
'cuda' \
)

# For Libs Caffe depends
PREFIX_PATH=/lustre/gi75/i75012/env/local
for s in "${PACKAGES[@]}"; do
  CPLUS_INCLUDE_PATH=$PREFIX_PATH/$s/include:$CPLUS_INCLUDE_PATH
  LD_LIBRARY_PATH=$PREFIX_PATH/$s/lib:$LD_LIBRARY_PATH
  PATH=$PREFIX_PATH/$s/bin:$PREFIX_PATH/$s/share:$PATH
done
export PATH=$PATH:${PREFIX_PATH}:$LD_LIBRARY_PATH:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH
export LIBRARY_PATH=$LD_LIBRARY_PATH:$LIBRARY_PATH

# setup for cuda
module load cuda/8.0.44

#export CUDA_HOME=/lustre/app/acc/cuda/8.0.61
#export CUDA_ROOT=${CUDA_HOME}
#export PATH=${CUDA_HOME}:${CUDA_HOME}/bin:${PATH}

# setup for cudnn
export CUDNN_ROOT=/lustre/gi75/i75012/env/local/cudnn7/cuda
export CUDNN_LIBRARY=${CUDNN_ROOT}/lib
export CUDNN_INCLUDE=${CUDNN_ROOT}/include
export CUDNN_DIR=${CUDNN_ROOT}
export CUDNN_PATH=${CUDNN_ROOT}

export LD_LIBRARY_PATH=${CUDNN_LIBRARY}:${LD_LIBRARY_PATH}
export PATH=${CUDNN_ROOT}:${PATH}

# For Python
export PYTHON_INCLUDE=/lustre/gi75/i75012/lustre/env/src/pyenv/versions/2.7.9/include
export PYTHON_LIB=/lustre/gi75/i75012/lustre/env/src/pyenv/versions/2.7.9:/home/gi75/i75012/env/src/pyenv/versions/2.7.9/lib:/home/env/src/pyenv/versions/2.7.9/lib/python2.7
export PYTHON_INCLUDE_DIRS=${PYTHON_INCLUDE}
export NUMPY_INCLUDE_DIR=/lustre/gi75/i75012/env/src/pyenv/versions/2.7.9/lib/python2.7/site-packages/numpy/core/include
export PYTHON_LIBRARIES=${PYTHON_LIB}
export PYTHONPATH=${PYTHON_INCLUDE_DIRS}:${PYTHON_LIBRARIES}:${NUMPY_INCLUDE_DIR}

#----- pyenv
export PYENV_ROOT=/lustre/gi75/i75012/env/src/pyenv
if [ -d "${PYENV_ROOT}" ]; then
   export PATH=${PYENV_ROOT}/bin:$PATH
   eval "$(pyenv init -)"
fi

# For Caffe2
CAFFE2_HOME=/lustre/gi75/i75012/dl/caffe2
CAFFE2_ROOT=${CAFFE2_HOME}
PYTHONPATH=${CAFFE2_HOME}/build:${PYTHONPATH}
PYTHONPATH=${CAFFE2_HOME}/caffe2/python:${PYTHONPATH}
PYTHONPATH=${CAFFE2_HOME}/local/caffe2/python:${PYTHONPATH}
PYTHONPATH=${CAFFE2_HOME}:${PYTHONPATH}

/lustre/gi75/i75012/dl/nvcaffe/examples/imagenet/create_imagenet.sh > ./output.log 2>&1
