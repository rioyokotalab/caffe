#ifndef CAFFE_UTIL_FP16_CONVERSION_H_
#define CAFFE_UTIL_FP16_CONVERSION_H_

#include <cuda_fp16.h>
// Host functions for converting between FP32 and FP16 formats
// Paulius Micikevicius (pauliusm@nvidia.com)

half cpu_float2half_rn(float f);
float cpu_half2float(half h);

#endif
