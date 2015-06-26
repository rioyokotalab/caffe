#ifndef CAFFE_UTIL_GET_H_
#define CAFFE_UTIL_GET_H

#include <cuda_fp16.h>
#include "caffe/util/fp16_conversion.hpp"

template <typename T> __inline__ __host__ __device__ T Get(unsigned int x);

template <> __inline__ __host__ __device__ unsigned int Get(unsigned int x) { return x; }
template <> __inline__ __host__ __device__ int Get(unsigned int x) { return (int)x; }
template <> __inline__ __host__ __device__ float Get(unsigned int x) { return (float)x; }
template <> __inline__ __host__ __device__ double Get(unsigned int x) { return (float)x; }
template <> __inline__ __host__ __device__ half Get(unsigned int x) {
#ifdef __CUDA_ARCH__
  half h;
  h.x = __float2half_rn(float(x));
  return h;
#else
  return cpu_float2half_rn(float(x));
#endif
}

template <typename T> __inline__ __host__ __device__ T Get(int x);

template <> __inline__ __host__ __device__ int Get(int x) { return x; }
template <> __inline__ __host__ __device__ float Get(int x) { return (float)x; }
template <> __inline__ __host__ __device__ double Get(int x) { return (float)x; }
template <> __inline__ __host__ __device__ half Get(int x) {
#ifdef __CUDA_ARCH__
  half h;
  h.x = __float2half_rn(float(x));
  return h;
#else
  return cpu_float2half_rn(float(x));
#endif
}

template <typename T> __inline__ __host__ __device__ T Get(float x);

template <> __inline__ __host__ __device__ unsigned int Get(float x) { return (unsigned int)x; }
template <> __inline__ __host__ __device__ int Get(float x) { return (int)x; }
template <> __inline__ __host__ __device__ float Get(float x) { return x; }
template <> __inline__ __host__ __device__ double Get(float x) { return (double)x; }
template <> __inline__ __host__ __device__ half Get(float x) {
#ifdef __CUDA_ARCH__
  half h;
  h.x = __float2half_rn(x);
  return h;
#else
  return cpu_float2half_rn(x);
#endif
}

template <typename T> __host__ __device__ T Get(double x);

template <> __inline__ __host__ __device__ int Get(double x) { return (int)x; }
template <> __inline__ __host__ __device__ float Get(double x) { return (float)x; }
template <> __inline__ __host__ __device__ double Get(double x) { return x; }
template <> __inline__ __host__ __device__ half Get(double x) {
#ifdef __CUDA_ARCH__
  half h;
  h.x = __float2half_rn(float(x));
  return h;
#else
  return cpu_float2half_rn(float(x));
#endif
}

template <typename T> __host__ __device__ T Get(half x);

template <> __inline__ __host__ __device__ int Get(half x) { 
#ifdef __CUDA_ARCH__
  return int(__half2float(x));
#else
  return int(cpu_half2float(x));
#endif
}
template <> __inline__ __host__ __device__ float Get(half x) {
#ifdef __CUDA_ARCH__
  return __half2float(x);
#else
  return cpu_half2float(x);
#endif
}
template <> __inline__ __host__ __device__ double Get(half x) {
#ifdef __CUDA_ARCH__
  return double(__half2float(x));
#else
  return double(cpu_half2float(x));
#endif
}
template <> __inline__ __host__ __device__ half Get(half x) { return x; }

#endif
