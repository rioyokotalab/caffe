#ifndef CAFFE_UTIL_GET_H_
#define CAFFE_UTIL_GET_H

#include <cfloat>
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

template <typename T> __host__ __device__ T Get(const half& x);

template <> __inline__ __host__ __device__ int Get(const half& x) { 
#ifdef __CUDA_ARCH__
  return int(__half2float(x));
#else
  return int(cpu_half2float(x));
#endif
}
template <> __inline__ __host__ __device__ float Get(const half& x) {
#ifdef __CUDA_ARCH__
  return __half2float(x);
#else
  return cpu_half2float(x);
#endif
}
template <> __inline__ __host__ __device__ double Get(const half& x) {
#ifdef __CUDA_ARCH__
  return double(__half2float(x));
#else
  return double(cpu_half2float(x));
#endif
}
template <> __inline__ __host__ __device__ half Get(const half& x) { return x; }


template <typename L, typename R> __host__ __device__ L& Incr(L& l, const R& r) {
    double ld = Get<double>(l);
    ld += Get<double>(r);
    l = Get<L>(ld);
    return l;
}

template <typename L, typename R> __host__ __device__ L& Decr(L& l, const R& r) {
    double ld = Get<double>(l);
    ld -= Get<double>(r);
    l = Get<L>(ld);
    return l;
}

template <typename L, typename R> __host__ __device__ L& Mult(L& l, const R& r) {
    double ld = Get<double>(l);
    ld *= Get<double>(r);
    l = Get<L>(ld);
    return l;
}

template <typename L, typename R> __host__ __device__ L& Div(L& l, const R& r) {
    double ld = Get<double>(l);
    ld /= Get<double>(r);
    l = Get<L>(ld);
    return l;
}

__inline__ __host__ __device__
bool operator < (const half& l, const half& r) {
    return Get<float>(l) < Get<float>(r);
}

template <typename T> __inline__ __host__ __device__ float tol(float t) {
    return t;
}

template <> __inline__ __host__ __device__ float tol<half>(float t) {
    return t < 1.e-4 ? 2.e-2 : t * 2.e2;
}

template <typename T> __inline__ __host__ __device__ float maxDtype() {
    return FLT_MAX;
}

template <> __inline__ __host__ __device__ float maxDtype<half>() {
    return 65504.0F;
}

#endif
