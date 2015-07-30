#ifndef CAFFE_UTIL_GET_H_
#define CAFFE_UTIL_GET_H

#include <cfloat>
#include "caffe/util/fp16_conversion.hpp"

#ifdef CPU_ONLY
  #define CAFFE_UTIL_GET_HD
  #define CAFFE_UTIL_GET_IHD inline
#else
  #define CAFFE_UTIL_GET_HD __host__ __device__
  #define CAFFE_UTIL_GET_IHD __inline__ __host__ __device__
#endif


template <typename T>
CAFFE_UTIL_GET_HD T Get(unsigned int x);
template <>
CAFFE_UTIL_GET_IHD unsigned int Get(unsigned int x) { return x; }
template <>
CAFFE_UTIL_GET_IHD int Get(unsigned int x) { return (int)x; }
template <>
CAFFE_UTIL_GET_IHD float Get(unsigned int x) { return (float)x; }
template <>
CAFFE_UTIL_GET_IHD double Get(unsigned int x) { return (float)x; }

template <typename T>
CAFFE_UTIL_GET_HD T Get(int x);
template <>
CAFFE_UTIL_GET_IHD int Get(int x) { return x; }
template <>
CAFFE_UTIL_GET_IHD float Get(int x) { return (float)x; }
template <>
CAFFE_UTIL_GET_IHD double Get(int x) { return (float)x; }

template <typename T>
CAFFE_UTIL_GET_HD T Get(float x);
template <>
CAFFE_UTIL_GET_IHD unsigned int Get(float x) { return (unsigned int)x; }
template <>
CAFFE_UTIL_GET_IHD int Get(float x) { return (int)x; }
template <>
CAFFE_UTIL_GET_IHD float Get(float x) { return x; }
template <>
CAFFE_UTIL_GET_IHD double Get(float x) { return (double)x; }

template <typename T>
CAFFE_UTIL_GET_HD T Get(double x);
template <>
CAFFE_UTIL_GET_IHD int Get(double x) { return (int)x; }
template <>
CAFFE_UTIL_GET_IHD float Get(double x) { return (float)x; }
template <>
CAFFE_UTIL_GET_IHD double Get(double x) { return x; }

template <typename L, typename R>
CAFFE_UTIL_GET_HD L& Incr(L& l, const R& r) {
    double ld = Get<double>(l);
    ld += Get<double>(r);
    l = Get<L>(ld);
    return l;
}

template <typename L, typename R>
CAFFE_UTIL_GET_HD L& Decr(L& l, const R& r) {
    double ld = Get<double>(l);
    ld -= Get<double>(r);
    l = Get<L>(ld);
    return l;
}

template <typename L, typename R>
CAFFE_UTIL_GET_HD L& Mult(L& l, const R& r) {
    double ld = Get<double>(l);
    ld *= Get<double>(r);
    l = Get<L>(ld);
    return l;
}

template <typename L, typename R>
CAFFE_UTIL_GET_HD L& Div(L& l, const R& r) {
    double ld = Get<double>(l);
    ld /= Get<double>(r);
    l = Get<L>(ld);
    return l;
}

template <typename T>
CAFFE_UTIL_GET_HD float tol(float t) {
    return t;
}

template <typename T>
CAFFE_UTIL_GET_HD float maxDtype() {
    return FLT_MAX;
}


#ifndef CPU_ONLY

template <>
CAFFE_UTIL_GET_IHD half Get(unsigned int x) {
#ifdef __CUDA_ARCH__
  half h;
  h.x = __float2half_rn(float(x));
  return h;
#else
  return cpu_float2half_rn(float(x));
#endif
}
template <>
CAFFE_UTIL_GET_IHD half Get(int x) {
#ifdef __CUDA_ARCH__
  half h;
  h.x = __float2half_rn(float(x));
  return h;
#else
  return cpu_float2half_rn(float(x));
#endif
}
template <>
CAFFE_UTIL_GET_IHD half Get(float x) {
#ifdef __CUDA_ARCH__
  half h;
  h.x = __float2half_rn(x);
  return h;
#else
  return cpu_float2half_rn(x);
#endif
}
template <>
CAFFE_UTIL_GET_IHD half Get(double x) {
#ifdef __CUDA_ARCH__
  half h;
  h.x = __float2half_rn(float(x));
  return h;
#else
  return cpu_float2half_rn(float(x));
#endif
}

template <typename T>
CAFFE_UTIL_GET_HD T Get(const half& x);
template <>
CAFFE_UTIL_GET_IHD int Get(const half& x) {
#ifdef __CUDA_ARCH__
  return int(__half2float(x));
#else
  return int(cpu_half2float(x));
#endif
}
template <>
CAFFE_UTIL_GET_IHD float Get(const half& x) {
#ifdef __CUDA_ARCH__
  return __half2float(x);
#else
  return cpu_half2float(x);
#endif
}
template <>
CAFFE_UTIL_GET_IHD double Get(const half& x) {
#ifdef __CUDA_ARCH__
  return double(__half2float(x));
#else
  return double(cpu_half2float(x));
#endif
}
template <>
CAFFE_UTIL_GET_IHD half Get(const half& x) { return x; }

CAFFE_UTIL_GET_IHD
bool operator < (const half& l, const half& r) {
    return Get<float>(l) < Get<float>(r);
}

template <>
CAFFE_UTIL_GET_IHD float tol<half>(float t) {
    return t < 1.e-4 ? 2.5e-2 : t * 2.5e2;
}

template <>
CAFFE_UTIL_GET_IHD float maxDtype<half>() {
    return 65504.0F;
}

#endif //ifndef CPU_ONLY

#endif
