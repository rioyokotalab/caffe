#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>
#include <thrust/inner_product.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <>
void caffe_gpu_gemm<float,float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm<double,double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm<half,float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const half* A, const half* B, const float beta,
    half* C) {
  // cudaDeviceSynchronize();
  // printf("%s\n", __FUNCTION__);
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemmEx(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, CUBLAS_DATA_HALF, ldb, A, CUBLAS_DATA_HALF, 
      lda, &beta, C, CUBLAS_DATA_HALF, N));
}

template <>
void caffe_gpu_gemv<float,float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<double,double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<half, float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const half* A, const half* x,
    const float beta, half* y) {
  // cudaDeviceSynchronize();
  // printf("%s : (%d,%d) %c\n", __FUNCTION__, M, N, (TransA == CblasNoTrans) ? 'N' : 'T');
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  int lda = (TransA == CblasNoTrans) ? N : M;
  int ldb = (TransA == CblasNoTrans) ? 1 : N;
  CUBLAS_CHECK(cublasSgemmEx(Caffe::cublas_handle(), cuTransA, cuTransA,
        M, 1, N, &alpha, A, CUBLAS_DATA_HALF, lda, x, CUBLAS_DATA_HALF, ldb, &beta, y, CUBLAS_DATA_HALF, lda));

}

template <>
void caffe_gpu_axpy<float,float>(const int N, const float alpha, const float* X,
    float* Y) {
  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_axpy<double,double>(const int N, const double alpha, const double* X,
    double* Y) {
  CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <typename T_STORE, typename T_MATH>
__global__
void axpy_kernel(const int N, const T_MATH alpha, const T_STORE *x, T_STORE *y)
{
  for (int idx = threadIdx.x + blockDim.x*blockIdx.x; idx < N; idx += blockDim.x*gridDim.x) {
    y[idx] = Get<T_STORE>( alpha * Get<T_MATH>(x[idx]) + Get<T_MATH>(y[idx]) );
  }
}

template <>
void caffe_gpu_axpy<half,float>(const int N, const float alpha, const half* x,
    half *y)
{
  axpy_kernel<half,float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, alpha, x, y);
  CUDA_POST_KERNEL_CHECK;
}

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

template <>
void caffe_gpu_scal<float,float>(const int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<double,double>(const int N, const double alpha, double *X) {
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <typename T_STORE, typename T_MATH>
__global__
void scal_kernel(const int N, const T_MATH alpha, T_STORE *X)
{
  for (int idx = threadIdx.x + blockDim.x*blockIdx.x; idx < N; idx += blockDim.x*gridDim.x) {
    X[idx] = Get<T_STORE>( alpha * Get<T_MATH>(X[idx]));
  }
}

template <>
void caffe_gpu_scal<half,float>(const int N, const float alpha, half *X) {
  scal_kernel<half,float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, alpha, X);
  CUDA_POST_KERNEL_CHECK;
}

template <>
void caffe_gpu_axpby<float,float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float,float>(N, beta, Y);
  caffe_gpu_axpy<float,float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double,double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_gpu_scal<double,double>(N, beta, Y);
  caffe_gpu_axpy<double,double>(N, alpha, X, Y);
}

template <typename T_STORE, typename T_MATH>
__global__
void axpby_kernel(const int N, const T_MATH alpha, const T_STORE* X,
    const T_MATH beta, T_STORE* Y)
{
  CUDA_KERNEL_LOOP(idx, N) {
    Y[idx] = Get<T_STORE>( alpha * Get<T_MATH>(X[idx]) + beta * Get<T_MATH>(Y[idx]) );
  }
}

template <>
void caffe_gpu_axpby<half,float>(const int N, const float alpha, const half* X,
    const float beta, half* Y)
{
  axpby_kernel<half,float><<<CAFFE_GET_BLOCKS(N),CAFFE_CUDA_NUM_THREADS>>>(N,alpha,X,beta,Y);
  CUDA_POST_KERNEL_CHECK;
}

template <>
void caffe_gpu_dot<float,float>(const int n, const float* x, const float* y,
    float* out) {
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<double,double>(const int n, const double* x, const double* y,
    double * out) {
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

struct half_dot_reduce {
  __host__ __device__
  float operator()(const float& x, const float& y)
  {
    return x + y;
  }
};

struct half_dot_mult {
  __host__ __device__
  float operator()(half& x, half& y)
  {
    return Get<float>(x) * Get<float>(y);
  }
};

template <typename Dtype, typename Mtype>
__global__
void gpu_dot_kernel(const int N, const Dtype *x, const Dtype *y, Mtype *out)
{
  __shared__ Mtype cache[256];

  const int tidx = threadIdx.x;
  cache[tidx] = Mtype(0);
  for (int i=tidx; i<N; i+=blockDim.x) {
    cache[tidx] += Get<Mtype>(x[i]) * Get<Mtype>(y[i]);
  }
  __syncthreads();
  for (int s=128; s > 0; s >>= 1) {
    if (tidx < s) cache[tidx] += cache[tidx+s];
    __syncthreads();
  }

  if (tidx == 0) *out = cache[tidx];
}

template <>
void caffe_gpu_dot<half, float>(const int n, const half* x, const half* y,
    float *out)
{
  // float ret = thrust::inner_product(x, x+n, y, init, half_dot_reduce(), half_dot_mult());
  // *out = ret;

  float *res;
  cudaMalloc(&res, sizeof(float));
  gpu_dot_kernel<half,float><<<1,256>>>(n, x, y, res);
  CUDA_POST_KERNEL_CHECK;
  cudaMemcpy(out,res,sizeof(float), cudaMemcpyDeviceToHost);
  CUDA_POST_KERNEL_CHECK;
}

template <>
void caffe_gpu_asum<float,float>(const int n, const float* x, float* y) {
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_asum<double,double>(const int n, const double* x, double* y) {
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x, 1, y));
}

struct half_asum_reduce
{
  __host__ __device__
  float operator()(const float& a, const half& b)
  {
    return a + fabs(Get<float>(b));
  }
};

template <typename Dtype, typename Mtype>
__global__
void gpu_asum_kernel(const int N, const Dtype *x, Mtype *out)
{
  __shared__ Mtype cache[256];

  const int tidx = threadIdx.x;
  cache[tidx] = Mtype(0);
  for (int i=tidx; i<N; i+=blockDim.x) {
    cache[tidx] += fabs(Get<Mtype>(x[i]));
  }
  __syncthreads();
  for (int s=128; s > 0; s >>= 1) {
    if (tidx < s) cache[tidx] += cache[tidx+s];
    __syncthreads();
  }

  if (tidx == 0) *out = cache[tidx];
}

template <>
void caffe_gpu_asum<half,float>(const int n, const half* x, float* y)
{
  // float init = 0.0f;
  // float result = thrust::reduce(x, x+n, init, half_asum_reduce());
  // *y = result;
  float *res;
  cudaMalloc(&res, sizeof(float));
  gpu_asum_kernel<half,float><<<1,256>>>(n,x,res);
  CUDA_POST_KERNEL_CHECK;
  cudaMemcpy(y,res,sizeof(float),cudaMemcpyDeviceToHost);
  CUDA_POST_KERNEL_CHECK;
}

template <>
void caffe_gpu_scale<float,float>(const int n, const float alpha, const float *x,
                            float* y) {
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <>
void caffe_gpu_scale<double,double>(const int n, const double alpha, const double *x,
                             double* y) {
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <typename T_STORE, typename T_MATH>
__global__
void scale_kernel(const int n, const T_MATH alpha, const T_STORE* x, T_STORE* y)
{
  CUDA_KERNEL_LOOP(idx, n) {
    y[idx] = Get<T_STORE>( alpha * Get<T_MATH>(x[idx]) );
  }
}

template <>
void caffe_gpu_scale<half,float>(const int n, const float alpha, const half *x,
    half *y)
{
  scale_kernel<half,float><<<CAFFE_GET_BLOCKS(n),CAFFE_CUDA_NUM_THREADS>>>(n,alpha,x,y);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype, typename Mtype>
__global__ void set_kernel(const int n, const Mtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = Get<Dtype>(alpha);
  }
}

template <typename Dtype, typename Mtype>
void caffe_gpu_set(const int N, const Mtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype,Mtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
  CUDA_POST_KERNEL_CHECK;
}

template void caffe_gpu_set<int,int>(const int N, const int alpha, int* Y);
template void caffe_gpu_set<float,float>(const int N, const float alpha, float* Y);
template void caffe_gpu_set<double,double>(const int N, const double alpha, double* Y);
template void caffe_gpu_set<half,float>(const int N, const float alpha, half* Y);

template <typename Dtype, typename Mtype>
__global__ void add_scalar_kernel(const int n, const Mtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = Get<Dtype>(alpha + Get<Mtype>(y[index]));
  }
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float,float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
  CUDA_POST_KERNEL_CHECK;
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<double,double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
  CUDA_POST_KERNEL_CHECK;
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, half* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<half,float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype, typename Mtype>
__global__ void add_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = Get<Dtype>( Get<Mtype>(a[index]) + Get<Mtype>(b[index]) );
  }
}

template <>
void caffe_gpu_add<float,float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<float,float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
  CUDA_POST_KERNEL_CHECK;
}

template <>
void caffe_gpu_add<double,double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<double,double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
  CUDA_POST_KERNEL_CHECK;
}

template <>
void caffe_gpu_add<half,float>(const int N, const half* a, const half* b,
    half* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<half, float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype, typename Mtype>
__global__ void sub_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = Get<Dtype>( Get<Mtype>(a[index]) - Get<Mtype>(b[index]) );
  }
}

template <>
void caffe_gpu_sub<float,float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float,float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
  CUDA_POST_KERNEL_CHECK;
}

template <>
void caffe_gpu_sub<double,double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<double,double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
  CUDA_POST_KERNEL_CHECK;
}

template <>
void caffe_gpu_sub<half,float>(const int N, const half* a, const half* b,
    half* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<half,float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype, typename Mtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = Get<Dtype>( Get<Mtype>(a[index]) * Get<Mtype>(b[index]) );
  }
}

template <>
void caffe_gpu_mul<float, float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float,float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
  CUDA_POST_KERNEL_CHECK;
}

template <>
void caffe_gpu_mul<double,double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double,double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
  CUDA_POST_KERNEL_CHECK;
}

template <>
void caffe_gpu_mul<half,float>(const int N, const half* a,
    const half* b, half* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<half,float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype, typename Mtype>
__global__ void div_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = Get<Dtype>( Get<Mtype>(a[index]) / Get<Mtype>(b[index]) );
  }
}

template <>
void caffe_gpu_div<float,float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<float,float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
  CUDA_POST_KERNEL_CHECK;
}

template <>
void caffe_gpu_div<double,double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<double,double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
  CUDA_POST_KERNEL_CHECK;
}

template <>
void caffe_gpu_div<half,float>(const int N, const half* a,
    const half* b, half* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<half,float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype, typename Mtype>
__global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = Get<Dtype>( abs(Get<Mtype>(a[index])) );
  }
}

template <>
void caffe_gpu_abs<float,float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<float,float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
  CUDA_POST_KERNEL_CHECK;
}

template <>
void caffe_gpu_abs<double,double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<double,double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
  CUDA_POST_KERNEL_CHECK;
}

template <>
void caffe_gpu_abs<half,float>(const int N, const half* a, half* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<half,float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype, typename Mtype>
__global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = Get<Dtype>( exp(Get<Mtype>(a[index])) );
  }
}

template <>
void caffe_gpu_exp<float,float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<float,float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
  CUDA_POST_KERNEL_CHECK;
}

template <>
void caffe_gpu_exp<double,double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<double,double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
  CUDA_POST_KERNEL_CHECK;
}

template <>
void caffe_gpu_exp<half,float>(const int N, const half* a, half* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<half,float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype, typename Mtype>
__global__ void powx_kernel(const int n, const Dtype* a,
    const Mtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = Get<Dtype>( pow(Get<Mtype>(a[index]), alpha) );
  }
}

template <>
void caffe_gpu_powx<float,float>(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<float,float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
  CUDA_POST_KERNEL_CHECK;
}

template <>
void caffe_gpu_powx<double,double>(const int N, const double* a,
    const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<double,double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
  CUDA_POST_KERNEL_CHECK;
}

template <>
void caffe_gpu_powx<half,float>(const int N, const half* a,
    const float alpha, half* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<half,float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
  CUDA_POST_KERNEL_CHECK;
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = Get<Dtype>( (Mtype(0) < Get<Mtype>(x[index]))
                                      - (Get<Mtype>(x[index]) < Mtype(0))) );
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = Get<Dtype>( signbit(Get<Mtype>(x[index]))) );

__global__ void popc_kernel(const int n, const float* a,
    const float* b, uint8_t* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = __popc(static_cast<uint32_t>(a[index]) ^
                      static_cast<uint32_t>(b[index]));
  }
}

__global__ void popcll_kernel(const int n, const double* a,
    const double* b, uint8_t* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = __popcll(static_cast<uint64_t>(a[index]) ^
                      static_cast<uint64_t>(b[index]));
  }
}

__global__ void popch_kernel(const int n, const half* a,
    const half* b, uint8_t* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = __popc(static_cast<uint32_t>(Get<float>(a[index])) ^
                      static_cast<uint32_t>(Get<float>(b[index])));
  }
}

template <>
uint32_t caffe_gpu_hamming_distance<float,float>(const int n, const float* x,
                                  const float* y) {
  // TODO: Fix caffe_gpu_hamming_distance (see failing unit test
  // TestHammingDistanceGPU in test_math_functions.cpp).
  NOT_IMPLEMENTED;
  thrust::device_vector<uint8_t> popcounts(n);
  // NOLINT_NEXT_LINE(whitespace/operators)
  popc_kernel<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
      n, x, y, thrust::raw_pointer_cast(popcounts.data()));
  return thrust::reduce(popcounts.begin(), popcounts.end(),
                        (uint32_t) 0, thrust::plus<uint32_t>());
}

template <>
uint32_t caffe_gpu_hamming_distance<double,double>(const int n, const double* x,
                                   const double* y) {
  // TODO: Fix caffe_gpu_hamming_distance (see failing unit test
  // TestHammingDistanceGPU in test_math_functions.cpp).
  NOT_IMPLEMENTED;
  thrust::device_vector<uint8_t> popcounts(n);
  // NOLINT_NEXT_LINE(whitespace/operators)
  popcll_kernel<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
      n, x, y, thrust::raw_pointer_cast(popcounts.data()));
  return thrust::reduce(popcounts.begin(), popcounts.end(),
                        /* NOLINT_NEXT_LINE(build/include_what_you_use) */
                        (uint32_t) 0, thrust::plus<uint32_t>());
}

template <>
uint32_t caffe_gpu_hamming_distance<half,float>(const int n, const half* x,
                                   const half* y) {
  // TODO: Fix caffe_gpu_hamming_distance (see failing unit test
  // TestHammingDistanceGPU in test_math_functions.cpp).
  NOT_IMPLEMENTED;
  thrust::device_vector<uint8_t> popcounts(n);
  // NOLINT_NEXT_LINE(whitespace/operators)
  popch_kernel<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
      n, x, y, thrust::raw_pointer_cast(popcounts.data()));
  return thrust::reduce(popcounts.begin(), popcounts.end(),
                        /* NOLINT_NEXT_LINE(build/include_what_you_use) */
                        (uint32_t) 0, thrust::plus<uint32_t>());
}

template <typename T_IN, typename T_OUT>
__global__
void convert_kernel(const int n, const T_IN* in, T_OUT* out)
{
  for (int idx=threadIdx.x+blockIdx.x*blockDim.x; idx<n; idx+=blockDim.x*gridDim.x) {
    out[idx] = Get<T_OUT>(in[idx]);
  }
}

template <typename T_IN, typename T_OUT>
void caffe_gpu_convert(const int n, const T_IN* in, T_OUT* out)
{
  convert_kernel<T_IN,T_OUT><<<n / 512 + 1, 512>>>(n, in, out);
}

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
}

template <>
void caffe_gpu_rng_uniform<float,float>(const int n, const float a, const float b,
                                  float* r) {
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    caffe_gpu_scal<float,float>(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    caffe_gpu_add_scalar<float,float>(n, a, r);
  }
}

template <>
void caffe_gpu_rng_uniform<double,double>(const int n, const double a, const double b,
                                   double* r) {
  CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    caffe_gpu_scal<double,double>(n, range, r);
  }
  if (a != static_cast<double>(0)) {
    caffe_gpu_add_scalar<double,double>(n, a, r);
  }
}

template <>
void caffe_gpu_rng_uniform<half,float>(const int n, const float a, const float b,
                                   half* r) {
  thrust::device_vector<float> rf(n);
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), thrust::raw_pointer_cast(rf.data()), n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    caffe_gpu_scal<float,float>(n, range, thrust::raw_pointer_cast(rf.data()));
  }
  if (a != static_cast<float>(0)) {
    caffe_gpu_add_scalar<float,float>(n, a, thrust::raw_pointer_cast(rf.data()));
  }
  caffe_gpu_convert<float,half>(n, thrust::raw_pointer_cast(rf.data()), r);
}

template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma,
                            float* r) {
  CURAND_CHECK(
      curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma,
                            double* r) {
  CURAND_CHECK(
      curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
}

}  // namespace caffe
