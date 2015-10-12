#ifdef USE_CUDNN
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

__global__ void sync_conv_groups() { }

template <typename Dtype, typename Mtype>
void CuDNNConvolutionLayer<Dtype,Mtype>::Forward_gpu(
    const vector<Blob<Dtype,Mtype>*>& bottom, const vector<Blob<Dtype,Mtype>*>& top) {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();




  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();

    // Forward through cuDNN in parallel over groups.
    for (int g = 0; g < this->group_; g++) {

      if (CuMem::usingPool()) {
        MemoryHandler::mallocGPU(&workspace[0], workspace_fwd_sizes_[i]);

      // Filters.
      // CUDNN_CHECK(cudnnConvolutionForward(handle_[g],
      CUDNN_CHECK(cudnnConvolutionForward(Caffe::cudnn_handle(),
            cudnn::dataType<Dtype>::one,
            bottom_descs_[i], bottom_data + bottom_offset_ * g,
            filter_desc_, weight + this->weight_offset_ * g,
            conv_descs_[i],
            fwd_algo_[i], workspace[0], workspace_fwd_sizes_[i],
            cudnn::dataType<Dtype>::zero,
            top_descs_[i], top_data + top_offset_ * g));

        MemoryHandler::freeGPU(workspace[0]);
        workspace[0] = NULL;
      } else {

    	    const int kernel_h = kernel_shape_data[0];
    	    const int kernel_w = kernel_shape_data[1];
    	    const size_t workspace_limit_bytes =
    	        kernel_h * kernel_w * this->channels_ * sizeof(int) + 1;

		 cudnnConvolutionFwdAlgo_t algo;

		 // pick the convolution algorithm
		 // TODO(shelhamer) this should be done during reshape
		 // TODO(shelhamer) the choice of automatic or manual algorithm picking
		 // should be exposed in proto
		 CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(Caffe::cudnn_handle(),
		   bottom_descs_[i],
		   filter_desc_,
		   conv_descs_[i],
		   top_descs_[i],
		   CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
		   workspace_limit_bytes,  // memoryLimitInBytes,
		   &algo));

		 // get minimum size of the workspace needed for the desired algorithm
		 size_t workspaceSizeInBytes_temp = 0;

		 CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(Caffe::cudnn_handle(),
		   bottom_descs_[i],
		   filter_desc_,
		   conv_descs_[i],
		   top_descs_[i],
		   algo,
		   &workspaceSizeInBytes_temp));

		 if (workspaceSizeInBytes_temp > workspaceSizeInBytes) {
		   workspaceSizeInBytes = workspaceSizeInBytes_temp;
		   // free the existing workspace and allocate a new (larger) one
		   cudaFree(this->workspace[0]);
		  cudaError_t err = cudaMalloc(&(this->workspace[0]), workspaceSizeInBytes);
		   if (err != cudaSuccess) {
			 // force zero memory path
			 algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
			 workspace[g] = NULL;
			 workspaceSizeInBytes = 0;
		   }
		 }




		CUDNN_CHECK(cudnnConvolutionForward(Caffe::cudnn_handle(),
			  cudnn::dataType<Dtype>::one,
			  bottom_descs_[i], bottom_data + bottom_offset_ * g,
			  filter_desc_, weight + this->weight_offset_ * g,
			  conv_descs_[i],
			  fwd_algo_[i], workspace[0], workspace_fwd_sizes_[i],
			  cudnn::dataType<Dtype>::zero,
			  top_descs_[i], top_data + top_offset_ * g));

      }

      // Bias.
      if (this->bias_term_) {
        const Dtype* bias_data = this->blobs_[1]->gpu_data();
        CUDNN_CHECK(cudnnAddTensor_v3
              (Caffe::cudnn_handle(),
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_data + bias_offset_ * g,
              cudnn::dataType<Dtype>::one,
              top_descs_[i], top_data + top_offset_ * g));
      }

    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    if (CuMem::usingPool()) {
      CUDA_CHECK(cudaStreamSynchronize(cudaStreamLegacy));
    } else {
      sync_conv_groups<<<1, 1>>>();
    }
  }
}


template <typename Dtype, typename Mtype>
void CuDNNConvolutionLayer<Dtype,Mtype>::Backward_gpu(const vector<Blob<Dtype,Mtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype,Mtype>*>& bottom) {
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
    caffe_gpu_set<Dtype,Mtype>(this->blobs_[0]->count(), Mtype(0), weight_diff);
  }
  Dtype* bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
    caffe_gpu_set<Dtype,Mtype>(this->blobs_[1]->count(), Mtype(0), bias_diff);
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Backward through cuDNN in parallel over groups and gradients.
    for (int g = 0; g < this->group_; g++) {
      // Gradient w.r.t. bias.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        CUDNN_CHECK(cudnnConvolutionBackwardBias(Caffe::cudnn_handle(),
              cudnn::dataType<Dtype>::one,
              top_descs_[i],  top_diff + top_offset_ * g,
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_diff + bias_offset_ * g));
      }

      // Gradient w.r.t. weights.
      if (this->param_propagate_down_[0]) {
#ifdef USE_CNMEM
        MemoryHandler::mallocGPU(&workspace[0], workspace_bwd_filter_sizes_[i]);
#endif
        const Dtype* bottom_data = bottom[i]->gpu_data();
        CUDNN_CHECK(cudnnConvolutionBackwardFilter_v3(
              Caffe::cudnn_handle(),
              cudnn::dataType<Dtype>::one,
              bottom_descs_[i], bottom_data + bottom_offset_ * g,
              top_descs_[i],    top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_filter_algo_[i], workspace[0], workspace_bwd_filter_sizes_[i],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight_diff + this->weight_offset_ * g));
#ifdef USE_CNMEM
        MemoryHandler::freeGPU(workspace[0]);
        workspace[0] = NULL;
#endif
      }

      // Gradient w.r.t. bottom data.
      if (propagate_down[i]) {
        if (weight == NULL) {
          weight = this->blobs_[0]->gpu_data();
        }
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
#ifdef USE_CNMEM
        MemoryHandler::mallocGPU(&workspace[0], workspace_bwd_data_sizes_[i]);
#endif
        CUDNN_CHECK(cudnnConvolutionBackwardData_v3(
              Caffe::cudnn_handle(),
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight + this->weight_offset_ * g,
              top_descs_[i], top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_data_algo_[i], workspace[0], workspace_bwd_data_sizes_[i],
              cudnn::dataType<Dtype>::zero,
              bottom_descs_[i], bottom_diff + bottom_offset_ * g));
#ifdef USE_CNMEM
        MemoryHandler::freeGPU(workspace[0]);
        workspace[0] = NULL;
#endif
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamLegacy));
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNConvolutionLayer);

}  // namespace caffe
#endif
