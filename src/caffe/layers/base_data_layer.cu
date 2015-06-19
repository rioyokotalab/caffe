#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype, typename Mtype>
void BasePrefetchingDataLayer<Dtype,Mtype>::Forward_gpu(
    const vector<Blob<Dtype,Mtype>*>& bottom, const vector<Blob<Dtype,Mtype>*>& top) {
  Batch<Dtype,Mtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->Reshape(batch->data_.num(), batch->data_.channels(),
      batch->data_.height(), batch->data_.width());
  // Copy the data
  caffe_copy<Dtype,Mtype>(batch->data_.count(), batch->data_.gpu_data(),
      top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    caffe_copy<Dtype,Mtype>(batch->label_.count(), batch->label_.gpu_data(),
        top[1]->mutable_gpu_data());
  }

  prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);

}  // namespace caffe
