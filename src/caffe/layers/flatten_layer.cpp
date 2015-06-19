#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype, typename Mtype>
void FlattenLayer<Dtype,Mtype>::Reshape(const vector<Blob<Dtype,Mtype>*>& bottom,
      const vector<Blob<Dtype,Mtype>*>& top) {
  vector<int> top_shape(2);
  top_shape[0] = bottom[0]->num();
  top_shape[1] = bottom[0]->count() / bottom[0]->num();
  top[0]->Reshape(top_shape);
  CHECK_EQ(top[0]->count(), bottom[0]->count());
}

template <typename Dtype, typename Mtype>
void FlattenLayer<Dtype,Mtype>::Forward_cpu(const vector<Blob<Dtype,Mtype>*>& bottom,
      const vector<Blob<Dtype,Mtype>*>& top) {
  top[0]->ShareData(*bottom[0]);
}

template <typename Dtype, typename Mtype>
void FlattenLayer<Dtype,Mtype>::Backward_cpu(const vector<Blob<Dtype,Mtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype,Mtype>*>& bottom) {
  bottom[0]->ShareDiff(*top[0]);
}

INSTANTIATE_CLASS(FlattenLayer);
REGISTER_LAYER_CLASS(Flatten);

}  // namespace caffe
