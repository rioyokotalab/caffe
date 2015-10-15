#include <boost/thread.hpp>
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype, typename Mtype>
void Layer<Dtype,Mtype>::InitMutex() {
  forward_mutex_.reset(new boost::mutex());
}

template <typename Dtype, typename Mtype>
void Layer<Dtype,Mtype>::Lock() {
  if (IsShared()) {
    forward_mutex_->lock();
  }
}

template <typename Dtype, typename Mtype>
void Layer<Dtype,Mtype>::Unlock() {
  if (IsShared()) {
    forward_mutex_->unlock();
  }
}

template
void Layer<half,half>::InitMutex();

template
void Layer<half,half>::Lock();

template
void Layer<half,half>::Unlock();

INSTANTIATE_CLASS(Layer);

}  // namespace caffe
