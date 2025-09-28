#include <algorithm>
#include <vector>

#include "caffe/layers/relu6_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void ReLU6Layer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->cpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_cpu_data<Ftype>();
  const int count = bottom[0]->count();
  float negative_slope = this->layer_param_.relu_param().negative_slope();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::min(std::max(bottom_data[i], Ftype(0)), Ftype(6))
        + negative_slope * std::min(bottom_data[i], Ftype(0));
  }
}

template <typename Ftype, typename Btype>
void ReLU6Layer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob*>& bottom) {
    NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS_FB(ReLU6Layer);

}  // namespace caffe
