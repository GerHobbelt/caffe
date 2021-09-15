#include <cfloat>
#include <vector>

#include "caffe/layers/shift_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void ShiftLayer<Ftype, Btype>::Forward_gpu(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
    NOT_IMPLEMENTED;
}

template <typename Ftype, typename Btype>
void ShiftLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
    NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(ShiftLayer);

}  // namespace caffe
