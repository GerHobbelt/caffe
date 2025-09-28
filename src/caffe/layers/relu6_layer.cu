#include <algorithm>
#include <vector>
#include <device_launch_parameters.h>

#include "caffe/layers/relu6_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ReLU6Forward(const int n, const Dtype* in, Dtype* out, float negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = !signbit(in[index]) ? (in[index] > Dtype(6) ? Dtype(6) : in[index]) : Dtype(in[index] * negative_slope);
  }
}

template <typename Dtype>
__global__ void ReLU6Forward0(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = !signbit(in[index]) ? (in[index] > Dtype(6) ? Dtype(6) : in[index]) : Dtype(0);
  }
}

template <typename Ftype, typename Btype>
void ReLU6Layer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->gpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_gpu_data<Ftype>();

  const int count = bottom[0]->count();
  float negative_slope = this->layer_param_.relu_param().negative_slope();
  if (negative_slope != 0.F) {
    // NOLINT_NEXT_LINE(whitespace/operators)
    ReLU6Forward <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream()>>>(
        count, bottom_data, top_data, negative_slope);
  } else {
    // NOLINT_NEXT_LINE(whitespace/operators)
    ReLU6Forward0 <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream()>>>(
        count, bottom_data, top_data);
  }
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));
}

template <typename Ftype, typename Btype>
void ReLU6Layer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob*>& bottom) {
    NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(ReLU6Layer);

}  // namespace caffe
