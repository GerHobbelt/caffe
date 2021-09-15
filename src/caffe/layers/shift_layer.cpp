#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/shift_layer.hpp"
#include "caffe/net.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void ShiftLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  const ScaleParameter& param = this->layer_param_.scale_param();
  if (bottom.size() == 1 && this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else if (bottom.size() == 1) {
    // scale is a learned parameter; initialize it
    axis_ = bottom[0]->CanonicalAxisIndex(param.axis());
    const int num_axes = param.num_axes();
    CHECK_GE(num_axes, -1) << "num_axes must be non-negative, "
                           << "or -1 to extend to the end of bottom[0]";
    if (num_axes >= 0) {
      CHECK_GE(bottom[0]->num_axes(), axis_ + num_axes)
          << "scale blob's shape extends past bottom[0]'s shape when applied "
          << "starting with bottom[0] axis = " << axis_;
    }
    this->blobs_.resize(3);
    const vector<int>::const_iterator& shape_start =
        bottom[0]->shape().begin() + axis_;
    const vector<int>::const_iterator& shape_end =
        (num_axes == -1) ? bottom[0]->shape().end() : (shape_start + num_axes);
    vector<int> scale_shape(shape_start, shape_end);
    this->blobs_[0] = Blob::create<Ftype>(scale_shape);
    this->blobs_[1] = Blob::create<Ftype>(scale_shape);
    this->blobs_[2] = Blob::create<Ftype>(1);
    FillerParameter filler_param(param.filler());
    if (!param.has_filler()) {
      // Default to unit (1) filler for identity operation.
      filler_param.set_type("constant");
      filler_param.set_value(1);
    }
    shared_ptr<Filler<Ftype> > filler(GetFiller<Ftype>(filler_param));
    filler->Fill(this->blobs_[0].get());
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Ftype, typename Btype>
void ShiftLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  const ScaleParameter& param = this->layer_param_.scale_param();
  Blob* scale = (bottom.size() > 1) ? bottom[1] : this->blobs_[0].get();
  // Always set axis_ == 0 in special case where scale is a scalar
  // (num_axes == 0). Mathematically equivalent for any choice of axis_, so the
  // actual setting can be safely ignored; and computation is most efficient
  // with axis_ == 0 and (therefore) outer_dim_ == 1. (Setting axis_ to
  // bottom[0]->num_axes() - 1, giving inner_dim_ == 1, would be equally
  // performant.)
  axis_ = (scale->num_axes() == 0) ?
      0 : bottom[0]->CanonicalAxisIndex(param.axis());
  CHECK_GE(bottom[0]->num_axes(), axis_ + scale->num_axes())
      << "scale blob's shape extends past bottom[0]'s shape when applied "
      << "starting with bottom[0] axis = " << axis_;
  for (int i = 0; i < scale->num_axes(); ++i) {
    CHECK_EQ(bottom[0]->shape(axis_ + i), scale->shape(i))
        << "dimension mismatch between bottom[0]->shape(" << axis_ + i
        << ") and scale->shape(" << i << ")";
  }
  outer_dim_ = bottom[0]->count(0, axis_);
  scale_dim_ = scale->count();
  inner_dim_ = bottom[0]->count(axis_ + scale->num_axes());
  if (bottom[0] == top[0]) {  // in-place computation
    temp_->ReshapeLike(bottom[0]);
  } else {
    top[0]->ReshapeLike(bottom[0]);
  }
  sum_result_.Reshape(vector<int>(1, outer_dim_ * scale_dim_));
  const int sum_mult_size = std::max(outer_dim_, inner_dim_);
  sum_multiplier_.Reshape(vector<int>(1, sum_mult_size));
  if (sum_multiplier_.cpu_data()[sum_mult_size - 1] != Btype(1)) {
    caffe_set<Btype>(sum_mult_size, Btype(1), sum_multiplier_.mutable_cpu_data());
  }
}

template <typename Ftype, typename Btype>
void ShiftLayer<Ftype, Btype>::Forward_cpu(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->cpu_data<Ftype>();
  if (bottom[0] == top[0]) {
    // In-place computation; need to store bottom data before overwriting it.
    // Note that this is only necessary for Backward; we could skip this if not
    // doing Backward, but Caffe currently provides no way of knowing whether
    // we'll need to do Backward at the time of the Forward call.
    caffe_copy<Ftype>(bottom[0]->count(), bottom[0]->cpu_data<Ftype>(),
               temp_->template mutable_cpu_data<Ftype>());
  }
  const Ftype* scale_data =
      this->blobs_[0].get()->template cpu_data<Ftype>();
  const Ftype* exponent_data =
      this->blobs_[1].get()->template cpu_data<Ftype>();
  const Ftype* zero_point =
      this->blobs_[2].get()->template cpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_cpu_data<Ftype>();
  for (int n = 0; n < outer_dim_; ++n) {
    for (int d = 0; d < scale_dim_; ++d) {
      const Ftype factor = scale_data[d];
      const Ftype exponent = exponent_data[d];
      const Ftype zp = zero_point[0];
      caffe_cpu_quant(inner_dim_, factor, exponent, zp, bottom_data, top_data);
      bottom_data += inner_dim_;
      top_data += inner_dim_;
    }
  }
}

template <typename Ftype, typename Btype>
void ShiftLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
    NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS_FB(ShiftLayer);
REGISTER_LAYER_CLASS(Shift);

}  // namespace caffe
