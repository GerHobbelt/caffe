#ifdef USE_HDF5
/*
TODO:
- only load parts of the file, in accordance with a prototxt param "max_mem"
*/

#include <stdint.h>
#include <vector>

#include "caffe/layers/binary_image_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void BinaryImageDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    const int batch_size = this->layer_param_.hdf5_data_param().batch_size();
    for (int i = 0; i < batch_size; ++i)
    {
        if (current_row_ == data_.size())
        {
            std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
            current_row_ = 0;
        }

        int size = width_in_ * height_in_;
        caffe_copy<Dtype>(size, data_[data_permutation_[current_row_]]->GetInput(), 
          &top[0]->mutable_gpu_data()[i * size]);
        size = width_out_ * height_out_;
        caffe_copy<Dtype>(size, data_[data_permutation_[current_row_]]->GetLabel(), 
          &top[1]->mutable_gpu_data()[i * size]);

        ++current_row_;
    }
  }

INSTANTIATE_LAYER_GPU_FUNCS(BinaryImageDataLayer);

}  // namespace caffe
#endif  // USE_HDF5
