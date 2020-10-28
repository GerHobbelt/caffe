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
        if (current_row_ == length_)
        {
            std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
            current_row_ = 0;
        }

        int data_row = data_permutation_[current_row_];
        int blob_index = std::floor(1.0 * data_row / count_per_blob_);
        int index_in_blob = data_row - blob_index * count_per_blob_;

        int size = width_in_ * height_in_;
        caffe_copy(size, &data_blobs_[0][blob_index]->cpu_data()[index_in_blob * size], 
          &top[0]->mutable_gpu_data()[i * size]);
        size = width_out_ * height_out_;
        caffe_copy(size, &data_blobs_[1][blob_index]->cpu_data()[index_in_blob * size], 
          &top[1]->mutable_gpu_data()[i * size]);

        ++current_row_;
    }
  }

INSTANTIATE_LAYER_GPU_FUNCS(BinaryImageDataLayer);

}  // namespace caffe
#endif  // USE_HDF5
