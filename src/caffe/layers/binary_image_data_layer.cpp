#ifdef USE_OPENCV

/*
TODO:
- load file in a separate thread ("prefetch")
- can be smarter about the memcpy call instead of doing it row-by-row
  :: use util functions caffe_copy, and Blob->offset()
  :: don't forget to update hdf5_daa_layer.cu accordingly
- add ability to shuffle filenames if flag is set
*/
#include <fstream> // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "stdint.h"

#include "caffe/layers/binary_image_data_layer.hpp"

namespace caffe
{
    template <typename Dtype>
    BinaryImageDataLayer<Dtype>::~BinaryImageDataLayer<Dtype>()
    {
    }

    template <typename Dtype>
    void BinaryImageDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top)
    {
        const string &source = this->layer_param_.hdf5_data_param().source();
        LOG(INFO) << "Loading binary file: " << source;

        std::ifstream stream(source, std::ios::binary);
        if (stream.bad() || stream.fail() || !stream.is_open())
        {
            LOG(FATAL) << "Failed opening Binary file: " << source;
        }

        // Get dimensions from file
        stream.read(reinterpret_cast<char *>(&width_in_), sizeof(int));
        stream.read(reinterpret_cast<char *>(&height_in_), sizeof(int));
        stream.read(reinterpret_cast<char *>(&width_out_), sizeof(int));
        stream.read(reinterpret_cast<char *>(&height_out_), sizeof(int));
        stream.read(reinterpret_cast<char *>(&length_), sizeof(int));

        // TODO: DEBUG
        // length_ = 1000;

        // Count how many blobs we need
        int max_size = std::max(width_in_ * height_in_, width_out_ * height_out_);
        count_per_blob_ = std::floor(1.0 * INT_MAX / max_size);
        int blobs_cnt = std::ceil(1.0 * length_ / count_per_blob_);

        // Setup data blobs
        data_blobs_.resize(2);
        for (int i = 0; i < 2; ++i)
        {
            data_blobs_[i].resize(blobs_cnt);
            for (int j = 0; j < blobs_cnt; ++j)
            {
                vector<int> blob_dims(4);
                blob_dims[0] = j == blobs_cnt - 1 ? length_ % count_per_blob_ : count_per_blob_;
                blob_dims[1] = 1;
                blob_dims[2] = i == 0 ? height_in_ : height_out_;
                blob_dims[3] = i == 0 ? width_in_ : width_out_;

                data_blobs_[i][j] = shared_ptr<Blob<Dtype>>(new Blob<Dtype>());
                data_blobs_[i][j]->Reshape(blob_dims);
            }
        }

        // Read until the end
        int count = 0;

        while (!stream.eof() && !stream.bad() && !stream.fail() && count < length_)
        {
            int blob_index = std::floor(1.0 * count / count_per_blob_);
            int index_in_blob = count - blob_index * count_per_blob_;

            stream.read(reinterpret_cast<char *>(&data_blobs_[0][blob_index]->mutable_cpu_data()[width_in_ * height_in_ * index_in_blob]),
                        width_in_ * height_in_ * sizeof(Dtype));
            stream.read(reinterpret_cast<char *>(&data_blobs_[1][blob_index]->mutable_cpu_data()[width_out_ * height_out_ * index_in_blob]),
                        width_out_ * height_out_ * sizeof(Dtype));
            count++;
        }

        if (count < length_)
        {
            LOG(FATAL) << "Failed reading Binary file, not enough data: " << source;
        }

        // // Check that data is ok (because we have issues !)
        // for (int i = 0; i < length; ++i)
        // {
        //     for (int j = 0; j < width_in_ * height_in_; ++j)
        //     {
        //         float val = data_[i]->GetInput()[j];
        //         if (val < -1.0f || val > 1.0f)
        //             std::cout << "Issue input ! " << i << ", " << j << ": " << val << std::endl;
        //     }
        //     for (int j = 0; j < width_out_ * height_out_; ++j)
        //     {
        //         float val = data_[i]->GetLabel()[j];
        //         if (val < -1.0f || val > 1.0f)
        //             std::cout << "Issue label ! " << i << ", " << j << ": " << val << std::endl;
        //     }
        // }

        // Reshape blobs.
        const int batch_size = this->layer_param_.hdf5_data_param().batch_size();

        vector<int> top_shape(4);
        top_shape[0] = batch_size;
        top_shape[1] = 1;
        top_shape[2] = height_in_;
        top_shape[3] = width_in_;

        top[0]->Reshape(top_shape);

        top_shape[0] = batch_size;
        top_shape[1] = 1;
        top_shape[2] = height_out_;
        top_shape[3] = width_out_;

        top[1]->Reshape(top_shape);

        // Setup data permutation
        data_permutation_.clear();
        data_permutation_.resize(length_);

        for (int i = 0; i < length_; ++i)
            data_permutation_[i] = i;

        std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
    }

    template <typename Dtype>
    void BinaryImageDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top)
    {
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
                       &top[0]->mutable_cpu_data()[i * size]);
            size = width_out_ * height_out_;
            caffe_copy(size, &data_blobs_[1][blob_index]->cpu_data()[index_in_blob * size],
                       &top[1]->mutable_cpu_data()[i * size]);

            ++current_row_;
        }
    }

#ifdef CPU_ONLY
    STUB_GPU_FORWARD(BinaryImageDataLayer, Forward);
#endif

    INSTANTIATE_CLASS(BinaryImageDataLayer);
    REGISTER_LAYER_CLASS(BinaryImageData);

} // namespace caffe

#endif