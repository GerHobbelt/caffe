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
    void BinaryImageDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
    {
        const string& source = this->layer_param_.hdf5_data_param().source();
        LOG(INFO) << "Loading binary file: " << source;

        std::ifstream stream(source, std::ios::binary);
        if (stream.bad() || stream.fail() || !stream.is_open())
        {
            LOG(FATAL) << "Failed opening Binary file: " << source;
        }

        // Get dimensions from file
        stream.read(reinterpret_cast<char*>(&width_in_), sizeof(int));
        stream.read(reinterpret_cast<char*>(&height_in_), sizeof(int));
        stream.read(reinterpret_cast<char*>(&width_out_), sizeof(int));
        stream.read(reinterpret_cast<char*>(&height_out_), sizeof(int));

        int length;
        stream.read(reinterpret_cast<char*>(&length), sizeof(int));

        // Read until the end
        int count = 0;
        while (!stream.eof() && !stream.bad() && !stream.fail() && count < length)
        {
            Dtype* data = new Dtype[width_in_ * height_in_];
            stream.read(reinterpret_cast<char*>(data), width_in_ * height_in_ * sizeof(Dtype));
            Dtype* label = new Dtype[width_out_ * height_out_];
            stream.read(reinterpret_cast<char*>(label), width_out_ * height_out_ * sizeof(Dtype));

            data_.emplace_back(std::make_shared<BinaryData<Dtype>>(data, label));
            count++;
        }

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
        data_permutation_.resize(length);

        for (int i = 0; i < length; ++i)
            data_permutation_[i] = i;

        std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
    }

    template <typename Dtype>
    void BinaryImageDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
    {
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
                              &top[0]->mutable_cpu_data()[i * size]);
            size = width_out_ * height_out_;
            caffe_copy<Dtype>(size, data_[data_permutation_[current_row_]]->GetLabel(),
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