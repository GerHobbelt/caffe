#ifndef CAFFE_BINARY_DATA_LAYER_HPP_
#define CAFFE_BINARY_DATA_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_data_layer.hpp"

namespace caffe
{
    template <typename Dtype>
    struct BinaryData
    {
    public:
        BinaryData(Dtype *input, Dtype *label) : input_(input), label_(label) {}

        ~BinaryData()
        {
            delete[] input_;
            delete[] label_;
        }

        const Dtype *GetInput() const { return input_; }
        const Dtype *GetLabel() const { return label_; }

    private:
        Dtype *input_;
        Dtype *label_;
    };

    /**
     * @brief Provides data to the Net from HDF5 files.
     *
     * TODO(dox): thorough documentation for Forward and proto params.
     */
    template <typename Dtype>
    class BinaryImageDataLayer : public Layer<Dtype>
    {
    public:
        explicit BinaryImageDataLayer(const LayerParameter &param) : Layer<Dtype>(param), current_row_(0) {}
        virtual ~BinaryImageDataLayer();
        virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top);
        // Data layers have no bottoms, so reshaping is trivial.
        virtual void Reshape(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {}

        virtual inline const char *type() const { return "BinaryImageData"; }
        virtual inline int ExactNumBottomBlobs() const { return 0; }
        virtual inline int MinTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top);
        virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top);
        virtual void Backward_cpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                                  const vector<Blob<Dtype> *> &bottom)
        {
        }
        virtual void Backward_gpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                                  const vector<Blob<Dtype> *> &bottom)
        {
        }

        size_t current_row_;
        // std::vector<BinaryData<Dtype>*> data_;
        std::vector<std::vector<shared_ptr<Blob<Dtype>>>> data_blobs_;
        int width_in_;
        int width_out_;
        int height_in_;
        int height_out_;
        int length_;
        int count_per_blob_;
        std::vector<unsigned int> data_permutation_;
    };

} // namespace caffe

#endif // CAFFE_BINARY_DATA_LAYER_HPP_
