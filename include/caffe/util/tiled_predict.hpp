#ifndef CAFFE_UTIL_TILED_PREDICT_H_
#define CAFFE_UTIL_TILED_PREDICT_H_

#include "caffe/caffe.hpp"
#include "caffe/util/vector_helper.hpp"

namespace caffe {

  template<typename T>
  void rotate90(Blob<T> const &in, Blob<T> &out, int n) {

    // Fix n to be in { 0, 1, 2, 3 }
    // (This allows to give negative n for inverse rotation)
    n = n % 4;
    if (n < 0) n += 4;

    if (n == 0) {
      if (&out == &in) return;
      out.ReshapeLike(in);
      std::memcpy(out.mutable_cpu_data(), in.cpu_data(),
                  in.count() * sizeof(T));
      return;
    }

    std::vector<int> outShape(in.shape());
    if (n % 2 == 1) {
      outShape[in.shape().size() - 2] = in.shape(in.shape().size() - 1);
      outShape[in.shape().size() - 1] = in.shape(in.shape().size() - 2);
    }

    Blob<T> *outReal = &out;
    if (&out == &in) outReal = new Blob<T>(outShape);
    else out.Reshape(outShape);

    int nSlices = outShape[0];
    for (int d = 1; d < outShape.size() - 2; ++d) nSlices *= outShape[d];
    int ny = outShape[outShape.size() - 2];
    int nx = outShape[outShape.size() - 1];
    int sliceSize = ny * nx;

    int xStride, yStride, xOffset, yOffset, xScale, yScale;
    switch (n)
    {
    case 1:
      yStride = 1;
      yOffset = 0;
      yScale = 1;
      xStride = ny;
      xOffset = nx - 1;
      xScale = -1;
      break;
    case 2:
      yStride = nx;
      yOffset = ny - 1;
      yScale = -1;
      xStride = 1;
      xOffset = nx - 1;
      xScale = -1;
      break;
    case 3:
      yStride = 1;
      yOffset = ny - 1;
      yScale = -1;
      xStride = ny;
      xOffset = 0;
      xScale = 1;
      break;
    default:
      break;
    }

    T *outP = outReal->mutable_cpu_data();
    for (int sliceIdx = 0; sliceIdx < nSlices; ++sliceIdx)
    {
      T const *inSlice = in.cpu_data() + sliceIdx * sliceSize;
      for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x, ++outP) {
          *outP = inSlice[(xScale * x + xOffset) * xStride +
                          (yScale * y + yOffset) * yStride];
        }
      }
    }

    if (&in == &out)
    {
      out.ReshapeLike(*outReal);
      std::memcpy(out.mutable_cpu_data(), outReal->cpu_data(),
                  outReal->count() * sizeof(T));
      delete outReal;
    }
  }

  template<typename T>
  void flip(Blob<T> const &in, Blob<T> &out,
            std::vector<bool> const &flipConfig) {

    CHECK_EQ(flipConfig.size(), in.shape().size() - 2)
        << "The number of flip dimensions must match the number of spatial "
        << "dimensions of the input blob";

    bool requiresFlip = false;
    for (size_t d = 0; d < flipConfig.size(); ++d)
        requiresFlip |= flipConfig[d];

    if (!requiresFlip) {
      if (&out == &in) return;
      out.ReshapeLike(in);
      std::memcpy(out.mutable_cpu_data(), in.cpu_data(),
                  in.count() * sizeof(T));
      return;
    }

    Blob<T> *outReal = &out;
    if (&out == &in) outReal = new Blob<T>(in.shape());
    else out.Reshape(in.shape());

    int nDims = in.shape().size() - 2;
    std::vector<int> strides(nDims, 1);
    for (int d = nDims - 2; d >= 0; --d)
        strides[d] = strides[d + 1] * in.shape(d + 3);
    int blockSize = strides[0] * in.shape(2);

    T* outP = outReal->mutable_cpu_data();
    for (int blockIdx = 0; blockIdx < in.shape(0) * in.shape(1); ++blockIdx) {
      T const *inBlock  = in.cpu_data() + blockIdx * blockSize;
      for (int i = 0; i < blockSize; ++i, ++outP) {
        const T* inP = inBlock;
        int tmp = i;
        for (int d = nDims - 1; d >= 0; --d) {
          ptrdiff_t src =
              ((flipConfig[d]) ?
               (in.shape(d + 2) - (tmp % in.shape(d + 2)) - 1) :
               (tmp % in.shape(d + 2)));
          tmp /= in.shape(d + 2);
          inP += src * strides[d];
        }
        *outP = *inP;
      }
    }

    if (&in == &out)
    {
      out.ReshapeLike(*outReal);
      std::memcpy(out.mutable_cpu_data(), outReal->cpu_data(),
                  outReal->count() * sizeof(T));
      delete outReal;
    }
  }

  template<typename T>
  void copyBlock(
      Blob<T> const &in, Blob<T> &out, std::vector<int> shape,
      std::vector<int> inPos, std::vector<int> outPos,
      bool padMirror) {

    CHECK_EQ(in.shape().size(), out.shape().size())
        << "Input and output blobs must have same dimensionality";
    CHECK_EQ(inPos.size(), out.shape().size())
        << "Input position dimensionality must match blob dimensionality";
    CHECK_EQ(outPos.size(), out.shape().size())
        << "Output position dimensionality must match blob dimensionality";
    CHECK_EQ(shape.size(), out.shape().size())
        << "block shape dimensionality must match blob dimensionality";

    std::vector<int> inShape(in.shape());
    std::vector<int> outShape(out.shape());

    int nBlobDims = inShape.size();

    // Intersect block to crop with output blob
    int nElements = 1;
    for (int d = 0; d < nBlobDims; ++d) {
      if (outPos[d] < 0) {
        inPos[d] += -outPos[d];
        shape[d] -= -outPos[d];
        if (shape[d] <= 0) return;
        outPos[d] = 0;
      }
      if (outPos[d] + shape[d] > outShape[d]) {
        shape[d] = outShape[d] - outPos[d];
        if (shape[d] <= 0) return;
      }
      nElements *= shape[d];
    }

    T const *inPtr = in.cpu_data();
    T *outPtr = out.mutable_cpu_data();

    bool fullInput = true, fullOutput = true;
    for (int d = 0; d < nBlobDims && fullInput; ++d)
        fullInput &= inPos[d] == 0 && shape[d] == inShape[d];
    for (int d = 0; d < nBlobDims && fullOutput; ++d)
        fullOutput &= outPos[d] == 0 && shape[d] == outShape[d];
    if (fullInput && fullOutput) {
      std::memcpy(outPtr, inPtr, nElements * sizeof(T));
      return;
    }

    std::vector<int> stridesIn(nBlobDims, 1);
    std::vector<int> stridesOut(nBlobDims, 1);
    for (int d = nBlobDims - 2; d >= 0; --d) {
      stridesIn[d] = stridesIn[d + 1] * inShape[d + 1];
      stridesOut[d] = stridesOut[d + 1] * outShape[d + 1];
    }

    if (fullInput) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (int i = 0; i < nElements; ++i) {
        T *outP = outPtr;
        int tmp = i;
        for (int d = nBlobDims - 1; d >= 0; --d) {
          outP += (outPos[d] + (tmp % shape[d])) * stridesOut[d];
          tmp /= shape[d];
        }
        *outP = inPtr[i];
      }
    }
    else {

      if (padMirror) {

        // Precompute lookup-table for input positions
        std::vector< std::vector<int> > rdPos(nBlobDims);
        for (int d = 0; d < nBlobDims; ++d) {
          rdPos[d].resize(shape[d]);
          for (int i = 0; i < shape[d]; ++i) {
            int p = inPos[d] + i;
            if (p < 0 || p >= inShape[d]) {
              if (p < 0) p = -p;
              int n = p / (inShape[d] - 1);
              if (n % 2 == 0) p = p - n * (inShape[d] - 1);
              else p = (n + 1) * (inShape[d] - 1) - p;
            }
            rdPos[d][i] = p;
          }
        }

        if (fullOutput) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
          for (int i = 0; i < nElements; ++i)
          {
            T const *inP = inPtr;
            int tmp = i;
            for (int d = nBlobDims - 1; d >= 0; --d) {
              int x = tmp % shape[d];
              tmp /= shape[d];
              inP += rdPos[d][x] * stridesIn[d];
            }
            outPtr[i] = *inP;
          }
        }
        else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
          for (int i = 0; i < nElements; ++i)
          {
            T const *inP = inPtr;
            T *outP = outPtr;
            int tmp = i;
            for (int d = nBlobDims - 1; d >= 0; --d) {
              int x = tmp % shape[d];
              tmp /= shape[d];
              inP += rdPos[d][x] * stridesIn[d];
              outP += (outPos[d] + x) * stridesOut[d];
            }
            *outP = *inP;
          }
        } // else (fullOutput)
      } // if (padMirror)
      else {

        if (fullOutput) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
          for (int i = 0; i < nElements; ++i) {
            T const *inP = inPtr;
            int tmp = i;
            int d = nBlobDims - 1;
            for (; d >= 0; --d) {
              int offs = tmp % shape[d];
              tmp /= shape[d];
              int p = offs + inPos[d];
              if (p < 0 || p >= inShape[d]) break;
              inP += p * stridesIn[d];
            }
            outPtr[i] = (d < 0) ? *inP : 0;
          }
        }
        else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
          for (int i = 0; i < nElements; ++i) {
            T const *inP = inPtr;
            T *outP = outPtr;
            int tmp = i;
            bool valid = true;
            for (int d = nBlobDims - 1; d >= 0; --d) {
              int offs = tmp % shape[d];
              tmp /= shape[d];
              int p = offs + inPos[d];
              valid &= p >= 0 && p < inShape[d];
              inP += p * stridesIn[d];
              outP += (offs + outPos[d]) * stridesOut[d];
            }
            *outP = valid ? *inP : 0;
          }
        }
      }
    }
  }

  // Tiled predict: score a model in overlap-tile strategy for passing large
  // images through caffe
  template <typename Dtype>
  void TiledPredict(
      const string& infileH5, const string& outfileH5, const string& model,
      const string& weights, int iterations,
      const string& gpu_mem_available_MB_str,
      const string& mem_available_px_str, const string& n_tiles_str,
      const string& tile_size_str, bool average_mirror = false,
      bool average_rotate = false);

}  // namespace caffe

#endif   // CAFFE_UTIL_TILED_PREDICT_H_
