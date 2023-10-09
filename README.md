# Caffe

The current repository is forked from [BVLC/caffe](https://github.com/BVLC/caffe) and updated to accommodate **Python 3**, **OpenCV 4**, **CUDA 11.X** and **cuDNN 8**.

## Use Caffe with Docker

It is recommended to pull the Docker image [duruyao/caffe](https://hub.docker.com/r/duruyao/caffe) with the updated Caffe built-in instead of the Docker image [bvlc/caffe](https://hub.docker.com/r/bvlc/caffe).

```shell
docker pull duruyao/caffe:cpu
docker pull duruyao/caffe:gpu
```

The following table lists some major differences between my own Docker image and the official Docker image.

| bvlc/caffe   | duruyao/caffe |
|--------------|---------------|
| ubuntu 16.04 | ubuntu 22.04  |
| cuda 8.0     | cuda 11.8.0   |
| cudnn 6      | cudnn 8       |
| python 2     | python 3      |
| opencv 3     | opencv 4      |

## Install Caffe from Source

Here is an example of building Caffe for Ubuntu 22.04.

```shell
sudo apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
  git \
  wget \
  libatlas-base-dev \
  libboost-all-dev \
  libgflags-dev \
  libgoogle-glog-dev \
  libhdf5-serial-dev \
  libleveldb-dev \
  liblmdb-dev \
  libopencv-dev \
  libprotobuf-dev \
  libsnappy-dev \
  protobuf-compiler \
  python3-dev \
  python3-numpy \
  python3-pip \
  python3-setuptools \
  python3-scipy

CAFFE_VERSION="v1.0.1"
CAFFE_SOURCE_DIR="caffe"
CAFFE_BUILD_DIR="caffe/build"
CAFFE_INSTALL_PREFIX="/opt/caffe"
git clone --branch "${CAFFE_VERSION}" --depth 1 https://github.com/duruyao/caffe.git "${CAFFE_SOURCE_DIR}"
for req in $(cat "${CAFFE_SOURCE_DIR}"/python/requirements.txt) pydot; do pip3 install --upgrade --no-cache-dir "${req}"; done
cmake -S "${CAFFE_SOURCE_DIR}" \
  -B "${CAFFE_BUILD_DIR}" \
  -G "Unix Makefiles" \
  -D CMAKE_CXX_STANDARD=11 \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX="${CAFFE_INSTALL_PREFIX}" \
  -D CPU_ONLY=OFF \
  -D USE_CUDNN=ON \
  -D USE_NCCL=OFF \
  -D BUILD_SHARED_LIBS=ON \
  -D BUILD_python=ON \
  -D python_version=3 \
  -D BUILD_matlab=OFF \
  -D BUILD_docs=ON \
  -D BUILD_python_layer=ON \
  -D USE_OPENCV=ON \
  -D USE_LEVELDB=ON \
  -D USE_LMDB=ON \
  -D ALLOW_LMDB_NOLOCK=OFF \
  -D USE_OPENMP=OFF
cmake --build "${CAFFE_BUILD_DIR}" --target all -- -j $(($(nproc) - 2))
cmake --build "${CAFFE_BUILD_DIR}" --target runtest -- -j $(($(nproc) - 2))
cmake --build "${CAFFE_BUILD_DIR}" --target install

export PYTHONPATH="${CAFFE_INSTALL_PREFIX}/python:${PYTHONPATH}"
export PATH="${CAFFE_INSTALL_PREFIX}/bin:${CAFFE_INSTALL_PREFIX}/python:${PATH}"
echo "${CAFFE_INSTALL_PREFIX}/lib" >>/etc/ld.so.conf.d/caffe.conf && ldconfig
```

## More

For more information about Caffe, visit the official website [caffe.berkeleyvision.org](https://caffe.berkeleyvision.org) and the original repository [BVLC/caffe](https://github.com/BVLC/caffe) please.