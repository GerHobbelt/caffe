FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

LABEL maintainer=duruyao@gmail.com

ARG LC_ALL="C"
ARG DEBIAN_FRONTEND="noninteractive"
ARG TZ="Etc/UTC"
RUN ln -snf /usr/share/zoneinfo/"${TZ}" /etc/localtime && echo "${TZ}" >/etc/timezone

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub && \
    rm -f /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get clean

RUN apt-get update && apt-get install -y --no-install-recommends \
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
      python3-scipy && \
    rm -rf /var/lib/apt/lists/*

ARG CAFFE_VERSION="v1.0.1"
ARG CAFFE_SOURCE_DIR="/caffe"
ARG CAFFE_BUILD_DIR="/caffe/build"
ARG CAFFE_INSTALL_PREFIX="/opt/caffe"
RUN git clone --branch "${CAFFE_VERSION}" --depth 1 https://github.com/duruyao/caffe.git "${CAFFE_SOURCE_DIR}" && \
    for req in $(cat "${CAFFE_SOURCE_DIR}"/python/requirements.txt) pydot; do pip3 install --upgrade --no-cache-dir "${req}"; done && \
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
      -D USE_OPENMP=OFF \
      -D CUDA_ARCH_NAME=Manual \
      -D CUDA_ARCH_BIN="50 52 60 61 70 75 80 86 89 90" \
      -D CUDA_ARCH_PTX="70" \
      -D CUDA_NVCC_FLAGS="-O3" && \
    cmake --build "${CAFFE_BUILD_DIR}" --target all -- -j $(($(nproc) - 0)) && \
    cmake --build "${CAFFE_BUILD_DIR}" --target install && \
    rm -rf "${CAFFE_SOURCE_DIR}"

ENV PYTHONPATH="${CAFFE_INSTALL_PREFIX}/python:${PYTHONPATH}"
ENV PATH="${CAFFE_INSTALL_PREFIX}/bin:${CAFFE_INSTALL_PREFIX}/python:${PATH}"
RUN echo "${CAFFE_INSTALL_PREFIX}/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

ENV THIS_DOCKER_REPOSITORY="https://hub.docker.com/r/duruyao/caffe"
ENV THIS_DOCKER_TAG="gpu"
ARG BUILD_DATE
ENV THIS_DOCKER_CREATED="${BUILD_DATE}"
ENV THIS_GIT_REPOSITORY="https://github.com/duruyao/caffe"
ENV THIS_GIT_TAG="v1.0.1"

WORKDIR /workspace
