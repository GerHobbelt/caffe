FROM nvcr.io/nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04


ENV TZ=Asia/Kolkata
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update
RUN apt install -y tzdata

RUN apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libatlas-base-dev \
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

RUN apt-get install -y gcc-8 g++-8 && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 800 --slave /usr/bin/g++ g++ /usr/bin/g++-8

#### Compile LIBBOOST against python3
RUN wget https://boostorg.jfrog.io/artifactory/main/release/1.70.0/source/boost_1_70_0.tar.gz
RUN tar -xzf boost_1_70_0.tar.gz
RUN cd boost_1_70_0 && ./bootstrap.sh --with-python=python3 --with-python-version=3.6
RUN cd boost_1_70_0 && ./b2  -j16 --libdir=/usr/lib/ --includedir=/usr/include/ && ./b2 headers -j16 --libdir=/usr/lib/ --includedir=/usr/include/ && ./b2 install -j16 --libdir=/usr/lib/ --includedir=/usr/include/
RUN ln -s /usr/lib/libboost_python36.so.1.70.0 /usr/lib/libboost_python3.so 
RUN ldconfig


ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

RUN git clone https://github.com/SuperSecureHuman/caffe-personal .
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 800 --slave /usr/bin/g++ g++ /usr/bin/g++-7

#RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/BVLC/caffe.git . && \
#    pip install --upgrade pip && \
#    cd python && for req in $(cat requirements.txt) pydot; do pip install $req; done && cd .. && \
#    git clone https://github.com/NVIDIA/nccl.git && cd nccl && make -j install && cd .. && rm -rf nccl && \
#    mkdir build && cd build && \
#    cmake -DUSE_CUDNN=1 -DUSE_NCCL=1 .. && \
#    make -j"$(nproc)"
#
#ENV PYCAFFE_ROOT $CAFFE_ROOT/python
#ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
#ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
#RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig
#
#WORKDIR /workspace
