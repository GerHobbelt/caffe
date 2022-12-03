FROM nvcr.io/nvidia/cuda:9.2-cudnn7-devel


ENV TZ=Asia/Kolkata
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt install tzdata curl -y

RUN apt install software-properties-common -y

RUN apt-get install -y --no-install-recommends build-essential 
RUN apt-get install -y --no-install-recommends cmake 
RUN apt-get install -y --no-install-recommends git 
RUN apt-get install -y --no-install-recommends wget 
RUN apt-get install -y --no-install-recommends python3-opencv
RUN apt-get install -y --no-install-recommends libopenblas-dev 
RUN apt-get install -y --no-install-recommends libboost-all-dev 
RUN apt-get install -y --no-install-recommends libgflags-dev 
RUN apt-get install -y --no-install-recommends libgoogle-glog-dev 
RUN apt-get install -y --no-install-recommends libhdf5-serial-dev 
RUN apt-get install -y --no-install-recommends libleveldb-dev 
RUN apt-get install -y --no-install-recommends liblmdb-dev 
RUN apt-get install -y --no-install-recommends libopencv-dev 
RUN apt-get install -y --no-install-recommends libprotobuf-dev 
RUN apt-get install -y --no-install-recommends libsnappy-dev 
RUN apt-get install -y --no-install-recommends protobuf-compiler 
RUN apt-get install -y --no-install-recommends python-dev 
RUN apt-get install -y --no-install-recommends python-numpy 
RUN apt-get install -y --no-install-recommends python-pip 
RUN apt-get install -y --no-install-recommends python-setuptools 
RUN apt-get install -y --no-install-recommends python-scipy
RUN apt-get install python-dev libxml2-dev libxslt-dev

ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT


#RUN git clone https://github.com/SuperSecureHuman/caffe-personal  . 
#RUN git pull 


#RUN cd python && for req in $(cat requirements.txt) pydot; do pip install $req; done

#RUN make -j"$(nproc)"

#RUN make test -j"$(nproc)"


#ENV PYCAFFE_ROOT $CAFFE_ROOT/python
#ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
#ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
#RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig
#
WORKDIR /workspace

