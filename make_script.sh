#!/bin/bash

ARGS=$@

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10

cd $DIR

BUILD_DEBUG="-DCMAKE_BUILD_TYPE=Debug"
BUILD_RELEASE="-DCMAKE_BUILD_TYPE=Release"

CMAKE_COMMON_FLAGS="-DCMAKE_C_COMPILER=/usr/bin/gcc-10 -DCMAKE_CXX_COMPILER=/usr/bin/g++-10 -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DCUDA_ARCH_NAME='Pascal' -DGPU_ARCHS='61' -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF"

mkdir -p ./build_d
cd ./build_d
cmake ${CMAKE_COMMON_FLAGS} ${BUILD_DEBUG} -DUSE_CUDNN=OFF ../
make -j12

cd ..
mkdir -p ./build_r
cd ./build_r
cmake ${CMAKE_COMMON_FLAGS} ${BUILD_RELEASE} -DUSE_CUDNN=OFF ../
make -j12
cd ..

mkdir -p ./build_d_cudnn
cd ./build_d_cudnn
cmake ${CMAKE_COMMON_FLAGS} ${BUILD_DEBUG} -DUSE_CUDNN=ON ../
make -j12

cd ..
mkdir -p ./build_r_cudnn
cd ./build_r_cudnn
cmake ${CMAKE_COMMON_FLAGS} ${BUILD_RELEASE} -DUSE_CUDNN=ON ../
make -j12 
cd ..

exit 0
