#!/bin/bash
# configure the project

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

echo -e "\e[35m\e[1mWITH_CMAKE: ${WITH_CMAKE}\e[0m"
echo -e "\e[35m\e[1mWITH_IO:    ${WITH_IO}\e[0m"
echo -e "\e[35m\e[1mWITH_CUDA:  ${WITH_CUDA}\e[0m"
echo -e "\e[35m\e[1mWITH_CUDNN: ${WITH_CUDNN}\e[0m"

if ! $WITH_CMAKE ; then
  source $BASEDIR/configure-make.sh
else
  source $BASEDIR/configure-cmake.sh
fi
