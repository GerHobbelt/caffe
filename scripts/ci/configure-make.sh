# raw Makefile configuration

LINE () {
  echo -e "\e[35m\e[1m$@\e[0m"
  echo "$@" >> Makefile.config
}

echo -e "\e[35m\e[1mcp Makefile.config.example Makefile.config\e[0m"
cp Makefile.config.example Makefile.config

LINE "BLAS := open"
LINE "WITH_PYTHON_LAYER := 1"

# TODO(lukeyeager) this path is currently disabled because of test errors like:
#   ImportError: dynamic module does not define init function (PyInit__caffe)
# LINE "PYTHON_LIBRARIES := python3.8m boost_python-py38"
# LINE "PYTHON_INCLUDE := /usr/include/python3.8 /usr/lib/python3/dist-packages/numpy/core/include"
# LINE "INCLUDE_DIRS := \$(INCLUDE_DIRS) \$(PYTHON_INCLUDE)"

if ! $WITH_IO ; then
  LINE "USE_OPENCV := 0"
  LINE "USE_LEVELDB := 0"
  LINE "USE_LMDB := 0"
fi

if $WITH_CUDA ; then
  # Only build SM50
  LINE "CUDA_ARCH := -gencode arch=compute_50,code=sm_50"
else
  LINE "CPU_ONLY := 1"
fi

if $WITH_CUDNN ; then
  LINE "USE_CUDNN := 1"
fi

