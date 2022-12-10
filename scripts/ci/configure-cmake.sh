# CMake configuration

echo -e "\e[35m\e[1mmkdir -p build; cd build\e[0m"
mkdir -p build
cd build

ARGS="-DCMAKE_BUILD_TYPE=Release -DBLAS=Open"

if $WITH_IO ; then
  ARGS="$ARGS -DUSE_OPENCV=On -DUSE_LMDB=On -DUSE_LEVELDB=On"
else
  ARGS="$ARGS -DUSE_OPENCV=Off -DUSE_LMDB=Off -DUSE_LEVELDB=Off"
fi

if $WITH_CUDA ; then
  # Only build SM50
  ARGS="$ARGS -DCPU_ONLY=Off -DCUDA_ARCH_NAME=Manual -DCUDA_ARCH_BIN=\"50\" -DCUDA_ARCH_PTX=\"\""
else
  ARGS="$ARGS -DCPU_ONLY=On"
fi

if $WITH_CUDNN ; then
  ARGS="$ARGS -DUSE_CUDNN=On"
else
  ARGS="$ARGS -DUSE_CUDNN=Off"
fi

echo -e "\e[35m\e[1mcmake .. ${ARGS}\e[0m"
cmake .. $ARGS
