# Cafeeeeeeee but custom

I came across some caffe models I needed to use, which needed me to hunt for some modifactions and custom layers I had to implement. So I decided to maintain a fork of caffe, which documents my hardships and the solutions I found.

# Caffe source modifications

1. https://github.com/asadalam/caffe - Some opencv4 changes
2. To support latest protobuf - file `src/caffe/util/io.cpp`

```
line - 58
--  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);
++  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit);
```



# Custom Layers addition

1. [SKNet](https://github.com/implus/SKNet)



# Environment

Arch linux
GCC 11.2
Anaconda Python 3.8
Boost (version 1.80) compiled against python 3.8
OpenCV4
Protobuf-21.10-1
Python-Protobuf 3.20.0