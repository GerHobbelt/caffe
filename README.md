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