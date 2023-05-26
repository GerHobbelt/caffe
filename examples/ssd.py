import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'

import numpy as np
import os
os.chdir('..')
caffe_root = './'
import sys
sys.path.insert(0, caffe_root + 'python')
import time

import caffe
from caffe.proto import caffe_pb2

caffe.set_device(0)
caffe.set_mode_gpu()
# caffe.set_mode_cpu()
# We create a solver that fine-tunes from a previously trained network.
solver = caffe.SGDSolver(caffe_root + 'models/VGGNet/VOC0712/SSD_300x300/solver.prototxt')
solver.net.copy_from(caffe_root + 'models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel')

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': solver.net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

net = solver.net
