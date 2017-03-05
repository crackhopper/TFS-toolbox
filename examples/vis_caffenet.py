from tfs.models import CaffeNet
import tensorflow as tf

net = CaffeNet()
net.build([None,227,227,3])

print net

from tfs.adapter import DeconvVisNet
visnet = DeconvVisNet(net)

invout = visnet._inv_out
print invout

import tensorflow as tf
import numpy as np
import pickle
f=open('/Users/crackhopper/proj/data/data_blob.npy','rb')
data_blob = pickle.load(f)
data_blob = data_blob.transpose((0,2,3,1))

from tfs.core.initializer import CaffeTensorflowLoader
init = CaffeTensorflowLoader(net,'/Users/crackhopper/proj/github/caffe-tensorflow/caffenet.npy')
net.run_initor(init)

from tfs.core.util import *

layer_name = 'fc8'
channelId=290
data_blob

img = visnet.vis_image(layer_name,channelId,data_blob)


import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.savefig('testplot.png')
