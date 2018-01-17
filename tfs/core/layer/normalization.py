import tensorflow as tf
import numpy as np

from tfs.core.layer import ops as ops
from tfs.core.layer.base import Layer
import tfs.core.initializer.init_func as init
from tfs.core.util import get_arg_dict

class LRN(Layer):
  def __init__(self,
               net,
               radius,
               alpha,
               beta,
               bias=1.0,
               name=None,
               print_names=['radius','alpha','beta']
  ):
    vtable = get_arg_dict(excludes=['self','net'])
    super(LRN,self).__init__(net,**vtable)

  def _build(self):
    inTensor = self._in
    output = tf.nn.local_response_normalization(
      inTensor,
      depth_radius=self.param.radius,
      alpha=self.param.alpha,
      beta=self.param.beta,
      bias=self.param.bias,
      name=self.name)
    return output

class BN(Layer):
  def __init__(self,
               net,
               scale_offset=True,
               activation=ops.relu,
               name=None,
               print_names=[]
  ):
    vtable = get_arg_dict(excludes=['self','net'])
    super(BN,self).__init__(net,**vtable)

  def _build(self):
    inTensor = self._in
    input_shape = inTensor.get_shape()
    scale_offset = self.param.scale_offset
    shape = [input_shape[-1]]
    if scale_offset:
      scale = self._make_variable('scale', shape=shape,init=init.constant())
      offset = self._make_variable('offset', shape=shape,init=init.constant())
    else:
      scale, offset = (None, None)
    output = tf.nn.batch_normalization(
      inTensor,
      mean=self._make_variable('mean', shape=shape,init=init.constant()),
      variance=self._make_variable('variance', shape=shape,init=init.constant()),
      offset=offset,
      scale=scale,
      # TODO: This is the default Caffe batch norm eps
      # Get the actual eps from parameters
      variance_epsilon=1e-5,
      name=self.name)
    if self.param.activation:
      output = self.param.activation(output)
    return output

