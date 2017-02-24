import tensorflow as tf
import numpy as np
import ops
from base import Layer

class LRN(Layer):
  def __init__(self,
               radius,
               alpha,
               beta,
               bias=1.0,
               name=None
  ):
    Layer._init(
      self,
      radius,
      alpha,
      beta,
      bias,
      name
    )
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
               scale_offset=True,
               activation=ops.relu,
               name=None
  ):
    Layer._init(
      self,
      scale_offset,
      activation,
      name
    )
  def _build(self):
    inTensor = self._in
    input_shape = inTensor.get_shape()
    scale_offset = self.param.scale_offset
    shape = [input_shape[-1]]
    if scale_offset:
      scale = self._make_variable('scale', shape=shape)
      offset = self._make_variable('offset', shape=shape)
    else:
      scale, offset = (None, None)
    output = tf.nn.batch_normalization(
      inTensor,
      mean=self._make_variable('mean', shape=shape),
      variance=self._make_variable('variance', shape=shape),
      offset=offset,
      scale=scale,
      # TODO: This is the default Caffe batch norm eps
      # Get the actual eps from parameters
      variance_epsilon=1e-5,
      name=self.name)
    if self.param.activation:
      output = self.param.activation(output)
    return output

