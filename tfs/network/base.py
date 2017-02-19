import tensorflow as tf
import numpy as np
from tfs.core.layer import func_table

def _layer_function(layerclass):
  def func(self,*args,**kwargs):
    layer = layerclass(*args,**kwargs)
    self.layers.append(layer)
    return self
  return func

def _network_meta(future_class_name, future_class_parents, future_class_attr):
  for k in func_table:
    future_class_attr[k]=_layer_function(func_table[k])
  return type(future_class_name, future_class_parents, future_class_attr)


class Network(object):
  __metaclass__ = _network_meta
  def __init__(self,input_shape):
    self._in = tf.placeholder(tf.float32,input_shape)
    self.layers=[]
    self.setup(input_shape)
    self._out = self.build()

  def setup(self,in_shape):
    '''Construct the network. '''
    raise NotImplementedError('Must be implemented by the subclass.')


  def build(self):
    tmp = self._in
    for l in self.layers:
      tmp = l.build(tmp)
      self._out = tmp
    return tmp
