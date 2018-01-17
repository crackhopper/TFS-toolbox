import tensorflow as tf
import numpy as np
from tfs.core.layer import ops as ops
from tfs.core.layer.base import Layer
from tfs.core.util import get_arg_dict

class Softmax(Layer):
  def __init__(self,
               net,
               name=None,
               print_names=[]
  ):
    vtable = get_arg_dict(excludes=['self','net'])
    super(Softmax,self).__init__(net,**vtable)

  def _build(self):
    inTensor = self._in
    output = tf.nn.softmax(inTensor,name=self.name)
    return output



