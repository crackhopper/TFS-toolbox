import tensorflow as tf
import numpy as np
from tfs.core.layer import ops as ops
from tfs.core.layer.base import Layer
from tfs.core.util import get_arg_dict

class Dropout(Layer):
  def __init__(self,
               net,
               keep_prob,
               name=None,
               print_names=['keep_prob']
  ):
    vtable = get_arg_dict(excludes=['self','net'])
    super(Dropout,self).__init__(net,**vtable)

  def _build(self):
    inTensor = self._in
    output = tf.nn.dropout(inTensor, self.param.keep_prob,
                           name=self.param.name)
    return output

