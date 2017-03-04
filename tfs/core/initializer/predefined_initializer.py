from base import DefaultValueInit, DefaultOpInit
import init_func
import tensorflow as tf
import numpy as np

class DefaultInitializer(DefaultValueInit):
  def init_layer_by_val(self,shape,dtype):
    layerDefaultOp = self._cur_node.initializers[self._cur_vname]
    return layerDefaultOp(shape,dtype)

class AllConstantInitializer(DefaultValueInit):
  def __init__(
      self,
      netobj,
      val=0.1
  ):
    super(AllConstantInitializer,self).__init__(
      netobj,
      val
    )

  def init_layer_by_val(self,shape,dtype):
    op = init_func.constant(self.param.val)
    return op(shape,dtype)

