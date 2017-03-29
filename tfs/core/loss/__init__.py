# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import inspect
from tfs.core.elem import Param
from tfs.core.layer.inference import Softmax


_EPSILON=1e-07

def _to_tensor(x, dtype):
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

class Loss(object):
    def __init__(self,netobj,*args):
        self.net=netobj
        argnames,_,_,_ = inspect.getargspec(type(self).__init__)
        self.param = Param(**{k:v for k,v in zip(argnames[2:],args)})

    @property
    def in_name(self):
        return self.net.loss_input_layer_name

    def compute(self,idx=None):
        in_name = self.in_name
        if len(self.net) is 0:
            return None
        if in_name is None:
            raise KeyError("please define loss_input_layer_name")
        else:
            if in_name not in self.net.net_def.names():
                raise KeyError("Loss input layer (%s) doesnot exist"%in_name)
            x1_node = self.net.node_by_name(in_name)
        if idx is None:
            x1 = x1_node.output
            x2 = self.net.true_output
        else:
            x1 = x1_node.output[idx]
            x2 = self.net.true_output[idx]

        return self._compute(x1,x2)

    def _compute(self,x1,x2):
        raise NotImplementedError

    def __str__(self):
        info = type(self).__name__+' (%s)'%self.in_name
        pstr = str(self.param)
        return info+'\n-----param-----\n'+pstr+'----------------'


class CrossEntropy(Loss):
    def __init__(self,netobj):
        Loss.__init__(self,netobj)

    def _compute(self,x1,x2):
        num_out = x1.get_shape().as_list()[-1]
        op = None
        if num_out==1:
            op = tf.nn.sigmoid_cross_entropy_with_logits
        else:
            op = tf.nn.softmax_cross_entropy_with_logits
        return tf.reduce_mean(op(labels=x2,logits=x1))

class SquareError(Loss):
    def __init__(self,netobj):
        Loss.__init__(self,netobj)
    def _compute(self,x1,x2):
        if x1.shape.ndims==2:
            axis=1
        else:
            axis=tuple(range(1,x1.shape.ndims))
        norm2=tf.norm(x2-x1,ord='euclidean',axis=axis)
        return tf.square(norm2)

DefaultLoss=CrossEntropy
