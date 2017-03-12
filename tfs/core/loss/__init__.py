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

    def compute(self):
        raise NotImplementedError

    def __str__(self):
        ""

class BinaryCrossentripy(Loss):
    def __init__(self,netobj,with_logits=True):
        Loss.__init__(self,netobj,with_logits)
    def compute(self):
        x1=self.net.output
        x2=self.net.true_output
        assert x1.graph==x2.graph
        if not isinstance(self.net.node_by_index(-1),Softmax):
            epsilon = _to_tensor(_EPSILON, x1.dtype.base_dtype)
            x1= tf.clip_by_value(x1, epsilon, 1 - epsilon)
            x1 = tf.log(x1 / (1 - x1))
        try:
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=x2,logits=x1))
        except TypeError:
            return tf.nn.sigmoid_cross_entropy_with_logits(x1, x2)
    def __str__(self):
        return ""

class CategoricalCrossentropy(Loss):
    def __init__(self,netobj,with_logits=True):
        Loss.__init__(self,netobj,with_logits)

    def compute(self):
        x1=self.net.output
        x2=self.net.true_output
        assert x1.graph==x2.graph
        if not isinstance(self.net.node_by_index(-1),Softmax):
            epsilon = _to_tensor(_EPSILON, x1.dtype.base_dtype)
            x1= tf.clip_by_value(x1, epsilon, 1 - epsilon)
            x1 = tf.log(x1 / (1 - x1))
        try:
            return tf.nn.softmax_cross_entropy_with_logits(labels=x2,logits=x1)
        except TypeError:
            return tf.nn.softmax_cross_entropy_with_logits(x1,x2)
    def __str__(self):
        return ""

class SquareError(Loss):
    def __init__(self,netobj):
        Loss.__init__(self,netobj)
    def compute(self):
        x1=self.net.output
        x2=self.net.true_output
        assert x1.graph==x2.graph
        if x1.shape.ndims==2:
            axis=1
        else:
            axis=tuple(range(1,x1.shape.ndims))
        norm2=tf.norm(x2-x1,ord='euclidean',axis=axis)
        return tf.square(norm2)
    def __str__(self):
        return ""

def DefaultLoss(netobj):
    return CategoricalCrossentropy(netobj,with_logits=True)
