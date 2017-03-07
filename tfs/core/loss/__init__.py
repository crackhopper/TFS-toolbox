# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import inspect
from tfs.core.elem import Param
from tfs.core.layer.inference import Softmax

class Objective(object):
    def __init__(self,netobj,*args):
        self.net=netobj
        argnames,_,_,_ = inspect.getargspec(type(self).__init__)
        self.param = Param(**{k:v for k,v in zip(argnames[2:],args)})
    def compute(self):
        raise NotImplementedError
    def __str__(self):
        ""

class categorical_crossentropy(Objective):
    def __init__(self,netobj,with_logits=True):
        Objective.__init__(self,netobj,with_logits)
    def compute(self):
        x1=self.net._out
        output_shape = x1.get_shape().as_list()
        output_dtype =x1.dtype
        with self.net.graph.as_default():
            x2=tf.placeholder(dtype=output_dtype,shape=output_shape)
            assert x2.graph is self.net.graph
        if isinstance(self.net.node_by_index(-1),Softmax):
            return tf.nn.softmax_cross_entropy_with_logits(labels=x2,logits=x1)
        else:
            x1/= tf.reduce_sum(x1,
                               reduction_indices=len(x1.get_shape()) - 1,
                               keep_dims=True)
            return -tf.reduce_sum(x2*tf.log(x1),
                                  reduction_indices=len(x1.get_shape())-1)
    def __str__(self):
        return ""
