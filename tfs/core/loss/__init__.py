# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import inspect
from tfs.core.elem import Param
from tfs.core.layer.inference import Softmax

class Loss(object):
    def __init__(self,netobj,*args):
        self.net=netobj
        argnames,_,_,_ = inspect.getargspec(type(self).__init__)
        self.param = Param(**{k:v for k,v in zip(argnames[2:],args)})

    def compute(self):
        raise NotImplementedError

    def __str__(self):
        ""

class CategoricalCrossentropy(Loss):
    def __init__(self,netobj,with_logits=True):
        Loss.__init__(self,netobj,with_logits)

    def compute(self):
        x1=self.net.output
        x2=self.net.true_output
        assert x1.graph==x2.graph
        if isinstance(self.net.node_by_index(-1),Softmax):
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=x1,logits=x2))
        else:
            x1/= tf.reduce_sum(x1,
                               reduction_indices=len(x1.get_shape()) - 1,
                               keep_dims=True)
            return -tf.reduce_sum(x2*tf.log(x1),
                                  reduction_indices=len(x1.get_shape())-1)
    def __str__(self):
        return ""


def DefaultLoss(netobj):
    return CategoricalCrossentropy(netobj,with_logits=True)
