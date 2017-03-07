import tensorflow as tf
import numpy as np
import inspect
from tfs.core.elem import Param
from tfs.core.layer.inference import Softmax

class Regularizer(object):
    def __init__(self,netobj,*args):
        self.net = netobj
        argnames,_,_,_ = inspect.getargspec(type(self).__init__)
        self.param = Param(**{k:v for k,v in zip(argnames[2:],args)})
    def compute(self):
        raise NotImplementedError
    def __str__(self):
        return ""

class L1(Regularizer):
    def __init__(self,netobj,l1=0.01):
        Regularizer.__init__(self,netobj,l1)
    def compute(self):
        for (var_name,var) in self.net.variables.iteritems():
            res=0.
            with self.net.graph.as_default():
                tmp=self.param.l1*tf.abs(var)
                res=tf.add(res,tf.reduce_sum(tmp))
        return res
    def __str__(self):
        return ""


