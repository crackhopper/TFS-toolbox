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
    def compute(self,index_nodes):
        raise NotImplementedError
    def __str__(self):
        return ""


class L1(Regularizer):
    def __init__(self,netobj,l1=0.,nodes_params=None):
        Regularizer.__init__(self,netobj,l1)
        self.nodes_params=nodes_params
    def compute(self):
        """
        param:
        nodes_params: a list of Param that give the Regularizer's parameters of each
                     node.
        example:
        for a given netobj that have 6 nodes in the netdef
        giving the nodes_params
        [None,None,0.01,0.02]
        then the variables in the nodes of index 0,1,4,5 do not use regularizer,
        as for the variables in the nodes of index 2,3 use the L1(Regularizer) for
        the given l1.
        """
        if self.nodes_params is None:
            self.nodes_params=[self.param.l1 for i in range(len(self.net))]
        res=0.
        for index,param in enumerate(self.nodes_params):
            for (var_name,var) in self.net.layers[index].variables.iteritems():
                tmp=param*tf.abs(var)
                res=tf.add(res,tf.reduce_sum(tmp,keep_dims=False))
        return res

    def __str__(self):
        return ""

class L2(Regularizer):
    def __init__(self,netobj,l2=0.,nodes_params=None):
        Regularizer.__init__(self,netobj,l2)
        self.nodes_params=nodes_params
    def compute(self):
        """
        param:
        nodes_params: a list of Param that give the Regularizer's parameters of each
                     node.
        example:
        for a given netobj that have 6 nodes in the netdef
        giving the nodes_params
        [None,None,0.01,0.02]
        then the variables in the nodes of index 0,1,4,5 do not use regularizer,
        as for the variables in the nodes of index 2,3 use the L1(Regularizer) for
        the given l1.
        """
        if self.nodes_params is None:
            self.nodes_params=[self.param.l2 for i in range(len(self.net))]
        res=0.
        for index,param in enumerate(self.nodes_params):
            for (var_name,var) in self.net.layers[index].variables.iteritems():
                tmp=param*tf.square(var)
                res=tf.add(res,tf.reduce_sum(tmp,keep_dims=False))
        return res

    def __str__(self):
        return ""

class DefaultRegularizer(Regularizer):
    def __init__(self,netobj):
        Regularizer.__init__(self,netobj)
    def compute(self):
        return tf.constant(0.)
    def __str__(self):
        return ""





