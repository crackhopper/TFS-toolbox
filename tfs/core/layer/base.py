import numpy as np
import inspect
from tfs.core.util import local_variable_scope,run_once_for_each_obj
import tensorflow as tf
from tfs.core.elem import Param

easyname_dict={
  "Layer":"Layer",
  "FullyConnect": "FC",
  "Conv2d":"Conv2d",
  "MaxPool":"MaxPool",
  "LRN":"LRN",
  "Softmax":"Softmax",
  "Dropout":"Drop",
  "BN":'BN',
}


class Layer(object):
  def __init__(self,name=None):
    self._init(name)

  def _init(self,*args):
    argnames,_,_,_ = inspect.getargspec(type(self).__init__)
    self.param = Param()
    for k,v in zip(argnames[1:],args):
      self.param.__dict__[k]=v
    self.param.name = self.get_unique_name(self.param.name)
    self.name = self.param.name
    self.net = None # it is set by class Network
    self._in = None
    self._out = None
    self._variables = {}
    self._initializers = {}

  @property
  def variables(self):
    return self._variables

  @property
  def initializers(self):
    return self._initializers

  @property
  def input(self):
    return self._in

  @property
  def output(self):
    return self._out

  @run_once_for_each_obj
  def get_unique_name(self,name):
    Layer._name_counter+=1
    if name: return name
    name = easyname_dict[type(self).__name__]
    return '%s_%d'%(name,Layer._name_counter)

  _name_counter=0
  # this is used when defining new network
  @classmethod
  def reset_counter(cls):
    Layer._name_counter=0

  def build(self,inTensor):
    self._in = inTensor
    self._out = self._build()
    return self._out

  @local_variable_scope
  def _make_variable(self,vname,shape,init):
    v=tf.get_variable(vname, shape=shape)
    v.tfs_node = self
    self._variables[vname]=v
    self._initializers[vname]=init
    return v

  def _build(self):
    '''Run the layer. '''
    raise NotImplementedError('Must be implemented by the subclass.')

  def inverse(self,outTensor):
    self._inv_in = outTensor
    self._inv_out = self._inverse()
    return self._inv_out

  def _inverse(self):
    print '%s doesn\'t define inverse op, ignore the layer'% type(self).__name__
    return self._inv_in

  def copy_to(self,to_net):
    cls = type(self)
    obj = cls(**self.param.__dict__)
    obj.net = to_net
    return obj

  def __str__(self):
    """
    The info of this layer
    """
    father_before   ="=={:15}==={:8}=".format(15*"=",8*"=")
    father_attribute="| {:<15} | {:<8} ".format("Name","Type")
    father_value    ="| {:<15} | {:<8} ".format(self.param.name,easyname_dict[type(self).__name__])
    # TODO: add compute_param(self) to the base class ?

    for key in self.param.__dict__.keys():
      if key!='name':
        if key!="activation":
          father_before  +="=={:<8}=".format("="*8)
          father_attribute+="| {:<8} ".format(key)
          father_value    +="| {:<8} ".format(self.param.__dict__[key])
        else:
          if  self.param.__dict__[key]:
            father_before  +="=={:<8}=".format("="*8)
            father_attribute+="| {:<8} ".format('activate')
            father_value    +="| {:<8} ".format(self.param.__dict__[key].func_name)
          else:
            father_before  +="=={:<10}=".format("="*10)
            father_attribute+="| {:<10} ".format(key)
            father_value    +="| {:<10} ".format(self.param.__dict__[key])
    return father_before+"|\n"+father_attribute+"|\n"+father_value+'|'
