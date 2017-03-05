import tensorflow as tf
import numpy as np
from tfs.core.layer import func_table,Layer
from tfs.core.initializer import DefaultInit,InitType,Initializer
from tfs.core.util import run_once_for_each_obj
from tensorflow.python.util.deprecation import deprecated
import pickle
import new
from tensorflow.python.client import device_lib

# for supporting multi-gpu:
# https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/models/image/cifar10/cifar10_multi_gpu_train.py#L174
#
# we use shared variables on CPU and model distributed on each GPU
#
#################### NetStructure
def _layer_function(layerclass):
  def func(self,*args,**kwargs):
    layer = layerclass(*args,**kwargs)
    self.append(layer)
    return self
  return func

def _net_sturcture_meta(future_class_name, future_class_parents, future_class_attr):
  for k in func_table:
    future_class_attr[k]=_layer_function(func_table[k])
  return type(future_class_name, future_class_parents, future_class_attr)

class NetStructure(object):
  __metaclass__ = _net_sturcture_meta
  """This class is used for define a network structure by using layers.
  """
  def __init__(self,net,nodes=None):
    nodes = nodes or []
    self._net = net
    self.nodes=nodes

  @property
  def in_shape(self):
    return self.net.in_shape

  def _adjust(self):
    def _setnet(n):
      n.net=self.net
    map(_setnet, self._nodes)

  @property
  def net(self):
    return self._net

  @net.setter
  def net(self,net):
    self._net = net
    self._adjust()

  @property
  def nodes(self):
    return self._nodes

  @nodes.setter
  def nodes(self,nodes):
    self._nodes = nodes
    self._need_built = True
    self._adjust()

  def append(self,l):
    self._need_built = True
    l.net = self.net
    self.nodes.append(l)

  def __getitem__(self,i):
    return self.nodes[i]

  def copy_to(self,net):
    res = NetStructure(net)
    for l in self.nodes:
      res.nodes.append(l.copy_to(net))
    return res

  def _built_lut(self):
    if not self._need_built: return
    self._lut = {}
    self._lut2 = {}
    for i,l in enumerate(self.nodes):
      self._lut[l.name]=l
      self._lut2[l.name]=i

  def by_name(self,name):
    self._built_lut()
    return self._lut[name]

  def find_index(self,l):
    self._built_lut()
    return self._lut2[l.name]

  def names(self):
    self._built_lut()
    return self._lut.keys()

  def __len__(self):
    return len(self.nodes)

  def save(self,filename):
    f=open(filename,'wb')
    nodes = [n.copy_to(None) for n in self.nodes]
    pickle.dump([nodes,self.in_shape],f)
    f.close()

  def load(self,filename):
    f=open(filename,'rb')
    self.nodes,in_shape = pickle.load(f)
    if in_shape:
      self.net.build(in_shape)
    f.close()

#################### Network
# decorators
def with_graph(f):
  def with_graph_run(self,*args,**kwargs):
    with self.graph.as_default():
      return f(self,*args,**kwargs)
  # this is important to make the decorator compatiable with run_once_each_obj.
  with_graph_run.__name__=f.__name__
  return with_graph_run

class Network(object):
  def __init__(self):
    self._struct = NetStructure(self)
    self._in = None
    self._out = None
    self._graph = tf.Graph()
    with self.graph.as_default():
      self._sess = tf.Session()
    self.variables = {}
    self.initializer = DefaultInit(self)
    self.num_gpu = 0

  @staticmethod
  def available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x for x in local_device_protos]

  def __len__(self):
    return len(self.net_def)

  @property
  @deprecated("2017-05-01", "Use `net_def` instead.")
  def layers(self):
    return self._struct

  @property
  def net_def(self):
    return self._struct

  def node_to_index(self,l):
    return self.net_def.find_index(l)

  def node_by_index(self,idx):
    return self.net_def[idx]

  @deprecated("2017-05-01", "Use `node_by_name` instead.")
  def layer_by_name(self,name):
    return self.net_def.by_name(name)

  def node_by_name(self,name):
    return self.net_def.by_name(name)

  def __del__(self):
    self.sess.close()

  def setup(self):
    '''Construct the network. '''
    raise NotImplementedError('Must be implemented by the subclass.')

  def setup_with_def(self,struct_def,in_shape=None):
    if isinstance(struct_def,list):
      struct_def = NetStructure(self,nodes=struct_def)
    self._struct = struct_def.copy_to(self)
    if in_shape:
      self.build(in_shape)

  @property
  def graph(self):
    return self._graph

  @property
  def input(self):
    return self._in

  @property
  def output(self):
    return self._out

  @property
  def sess(self):
    return self._sess

  def _init_in_out_size(self):
    if self.num_gpu and self._in is None and self._out is None:
      self._in = [None]*self.num_gpu
      self._out = [None]*self.num_gpu

  @with_graph
  @run_once_for_each_obj
  def build(self,input_shape,dtype=tf.float32):
    Layer.reset_counter()
    """Build the computational graph
    inTensor: the network input tensor.
    """
    if not self.num_gpu:
      self._build(input_shape,dtype)
    else:
      for i in range(self.num_gpu):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % ('GPU', i)) as scope:
            self._build(input_shape,dtype,i)
            tf.get_variable_scope().reuse_variables()

    self.build_variables_table()
    self._initialize()
    return self.output

  def tf_graph_str(self):
    info=[]
    for n in self.graph.as_graph_def().node:
      s = '%-20s@%20s'%(n.name,n.device)
      if hasattr(n,'tfs_nodename'):
        s=s+' --%s'%n.tfs_nodename
      info.append(s)
    return '\n'.join(info)

  def _build(self,input_shape,dtype,idx=None):
    self._init_in_out_size()
    tmp = tf.placeholder(dtype,input_shape)
    if idx is None:
      self._in = tmp
    else:
      self._in[idx] = tmp

    for l in self.layers:
      tmp = l.build(tmp,idx)

    if idx is None:
      self._out = tmp
    else:
      self._out[idx] = tmp

    return tmp

  def _initialize(self):
    # TODO: check if need initialize
    self.run_initor(self.initializer)

  def _get_init_op(self,initor):
    t,initor = initor.init_table
    if t == InitType.values:
      return self._get_init_op_by_val(initor)
    elif t == InitType.ops:
      return self._get_init_op_by_ops(initor)
    else:
      assert isinstance(initor,tf.Operation)
      return initor


  def _get_init_op_by_val(self,tbl):
    return self._get_init_op_by_ops({
      n:tf.assign(self.variables[n],val)
      for n,val in tbl.items()
    })

  def _get_init_op_by_ops(self,initor):
    return tf.group(*initor.values())

  def build_variables_table(self):
    for l in self.net_def:
      for k in l.variables:
        v = l.variables[k]
        self.variables[v.name] = v

  def has_built(self):
    if hasattr(self,'_has_run'):
      if Network.build.__name__ in self._has_run:
        return True
    return False

  def run(self,eval_list,feed_dict=None):
    return self.sess.run(eval_list, feed_dict=feed_dict)

  def run_initor(self,initor):
    assert isinstance(initor,Initializer)
    op = self._get_init_op(initor)
    return self.sess.run(op)

  def save(self,filename):
    self.save_def(filename)
    to_save={}
    for k,v in self.variables.items():
      to_save[k]=self.run(v)
    f=open(filename+'.model','wb')
    pickle.dump(to_save,f)
    f.close()

  def save_def(self,filename):
    self.net_def.save(filename+'.def')

  def load(self,filename):
    self.load_def(filename)
    f=open(filename+'.model','rb')
    data_dict=pickle.load(f)
    f.close()
    if self.has_built():
      op = self._get_init_op_by_val(data_dict)
      self.run(op)

  def load_def(self,filename):
    self.net_def.load(filename+'.def')

  @property
  def in_shape(self):
    if self._in is not None:
      return self._in.get_shape().as_list()
    return None

  def copy(self):
    obj = Network()
    obj.setup_with_def(self.net_def,self.in_shape)
    return obj

  def __str__(self):
    # TODO:
    result=''
    for (i,layer) in enumerate(self.layers):
      result+="Layer {} :\n".format(i)+str(layer)+"\n"
    return result

  def subnet(self,begin_index,end_index):
    obj = Network()
    obj.setup_with_def(self.layers[begin_index:end_index])
    return obj


class CustomNetwork(Network):
  """Automatically called setup and build when construct
  """
  def __init__(self,in_shape=None):
    Network.__init__(self)
    self.default_in_shape = None
    self.setup()
    in_shape = self.default_in_shape
    if not in_shape:
      raise ValueError("must sepecify the default_in_shape attributes, or pass the shape as an argument when construction")

  def setup(self):
    raise NotImplementedError("CustomNetwork Must Implement setup Method")

  def build(self,inshape=None):
    inshape = inshape or self.default_in_shape
    return Network.build(self,inshape)


