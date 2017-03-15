import tensorflow as tf
import numpy as np
from tfs.core.layer import func_table,Layer
from tfs.core.util import run_once_for_each_obj

from tfs.core.initializer import DefaultInit
from tfs.core.loss import DefaultLoss
from tfs.core.regularizers import DefaultRegularizer
from tfs.core.monitor import DefaultMonitor
from tfs.core.optimizer import DefaultOptimizer

from tensorflow.python.util.deprecation import deprecated
import pickle
import new
from tensorflow.python.client import device_lib
from sklearn import metrics

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
    pickle.dump([nodes,self.in_shape,self.net.loss_input_layer_name],f)
    f.close()

  def load(self,filename):
    f=open(filename,'rb')
    self.nodes,in_shape,self.net.loss_input_layer_name = pickle.load(f)
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
    self._init_graph_sess()
    self._struct = NetStructure(self)

    self._true_out=None 

    self._in = None
    self._out = None
    self._loss=None

    self.variables = {}
    self.initializer = DefaultInit(self)
    self.losser = DefaultLoss(self)
    self.regularizer =DefaultRegularizer(self)
    self.monitor = DefaultMonitor(self)
    self._optimizer = DefaultOptimizer(self)

    # this must be set when define a network
    self.loss_input_layer_name = None

    self._regulization=None
    self.grads = None
    self._train_op = None

    self.num_gpu = 0
    self.i_step = 0
    self.n_epoch = 0
    self._dtype = None

  def _init_graph_sess(self):
    self._graph = tf.Graph()
    with self.graph.as_default():
      self._sess = tf.Session()

  @property
  def optimizer(self):
    return self._optimizer

  @optimizer.setter
  def optimizer(self,opt):
    self.grads=None
    self._optimizer=opt


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
  def true_output(self):
    return self._true_out

  @property
  def sess(self):
    return self._sess

  def _init_in_out_size(self):
    if self.num_gpu and self._in is None and self._out is None:
      self._in = [None]*self.num_gpu
      self._out = [None]*self.num_gpu
      self._true_out = [None]*self.num_gpu
      self._loss = [None]*self.num_gpu

  def tf_graph_str(self):
    info=[]
    for n in self.graph.as_graph_def().node:
      s = '%-20s@%20s'%(n.name,n.device)
      if hasattr(n,'tfs_nodename'):
        s=s+' --%s'%n.tfs_nodename
      info.append(s)
    return '\n'.join(info)

  @with_graph
  @run_once_for_each_obj
  def build(self,input_shape,dtype=tf.float32):
    self._dtype = dtype
    Layer.reset_counter()
    """Build the computational graph
    inTensor: the network input tensor.
    """
    if not self.num_gpu:
      self._build(input_shape,dtype)
    else:
      tower_grads = []
      for i in range(self.num_gpu):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % ('GPU', i)) as scope:
            self._build(input_shape,dtype,i)
            tf.get_variable_scope().reuse_variables()
            _loss = self.loss[i]
            tower_grads.append(_grad)

    self.build_variables_table()
    self._initialize()
    self.compute_gradients()
    return self.output

  def compute_gradients(self):
    if self.loss is None:
      return
    if not self.num_gpu:
      self.grads = self.optimizer.compute_gradients(self.loss,self.variables.values())
    else:
      tower_grads = []
      for i in range(self.num_gpu):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % ('GPU', i)) as scope:
            tf.get_variable_scope().reuse_variables()
            _loss = self.loss[i]
            _grad = self.optimizer.compute_gradients(_loss,self.variables.values())
            tower_grads.append(_grad)
      self.grads = self.average_gradients(tower_grads)

  def average_gradients(self,tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
      # Note that each grad_and_vars looks like the following:
      #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
      grads = []
      for g, _ in grad_and_vars:
        expanded_g = tf.expand_dims(g, 0)
        grads.append(expanded_g)
      grad = tf.concat(axis=0, values=grads)
      grad = tf.reduce_mean(grad, 0)
      v = grad_and_vars[0][1]
      grad_and_var = (grad, v)
      average_grads.append(grad_and_var)
    return average_grads

  # this function is called only in build() under current graph.
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
      output_shape=self._out.get_shape().as_list()
      output_dtype=self._out.dtype
      self._true_out=tf.placeholder(dtype=output_dtype,shape=output_shape)
      self._loss = self._compute_loss(idx)
    else:
      self._out[idx] = tmp
      output_shape=self._out[idx].get_shape().as_list()
      output_dtype=self._out[idx].dtype
      self._true_out[i]=tf.placeholder(dtype=output_dtype,shape=output_shape)
      self._loss[idx] = self._compute_loss(idx)
    return self

  def _initialize(self):
    self.run_initor(self.initializer)

  def _compute_loss(self,idx):
    loss =  self.losser.compute(idx)
    if loss is None:
      return loss
    return loss + self.regularizer.compute()

  @property
  def loss(self):
    return self._loss

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

  def fit(self,dataset,batch_size,n_epoch,
          shuffle_epoch=True,max_step=10000000):
    train_set = dataset.train
    test_set = dataset.test
    train_set.before_iter()
    self.i_step = 0
    self.n_epoch = 0
    while True:
      self.i_step += 1
      self.n_epoch = train_set.epochs_completed
      X,y = train_set.next_batch(batch_size,shuffle=shuffle_epoch)
      self.step(X,y,self.i_step)
      self.monitor.status(train_set,test_set,self.i_step,self.n_epoch)
      if self.n_epoch>=n_epoch:
        break
      if self.i_step >= max_step:
        break
    return self

  @property
  def train_op(self):
    if self._train_op is None:
      self._train_op = self._get_train_op()
    return self._train_op

  @with_graph
  def _get_train_op(self,step=None):
    if self.loss is None:
      return None
    if self.grads is None:
      self.compute_gradients()
    op = self.optimizer.apply_gradients(self.grads,step)
    # initialize the uninitalized variable (the optimizer would introduce
    # uninitalized variable)
    vars = self.optimizer.variables
    self.run(tf.initialize_variables(vars.values()))
    return op

  def step(self,X,y,step):
    self.run(self.train_op,feed_dict={self.input:X,self.true_output:y})

  def predict(self,X):
    if self.num_gpu==0:
      _in = self.input
      _out = self.output
    else:
      _in = self.input[0]
      _out = self.output[0]
    return self.run(_out,feed_dict={_in:X})

  def score(self,datasubset):
    y_pred = self.predict(datasubset.data)
    y_pred = np.argmax(y_pred,1)
    y_true = datasubset.labels
    y_true = np.argmax(y_true,1)
    return metrics.accuracy_score(y_true,y_pred)

  def measure_loss(self,X,y):
    if self.num_gpu==0:
      _in = self.input
      _true_out = self.true_output
      _loss = self.loss
    else:
      _in = self.input[0]
      _true_out = self.true_output[0]
      _loss = self.loss[0]

    return self.run(_loss,feed_dict={_in:X,_true_out:y})

  def run(self,eval_list,feed_dict=None):
    return self.sess.run(eval_list, feed_dict=feed_dict)

  def run_initor(self,initor):
    op = initor.compute()
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
    self._init_graph_sess()
    self.load_def(filename)
    f=open(filename+'.model','rb')
    data_dict=pickle.load(f)
    f.close()
    if self.has_built():
      with self._graph.as_default():
        op = self.initializer.op_by_value_table(data_dict)
      self.run(op)

  def load_def(self,filename):
    self.net_def.load(filename+'.def')

  @property
  def in_shape(self):
    if self._in is not None:
      if self.num_gpu==0:
        return self._in.get_shape().as_list()
      else:
        return self._in[0].get_shape().as_list()
    return None

  @property
  def dtype(self):
    return self._dtype

  @property
  def out_shape(self):
    if self._out:
      if self.num_gpu==0:
        return self._out.get_shape().as_list()
      else:
        return self._out[0].get_shape().as_list()
    return None

  def copy(self):
    obj = Network()
    obj.loss_input_layer_name = self.loss_input_layer_name
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


