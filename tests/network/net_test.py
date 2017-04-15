import pytest
import tensorflow as tf
import numpy as np
from tfs.network.base import Network,CustomNetwork
import shutil
import tfs.g

class MyNet(CustomNetwork):
  def setup(self):
    self.default_in_shape=[1,10,10,2]
    (self.layers
     .conv2d([3,3],4,[1,1],group=2)
     .maxpool([2,2],[2,2],name='out'))
    self.loss_input_layer_name = 'out'

@pytest.fixture
def n():
  return MyNet()


class TestNetwork:
  def test_init(self):
    n = Network()
    n.build([1,1,1,1])

  def test_build(self,n):
    assert not n.has_built()
    n.build([None,10,10,4])
    with pytest.raises(AssertionError):
      n.build([None,10,10,4])
    assert n.has_built()

  def test_copy(self,n):
    n1 = n.copy()
    assert not n1.has_built()
    for l,l1 in zip(n.layers,n1.layers):
      assert l.name == l1.name
      assert l.net != l1.net

    n.build([None,10,10,4])
    n1 = n.copy()
    assert n1.has_built()
    for l,l1 in zip(n.layers,n1.layers):
      assert l.name == l1.name
      assert l.net != l1.net

    assert n.graph != n1.graph

  def test_save_load(self,n,tmpdir):
    tmpdir = str(tmpdir)
    n.save(tmpdir+'unbuild')
    n1 = Network()
    n1.load(tmpdir+'unbuild')
    assert not n1.has_built()
    for i,node in enumerate(n1.net_def):
      assert node.param == n.node_by_index(i).param
      assert node.name == n.node_by_index(i).name
      assert node.net == n1
      assert node.variables=={}
      assert node.initializers=={}
      assert node.input == None
      assert node.output == None

    n.build([None,10,10,4])
    n.save(tmpdir+'build')
    n1 = Network()
    n1.load(tmpdir+'build')
    assert n1.has_built()
    for k,v in n1.variables.items():
      cmp=(n1.run(v)==n.run(n.variables[k]))
      assert cmp.all()
    assert n1.in_shape == n.in_shape
    shutil.rmtree(tmpdir)

  def test_save_load2(self,n,tmpdir):
    n.build([None,10,10,4])
    from tfs.core.optimizer import GradientDecentOptimizer
    from tfs.core.loss import SquareError
    from tfs.core.regularizers import L1
    n.optimizer = GradientDecentOptimizer(n)
    n.losser = SquareError(n)
    n.regularizer = L1(n)

    tmpdir = str(tmpdir)

    n.save(tmpdir+'unbuild')

    n1 = Network()
    n1.load(tmpdir+'unbuild')
    assert isinstance(n1.optimizer,GradientDecentOptimizer)
    assert isinstance(n1.losser,SquareError)
    assert isinstance(n1.regularizer,L1)
    assert n1.optimizer.param ==n.optimizer.param

    shutil.rmtree(tmpdir)

  def test_device(self,n,capsys):
    with capsys.disabled():
      print ''
      print [d.name for d in n.available_devices()]
    has_gpu = ('GPU' in [d.device_type for d in n.available_devices()])
    if has_gpu:
      n.num_gpu=2
      n.build([None,10,10,4])
      with capsys.disabled():
        for i,o in zip(n.input,n.output):
          print i.name,i.device
          print o.name,o.device
        #print n.tf_graph_str()


    # TODO: after adding initializer, test the results are same
  def test_subnet(self,n):
    sub = n.subnet(0,1)

  def test_inference(self,n):
    # TODO: after adding initializer, test inference result
    pass
