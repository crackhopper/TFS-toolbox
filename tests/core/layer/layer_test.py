import pytest
# name-convention: http://doc.pytest.org/en/latest/goodpractices.html#test-discovery
# fixtures: http://docs.pytest.org/en/latest/fixture.html#fixtures-as-function-arguments

import tensorflow as tf
import numpy as np


import tfs.core.layer.base as base
import tfs.core.layer.ops as ops
from tfs.network import Network
net = Network()

# test param
Param = base.Param

@pytest.fixture
def param():
  p = Param()
  p.name = 'name'
  p.outdim = 10
  p.activation = ops.relu
  return p

class TestParam:
  def test_copy(self,param):
    cp = param.copy()
    assert param.__dict__ == cp.__dict__

  def test_str(self,param,capsys):
    with capsys.disabled():
      print(param,)

# test Layer
Layer = base.Layer
@pytest.fixture
def layer():
  l= Layer(net,name='test')
  return l

class TestLayer:
  def test_init(self,layer):
    l = Layer(net)
    assert l.name =='Layer_2'
    assert layer.name=='test'

  def test_build(self,layer):
    _in = tf.constant(0)
    with pytest.raises(NotImplementedError):
      layer.build(_in)
    assert layer._in == _in
    assert layer._out == None

  def test_inverse(self,layer, capsys):
    _in = tf.constant(0)
    layer.inverse(_in)
    assert layer._inv_in==_in
    assert layer._inv_out==_in
    out, err = capsys.readouterr()
    assert out == 'Layer doesn\'t define inverse op, ignore the layer\n'

  def test_copy_to(self,layer):
    with pytest.raises(AssertionError):
      cp = layer.copy_to(None)

    cp = layer.copy_to(net)
    assert type(cp) is type(layer)
    assert cp.param.__dict__ == layer.param.__dict__
    assert not (cp.param.__dict__ is layer.param.__dict__)
    assert cp.net is net

