import pytest
import tensorflow as tf
import numpy as np
import tfs.core.layer.ops as ops

from tfs.core.layer.fc import FullyConnect
from tfs.network import Network
net = Network()

@pytest.fixture
def l():
  l = FullyConnect(
    net,
    10,
    activation = ops.relu,
    name=None
  )
  return l

class TestFC:
  def test_build_inverse(self,l):
    _in = tf.zeros([1,10,10,4])
    _out=l.build(_in)
    assert _out.get_shape().as_list()==[1,10]
    _inv_out = l.inverse(_out)
    assert _inv_out.get_shape().as_list()==[1,10,10,4]


