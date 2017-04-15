import pytest
import tensorflow as tf
import numpy as np
import tfs.core.layer.ops as ops

from tfs.core.layer.conv import Conv2d
from tfs.network import Network
net = Network()
@pytest.fixture
def l():
  l = Conv2d(
    net,
    ksize=[3,3],
    knum=10,
    strides=[1,1],
    activation=ops.relu,
    padding='VALID',
    group=2,
    biased=True,
    name=None
  )
  return l

class TestConv:
  def test_build_inverse(self,l):
    _in = tf.zeros([1,10,10,4])
    _out=l.build(_in)
    assert _out.get_shape().as_list()==[1,8,8,10]
    weights = l._variables['weights']
    biases = l._variables['biases']
    assert weights.get_shape().as_list()==[3,3,2,10]
    assert biases.get_shape().as_list()==[10]

    class net(object):
      pass
    l.net=net()
    l.net.nsamples = 1
    _inv_out = l.inverse(_out)
    assert _inv_out.get_shape().as_list()==[1,10,10,4]

