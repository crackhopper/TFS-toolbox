import pytest
import tensorflow as tf
import numpy as np
import tfs.core.layer.ops as ops

from tfs.core.layer.pool import MaxPool,AvgPool
from tfs.network import Network
net = Network()

@pytest.fixture
def l():
  l = MaxPool(
    net,
    ksize=[2,2],
    strides=[2,2],
    padding='SAME',
    name=None
  )
  return l

class TestMaxPool:
  def test_build_inverse(self,l):
    _in = tf.zeros([1,10,10,4])
    _out=l.build(_in)
    assert _out.get_shape().as_list()==[1,5,5,4]
    _inv_out = l.inverse(_out)
    assert _inv_out.get_shape().as_list()==[1,10,10,4]
