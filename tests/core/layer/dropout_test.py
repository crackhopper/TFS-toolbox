import pytest
import tensorflow as tf
import numpy as np
import tfs.core.layer.ops as ops

from tfs.core.layer.dropout import Dropout
from tfs.network import Network
net = Network()

@pytest.fixture
def l():
  l = Dropout(
    net,
    keep_prob=1.0,
  )
  return l

class TestDropout:
  def test_build_inverse(self,l):
    _in = tf.zeros([1,10,10,4])
    _out=l.build(_in)
    assert _out.get_shape().as_list()==[1,10,10,4]

