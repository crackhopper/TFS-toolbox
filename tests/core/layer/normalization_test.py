import pytest
import tensorflow as tf
import numpy as np
import tfs.core.layer.ops as ops

from tfs.core.layer.normalization import LRN,BN
from tfs.network import Network
net = Network()

@pytest.fixture
def l():
  l = LRN(
    net,
    radius=1,
    alpha=.1,
    beta=0.01,
    bias=1.0,
    name=None
  )
  return l
class TestLRN:
  def test_build_inverse(self,l):
    _in = tf.zeros([1,10,10,4])
    _out=l.build(_in)
    assert _out.get_shape().as_list()==[1,10,10,4]

@pytest.fixture
def l():
  l = BN(
    net,
    scale_offset=True,
    activation=ops.relu,
  )
  return l
class TestBN:
  def test_build_inverse(self,l):
    _in = tf.zeros([1,10,10,4])
    _out=l.build(_in)
    assert _out.get_shape().as_list()==[1,10,10,4]


