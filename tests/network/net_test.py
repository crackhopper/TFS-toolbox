import pytest
import tensorflow as tf
import numpy as np

from tfs.network.base import Network,CustomNetwork

class MyNet(CustomNetwork):
  def setup(self):
    self.default_in_shape=[1,10,10,2]
    (self.layers
     .conv2d([3,3],4,[1,1],group=2)
     .maxpool([2,2],[2,2]))

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


    # TODO: after adding initializer, test the results are same
  def test_subnet(self,n):
    sub = n.subnet(0,1)

  def test_inference(self,n):
    # TODO: after adding initializer, test inference result
    pass
