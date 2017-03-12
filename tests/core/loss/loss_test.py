import pytest
import tensorflow as tf
import numpy as np
from tfs.models import LeNet
from tfs.core.loss import Loss,CategoricalCrossentropy,DefaultLoss,BinaryCrossentripy,SquareError
from tfs.network.base import CustomNetwork

class TenoutNet(CustomNetwork):
  def setup(self):
    """http://ethereon.github.io/netscope/#/gist/87a0a390cff3332b476a
    Note : lr_mult parameter is different.
    """
    self.default_in_shape = [None,28,28,1]
    (self.net_def
     .conv2d([5,5],20,[1,1],activation=None,name='conv1',padding='VALID')
     .maxpool([2,2],[2,2],name='pool1',padding='VALID')
     .conv2d([5,5],50,[1,1],name='conv2',padding='VALID')
     .maxpool([2,2],[2,2],name='pool2',padding='VALID')
     .fc(500,name='ip1')
     .fc(10, activation=None,name='ip2')
    )

class TwooutNet(CustomNetwork):
  def setup(self):
    """http://ethereon.github.io/netscope/#/gist/87a0a390cff3332b476a
    Note : lr_mult parameter is different.
    """
    self.default_in_shape = [None,28,28,1]
    (self.net_def
     .conv2d([5,5],20,[1,1],activation=None,name='conv1',padding='VALID')
     .maxpool([2,2],[2,2],name='pool1',padding='VALID')
     .conv2d([5,5],50,[1,1],name='conv2',padding='VALID')
     .maxpool([2,2],[2,2],name='pool2',padding='VALID')
     .fc(500,name='ip1')
     .fc(2, activation=None,name='ip2')
    )
class TwooutNetwithsoftmax(CustomNetwork):
  def setup(self):
    """http://ethereon.github.io/netscope/#/gist/87a0a390cff3332b476a
    Note : lr_mult parameter is different.
    """
    self.default_in_shape = [None,28,28,1]
    (self.net_def
     .conv2d([5,5],20,[1,1],activation=None,name='conv1',padding='VALID')
     .maxpool([2,2],[2,2],name='pool1',padding='VALID')
     .conv2d([5,5],50,[1,1],name='conv2',padding='VALID')
     .maxpool([2,2],[2,2],name='pool2',padding='VALID')
     .fc(500,name='ip1')
     .fc(2, activation=None,name='ip2')
     .softmax(name='prob')
    )


class TestLoss:
    def test_categoricalcrossentropy(self,capsys):
        netobj=LeNet()
        netobj.build([100,28,28,1])
        assert netobj.true_output.get_shape().as_list()==netobj.output.get_shape().as_list()
        assert netobj.loss.get_shape().as_list()==[100]
        netobj=TenoutNet()
        netobj.build([100,28,28,1])
        assert netobj.true_output.get_shape().as_list()==netobj.output.get_shape().as_list()
        assert netobj.loss.get_shape().as_list()==[100]

    def test_binarycrossentropy(self,capsys):
        netobj=TwooutNetwithsoftmax()
        netobj.build([100,28,28,1])
        assert netobj.true_output.get_shape().as_list()==netobj.output.get_shape().as_list()
        assert netobj.loss.get_shape().as_list()==[100]
        netobj=TwooutNet()
        netobj.build([100,28,28,1])
        assert netobj.true_output.get_shape().as_list()==netobj.output.get_shape().as_list()
        assert netobj.loss.get_shape().as_list()==[100]
    def test_squareerror(self,capsys):
        netobj=LeNet()
        netobj.Loss=SquareError(netobj)
        netobj.build([100,28,28,1])
        assert netobj.true_output.get_shape().as_list()==netobj.output.get_shape().as_list()
        with capsys.disabled():
            print ""
            print "The shape of the loss:"
            print netobj.loss.get_shape().as_list()
