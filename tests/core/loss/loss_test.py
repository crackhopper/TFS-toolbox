import pytest
import tensorflow as tf
import numpy as np
from tfs.models import LeNet
from tfs.core.loss import Loss,DefaultLoss,CrossEntropyByLogitLabel,SquareError
from tfs.network.base import CustomNetwork
from tfs.core.regularizers import Regularizer,L1,L2

from tfs.dataset.predefined import Mnist
mnist=Mnist().to_one_hot()
train_data=mnist.train
X=train_data.data
Y=train_data.labels

class TenoutNet(CustomNetwork):
  def setup(self):
    """http://ethereon.github.io/netscope/#/gist/87a0a390cff3332b476a
    Note : lr_mult parameter is different.
    """
    self.default_in_shape = [None,28,28,1]
    (self
     .conv2d([5,5],20,[1,1],activation=None,name='conv1',padding='VALID')
     .maxpool([2,2],[2,2],name='pool1',padding='VALID')
     .conv2d([5,5],50,[1,1],name='conv2',padding='VALID')
     .maxpool([2,2],[2,2],name='pool2',padding='VALID')
     .fc(500,name='ip1')
     .fc(10, activation=None,name='ip2')
    )
    self.loss_input_layer_name='ip2'

class TwooutNet(CustomNetwork):
  def setup(self):
    """http://ethereon.github.io/netscope/#/gist/87a0a390cff3332b476a
    Note : lr_mult parameter is different.
    """
    self.default_in_shape = [None,28,28,1]
    (self
     .conv2d([5,5],20,[1,1],activation=None,name='conv1',padding='VALID')
     .maxpool([2,2],[2,2],name='pool1',padding='VALID')
     .conv2d([5,5],50,[1,1],name='conv2',padding='VALID')
     .maxpool([2,2],[2,2],name='pool2',padding='VALID')
     .fc(500,name='ip1')
     .fc(2, activation=None,name='ip2')
    )
    self.loss_input_layer_name='ip2'

class TwooutNetwithsoftmax(CustomNetwork):
  def setup(self):
    """http://ethereon.github.io/netscope/#/gist/87a0a390cff3332b476a
    Note : lr_mult parameter is different.
    """
    self.default_in_shape = [None,28,28,1]
    (self
     .conv2d([5,5],20,[1,1],activation=None,name='conv1',padding='VALID')
     .maxpool([2,2],[2,2],name='pool1',padding='VALID')
     .conv2d([5,5],50,[1,1],name='conv2',padding='VALID')
     .maxpool([2,2],[2,2],name='pool2',padding='VALID')
     .fc(500,name='ip1')
     .fc(2, activation=None,name='ip2')
     .softmax(name='prob')
    )
    self.loss_input_layer_name='ip2'


class TestLoss:
    def test_categoricalcrossentropy(self,capsys):
        netobj=LeNet()
        netobj.build([100,28,28,1])
        assert netobj.true_output.get_shape().as_list()==netobj.output.get_shape().as_list()
        assert netobj.loss.get_shape().as_list()==[]
        loss=netobj.loss.eval(session=netobj.sess,feed_dict= {netobj.input:X[:100].reshape((100,28,28,1)),netobj.true_output: Y[:100]})
        with capsys.disabled():
          print("")
          print(loss)

        netobj=TenoutNet()
        netobj.build([100,28,28,1])
        assert netobj.true_output.get_shape().as_list()==netobj.output.get_shape().as_list()
        assert netobj.loss.get_shape().as_list()==[]
        loss=netobj.loss.eval(session=netobj.sess,feed_dict= {netobj.input:X[:100].reshape((100,28,28,1)),netobj.true_output: Y[:100]})
        with capsys.disabled():
          print("")
          print(loss)
    def test_regulization(self,capsys):
      netobj=LeNet()
      netobj.Regularizer=L2(netobj,l2=0.2)
      netobj.build([100,28,28,1])
      loss=netobj.loss.eval(session=netobj.sess,feed_dict={netobj.input: X[:100].reshape((100,28,28,1)),netobj.true_output: Y[:100]})
      with capsys.disabled():
        print("")
        print(loss)

    def test_binarycrossentropy(self,capsys):
        netobj=TwooutNetwithsoftmax()
        netobj.build([100,28,28,1])
        assert netobj.true_output.get_shape().as_list()==netobj.output.get_shape().as_list()
        assert netobj.loss.get_shape().as_list()==[]
        # TODO: dataset that just have 2 class.
        netobj=TwooutNet()
        netobj.build([100,28,28,1])
        assert netobj.true_output.get_shape().as_list()==netobj.output.get_shape().as_list()
        assert netobj.loss.get_shape().as_list()==[]
    def test_squareerror(self,capsys):
        netobj=LeNet()
        netobj.Loss=SquareError(netobj)
        netobj.build([100,28,28,1])
        assert netobj.true_output.get_shape().as_list()==netobj.output.get_shape().as_list()
        loss=netobj.loss.eval(session=netobj.sess,feed_dict={netobj.input:X[:100].reshape((100,28,28,1)),netobj.true_output:Y[:100]})
        with capsys.disabled():
          print("")
          print(loss)
