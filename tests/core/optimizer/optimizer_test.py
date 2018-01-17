import pytest
import tensorflow as tf
import numpy as np
from tfs.models import LeNet

from tfs.dataset.predefined import Mnist
import tfs.g

@pytest.fixture
def data():
  mnist=Mnist().to_one_hot()
  return mnist

@pytest.fixture
def net():
  n = LeNet()
  n.build()
  return n

class TestOptimizer:
  def test_fit(self,data,net,capsys):
    if tfs.g.config['test']['optimizer']:
      with capsys.disabled():
        net.monitor['default'].interval=1
        net.fit(data,batch_size=300,n_epoch=1,max_step=20)
