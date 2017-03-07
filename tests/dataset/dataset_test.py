import pytest
import tensorflow as tf
import numpy as np

from tfs.dataset import *
import tfs.dataset.data_tool as dtool
import shutil

@pytest.fixture
def data():
  return MakeBlobs(test_percent=0.3,n_samples=100)

class TestDataTool:
  def test_split(self,capsys):
    d=np.arange(10)
    ds=dtool.split_n(d,5)
    assert len(ds)==5
    for dd in ds:
      assert len(dd)==2

class TestDataset:
  def test_dataset(self,data):
    assert data.train.shape[0]==70

  def test_cv(self,data):
    i=0
    for train,test in data.train.train_test_for_cv(7):
      i=i+1
      assert train.shape[0]==60
      assert test.shape[0]==10
    assert i==7

  def test_batch(self,data):
    first_data=data.train.data[0]
    for i in range(8):
      x,y = data.train.next_batch(10,False)
      assert x.shape[0]==10 and y.shape[0]==10
    assert data.train.epochs_completed==1
    assert (x[0] == first_data).all()

  def test_cifar10(self,capsys):
    with capsys.disabled():
      data = Cifar10()

  def test_mnist(self,capsys):
    with capsys.disabled():
      data = Mnist()



