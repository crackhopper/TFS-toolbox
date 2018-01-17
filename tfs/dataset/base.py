import tensorflow as tf
import numpy as np

from tfs.dataset import data_tool as dtool
from tfs.dataset.data_tool import *
import tfs.g
import os

from tfs.dataset.subset import DataSubset
from tfs.data_processor import LabelBinarizer
class Dataset(object):
  def __init__(self,data_dir=None):
    if data_dir:
      self.data_dir = data_dir
    else:
      self.data_dir = tfs.g.config.dataset.getdir(self)
    self.prepare()
    _trainX, _trainY, _testX, _testY = self.load_train_test()
    self._train = DataSubset(_trainX,_trainY)
    self._test = DataSubset(_testX,_testY)

    # one hot support
    self._one_hot_processor = LabelBinarizer()
    self._is_one_hot = False

  def to_one_hot(self):
    if not self._is_one_hot:
      self._is_one_hot = True
      self.process(self._one_hot_processor)
    return self

  def to_raw_label(self):
    if self._is_one_hot:
      self._is_one_hot = False
      self.inv_process(self._one_hot_processor)
    return self

  def load_train_test(self):
    return None,None,None,None

  def process(self,processor,save=True):
    train = processor.fit_transform(self.train)
    test = processor.transform(self.test)
    if save:
      self._train = train
      self._test = test
    return train,test
  def inv_process(self,processor,save=True):
    train = processor.inverse_transform(self.train)
    test = processor.inverse_transform(self.test)
    if save:
      self._train = train
      self._test = test
    return train,test

  @property
  def train(self):
    return self._train

  @property
  def test(self):
    return self._test

  def prepare(self):
    pass

  def data_full_path(self,basename):
    return os.path.join(self.data_dir,basename)

  @property
  def data_dir(self):
    return self._data_dir

  @data_dir.setter
  def data_dir(self,_dir):
    tfs.g.config.dataset.setdir(self,_dir)
    self._data_dir = _dir


