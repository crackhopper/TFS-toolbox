from __future__ import division
import tensorflow as tf
import numpy as np
from data_tool import *
import tfs
import os

class DataSubset(object):
  def __init__(self,
               data,
               labels
  ):
    """Construct a DataSubset.
    The first dimension should be sample id.
    """
    self._num_examples = data.shape[0]

    self._data = data
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  def train_test_for_cv(self,n_fold):
    """return a iterator
    each time return a split for cross validation
    """
    n = self.num_examples
    curs = [(n//n_fold)*i for i in range(n_fold)]
    curs.append(n)
    for i in range(n_fold):
      test_idx = np.array([False]*self.num_examples)
      test_idx[curs[i]:curs[i+1]]=True
      train_idx = ~test_idx
      train = DataSubset(
        self.data[train_idx],
        self.labels[train_idx]
        )
      test = DataSubset(
        self.data[test_idx],
        self.labels[test_idx]
        )
      yield train,test

  @property
  def shape(self):
    if isinstance(self._data,tf.Tensor):
      return self._data.get_shape().as_list()
    return self._data.shape

  @property
  def data(self):
    return self._data

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, shuffle=True):
    """Return the next `batch_size` examples from this data set."""

    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._data = self.data[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      data_rest_part = self._data[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._data = self.data[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      data_new_part = self._data[start:end]
      labels_new_part = self._labels[start:end]
      return np.concatenate((data_rest_part, data_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._data[start:end], self._labels[start:end]

class Dataset(object):
  def __init__(self,data_dir=None):
    if data_dir:
      self.data_dir = data_dir
    else:
      self.data_dir = tfs.config.dataset.getdir(self)
    self.prepare()
    self._trainX, self._trainY, self._testX, self._testY = self.load_train_test()
    self._train = None
    self._test = None

  @property
  def train(self):
    if self._train is None:
      self._train = DataSubset(self._trainX,self._trainY)
    return self._train

  @property
  def test(self):
    if self._test is None:
      self._test = DataSubset(self._testX,self._testY)
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
    tfs.config.dataset.setdir(self,_dir)
    self._data_dir = _dir

  def load_train_test(self):
    raise NotImplementedError()


