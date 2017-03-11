from __future__ import division
import tensorflow as tf
import numpy as np
from data_tool import *
import tfs
import os

class DataSubset(object):
  def __init__(self,
               data=None,
               labels=None
  ):
    """Construct a DataSubset.
    The first dimension should be sample id.
    """
    if data is not None:
      assert isinstance(data,np.ndarray)
    if labels is not None:
      assert isinstance(labels,np.ndarray)
    if data is not None:
      self._num_examples = data.shape[0]
    else:
      self._num_examples = 0

    self._data = data
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._in_cv = False

  def cross_validation_loop(self,n_fold):
    """return a iterator
    each time return a split for cross validation
    """
    self._in_cv = True
    n = self.num_examples
    curs = [(n//n_fold)*i for i in range(n_fold)]
    curs.append(n)
    for i in range(n_fold):
      val_idx = np.array([False]*self.num_examples)
      val_idx[curs[i]:curs[i+1]]=True
      train_idx = ~val_idx

      trX,trY,teX,teY=None,None,None,None
      if self.data is not None:
        trX = self.data[train_idx]
        teX = self.data[val_idx]
      if self.labels is not None:
        trY = self.labels[train_idx]
        teY = self.labels[val_idx]

      train = DataSubset(trX,trY)
      val = DataSubset(teX,teY)
      yield train,val
    self._in_cv = False

  @property
  def shape(self):
    if isinstance(self._data,tf.Tensor):
      return self._data.get_shape().as_list()
    return self._data.shape

  def __len__(self):
    return self.num_examples

  @property
  def data(self):
    return self._data

  @data.setter
  def data(self,_data):
    if self._in_cv:
      raise RuntimeError("Cannot set data during cross validation")
    self._data=_data

  @property
  def labels(self):
    return self._labels

  @labels.setter
  def labels(self,_labels):
    if self._in_cv:
      raise RuntimeError("Cannot set labels during cross validation")
    self._labels=_labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def shuffle(self):
    perm = np.arange(self.num_examples)
    np.random.shuffle(perm)
    self.data = self.data[perm]
    self.labels = self.labels[perm]

  def next_batch(self, batch_size, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    assert batch_size < self.num_examples
    start = self._index_in_epoch

    if start + batch_size > self._num_examples:
      self._epochs_completed += 1
      rest_num_examples = self._num_examples - start
      data_rest_part = self._data[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      if shuffle:
        self.shuffle()
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

# must import here, because LabelBinarizer need DataSubset
from tfs.data_processor import LabelBinarizer
class Dataset(object):
  def __init__(self,data_dir=None):
    if data_dir:
      self.data_dir = data_dir
    else:
      self.data_dir = tfs.config.dataset.getdir(self)
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
    tfs.config.dataset.setdir(self,_dir)
    self._data_dir = _dir


