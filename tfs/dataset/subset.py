import tensorflow as tf
import numpy as np

from tfs.dataset import data_tool as dtool
from tfs.dataset.data_tool import *
import tfs.g
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
    self._in_cv = False
    self.before_iter()

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

  def split(self,percents,axis=0):
    """split dataset into several dataset:
    percent : list of float numbers, and sum(percent)<=1.  if percent much less than 1, then it would add the remainder automatically
    return : dset_1, ..., dset_n.
    """
    data_arr = dtool.split(self.data,percents,False,axis)
    label_arr = dtool.split(self.labels,percents,False,axis)
    return [DataSubset(d,l) for d,l in zip(data_arr,label_arr)]

  def feature_select(self,columns):
    data = self.data[:,columns]
    return DataSubset(data,self.labels)

  @staticmethod
  def join(list_of_dataset,axis=0):
    ds = []
    ls = []
    for d in list_of_dataset:
      ds.append(d.data)
      ls.append(d.labels)
    data = np.concatenate(ds,axis)
    labels = np.concatenate(ls,axis)
    return DataSubset(data,labels)

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

  def before_iter(self):
    self._epochs_completed=0
    self._index_in_epoch=0

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
      self.current_batch = (np.concatenate((data_rest_part, data_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0))
      return self.current_batch
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      self.current_batch = (self._data[start:end], self._labels[start:end])
      return self.current_batch
