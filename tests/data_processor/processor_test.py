import pytest
import tensorflow as tf
import numpy as np

from tfs.dataset.skdata import *
from tfs.data_processor import StandardScaler,MinMaxScaler,Normalizer

@pytest.fixture
def data():
  return MakeBlobs(test_percent=0.3,n_samples=100)

class TestSKlearnProcessor:
  def test_standard_scaler(self,data):
    p = StandardScaler()
    d = data.train.data.copy()
    data.process(p)
    np.testing.assert_approx_equal(np.std(data.train.data),1)
    data.inv_process(p)
    np.testing.assert_array_almost_equal(d,data.train.data)


  def test_minmax_scaler(self,data):
    p = MinMaxScaler()
    d = data.train.data.copy()
    data.process(p)
    np.testing.assert_approx_equal(np.min(data.train.data),0)
    np.testing.assert_approx_equal(np.max(data.train.data),1)
    data.inv_process(p)
    np.testing.assert_array_almost_equal(d,data.train.data)

  def test_normalizer(self,data):
    p = Normalizer()
    data.process(p)
    dd = data.train.data
    norms = np.linalg.norm(dd,axis=1)
    np.testing.assert_approx_equal(np.min(norms),1)
    np.testing.assert_approx_equal(np.max(norms),1)





