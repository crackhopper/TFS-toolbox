import pytest
import tensorflow as tf
import numpy as np
from tfs.models import LeNet
from tfs.core.initializer import DefaultInitializer,AllConstantInitializer

class TestInitConstant:
  def test_print(self,capsys):
    netobj = LeNet()
    netobj.build()
    with capsys.disabled():
      print('')
      print(netobj.initializer)

    netobj = LeNet()
    netobj.initializer = AllConstantInitializer(netobj)
    netobj.build()
    with capsys.disabled():
      print('')
      print(netobj.initializer)

  def test_constant(self,capsys):
    netobj = LeNet()
    netobj.initializer = AllConstantInitializer(netobj)
    netobj.build()
    for key,v in netobj.variables.items():
      val = netobj.initializer.param.val
      val = val*np.ones(v.get_shape().as_list(),v.dtype.as_numpy_dtype)
      var = v.eval(netobj.sess)
      cmp = (var==val)
      assert cmp.all()


