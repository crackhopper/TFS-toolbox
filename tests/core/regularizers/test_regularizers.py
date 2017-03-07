import tensorflow as tf
import numpy as np
from tfs.models import LeNet
from tfs.core.regularizers import Regularizer,L1

class TestRegulization:
    def test_regulization(self,capsys):
        netobj=LeNet()
        netobj.build()
        netobj.regularizer=L1(netobj,l1=0.02)
        assert netobj.regularizer.param.l1==0.02
        with capsys.disabled():
            print ""
            print netobj.graph
            print netobj.regularizer.compute().graph
        assert isinstance(netobj.regularizer.compute(),tf.Tensor)
