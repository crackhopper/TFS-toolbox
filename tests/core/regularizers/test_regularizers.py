import tensorflow as tf
import numpy as np
from tfs.models import LeNet
from tfs.core.regularizers import Regularizer,L1,L2

class TestRegulization:

    def test_default(self,capsys):
        netobj=LeNet(in_shape=[1,28,28,1])
        netobj.build()
        assert netobj.regulization.graph==netobj.graph

    def test_l1(self,capsys):
        netobj=LeNet(in_shape=[1,28,28,1])
        netobj.Regularizer=L1(netobj,l1=0.01,nodes_params=[0,0,0,0.01])
        netobj.build()
        assert netobj.Regularizer.nodes_params==[0,0,0,0.01]
        assert netobj.regulization.graph==netobj.graph

    def test_l2(self,capsys):
        netobj=LeNet(in_shape=[1,28,28,1])
        netobj.Regularizer=L2(netobj,l2=0.02,nodes_params=[0,0,0,0.01])
        netobj.build()
        assert netobj.Regularizer.nodes_params==[0,0,0,0.01]
        assert netobj.regulization.graph==netobj.graph
        netobj
