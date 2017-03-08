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
            print "The Graph of the netobj"
            print netobj.graph
            print "The Graph of the regulization"
            print netobj.regularizer.compute().graph
            print "The Graph of the parameters : "
            for (name,elem) in netobj.variables.iteritems():
                print name
                print elem.graph
            print netobj.regularizer.compute()
        assert isinstance(netobj.regularizer.compute(),tf.Tensor)
