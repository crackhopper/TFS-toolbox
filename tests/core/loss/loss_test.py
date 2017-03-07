import pytest
import tensorflow as tf
import numpy as np
from tfs.models import LeNet
from tfs.core.loss import Objective,categorical_crossentropy

class TestLoss:
    def test_loss(self,capsys):
        netobj=LeNet()
        netobj.build()
        netobj.objective=categorical_crossentropy(netobj,with_logits=True)
        with capsys.disabled():
            print ""
            print netobj.objective.param
        assert isinstance(netobj.objective.compute(),tf.Tensor)
