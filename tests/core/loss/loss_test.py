import pytest
import tensorflow as tf
import numpy as np
from tfs.models import LeNet
from tfs.core.loss import Loss,CategoricalCrossentropy,DefaultLoss

class TestLoss:
    def test_loss(self,capsys):
        netobj=LeNet()
        netobj.build([1,28,28,1])
        assert netobj.true_output.get_shape().as_list()==netobj.output.get_shape().as_list()
        assert netobj.loss.get_shape().as_list()==netobj.regulization.get_shape().as_list()
        assert netobj.loss.get_shape().as_list()==netobj.objective.get_shape().as_list()
