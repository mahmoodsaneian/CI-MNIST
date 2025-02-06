import numpy as np

from mytorch import Tensor
from mytorch.activation import softmax


def CategoricalCrossEntropy(preds: Tensor, label: Tensor):
    """TODO: implement Categorical Cross Entropy loss"""
    return -(label * preds.log()).sum()
