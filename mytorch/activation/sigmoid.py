import numpy as np
from mytorch import Tensor, Dependency

from mytorch.tensor import _tensor_exp


def sigmoid(x: Tensor) -> Tensor:
    """
    TODO: implement sigmoid function
    hint: you can do it using function you've implemented (not directly define grad func)
    """
    one = Tensor(np.ones(x.shape))
    two = Tensor(np.ones(x.shape)) + ((-x).exp())
    two = two ** -1
    return one * two
