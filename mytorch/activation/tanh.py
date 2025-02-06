import numpy as np
from mytorch import Tensor

from mytorch.tensor import _tensor_exp, _tensor_neg, _tensor_pow


def tanh(x: Tensor) -> Tensor:
    """
    TODO: (optional) implement tanh function
    hint: you can do it using function you've implemented (not directly define grad func)
    """
    one = x.exp() - (-x).exp()
    two = x.exp() + (-x).exp()
    two = two ** -1
    return one * two
