import numpy as np

from mytorch import Tensor


def MeanSquaredError(preds: Tensor, actual: Tensor):
    """TODO: implement Mean Squared Error loss"""
    error = preds - actual
    er2 = error ** 2
    mse = er2
    size = Tensor(np.array([er2.data.size], dtype=np.float64))
    size = size ** -1
    return mse * size
