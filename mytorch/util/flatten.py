import numpy as np
from mytorch import Tensor, Dependency


def flatten(x: Tensor) -> Tensor:
    """
    TODO: implement flatten. 
    this methods transforms a n dimensional array into a flat array
    hint: use numpy flatten
    """
    # Flatten the data
    data = np.flatten(x.data)

    # Determine if gradients are required
    req_grad = x.requires_grad

    # If gradients are required, create a dependency
    if req_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad.reshape(x.shape)  # Reshape to the original shape of the input tensor

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    # Return a new tensor with the flattened data
    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)
