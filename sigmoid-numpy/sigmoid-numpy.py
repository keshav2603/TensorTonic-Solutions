import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    # Write code here
    a=np.array(x,dtype=float)
    a=1/(1+np.exp(-a))
    return a