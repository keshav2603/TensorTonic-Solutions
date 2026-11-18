import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """

    w=np.zeros(X.shape[1])
    b=np.zeros(1)
    n=X.shape[0]

    for step in range(steps):
        #prediction 
        z=X@w+b
        # activation 
        a=_sigmoid(z)

        loss=-np.mean(y*np.log(a)+(1-y)*(np.log(1-a)))
        
        error = a - y

        dw = (X.T @ error) / n
        db = np.mean(error)

        # Parameter updates
        w -= lr * dw
        b -= lr * db


    return w,b