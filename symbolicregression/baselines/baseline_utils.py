from sklearn.metrics import r2_score
from numpy import ndarray

__all__ = ("variance_weighted_r2_score", )

def variance_weighted_r2_score(y_true: ndarray, y_pred: ndarray) -> float:
    return r2_score(y_true=y_true, y_pred=y_pred, multioutput="variance_weighted")