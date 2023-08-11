from typing import List
from numpy import ndarray
from sklearn.metrics import r2_score
import json

__all__ = ("variance_weighted_r2_score", "read_pareto_front_from_json")

def variance_weighted_r2_score(y_true: ndarray, y_pred: ndarray) -> float:
    return r2_score(y_true=y_true, y_pred=y_pred, multioutput="variance_weighted")

def read_pareto_front_from_json(path: str) -> List:
    eqs = []
    with open(path, "r") as fin:
        for line in fin:
            eqs.extend(
                json.loads(("{" if l[0] != "{" else "") + l + ("}" if l[-1] != "}" else "")) 
                for l in line.split("}{")
            ) 
    return eqs