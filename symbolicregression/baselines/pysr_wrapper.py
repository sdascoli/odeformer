from typing import Callable, Dict, List, Union
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import itertools
import tempfile
from pysr import PySRRegressor
from symbolicregression.model.mixins import (
    GridSearchMixin,
    BatchMixin, 
    PredictionIntegrationMixin, 
    FiniteDifferenceMixin,
)
from symbolicregression.baselines.baseline_utils import variance_weighted_r2_score

class PySRWrapper(
    PySRRegressor, BatchMixin, PredictionIntegrationMixin, FiniteDifferenceMixin, GridSearchMixin,
):

    """Documentation of PySR: https://astroautomata.com/PySR/api/"""

    def __init__(
        self, 
        feature_names: Union[None, List[str]] = None,
        finite_difference_order: Union[None, int] = None,
        smoother_window_length: Union[None, int] = None,
        niterations: int = 40,
        binary_operators: Union[None, List[str]] = None,
        unary_operators: Union[None, List[str]] = None,
        loss: str = "loss(x, y) = (x - y)^2",
        procs: int = 1,
        equation_file: Union[None, str] = "./pysr_hof.csv",
    ):
        if binary_operators is None:
            binary_operators = ["plus", "sub", "mult", "pow", "div"] 
        if unary_operators is None:
            unary_operators=["cos", "exp", "sin", "neg", "log", "sqrt",]
        super().__init__(
            niterations=niterations,
            binary_operators=binary_operators, 
            unary_operators=unary_operators,
            loss=loss,
            procs=procs,
            equation_file=equation_file,
        )
        self.feature_names = feature_names
        self.finite_difference_order = finite_difference_order
        self.smoother_window_length = smoother_window_length
        
    def get_hyper_grid(self) -> Dict:
        return {
            "finite_difference_order": [3],
            "smoother_window_length": [None],
        }
        
    def score(
        self,
        times: np.ndarray,
        trajectories: np.ndarray,
        metric: Callable = variance_weighted_r2_score
    ) -> float:
        try:
            candidates = self._get_equations()
            assert len(candidates) > 0, candidates
            pred_trajectory = self.integrate_prediction(
                times, y0=trajectories[0], prediction=candidates[0][0]
            )
            assert pred_trajectory is not None, f"pred_trajectory is None."
            return metric(trajectories, pred_trajectory)
        except AssertionError as e:
            return np.nan
        
    def fit(
        self, 
        times: Union[List[np.ndarray], np.ndarray], 
        trajectories: Union[List[np.ndarray], np.ndarray],
    ) -> Dict[int, Union[None, List[str]]]:
        if isinstance(trajectories, List):
            return self.fit_all(times=times, trajectories=trajectories)
        if self.feature_names is None:
            feature_names = [f"x_{i}" for i in range(trajectories.shape[1])]
        else:
            feature_names = self.feature_names
        super().fit(
            X=trajectories, 
            y=self.approximate_derivative(
                trajectory=trajectories, 
                times=times
            ).squeeze(),
            variable_names=feature_names,
        )
        return self._get_equations()
        
    def _get_equations(
        self,
        hof: Union[None, pd.DataFrame] = None,
        by: str = "score"
    ) -> Dict[int, List[str]]:
        if hof is None:
            hof = self.get_hof()
        if isinstance(hof, List):
            # We have a list of candidates per dimension and combine them via a Cartesian product.
            # The first returned equation in the result correspond to the pair of best 
            # equations per dimension, for subsequent returned equations the order is arbitrary.
            eqs = []
            for hof_i in hof: # iter across dimensions of ODE system
                eqs.append(list(self._get_equations(hof_i).values())[0]) # List[List[str]]
            return {0: list(" | ".join(vs) for vs in itertools.product(*eqs))}
        return {0: hof.sort_values(by=by, ascending=False).sympy_format.apply(str).values.tolist()}
