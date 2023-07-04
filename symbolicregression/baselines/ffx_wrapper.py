from typing import Any, Callable, Dict, List, Union
from sklearn.metrics import r2_score
from ffx import FFXRegressor
from symbolicregression.model.mixins import (
    GridSearchMixin,
    BatchMixin,
    PredictionIntegrationMixin,
    FiniteDifferenceMixin,
    MultiDimMixin,
    SympyMixin,
)
from symbolicregression.baselines.baseline_utils import variance_weighted_r2_score
import re
import numpy as np

__all__ = ("FFXWrapper")

class FFXWrapper(
    FFXRegressor, 
    BatchMixin, 
    FiniteDifferenceMixin, 
    MultiDimMixin, 
    PredictionIntegrationMixin, 
    SympyMixin,
    GridSearchMixin,
):
    def __init__(
        self, 
        finite_difference_order: Union[None, int] = 2,
        smoother_window_length: Union[None, int] = None,
    ):
        fd_kwargs = {}
        if finite_difference_order is not None:
            fd_kwargs["finite_difference_order"] = finite_difference_order
        if smoother_window_length is not None:
            fd_kwargs["smoother_window_length"] = smoother_window_length
        FiniteDifferenceMixin.__init__(self, **fd_kwargs)
    
    def get_hyper_grid(self) -> Dict[str, List[Any]]:
        return {
            "finite_difference_order": list(set([2,3,4, self.finite_difference_order])),
            "smoother_window_length": list(set([None, 15, self.smoother_window_length])),
        }
        
    def get_n_jobs(self) -> int:
        return 6
    
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
        trajectories: Union[List, np.ndarray],
        derivatives: Union[None, np.ndarray] = None,
        *args, **kwargs, # ignored, for compatibility only
    ) -> Dict[int, List[str]]:
        if isinstance(trajectories, List):
            self.final_equations = self.fit_all(times=times, trajectories=trajectories)
            return self.final_equations
        if derivatives is None:
            derivatives = self.approximate_derivative(
                trajectory=trajectories, 
                times=times,
            ).squeeze()
            if len(derivatives.shape) > 1:
                self.final_equations = self.fit_components(
                    times=times,
                    trajectories=trajectories,
                    derivatives=derivatives
                )
                return self.final_equations
        # TODO: How to restrict function class, e.g. no "max", log10?
        super().fit(X=trajectories, y=derivatives)
        self.final_equations = self.__get_equations()
        return self.final_equations
    
    def _get_equations(self) -> Dict[int, List[str]]:
        if self.final_equations is None:
            print("Warning: model does not seem to be fitted.")
        return self.final_equations
    
    def __get_equations(self) -> Dict[int, List[str]]:
        return {0: self._clean_equation([m.str2() for m in self.models_[::-1]])}
    
    def _clean_equation(self, eqs: List[str]) -> List[str]:
        return [
            re.sub(pattern=r"(X)(\d+)", repl=r"x_\2", string=eq).replace("^", "**") 
            for eq in eqs if self._is_valid(eq)
        ]
        
    def _is_valid(self, eq) -> bool:
        # sympy and numexpr can parse log10 and floats in scientific notation, e.g. 2.00e-6
        illegal_ops = ["max", "min"]
        for op in illegal_ops:
            if op in eq:
                return False
        return True