from typing import Dict, List, Union
from ffx import FFXRegressor
from symbolicregression.model.mixins import (
    BatchMixin, PredictionIntegrationMixin, FiniteDifferenceMixin, MultiDimMixin, SympyMixin,
)

import re
import numpy as np

__all__ = ("FFXWrapper",)

class FFXWrapper(
    FFXRegressor, 
    BatchMixin, 
    FiniteDifferenceMixin, 
    MultiDimMixin, 
    PredictionIntegrationMixin, 
    SympyMixin,
):
    def fit(
        self, 
        times: Union[List[np.ndarray], np.ndarray], 
        trajectories: Union[List, np.ndarray],
        derivatives: Union[None, np.ndarray] = None,
        finite_difference_order: Union[None, int] = None,
        smoother_window_length: Union[None, int] = None,
        *args, 
        **kwargs,
    ) -> Dict[int, List[str]]:
        if isinstance(trajectories, List):
            return self.fit_all(
                times=times, 
                trajectories=trajectories,
                finite_difference_order=finite_difference_order,
                smoother_window_length=smoother_window_length,
                *args, **kwargs,
            )
        if derivatives is None:
            derivatives = self.approximate_derivative(
                trajectory=trajectories, 
                times=times,
                finite_difference_order=finite_difference_order,
                smoother_window_length=smoother_window_length,
            ).squeeze()
            if len(derivatives.shape) > 1:
                return self.fit_components(
                    times, trajectories, derivatives, args, kwargs
                )
        # TODO: How to restrict function class, e.g. no "max", log10?
        super().fit(X=trajectories, y=derivatives, **kwargs)
        return {0: self._clean_equation(self._get_equations())}
    
    def _get_equations(self) -> List[str]:
        return [m.str2() for m in self.models_[::-1]]
    
    def _clean_equation(self, eqs: List[str]):
        return [re.sub(pattern=r"(X)(\d+)", repl=r"x_\2", string=eq).replace("^", "**") for eq in eqs]