from typing import Dict, List, Union
from ffx import FFXRegressor
from symbolicregression.model.mixins import (
    BatchMixin, PredictionIntegrationMixin, FiniteDifferenceMixin, MultiDimMixin, SympyMixin,
)

import re
import numpy as np

__all__ = ("FFXWrapper",)

# TODO: restrict function class, e.g. no "max", log10?

class FFXWrapper(
    FFXRegressor, BatchMixin, FiniteDifferenceMixin, MultiDimMixin, PredictionIntegrationMixin, SympyMixin,
):
    def fit(
        self, 
        times: Union[List[np.ndarray], np.ndarray], 
        trajectories: Union[List, np.ndarray],
        derivatives: Union[None, np.ndarray] = None,
        *args, 
        **kwargs,
    ) -> Dict[int, List[str]]:  
        if isinstance(trajectories, List):
            return self.fit_all(times=times, trajectories=trajectories)
        if derivatives is None:
            derivatives = self.approximate_derivative(trajectory=trajectories, times=times).squeeze()
            if len(derivatives.shape) > 1:
                return self.fit_components(times, trajectories, derivatives, args, kwargs)
        super().fit(X=trajectories, y=derivatives, **kwargs)
        return {0: self._clean_equation(self._get_equations())}
    
    def _get_equations(self) -> List[str]:
        return [m.str2() for m in self.models_[::-1]]
    
    def _clean_equation(self, eqs: List[str]):
        return [re.sub(pattern=r"(X)(\d+)", repl=r"x_\2", string=eq).replace("^", "**") for eq in eqs]