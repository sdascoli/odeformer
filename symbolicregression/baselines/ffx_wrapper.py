from typing import List, Union
from ffx import FFXRegressor
from symbolicregression.model.mixins import (BatchMixin, 
                                             PredictionIntegrationMixin, 
                                             FiniteDifferenceMixin)

import numpy as np

__all__ = ("FFXWrapper",)

class FFXWrapper(FFXRegressor, BatchMixin, PredictionIntegrationMixin, FiniteDifferenceMixin):
    
    # TODO: Does not support vector-values functions.
    
    def fit(
        self, 
        times: Union[List[np.ndarray], np.ndarray], 
        trajectories: Union[List, np.ndarray], 
        **kwargs,
    ):  
        if isinstance(trajectories, List):
            self.fit_all(times=times, trajectories=trajectories)
        y = self.approximate_derivative(trajectory=trajectories, t=times)
        super().fit(X=trajectories, y=y, **kwargs)
        return self._get_equations()
    
    def _get_equations(self):
        return [m.str2() for m in self.models_[::-1]]