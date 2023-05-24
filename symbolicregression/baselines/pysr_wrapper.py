from typing import Dict, List, Union
import numpy as np
import pandas as pd
import itertools
import tempfile
from pysr import PySRRegressor
from symbolicregression.model.mixins import (
    BatchMixin, 
    PredictionIntegrationMixin, 
    FiniteDifferenceMixin,
)

class PySRWrapper(PySRRegressor, BatchMixin, PredictionIntegrationMixin, FiniteDifferenceMixin):

    """Documentation of PySR: https://astroautomata.com/PySR/api/"""

    def __init__(
        self, 
        feature_names: Union[None, List[str]] = None, 
        niterations: int = 40,
        binary_operators: Union[None, List[str]] = None,
        unary_operators: Union[None, List[str]] = None,
        loss: str = "loss(x, y) = (x - y)^2",
        procs: int = 1,
        equation_file: Union[None, str] = "./pysr_hof.csv",
        *args, 
        **kwargs,
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
            *args, 
            **kwargs,
        )
        self.feature_names = feature_names
        
    def fit(
        self, 
        times: Union[List, np.ndarray], 
        trajectories: Union[List, np.ndarray],
        finite_difference_order: Union[None, int] = None,
        smoother_window_length: Union[None, int] = None,
        *args, 
        **kwargs,
    ) -> Dict[int, Union[None, List[str]]]:
        if isinstance(trajectories, List):
            return self.fit_all(
                times=times, 
                trajectories=trajectories,
                finite_difference_order=finite_difference_order,
                smoother_window_length=smoother_window_length,
            )
        if self.feature_names is None:
            feature_names = [f"x_{i}" for i in range(trajectories.shape[1])]
        else:
            feature_names = self.feature_names
        super().fit(
            X=trajectories, 
            y=self.approximate_derivative(
                trajectory=trajectories, 
                times=times,
                finite_difference_order=finite_difference_order,
                smoother_window_length=smoother_window_length,
            ).squeeze(),
            variable_names=feature_names,
        )
        return self._hof_to_equations(self.get_hof())
        
    def _hof_to_equations(
        self,
        hof: Union[List, pd.DataFrame],
        by: str = "score",
    ) -> Dict[int, Union[str, List[str]]]:
        
        if isinstance(hof, List):
            # We have a list of candidates per dimension and combine them via a Cartesian product.
            # The first returned equation in the result correspond to the pair of best 
            # equations per dimension, for subsequent returned equations the order is arbitrary.
            eqs = []
            for hof_i in hof: # iter across dimensions of ODE system
                eqs.append(list(self._hof_to_equations(hof_i).values())[0]) # List[List[str]]
            return {0: list(" | ".join(vs) for vs in itertools.product(*eqs))}
        return {0: hof.sort_values(by=by, ascending=False).sympy_format.apply(str).values.tolist()}
