from typing import Any, Callable, Dict, List, Union
from sklearn.metrics import r2_score
import re
import traceback
import numpy as np
from pysindy import ConcatLibrary, CustomLibrary, PolynomialLibrary, SINDy, optimizers
from symbolicregression.model.mixins import (
    GridSearchMixin, BatchMixin, FiniteDifferenceMixin, PredictionIntegrationMixin,
)
from symbolicregression.baselines.baseline_utils import variance_weighted_r2_score

__all__ = ("SINDyWrapper", "create_library")

class SINDyWrapper(
    SINDy, BatchMixin, FiniteDifferenceMixin, PredictionIntegrationMixin, GridSearchMixin
):
    
    """SINDy with default values. You only need to set the names of variables."""
    
    def __init__(
        self, 
        polynomial_degree: Union[None, int] = None,
        functions: Union[None, List[str]] = None,
        optimizer_threshold: Union[None, float] = None,
        optimizer_alpha: Union[None, float] = None,
        optimizer_max_iter: Union[None, int] = None,
        finite_difference_order: Union[None, int] = None,
        smoother_window_length: Union[None, int] = None,
        debug: bool = False,
        grid_search_polynomial_degree: bool = False,
    ):
        fd_kwargs = {"smoother_window_length": smoother_window_length}
        if finite_difference_order is not None:
            fd_kwargs["finite_difference_order"] = finite_difference_order
        FiniteDifferenceMixin.__init__(self, **fd_kwargs)
        self.polynomial_degree = polynomial_degree
        self.functions = functions
        self.optimizer_threshold = optimizer_threshold
        self.optimizer_alpha = optimizer_alpha
        self.optimizer_max_iter = optimizer_max_iter
        self.debug = debug
        self.grid_search_polynomial_degree = grid_search_polynomial_degree
        
        feature_library = create_library(
            degree=self.polynomial_degree, functions=self.functions,
        )
        optim_kwargs = {}
        if optimizer_threshold is not None:
            optim_kwargs["threshold"] = self.optimizer_threshold
        if optimizer_alpha is not None:
            optim_kwargs["alpha"] = self.optimizer_alpha
        if optimizer_max_iter is not None:
            optim_kwargs["max_iter"] = self.optimizer_max_iter
        optimizer = optimizers.STLSQ(**optim_kwargs)
        super().__init__(
            feature_library=feature_library,
            optimizer=optimizer,
            differentiation_method=self.get_differentiation_method(),
        )
    
    def _format_equation(self, expr: str) -> str:
        # x0, x1, ... -> x_0, x_1, ...
        expr = re.sub(fr"(x)(\d)", repl=r"\1_\2", string=expr)
        # <coef> <space> 1 -> <coef> * 1
        expr = re.sub(r"(\d+\.?\d*) (1)", repl=r"\1 * \2", string=expr)
        # <coef> <space> <var> -> <coef> * <var>
        expr = re.sub(r"(\d+\.?\d*) (x_\d+)", repl=r"\1 * \2", string=expr)
        # <var> <space> <var> -> <coef> * <var>
        expr = re.sub(r"(x_\d+) (x_\d+)", repl=r"\1 * \2", string=expr)
        # python power symbol
        expr = expr.replace("^", "**")
        return expr
    
    def get_feature_names(self) -> List[str]:
        return [re.sub(fr"(x)(\d)", repl=r"\1_\2", string=name) for name in self.feature_names]

    def fit(
        self,
        times: Union[List, np.ndarray],
        trajectories: Union[List, np.ndarray],
        average_trajectories: bool = False,
    ) -> Dict[int, Union[None, List[str]]]:
        
        if isinstance(trajectories, List) and not average_trajectories:
            # we have multiple trajectories but do not want to average
            return super().fit_all(times, trajectories, average_trajectories=False)
        try:
            super().fit(trajectories, t=times, multiple_trajectories=average_trajectories, quiet=not self.debug)
            return self._get_equations()
        except Exception as e:
            print(traceback.format_exc())
            return {0: [None]}
        
    def _get_equations(self) -> Dict[int, List[Union[None, str]]]:
        return {0: [" | ".join([self._format_equation(eq) for eq in self.equations()])]}
    
    def score(
        self,
        times: np.ndarray,
        trajectories: np.ndarray,
        metric: Callable = variance_weighted_r2_score,
    ) -> float:
        return super().score(x=trajectories, t=times, metric=metric)
    
    def get_hyper_grid(self) -> Dict[str, List[Any]]:
        hparams = {
            "optimizer_threshold": list(set([0.05, 0.1, 0.15, self.optimizer_threshold])),
            "optimizer_alpha": list(set([0.025, 0.05, 0.075, self.optimizer_alpha])),
            "optimizer_max_iter": list(set([20, 100, self.optimizer_max_iter])),
            # "finite_difference_order": list(set([2, self.finite_difference_order])),
            # "smoother_window_length": list(set([None, 9, self.smoother_window_length])),
        }
        if self.grid_search_polynomial_degree:
            hparams["polynomial_degree"] = np.arange(self.polynomial_degree+1, dtype=int)
        return hparams
    
    def get_n_jobs(self) -> Union[int, None]:
        return 24

        
def _logarithm_with_error(x):
    y = np.log(x)
    if np.any(~np.isfinite(y)):
        raise ValueError("log(x) is not finite")
    return y

def _exponential_with_error(x):
    y = np.exp(x)
    if np.any(~np.isfinite(y)):
        raise ValueError("exp(x) is not finite")
    return y

def _sqrt_with_error(x):
    y = np.sqrt(x)
    if np.any(~np.isfinite(y)):
        raise ValueError("sqrt(x) is not finite")
    return y

def _one_over_x_with_error(x):
    y = 1.0 / x
    if np.any(~np.isfinite(y)):
        raise ValueError("1/x is not finite")
    return y
        
def create_library(
    degree: Union[None, int] = 3, 
    functions: Union[None, str, List[str]] = None,
):
    if functions is None:
        functions = ["sin", "cos", "exp", "log", "sqrt", "one_over_x"]
    elif isinstance(functions, str):
        functions = [functions]
    assert degree >= 0, f"Degree may not be negative but is {degree}."
    _basis_functions = {
        "sin": lambda x: np.sin(x),
        "cos": lambda x: np.cos(x),
        "exp": lambda x: _exponential_with_error(x),
        "log": lambda x: _logarithm_with_error(x),
        "sqrt": lambda x: _sqrt_with_error(x),
        "one_over_x": lambda x: _one_over_x_with_error(x), 
    }
    _basis_function_names = {
        "sin": lambda x: f"* sin({x})",
        "cos": lambda x: f"* cos({x})",
        "exp": lambda x: f"* exp({x})",
        "log": lambda x: f"* log({x})",
        "sqrt": lambda x: f"* sqrt({x})",
        "one_over_x": lambda x: f"* 1/({x})"
    }   
    
    libs = []
    if degree:
        libs.append(
            PolynomialLibrary(
                degree=degree, 
                include_interaction=True, 
                include_bias=True
            )
        )
    if functions is not None:
        used_funcs = {}
        used_names = {}   
        for f in functions:
            used_funcs[f] = _basis_functions[f]
            used_names[f] = _basis_function_names[f]
        custom_lib = CustomLibrary(
            library_functions=list(used_funcs.values()),
            function_names=list(used_names.values()),
        )
        libs.append(custom_lib)
    return ConcatLibrary(libs)