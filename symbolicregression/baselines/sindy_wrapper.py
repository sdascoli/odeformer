from typing import Dict, List, Union
import re
import traceback
import numpy as np
from pysindy import ConcatLibrary, CustomLibrary, PolynomialLibrary, SINDy, optimizers
from symbolicregression.model.mixins import (
    BatchMixin, FiniteDifferenceMixin, PredictionIntegrationMixin,
)

__all__ = ("SINDyWrapper", "create_library")

class SINDyWrapper(SINDy, BatchMixin, FiniteDifferenceMixin, PredictionIntegrationMixin):
    
    """SINDy with default values. You only need to set the names of variables."""
    
    def __init__(
        self, 
        feature_names: List[str], 
        feature_library = None, # The default library consists of polynomials of degree 2
        optimizer = None,
        differentiation_method = None,
        polynomial_degree: Union[None, int] = None,
        functions: Union[None, List[str]] = None,
        optimizer_threshold: Union[None, float] = None,
        optimizer_alpha: Union[None, float] = None,
        optimizer_max_iter: Union[None, int] = None,
        finite_difference_order: Union[None, int] = None,
        smoother_window_length: Union[None, int] = None,
        debug: bool = False,
        *args,
        **kwargs,
    ):
        if feature_library is not None:
            assert functions is None, \
                "You can only supply feature library or functinos, not both."
            assert polynomial_degree is None, \
                "You can only supply feature library or degree, not both."
        if (functions is not None) or (polynomial_degree is not None):
            feature_library = create_library(
                degree=polynomial_degree, functions=functions,
            )
        if differentiation_method is None:
            differentiation_method = self.get_differentiation_method(
                finite_difference_order=finite_difference_order, 
                smoother_window_length=smoother_window_length,
            )
        if optimizer is None:
            optim_kwargs = {}
            if optimizer_threshold is not None:
                optim_kwargs["threshold"] = optimizer_threshold
            if optimizer_alpha is not None:
                optim_kwargs["alpha"] = optimizer_alpha
            if optimizer_max_iter is not None:
                optim_kwargs["max_iter"] = optimizer_max_iter
            optimizer = optimizers.STLSQ(**optim_kwargs)
        super().__init__(
            feature_names=feature_names,
            feature_library=feature_library,
            optimizer=optimizer,
            differentiation_method=differentiation_method,
            *args, 
            **kwargs,
        )
        self.debug = debug
    
    def _format_equation(self, expr: str):
        expr = re.sub(r"(\d+\.?\d*) (1)", repl=r"\1 * \2", string=expr)
        for var_name in self.feature_names:
            expr = re.sub(fr"(\d+\.?\d*) ({var_name})", repl=r"\1 * \2", string=expr)
        expr = expr.replace("^", "**")
        return expr
    
    def fit(
        self, 
        times: Union[List, np.ndarray], 
        trajectories: Union[List, np.ndarray], 
        average_trajectories: bool = False, 
        *args, 
        **kwargs,
    ) -> Dict[int, Union[None, List[str]]]:
        
        if isinstance(trajectories, List) and not average_trajectories:
            # we have multiple trajectories but do not want to average
            return super().fit_all(times, trajectories, average_trajectories=False)
        try:
            super().fit(trajectories, t=times, multiple_trajectories=average_trajectories, quiet=not self.debug)
            return {0: [" | ".join([self._format_equation(eq) for eq in self.equations()])]}
        except Exception as e:
            print(traceback.format_exc())
            return {0: [None]}
        
        
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