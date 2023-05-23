from typing import List, Union
import re
import traceback
import numpy as np
from pysindy import ConcatLibrary, CustomLibrary, PolynomialLibrary, SINDy
from symbolicregression.model.mixins import PredictionIntegrationMixin, BatchMixin

__all__ = ("SINDyWrapper", "PolynomialSINDy", "create_library")


class SINDyWrapper(SINDy, BatchMixin, PredictionIntegrationMixin):
    
    """SINDy with default values. You only need to set the names of variables."""
    
    def __init__(
        self, 
        feature_names: List[str], 
        feature_library = None, 
        optimizer = None, 
        debug: bool = False, 
        *args, 
        **kwargs
    ):
        # The default library consists of polynomials of degree 2.
        super().__init__(
            feature_names=feature_names, feature_library=feature_library, optimizer=optimizer, *args, **kwargs,
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
    ):
        
        if isinstance(trajectories, List) and not average_trajectories:
            # we have multiple trajectories but do not want to average
            return super().fit_all(times, trajectories, average_trajectories=False)
        try:
            super().fit(trajectories, t=times, multiple_trajectories=average_trajectories, quiet=not self.debug)
            return {0: [" | ".join([self._format_equation(eq) for eq in self.equations()])]}
        except Exception as e:
            print(traceback.format_exc())
            return None
        
        
class PolynomialSINDy(SINDyWrapper):
    
    """SINDy with polynomial library of custom degree."""
    
    def __init__(self, feature_names: List[str], degree: int, debug: bool=False, *args, **kwargs):
        super().__init__(
            feature_names=feature_names,
            feature_library = PolynomialLibrary(
                degree=degree, 
                include_interaction=True, 
                include_bias=True,
            ),
            *args, 
            **kwargs,
        )
        
        
def _logarithm_with_error(self, x):
    y = np.log(x)
    if np.any(~np.isfinite(y)):
        raise ValueError("log(x) is not finite")
    return y

def _exponential_with_error(self, x):
    y = np.exp(x)
    if np.any(~np.isfinite(y)):
        raise ValueError("exp(x) is not finite")
    return y

def _sqrt_with_error(self, x):
    y = np.sqrt(x)
    if np.any(~np.isfinite(y)):
        raise ValueError("sqrt(x) is not finite")
    return y

def _one_over_x_with_error(self, x):
    y = 1.0 / x
    if np.any(~np.isfinite(y)):
        raise ValueError("1/x is not finite")
    return y

        
def create_library(degree, functions: List=["sin", "cos", "exp", "log", "sqrt", "one_over_x"]):
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
    used_funcs = {}
    used_names = {}   
    for f in functions:
        used_funcs[f] = _basis_functions[f]
        used_names[f] = _basis_function_names[f]
        
    poly_lib = PolynomialLibrary(
        degree=degree, 
        include_interaction=True, 
        include_bias=True
    )
    custom_lib = CustomLibrary(
        library_functions=list(used_funcs.values()),
        function_names=list(used_names.values()),
    )
    return ConcatLibrary([poly_lib, custom_lib])