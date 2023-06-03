from typing import Dict, List, Union
from typing_extensions import Literal
from tqdm.auto import tqdm
from collections import defaultdict
import sympy
import torch
import numpy as np
import itertools
import traceback
from pysindy.differentiation import FiniteDifference, SmoothedFiniteDifference
from symbolicregression.envs.generators import integrate_ode

__all__ = ("BatchMixin", "FiniteDifferenceMixin", "PredictionIntegrationMixin",)
            

class MultiDimMixin:
    """
    Mixin for vector valued output. Each component of the output is fit individually.
    """
    def fit_components(
        self, 
        times: np.ndarray, 
        trajectories: np.ndarray, 
        derivatives: np.ndarray, 
        *args, 
        **kwargs,
    ) -> Dict[int, List[str]]:
        assert len(trajectories.shape) == 2, f"len(times.shape) == {len(times.shape)}"
        predictions_per_component: List[List[str]] = []
        for _deriv in derivatives.T:
            # supply all trajectories as input but only single output component to learn a func R^n -> R
            predictions_per_component.append(self.fit(times, trajectories, _deriv, args, kwargs,)[0])
        return {0: list(" | ".join(vs) for vs in itertools.product(*predictions_per_component))}
        


class BatchMixin:
    """
    This class lets models iteratively process a list of trajectories.
    The base model needs to implement a fit() method.
    
    Methods
    -------
    fit_all(times, trajectories):
        Calls model.fit for every element in times and trajectories.
    """
    def fit_all(
        self,
        times: np.ndarray,
        trajectories: np.ndarray,
        *args, 
        **kwargs,
    ) -> Dict[int, Union[None, List[str]]]:

        assert isinstance(trajectories, List)
        predictions_per_equation = defaultdict(list)
        for trajectory_i, (trajectory, time) in tqdm(
            enumerate(zip(trajectories, times)), total=len(trajectories)
        ):
            try:
                candidates: Dict[int, List[str, None]] = self.fit(
                    times=time, trajectories=trajectory, *args, **kwargs
                )
                predictions_per_equation[trajectory_i].extend(candidates[0])
            except Exception as e:
                print(traceback.format_exc())
                predictions_per_equation[trajectory_i].append(None)
        return dict(predictions_per_equation)

class FiniteDifferenceMixin:
    """
    A class to approximate derivatives via pysindy's implementation.
    
    Methods
    -------
    approximate_derivative(trajectory, times, finite_difference_order, smoother_window_length):
        Approximate derivatives.
        
    get_differentiation_method(finite_difference_order, smoother_window_length):
        Create differentiation method instance.
    """
    def approximate_derivative(
        self,
        trajectory: np.ndarray,
        times: np.ndarray,
        finite_difference_order: Union[None, int] = 2,
        smoother_window_length: Union[None, int] = None,
    ) -> np.ndarray:
        times = times.squeeze()
        assert len(times.shape) == 1 or np.all(times.shape == trajectory.shape), f"{times.shape} vs {trajectory.shape}"
        
        fd = self.get_differentiation_method(
            finite_difference_order = finite_difference_order, 
            smoother_window_length = smoother_window_length,
        )
        return fd._differentiate(trajectory, times)
    
    def get_differentiation_method(
        self, 
        finite_difference_order: Union[None, int] = None, 
        smoother_window_length: Union[None, int] = None,
    ):
        if finite_difference_order is None:
            finite_difference_order = 2
        if smoother_window_length is None:
            return FiniteDifference(order=finite_difference_order)
        return SmoothedFiniteDifference(
            order=finite_difference_order,
            smoother_kws={'window_length': smoother_window_length},
        )

class PredictionIntegrationMixin:
    """
    Implements integration of a predicted function.
    
    Methods
    -------
    integrate_prediction(times, y0, prediction, ode_integrator):
        Integrate the prediction using ode_integrator over times interval, starting with y0.
    """
    @torch.no_grad()
    def integrate_prediction(
        self, 
        times: np.ndarray,
        y0: np.ndarray, 
        prediction = None, 
        ode_integrator: Union[None, str] = None,
    ) -> np.ndarray:
        
        _default_ode_integrator = "solve_ivp"
        if prediction is None:
            return None
        # integrate the ODE
        if ode_integrator is not None:
            # default to passed argument
            pass
        elif hasattr(self, "params") and self.params:
            try:
                if isinstance(self.params, Dict):
                    ode_integrator = self.params["ode_integrator"]
                else:
                    ode_integrator = self.params.ode_integrator
            except Exception:
                ode_integrator = _default_ode_integrator
        else:
            ode_integrator = _default_ode_integrator
        return integrate_ode(y0, times, prediction, ode_integrator=ode_integrator)
