from typing import Union, List
from tqdm.auto import tqdm
from collections import defaultdict
import traceback
import torch
import numpy as np
from pysindy.differentiation import FiniteDifference
from symbolicregression.envs.generators import integrate_ode

__all__ = ("BatchMixin", "FiniteDifferenceMixin", "PredictionIntegrationMixin",)

class BatchMixin:
    
    def fit_all(
        self,
        times,
        trajectories,
        *args, 
        **kwargs,
    ):
        assert isinstance(trajectories, List)
        predictions_per_equation = defaultdict(list)
        for trajectory_i, (trajectory, time) in tqdm(
            enumerate(zip(trajectories, times)), total=len(trajectories)
        ):
            try:
                candidates = self.fit(times=time, trajectories=trajectory, *args, **kwargs)                    
                predictions_per_equation[trajectory_i].extend(candidates[0])
            except Exception as e:
                print(traceback.format_exc())
        return dict(predictions_per_equation)

class FiniteDifferenceMixin:
    
    def approximate_derivative(
        self, 
        trajectory: np.ndarray, 
        times: np.ndarray,
        finite_difference_order: int = 7,
        smoother_window_length: Union[None, int] = None,
    ) -> np.ndarray:
        
        assert len(times.shape) == 1, f"{times.shape}"
        assert times.shape[0] == trajectory.shape[0], f"{times.shape} vs {trajectory.shape}"
        if smoother_window_length is None:
            fd = FiniteDifference(order=finite_difference_order)
        else:
            fd = SmoothedFiniteDifference(
                order=finite_difference_order,
                smoother_kws={'window_length': smoother_window_length},
            )
        return fd._differentiate(trajectory, times)
    
    
class PredictionIntegrationMixin:
    
    @torch.no_grad()
    def integrate_prediction(self, times, y0, prediction=None, ode_integrator=None):   
        
        print("prediction", prediction)
        
        if prediction is None:
            return None
        # integrate the ODE
        if ode_integrator is not None:
            # default to passed argument
            pass
        elif hasattr(self, "params") and self.params:
            ode_integrator = self.params.ode_integrator
        else:
            ode_integrator = "solve_ivp"
        trajectory = integrate_ode(y0, times, prediction, ode_integrator=ode_integrator)
        
        return trajectory