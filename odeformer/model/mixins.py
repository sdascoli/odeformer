from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable, Dict, List, Union
from typing_extensions import Literal
from tqdm.auto import tqdm
from collections import defaultdict
from sklearn.model_selection import GridSearchCV
import os
import json
import math
import sympy
import torch
import numpy as np
import itertools
import traceback
from pysindy.differentiation import FiniteDifference, SmoothedFiniteDifference
from odeformer.envs.generators import integrate_ode
from odeformer.metrics import compute_metrics

__all__ = (
    "BatchMixin", 
    "GridSearchMixin",
    "FiniteDifferenceMixin", 
    "MultiDimMixin", 
    "PredictionIntegrationMixin", 
    "SympyMixin",
)

class GridSearchMixin(ABC):
    
    @abstractmethod
    def integrate_prediction(self) -> np.ndarray:
        ...
        
    @abstractmethod
    def fit(self) -> Dict:
        ...

    @abstractmethod
    def get_hyper_grid(self) -> Dict:
        ...
    
    def get_n_jobs(self) -> Union[None, int]:
        return None
    
    def get_grid_search(
        self, 
        train_idcs: np.ndarray, 
        test_idcs: np.ndarray,
        n_jobs: int = None,
        verbose: int = 4,
    ) -> GridSearchCV:
        return GridSearchCV(
            estimator=self,
            param_grid=self.get_hyper_grid(),
            refit = True,
            cv=[(train_idcs, test_idcs)],
            verbose=verbose,
            error_score=np.nan,
            n_jobs=(self.get_n_jobs() if n_jobs is None else n_jobs),
        )
    
    def save_pareto_front(self, equations: Dict):
        with open(os.path.join(self.model_dir, self.filename_pareto_front), "a") as fout:
            json.dump(fp=fout, obj=equations)
    
    def fit_grid_search(
        self,
        times: np.ndarray,
        trajectory: np.ndarray,
    ):
        self.grid_search_is_running = True
        train_idcs = np.arange(
            int(np.floor((1-self.hyper_opt_eval_fraction)*len(times))), 
            dtype=int
        )
        test_idcs = np.arange(
            int(np.floor((1-self.hyper_opt_eval_fraction)*len(times))), 
            len(times),
            dtype=int
        )
        assert len(set(train_idcs).intersection(test_idcs)) == 0, "`train_idcs` and `test_idcs` overlap."
        model = self.get_grid_search(train_idcs, test_idcs)
        model.fit(times, trajectory)
        self.grid_search_is_running = False
        all_candidates = model.best_estimator_._get_equations()
        self.save_pareto_front(all_candidates)
        assert len(all_candidates) == 1, f"len(all_candidates) = {len(all_candidates)}"
        if len(all_candidates[0]) > 1:
            all_candidates = self.sort_candidates(
                candidates=all_candidates[0],
                trajectory=trajectory,
                times=times,
                sorting_metric=self.sorting_metric,
            )
        return all_candidates
        
    def sort_candidates(
        self,
        candidates: List[str],
        trajectory: np.ndarray,
        times: np.ndarray,
        sorting_metric: str,
    ):
        if "r2" in sorting_metric:
            descending = True
        else: 
            descending = False
        _scores = []
        for candidate in candidates:
            _score = compute_metrics(
                predicted = self.integrate_prediction(
                    times=times,
                    y0=trajectory[0],
                    prediction=candidate,
                ),
                true = trajectory, 
                metrics = sorting_metric,
            )[sorting_metric][0]
            if math.isnan(_score): 
                _score = -np.infty if descending else np.infty
            _scores.append(_score)
        sorted_idx = np.argsort(_scores)  
        if descending: sorted_idx = list(reversed(sorted_idx))
        return [candidates[i] for i in sorted_idx]
        
        

class SympyMixin:
    def to_sympy(
        self, 
        eqs: Union[str, List[str], Dict[int, List[str]]], 
        var_names: List[str], # e.g. "x_0,x_1" or ["x_0", "x_1"]
        return_type: Literal["expr", "func", "str"] = "expr",
        evaluate: bool = False
    ):
        keys, values = [], []
        if isinstance(eqs, Dict):
            for key in eqs.keys():
                keys.append(key)
                values.append(
                    self.to_sympy(eqs=eqs[key], var_names=var_names, return_type=return_type, evaluate=evaluate)
                )
            return dict(zip(keys, values))
        if isinstance(eqs, List):
            return [
                self.to_sympy(eqs=eq, var_names=var_names, return_type=return_type, evaluate=evaluate) for eq in eqs
            ]
        assert isinstance(eqs, str), f"Expected eqs to be instanceof str but found {type(eqs)}."
        symbols = []
        for var_name in var_names:
            symbols.append(sympy.Symbol(var_name, real=True, finite=True))
        
        expr = [
            sympy.parse_expr(e, evaluate=evaluate, local_dict=dict(zip(var_names, symbols))) for e in eqs.split("|")
        ]
        if return_type == "expr":
            return expr
        elif return_type == "func":
            funcs = []
            for e in expr:
                funcs.append(sympy.lambdify(",".join(var_names), e))
            
            def wrap_system(*args):
                output = []
                for f in funcs:
                    output.append(f(*args))
                return np.array(output).squeeze()
                
            return wrap_system
            
        elif return_type == "str":
            return " | ".join([str(e) for e in expr])
        else:
            raise ValueError(f"Unknown return type: {return_type}.")
    
    def simplify(
        self, 
        eqs: Union[str, List[str], Dict[int, List[str]]], 
        var_names: List[str],
        evaluate=True,
    ):
        return self.to_sympy(eqs=eqs, var_names=var_names, return_type="str", evaluate=evaluate)
    
    def lambdify(
        self, 
        eqs: Union[str, List[str], Dict[int, List[str]]], 
        var_names: List[str],
        evaluate=True,
    ):
        return self.to_sympy(eqs=eqs, var_names=var_names, return_type="func", evaluate=evaluate)        

class MultiDimMixin(ABC):
    """
    Mixin for vector valued output. Each component of the output is fit individually.
    """
    
    @abstractmethod
    def fit(self, times: np.ndarray, trajectories: np.ndarray, derivatives: np.ndarray) -> Dict:
        ...
    
    def fit_components(
        self, 
        times: np.ndarray, 
        trajectories: np.ndarray, 
        derivatives: np.ndarray, 
    ) -> Dict[int, List[str]]:
        assert len(trajectories.shape) == 2, f"len(trajectories.shape) == {len(trajectories.shape)}"
        predictions_per_component: List[List[str]] = []
        for _deriv in derivatives.T:
            # supply all trajectories as input but only single output component to learn a func R^n -> R
            predictions_per_component.append(self.fit(times, trajectories, _deriv)[0])
        return {0: list(" | ".join(vs) for vs in itertools.product(*predictions_per_component))}
        


class BatchMixin(ABC):
    """
    This class lets models iteratively process a list of trajectories.
    The base model needs to implement a fit() method.
    
    Methods
    -------
    fit_all(times, trajectories):
        Calls model.fit for every element in times and trajectories.
    """
    
    @abstractmethod
    def fit(self, times: np.ndarray, trajectories: np.ndarray, derivatives: np.ndarray) -> Dict:
        ...
    
    def fit_all(
        self,
        times: np.ndarray,
        trajectories: np.ndarray,
        *args, 
        **kwargs,
    ) -> Dict[int, Union[None, List[str]]]:

        assert isinstance(trajectories, List)
        predictions_per_equation = defaultdict(list)
        for trajectory_i, (trajectory, time) in tqdm(enumerate(zip(trajectories, times)), total=len(trajectories)):
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
    
    Arguments:
    ----------
    finite_difference_order: int:
        Approximation order.
    
    smoother_window_length: Union[None, int]:
        Ignored if 'None'. Else, window length for smoothing the trajectory before estimating the derivative.
    
    Methods
    -------
    approximate_derivative(trajectory, times, finite_difference_order, smoother_window_length):
        Approximate derivatives.
        
    get_differentiation_method(finite_difference_order, smoother_window_length):
        Create differentiation method instance.
    """
    def __init__(
        self, 
        finite_difference_order: int = 2,
        smoother_window_length: Union[None, int] = None,
    ):
        if hasattr(self, "finite_difference_order"):
            # constructur has already been called before
            # https://stackoverflow.com/questions/34884567/python-multiple-inheritance-passing-arguments-to-constructors-using-super
            return
        self.finite_difference_order = finite_difference_order
        self.smoother_window_length = smoother_window_length
    
    def approximate_derivative(
        self,
        trajectory: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        times = times.squeeze()
        assert len(times.shape) == 1 or np.all(times.shape == trajectory.shape), f"{times.shape} vs {trajectory.shape}"
        return self.get_differentiation_method()._differentiate(trajectory, times)
    
    def get_differentiation_method(self):
        if self.smoother_window_length is None:
            return FiniteDifference(order=self.finite_difference_order)
        return SmoothedFiniteDifference(
            order=self.finite_difference_order,
            smoother_kws={'window_length': self.smoother_window_length},
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
        
        times = np.array(times)
        sort_idx = np.argsort(times)
        unsort_idx = np.argsort(sort_idx) 
        sorted_times = times[sort_idx]
        
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
        trajectory = integrate_ode(y0, sorted_times, prediction, ode_integrator=ode_integrator)

        if trajectory is not None:
            trajectory = np.array(trajectory)
            trajectory = trajectory[unsort_idx]

        return trajectory
