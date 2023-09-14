from typing import Any, Callable, Dict, List, Literal, Union, get_args
from multiprocessing import Pool
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from ProGED.equation_discoverer import EqDisco
from odeformer.model.mixins import (
    PredictionIntegrationMixin,
    BatchMixin,
    GridSearchMixin,
)
from odeformer.baselines.baseline_utils import variance_weighted_r2_score
import re
import time
import numpy as np
import pandas as pd
import traceback

__all__ = ("ProGEDWrapper")

# https://github.com/brencej/ProGED/blob/main/ProGED/generators/grammar_construction.py#L308
GRAMMARS = Literal["universal", "rational", "simplerational", "trigonometric", "polynomial"]

class ProGEDWrapper(BaseEstimator, PredictionIntegrationMixin, BatchMixin, GridSearchMixin):
    
    def __init__(
        self,
        model_dir: str,
        num_candidates: int = 32, 
        verbosity: int = 1, 
        num_workers: int = 1,
        debug: bool = True,
        generator_template_name: GRAMMARS = "polynomial",
        include_time_as_variable: bool = False,
        grid_search_generator_template_name: bool = True,
        optimize_hyperparams: bool = True,
        hyper_opt_eval_fraction: Union[None, float] = None,
        sorting_metric: str = "r2", 
        grid_search_is_running: bool = False,
    ):
        self.model_dir = model_dir
        self.filename_pareto_front = f"equations_{time.strftime('%Y-%m-%d-%H-%M-%S-%MS')}.json"
        self.num_candidates = num_candidates
        self.verbosity = verbosity
        self.num_workers = num_workers
        self.debug = debug
        self.generator_template_name = generator_template_name
        self.include_time_as_variable = include_time_as_variable
        self.grid_search_generator_template_name = grid_search_generator_template_name
        self.optimize_hyperparams = optimize_hyperparams
        self.hyper_opt_eval_fraction = hyper_opt_eval_fraction
        self.sorting_metric = sorting_metric
        self.grid_search_is_running = grid_search_is_running
    
        
    def get_hyper_grid(self) -> Dict[str, Any]:
        hparams = {}
        if self.grid_search_generator_template_name:
            hparams["generator_template_name"] = list(get_args(GRAMMARS))
        return hparams
    
    def get_n_jobs(self) -> int:
        return 48
        
    def set_params(self, **params: Dict):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        return self
    
    def score(
        self,
        times: np.ndarray,
        trajectories: np.ndarray,
        metric: Callable = variance_weighted_r2_score,
    ) -> float:
        try:
            candidates = self._get_equations()
            assert len(candidates.keys()) > 0, candidates.keys()
            pred_trajectory = self.integrate_prediction(
                times, y0=trajectories[0], prediction=candidates[0][0]
            )
            assert pred_trajectory is not None, f"pred_trajectory is None."
            return metric(trajectories, pred_trajectory)
        except AssertionError as e:
            print(traceback.format_exc())
            return np.nan
    
    def fit(
        self,
        times: Union[List[np.ndarray], np.ndarray],
        trajectories: Union[List[np.ndarray], np.ndarray],
        verbose: Union[None, bool] = None,
        generator_template_name: Union[None, Literal["polynomial",]] = None,
        *args, **kwargs, # ignored, for compatibility only
    ) -> Dict[int, List[Union[None, str]]]:
        
        if self.optimize_hyperparams and not self.grid_search_is_running:
            if isinstance(trajectories, List):
                assert len(trajectories) == 1, len(trajectories)
                trajectories = trajectories[0]
            assert isinstance(trajectories, np.ndarray)
            return self.fit_grid_search(times=times, trajectory=trajectories)
        
        if isinstance(trajectories, List):
            return self.fit_all(times=times, trajectories=trajectories)
        # trajectories needs to have shape (len(time-series), #vars)
        assert len(times.shape) == 1, f"len(times.shape) = {len(times.shape)}"
        assert times.shape[0] == trajectories.shape[0], \
            f"{times.shape[0]} vs {trajectories.shape[0]}"
        if generator_template_name is None:
            generator_template_name = self.generator_template_name
        feature_names = [f"x_{i}" for i in range(trajectories.shape[1])]
        data = pd.DataFrame(
            np.hstack((times.reshape(-1,1), trajectories)), 
            columns=['t']+feature_names,
        )
        self.ED = EqDisco(
            data = data,
            task_type="differential", 
            lhs_vars = feature_names,
            rhs_vars = ["t"] + feature_names if self.include_time_as_variable else feature_names,
            sample_size = self.num_candidates,
            system_size = len(feature_names),
            generator_template_name = generator_template_name,
            strategy_settings = {"max_repeat": 100},
            verbosity=verbose if verbose is not None else self.verbosity,
        )
        if self.debug: 
            print("Generating models...")
        self.ED.generate_models()
        if self.debug: 
            print("Estimating parameters...")
        if self.num_workers > 1:
            with Pool(self.num_workers) as p:
                self.ED.fit_models(pool_map=p.map)
        else:
            self.ED.fit_models()
        return self._get_equations()
    
    def _get_equations(self) -> Dict[int, List[Union[None, str]]]:
        results = self.ED.get_results(self.num_candidates)
        candidates, errors = [], []
        for eq in results:
            clean_eq = self._clean_equation(eq)
            candidates.append(clean_eq)
            errors.append(eq.get_error())
        order = np.argsort(errors)
        return {0: [candidates[i] for i in order]}
    
    def _clean_equation(self, eq) -> Union[None, str]:
        expr = eq.get_full_expr()
        if isinstance(expr, List):
            eq = " | ".join([str(e) for e in expr])
        else: 
            assert isinstance(expr, str), type(expr)
            eq = str(expr)  
        if len(re.findall(r"C\d", eq)) > 0:
            # some equations have unspecified constants, e.g. C0, C1
            return None
        return eq
