from typing import Dict, List, Literal, Union
from multiprocessing import Pool
from sklearn.base import BaseEstimator
import re
import numpy as np
import pandas as pd
from ProGED.equation_discoverer import EqDisco
from symbolicregression.model.mixins import PredictionIntegrationMixin, BatchMixin

class ProGEDWrapper(BaseEstimator, PredictionIntegrationMixin, BatchMixin):
    def __init__(
        self, 
        feature_names: List[str],
        num_candidates: int = 10, 
        verbosity: int = 1, 
        num_workers: int = 1,
        debug: bool = True,
        generator_template_name: Literal["polynomial",] = "polynomial"
    ):
        self.feature_names = feature_names
        self.num_candidates = num_candidates
        self.verbosity = verbosity
        self.num_workers = num_workers
        self.debug = debug
        self.generator_template_name = generator_template_name
    
    def get_params(self, *args, **kwargs):
        return {"num_candidates": self.num_candidates, 
                "verbosity": self.verbosity}
        
    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        return self
    
    def score(self, X):
        raise NotImplementedError()
    
    def fit(
        self,
        times: np.ndarray,
        trajectories: np.ndarray, # trajectories needs to have shape (len(time-series), #vars)
        sort_candidates: bool = True, # this will be ignored, only here for compatibility
        verbose: Union[None, bool] = None,
        generator_template_name: Union[None, Literal["polynomial",]] = None,
    ) -> Dict[int, List[Union[None, str]]]:
        
        if isinstance(trajectories, List):
            return self.fit_all(times=times, trajectories=trajectories, )
        assert len(times.shape) == 1, f"len(times.shape) = {len(times.shape)}"
        assert times.shape[0] == trajectories.shape[0], \
            f"{times.shape[0]} vs {trajectories.shape[0]}"
        if generator_template_name is None:
            generator_template_name = self.generator_template_name
        data = pd.DataFrame(
            np.hstack((times.reshape(-1,1), trajectories)), 
            columns=['t']+self.feature_names,
        )
        self.ED = EqDisco(
            data = data,
            task_type="differential", 
            lhs_vars = self.feature_names,
            rhs_vars = ["t"] + self.feature_names,
            sample_size = self.num_candidates,
            system_size = len(self.feature_names),
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
        candidates = []
        for eq in results:
            candidates.append(self._clean_equation(eq))
        return {0: candidates}
    
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