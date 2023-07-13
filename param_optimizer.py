from tqdm import tqdm
from copy import deepcopy
from scipy.optimize import minimize
from typing import Dict, Literal, List, Tuple
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.exceptions import NotFittedError

import os
import regex
import sympy
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evaluate import *
from symbolicregression.model.mixins import PredictionIntegrationMixin
from symbolicregression.baselines.baseline_utils import variance_weighted_r2_score

class ConstantOptimizer(PredictionIntegrationMixin):
    
    def __init__(
        self,
        eq: str,
        y0: np.ndarray,
        time: np.ndarray,
        observed_trajectory: np.ndarray,
        init_random: bool,
        optimization_objective: Literal["mse", "r2"] = "r2",
        eval_objective: Literal["mse", "r2"] = "r2",
        track_eval_history: bool = True,
    ):
        """
        NOTE: for eval_objective, a smaller objective value should imply better parameters.
        """
        
        self.eq = eq
        self.y0 = y0
        self.time = time
        self.observed_trajectory = observed_trajectory
        self.init_random = init_random
        self.optimization_objective = optimization_objective
        self.eval_objective = eval_objective
        self.track_eval_history = track_eval_history
        self.CONSTANTS_PATTERN = \
            r"(?:(?<!_\d*))(?:(?<!\*\*))(?:[-+]?)?(?:(?<=\()[-+]?)?(?:(?<=^)[-+]?)?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?"
        self.orig_params = self.get_params()
        self.eval_history = []
        
    
    def get_params(self) -> Dict[str, float]:
            
        def replace_constant_by_param(match: regex.Match, param_prior: Dict[str, float]) -> str:
            param_counter = len(param_prior)        
            param_prior[f"p_{param_counter}"] = float(match.group(0))
            return f"p_{param_counter}"
        
        param_priors = {}
        _ = regex.sub(
            pattern = self.CONSTANTS_PATTERN,
            repl=lambda m: replace_constant_by_param(m, param_priors),
            string=self.eq
        )
        return param_priors
    
    def set_params(self, params: np.ndarray) -> str:
        params = deepcopy(params)
        if isinstance(params, np.ndarray):
            params = params.tolist()
        assert isinstance(params, List), type(params)
        params = params[::-1]
        
        def replace_constant_by_param(params: List) -> str:
            return f"{params.pop()}"
        
        return regex.sub(
            pattern = self.CONSTANTS_PATTERN,
            repl=lambda _match: replace_constant_by_param(params), # do not remove lambda, this needs to be a Callable
            string=self.eq
        )
    
    def simulate(self, params: Union[List, np.ndarray]) -> Tuple[np.ndarray, str]:
        return self.integrate_prediction(
            times=self.time,
            y0=self.y0, 
            prediction=self.set_params(params),
        )
        
        
    def objective(self, params: np.ndarray) -> float:
        simulated_trajectory = self.simulate(params)
        if self.track_eval_history:
            self.eval_history.append([self._objective(simulated_trajectory, self.eval_objective), *params])
        return self._objective(simulated_trajectory, self.optimization_objective)
        
        
    def _objective(self, simulated_trajectory, objective: str):
        if simulated_trajectory is None or not np.isfinite(simulated_trajectory).any():
            return np.inf
        try:
            if objective == "r2":
                return -1*variance_weighted_r2_score(self.observed_trajectory, simulated_trajectory)        
            elif objective == "mse":
                return mean_squared_error(self.observed_trajectory, simulated_trajectory)
            else:
                raise ValueError(f"Unknown objective: {self.optimization_objective}")
        except:
            return np.inf
    
    def optimize(self) -> Tuple[str, np.ndarray, np.ndarray]:
        param_priors = np.array(list(self.get_params().values()))
        info = minimize(
            fun=lambda _params: self.objective(_params),
            x0=(np.random.randn(len(param_priors)) if self.init_random else param_priors),
        )
        if self.track_eval_history:
            # take best params
            optimal_params = self._get_optimal_params()
        else:
            # take final params
            optimal_params = info["x"]  
        return self.set_params(optimal_params), optimal_params, self.simulate(optimal_params)
        
    def _get_optimal_params(self) -> np.ndarray:
        if len(self.eval_history) == 0:
            raise NotFittedError()
        history = np.array(self.eval_history)
        objective_history = history[:, 0]
        param_history = history[:, 1:]
        if np.all(~np.isfinite(objective_history)):
            print("Warning: the entire objective history is non-finite.") 
        opt_idx = np.nanargmin(objective_history)
        return param_history[opt_idx]

def main(args, result_dir, result_file):
    if hasattr(args, "random_seed"):
        print(f"Using random seed {args.random_seed}")
        np.random.seed(args.random_seed)
        
    pmlb_iterator = pd.read_pickle(args.path_dataset)
    all_scores = pd.read_csv(args.path_scores, delimiter="\t")
    
    final_scores = deepcopy(all_scores)
    final_scores.loc[:, "optimize_params"] = True
    final_preds, final_r2s = [], []
    trajetory_counter = 0
    for samples_i, (samples, _) in enumerate(tqdm(pmlb_iterator)):
        
        times = samples["times"]
        trajectories = samples["trajectory"]
        assert isinstance(times, List), type(times)
        assert isinstance(trajectories, List), type(trajectories)
        for _trajectory_i, (_times, _trajectory) in enumerate(zip(times, trajectories)):
            
            scores = all_scores.iloc[trajetory_counter]
            pred_eq = " | ".join([str(sympy.parse_expr(e)) for e in scores.predicted_trees.split("|")])
            
            param_optimizer = ConstantOptimizer(
                eq=pred_eq,
                y0=_trajectory[0],
                time=_times,
                observed_trajectory=_trajectory,
                optimization_objective="r2",
                eval_objective="r2",
                init_random=False,
                track_eval_history=False
            )
            
            final_eq, estimated_params, simulated_trajectory = param_optimizer.optimize()
            
            try:
                r2 = variance_weighted_r2_score(_trajectory, simulated_trajectory)
            except Exception as e:
                print(e)
                r2 = np.nan
            try:
                mse1 = mean_squared_error(simulated_trajectory, _trajectory)
            except Exception as e:
                print(e)
                mse1 = np.nan
            try:
                mse2 = mean_squared_error(simulated_trajectory, _trajectory)
            except Exception as e:
                print(e)
                mse2 = np.nan
            final_r2s.append([scores.r2, r2, mse1, mse2])
            print(final_r2s[-1])
            final_preds.append(final_eq)
            trajetory_counter += 1
    
    final_scores.loc[:, "predicted_trees"] = final_preds
    final_scores.to_csv(os.path.join(result_dir, result_file))
    print(f"final_scores: {final_scores}")
    print(f"final_scores (mean): {np.nanmean(np.array(final_r2s), axis=0)}")
    print(f"final_scores (median): {np.nanmedian(np.array(final_r2s), axis=0)}")
            
    return 0

def str2bool(arg):
    if isinstance(arg, bool):
        return arg
    if arg.lower() == "true":
        return True
    elif arg.lower() == "false":
        return False
    else:
        raise ValueError(f"Unknown argument {arg}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_scores", type=str, default="./experiments/odeformer/scores.csv")
    parser.add_argument("--path_dataset", type=str, default="./datasets/strogatz.pkl")
    parser.add_argument("--num_simulations", type=int, default=10000)
    parser.add_argument("--num_posterio_samples", type=int, default=100000)
    parser.add_argument("--random_seed", type=int, default=2023)
    parser.add_argument("--random_param_initialization", type=str2bool, default=False)
    args = parser.parse_args()
    
    result_dir = Path(args.path_scores).parent / "optimize"
    result_file = f"{str(Path(args.path_scores).stem)}_optimize.csv"
    os.makedirs(result_dir, exist_ok=True)
    
    with open(result_dir / result_file, "wb") as fout:
        print(f"Saving args at {result_dir / result_file}")
        pickle.dump(obj = dict(vars(args)), file=fout)

    main(args, result_dir, result_file)
    
    # TODO: optimization fails for 1 example, why? Maybe try better loss function?
    # TODO: start from random initial guesses
    # TODO: write up in manuscript