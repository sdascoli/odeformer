# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from tqdm import tqdm
from copy import deepcopy
from timeit import default_timer as timer
from typing import Any, Dict, List, Union, Literal, Tuple
from pathlib import Path
from collections import OrderedDict, defaultdict

import os
import json
import glob
import sympy
import torch
import wandb
import pickle
import numpy as np
import pandas as pd
import traceback

import symbolicregression
from parsers import get_parser
from symbolicregression.slurm import init_signal_handler, init_distributed_mode
from symbolicregression.utils import initialize_exp
from symbolicregression.model import build_modules
from symbolicregression.envs import build_env
from symbolicregression.envs.generators import NodeList
from symbolicregression.trainer import Trainer
from symbolicregression.model.sklearn_wrapper import SymbolicTransformerRegressor
from symbolicregression.model.model_wrapper import ModelWrapper
from symbolicregression.metrics import compute_metrics
from param_optimizer import ParameterOptimizer
# np.seterr(all="raise")

def setup_odeformer(trainer) -> SymbolicTransformerRegressor:
    embedder = (
        trainer.modules["embedder"].module
        if trainer.params.multi_gpu
        else trainer.modules["embedder"]
    )
    encoder = (
        trainer.modules["encoder"].module
        if trainer.params.multi_gpu
        else trainer.modules["encoder"]
    )
    decoder = (
        trainer.modules["decoder"].module
        if trainer.params.multi_gpu
        else trainer.modules["decoder"]
    )
    embedder.eval()
    encoder.eval()
    decoder.eval()

    model_kwargs = {
        'beam_length_penalty': trainer.params.beam_length_penalty,
        'beam_size': trainer.params.beam_size,
        'max_generated_output_len': trainer.params.max_generated_output_len,
        'beam_early_stopping': trainer.params.beam_early_stopping,
        'beam_temperature': trainer.params.beam_temperature,
        'beam_type': trainer.params.beam_type,
    }

    mw = ModelWrapper(
        env=trainer.env,
        embedder=embedder,
        encoder=encoder,
        decoder=decoder,
        **model_kwargs
    )
    return SymbolicTransformerRegressor(
        model=mw,
        from_pretrained=trainer.params.from_pretrained,
        max_input_points=trainer.params.max_points,
        rescale=trainer.params.rescale,
        params=trainer.params,
        model_kwargs=model_kwargs,
    )

def read_file(filename, label="target", sep=None):
    if filename.endswith("gz"):
        compression = "gzip"
    else:
        compression = None
    if sep:
        input_data = pd.read_csv(filename, sep=sep, compression=compression)
    else:
        input_data = pd.read_csv(
            filename, sep=sep, compression=compression, engine="python"
        )
    feature_names = [x for x in input_data.columns.values if x != label]
    feature_names = np.array(feature_names)
    X = input_data.drop(label, axis=1).values.astype(float)
    y = input_data[label].values
    assert X.shape[1] == feature_names.shape[0]
    return X, y, feature_names

class Evaluator(object):

    def __init__(self, trainer, model):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.model = model
        self.params = trainer.params
        self.env = trainer.env
        self.save_path = (
            self.params.eval_dump_path
            if self.params.eval_dump_path
            else self.params.dump_path
            if self.params.dump_path
            else self.params.reload_checkpoint
        )
        if hasattr(self.params, "eval_integration_timeout"):
            self.eval_integration_timeout = self.params.eval_integration_timeout
        else:
            self.eval_integration_timeout = 1
        self.trainer.logger.info(f"Setting eval_integration_timeout to {self.eval_integration_timeout}")
        
        if not os.path.exists(self.save_path): os.makedirs(self.save_path)
        
        if hasattr(self.params, "eval_size"):
            self.eval_size = self.params.eval_size
        else:
            self.eval_size = -1

        self.ablation_to_keep = list(
            map(lambda x: "info_" + x, self.params.ablation_to_keep.split(","))
        )
        
    def prepare_test_trajectory(
        self,
        samples: Dict[str, Dict[str, Any]],
        evaluation_task: Literal["debug", "interpolation", "forecasting", "y0_generalization"],
        rng: np.random.RandomState,
        y0: Union[None, np.ndarray],
        y0_generalization_delta: Union[None, np.ndarray],
    ) -> Dict[str, Dict[str, Any]]:
        
        if "train" not in samples.keys():
            samples["train"] = {"times": samples["times"], "trajectories": samples["trajectory"]}
            del samples["times"], samples["trajectory"]
        
        # samples["test"] = {"times":[], "trajectories":[]}
        
        if "interpolation" in evaluation_task:
            self.trainer.logger.info("Setting up test trajectory for interpolation")
            samples["test"] = deepcopy(samples["train"])
            return samples

        # elif "forecasting" in evaluation_task:
        #     for time, trajectory, tree in zip(samples["train"]["times"], samples["train"]["trajectories"], samples["tree"]):
        #         if y0 is None:
        #             y0 = trajectory[-1]
        #             self.trainer.logger.info(f"forecasting: y0 is None. Setting y0 = {y0}")
        #         self.trainer.logger.info(f"forecasting: using y0 = {y0}")
        #         t0 = time[-1]
        #         self.trainer.logger.info(f"Using y0 = {y0}")
        #         if hasattr(self.params, "forecasting_window_length"):
        #             forecasting_window_length = self.params.forecasting_window_length
        #         else:
        #             forecasting_window_length = 5
        #         teval = np.linspace(t0, t0+forecasting_window_length, 512, endpoint=True)
        #         test_trajectory = self.model.integrate_prediction(
        #             teval, y0=y0, prediction=tree, timeout=self.eval_integration_timeout
        #         )
        #         samples["test"]["trajectories"].append(test_trajectory)
        #         samples["test"]["times"].append(teval)
        #     return samples
        
        elif "y0_generalization" in evaluation_task:
            
            if "test" in samples.keys():
                self.trainer.logger.info("y0_generalization: Using predefined test trajectories.")
                return samples
            self.trainer.logger.info("y0_generalization: creating new test trajectories.")
            samples["test"] = {"times":[], "trajectories":[]}
            
            for time, trajectory, tree, dimension in zip(samples["train"]["times"], samples["train"]["trajectories"], samples["tree"], samples["infos"]["dimension"]):
                if y0 is None:
                    y0 = trajectory[0]
                    self.trainer.logger.info(f"y0_generalization: y0 is None. Setting y0 = {y0}")
                if y0_generalization_delta is not None:
                    y0 = y0 + y0_generalization_delta
                self.trainer.logger.info(f"y0_generalization: using y0 = {y0}")
                test_trajectory = self.model.integrate_prediction(
                    time, y0=y0, prediction=tree, timeout=self.eval_integration_timeout
                )
                samples["test"]["trajectories"].append(test_trajectory)
                samples["test"]["times"].append(time)
            return samples
        
        else:
            raise ValueError(f"Unknown evaluation_task: {evaluation_task}")
            
            
    def _optimize_constants(
        self,
        eq: str,
        times: np.ndarray,
        observations: np.ndarray,
        init_random: bool,
        seed: int = 2023,
    ) -> str:
        
        # TODO: allow input to be of type List or Dict
        try:
            optimizer = ParameterOptimizer(
                eq=eq,
                y0=observations[0],
                time=times,
                observed_trajectory=observations,
                optimization_objective="r2",
                eval_objective="r2",
                init_random=init_random,
                track_eval_history=True,
                seed=seed,
            )
            eq_optimized, _, _ = optimizer.optimize()
            self.trainer.logger.info(f"orig eq: {eq}\noptimized eq: {eq_optimized}")
            return eq_optimized
        except Exception as e:
            traceback.print_exc()
            return np.nan

    def _evaluate(
        self,
        times: List[Dict],
        trajectories: List[Dict],
        trees: List[Union[None, NodeList]],
        all_candidates: Union[List, Dict],
        all_durations: Union[List, Dict],
        validation_metrics: str,
        y0: Union[None, np.ndarray] = None
    ) -> Tuple[Dict, Dict]:
        
        best_results = {metric: [] for metric in validation_metrics.split(',')}
        best_results["duration_fit"], best_results["pareto_front"], best_candidates = [], [], []
        zipped = [times, trajectories, trees, (all_candidates.values() if isinstance(all_candidates, Dict) else all_candidates)]
        pred_trajectories = []
        if all_durations is not None:
            zipped.append(all_durations)
        for items in zip(*zipped):
            if len(items) == 5:
                time, trajectory, tree, candidates, duration_fit  = items
            else:
                time, trajectory, tree, candidates = items
            if not candidates or trajectory is None: 
                for k in best_results:
                    best_results[k].append(np.nan)
                best_candidates.append(None)
                pred_trajectories.append(np.nan)
                continue
            best_results["pareto_front"].append(candidates)
            time, idx = sorted(time), np.argsort(time)
            trajectory = trajectory[idx]
            if isinstance(candidates, List):
                best_candidate = candidates[0]
            else:
                best_candidate = candidates

            if isinstance(best_candidate, str) and (not hasattr(self.params, "convert_prediction_to_tree") or self.params.convert_prediction_to_tree):
                try: best_candidate = self.str_to_tree(best_candidate)
                except: pass
            if y0 is None:
                y0=trajectory[0]
            pred_trajectory = self.model.integrate_prediction(
                time, y0=y0, prediction=best_candidate, timeout=self.eval_integration_timeout
            )
            if not hasattr(self.params, "convert_prediction_to_tree") or self.params.convert_prediction_to_tree:
                try: best_candidate = self.env.simplifier.simplify_tree(best_candidate, expand=True)
                except: pass
            pred_trajectories.append(pred_trajectory)
            best_result = compute_metrics(
                pred_trajectory, 
                trajectory, 
                predicted_tree=best_candidate, 
                tree=tree,
                metrics=validation_metrics
            )
            
            if len(items) == 5:
                best_result["duration_fit"] = [duration_fit]
            for k, v in best_result.items():
                best_results[k].append(v[0])
            best_candidates.append(best_candidate)
        
        return best_results, best_candidates, pred_trajectories

    def save_results(self, batch_results: Dict, save_file: str):
        batch_results = pd.DataFrame.from_dict(batch_results)
        batch_results.to_pickle(save_file.replace(".csv", ".pkl"))
        batch_results.drop(
            labels=[
                "times_train", "trajectories_train", "times_test", "trajectories_test", 
                "trajectory_train_pred", "trajectory_test_pred",
            ], 
            axis=1,
            inplace=True,
        )
        batch_results.to_csv(save_file, index=False)

    def evaluate_on_iterator(self, iterator, name="in_domain"):
        save_file = os.path.join(self.save_path, f"eval_{name}.csv")
        if os.path.exists(save_file):
            self.trainer.logger.info(f"{save_file} exists. Skipping.")
            return    
        
        self.trainer.logger.info("evaluate_on_iterator")
        scores = OrderedDict({"epoch": self.trainer.epoch})
        batch_results = defaultdict(list)
        _total = len(iterator)
        if hasattr(self.params, "reload_scores_path") and (self.params.reload_scores_path is not None):
            self.trainer.logger.info(f"Reloading scores from {self.params.reload_scores_path}")
            if self.params.reload_scores_path.endswith("csv"):
                reloaded_scores = pd.read_csv(self.params.reload_scores_path)
            elif self.params.reload_scores_path.endswith(".pkl"):
                reloaded_scores = pd.read_pickle(self.params.reload_scores_path)
            else:
                raise ValueError(f"Unknown file suffix: {self.params.reload_scores_path}")
            num_reloaded_preds = reloaded_scores.shape[0]
            self.trainer.logger.info(f"Reloaded {num_reloaded_preds} existing predictions")
            if num_reloaded_preds < _total and not self.params.continue_fitting:
                self.trainer.logger.info(
                    "evaluate_on_iterator:"
                    f"number of reloaded predictions ({num_reloaded_preds}) < len(iterator ({_total}))"
                    "Only evaluating on samples with reloaded prediction."
                )
                _total = num_reloaded_preds
        
        if self.eval_size > 0 and self.eval_size < _total:
            self.trainer.logger(
                "evaluate_on_iterator: evaluating on self.eval_size = {self.eval_size} / {_total} samples." 
            )
            _total = self.eval_size
        
        fit_counter = 0
        
        for samples_i, samples in enumerate(tqdm(iterator[:_total], total=_total)):
            
            if samples_i == self.eval_size:
                self.trainer.logger.info(f"Reached self.eval_size = {self.eval_size} iterations. Stopping evaluation.")
                break
            
            y0_generalization_delta=None
            if "y0_generalization" in self.params.evaluation_task:
                y0_generalization_delta = self.params.y0_generalization_delta        
        
            samples = self.prepare_test_trajectory(
                samples=samples, 
                evaluation_task=self.params.evaluation_task,
                rng=np.random.RandomState(self.params.test_env_seed + samples_i),
                y0=None,
                y0_generalization_delta=y0_generalization_delta,
            )
            
            times, trajectories, infos = samples["train"]["times"], samples["train"]["trajectories"], samples["infos"]
            for k, v in infos.items():
                if isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
                    infos[k] = v.tolist()
                elif isinstance(v, List):
                    infos[k] = v
                else:
                    raise TypeError(f"v should be of type List of np.ndarray but has type: {type(v)}")
            
            if "tree" in samples.keys():
                trees = [self.env.simplifier.simplify_tree(tree, expand=True) for tree in samples["tree"]]
                batch_results["trees"].extend(
                    [None if tree is None else tree.infix() for tree in trees]
                )
            else:
                trees = [None] * len(times)

            # corrupt training data
            for i, (time, trajectory) in enumerate(zip(times, trajectories)):
                if self.params.eval_noise_gamma:
                    noise = self.env._create_noise(
                        train=False,
                        trajectory=trajectory,
                        gamma=self.params.eval_noise_gamma,
                        noise_type=self.params.eval_noise_type,
                        seed=self.params.test_env_seed + i,
                    )
                    if self.params.eval_noise_type in ["additive", "adaptive_additive"]:
                        trajectory = trajectory + noise
                    elif self.params.eval_noise_type == "multiplicative":
                        trajectory = trajectory * noise
                    else:
                        raise ValueError(f"Unknown noise type: {self.params.eval_noise_type}")
                    
                if self.params.eval_subsample_ratio:
                    time, trajectory = self.env._subsample_trajectory(
                        time,
                        trajectory,
                        subsample_ratio=self.params.eval_subsample_ratio,
                        seed=self.params.test_env_seed,
                    )
                times[i] = time
                trajectories[i] = trajectory

            # fit
            start_time_fit = timer()
            if (hasattr(self.params, "reload_scores_path") and \
                (self.params.reload_scores_path is not None) and \
                (reloaded_scores.shape[0] > fit_counter)):
                
                self.trainer.logger.info(f"Reusing prediction for sample {fit_counter}")
                
                # we want to reload predictions and there is a prediction for this sample
                all_candidates = [reloaded_scores.iloc[fit_counter].predicted_trees]
                
                if (hasattr(self.params, "optimize_constants") and \
                    self.params.optimize_constants and \
                    (not "optimize_constants" in self.params.reload_scores_path)):
                    
                    all_candidates[0] = self._optimize_constants(
                        eq=all_candidates[0],
                        times=reloaded_scores.iloc[fit_counter].times_train,
                        observations=reloaded_scores.iloc[fit_counter].trajectories_train,
                        init_random=self.params.optimize_constants_init_random,
                        seed=self.params.test_env_seed + fit_counter,
                    )
                    
                if all_candidates[0] is np.nan or pd.isnull(all_candidates[0]):
                    all_candidates[0] = str(all_candidates[0])
            else:
                self.trainer.logger.info(f"Obtaining new prediction for sample {fit_counter}")
                all_candidates = self.model.fit(times, trajectories, verbose=False, sort_candidates=True)
                
            all_duration_fit = [timer() - start_time_fit] * len(times)
            fit_counter += 1
            
            # evaluate on train data
            best_results, best_candidates, trajectories_train_pred = self._evaluate(
                times, trajectories, trees, all_candidates, all_duration_fit, self.params.validation_metrics
            )
            if "y0_generalization" in self.params.evaluation_task:
                y0=None # trajectories_train_pred[0][0]
            elif "forecasting" in self.params.evaluation_task:
                # TODO could be None
                y0=trajectories_train_pred[0][-1]
            else:
                y0=None
                
            # evaluate on test data
            test_results, _, trajectories_test_pred = self._evaluate(
                times=samples["test"]["times"],
                trajectories=samples["test"]["trajectories"],
                trees=trees,
                all_candidates=best_candidates,
                all_durations=None,
                validation_metrics=self.params.validation_metrics,
                y0=y0
            )
            # collect results
            batch_results["predicted_trees"].extend([tree.infix() if hasattr(tree, 'infix') else tree for tree in best_candidates])
            
            batch_results["times_train"].extend(times)
            batch_results["trajectories_train"].extend(trajectories)
            batch_results["times_test"].extend(samples["test"]["times"])
            batch_results["trajectories_test"].extend(samples["test"]["trajectories"])
            batch_results["trajectory_train_pred"].extend(trajectories_train_pred)
            batch_results["trajectory_test_pred"].extend(trajectories_test_pred)
            
            for k, v in infos.items():
                batch_results["info_" + k].extend(v)
            for k, v in best_results.items():
                batch_results[k].extend(v)
            for k, v in test_results.items():
                if k == "duration_fit": continue
                batch_results['test_'+k].extend(v)

            # save intermediate_results
            self.save_results(
                batch_results=batch_results,
                save_file=save_file.replace(".csv", "_intermediate.csv")
            )

        self.save_results(batch_results=batch_results, save_file=save_file)
        
        self.trainer.logger.info("Saved {} equations to {}".format(len(batch_results), save_file))
        try:
            df = pd.read_csv(save_file, na_filter=True)
        except:
            self.trainer.logger.info("WARNING: no results")
            return
        
        info_columns = [x for x in list(df.columns) if x.startswith("info_")]
        df = df.drop(columns=filter(lambda x: x not in self.ablation_to_keep, info_columns))
        df = df.drop(columns=["predicted_trees", "pareto_front"])
        if "trees" in df: df = df.drop(columns=["trees"])
        if "info_name" in df.columns: df = df.drop(columns=["info_name"])

        for metric in self.params.validation_metrics.split(','):
            for prefix in ["", "test_"]:
                scores[metric] = df[metric].mean()
                scores[prefix+metric+'_median'] = df[metric].median()

        scores["duration_fit"] = df["duration_fit"].mean()

        if self.params.use_wandb:
            wandb.log({name+"_"+metric: score for metric, score in scores.items()})
        return scores
        
    def evaluate_in_domain(self, task):
        self.model.rescale = False
        self.trainer.logger.info(
            "====== STARTING EVALUATION IN DOMAIN (multi-gpu: {}) =======".format(
                self.params.multi_gpu
            )
        )
        iterator = self.env.create_test_iterator(
            task,
            data_path=self.trainer.data_path,
            batch_size=self.params.batch_size_eval,
            params=self.params,
            size=self.params.eval_size,
            test_env_seed=self.params.test_env_seed,
        )
        return self.evaluate_on_iterator(iterator, name = "in_domain")

    def evaluate_on_pmlb(self, path_dataset=None):
        if path_dataset is not None and os.path.exists(path_dataset):
            iterator = pd.read_pickle(path_dataset)
        else:
            def format_strogatz_equation(eq):
                return " | ".join(
                    [
                        str(
                            sympy.parse_expr(
                                comp.replace("u(1)", "x_0").replace("u(2)", "x_1").replace("^", "**")
                            )
                        )
                        for comp in eq.split("|")
                    ]
                )
            strogatz_equations = {
                "strogatz_bacres1": '20-u(1) - (u(1)*u(2)/(1+0.5*u(1)^2)) | 10 - (u(1)*u(2)/(1+0.5*u(1)^2))',
                "strogatz_barmag1": '0.5*sin(u(1)-u(2))-sin(u(1)) | 0.5*sin(u(2)-u(1)) - sin(u(2))',
                "strogatz_glider1": '-0.05*u(1)^2-sin(u(2)) | u(1) - cos(u(2))/u(1)',
                "strogatz_lv1": '3*u(1)-2*u(1)*u(2)-u(1)^2 | 2*u(2)-u(1)*u(2)-u(2)^2',
                "strogatz_predprey1": 'u(1)*(4-u(1)-u(2)/(1+u(1))) | u(2)*(u(1)/(1+u(1))-0.075*u(2))',
                "strogatz_shearflow1": '(cos(u(2))/sin(u(2)))*cos(u(1)) | (cos(u(2))^2+0.1*sin(u(2))^2)*sin(u(1))', # replaced cot(x) with cos(x) / sin(x)
                "strogatz_vdp1": '10*(u(2)-(1/3*(u(1)^3-u(1)))) | -1/10*u(1)',
            }
            
            self.model.rescale = self.params.rescale
            self.trainer.logger.info(
                "====== STARTING EVALUATION PMLB (multi-gpu: {}) =======".format(self.params.multi_gpu)
            )
            iterator = []
            from pmlb import fetch_data, dataset_names
            strogatz_names = [name for name in dataset_names if "strogatz" in name and "2" not in name]
            times = np.linspace(0, 10, 100)
            for name in strogatz_names:
                data = fetch_data(name)
                x = data['x'].values.reshape(-1,1)
                y = data['y'].values.reshape(-1,1)
                
                infos = {
                    'dimension': [2],
                    'n_unary_ops': [0],
                    'n_input_points': [100],
                    'name': [name],
                }   
                for j in range(4):
                    samples = {"train": defaultdict(list)}
                    start = j * len(times)
                    stop = (j+1) * len(times)
                    trajectory = np.concatenate((x[start:stop], y[start:stop]),axis=1)
                    samples["train"]['times'].append(deepcopy(times))
                    samples["train"]['trajectories'].append(trajectory)
                    samples['tree'] = [self.str_to_tree(format_strogatz_equation(strogatz_equations[name]))]
                    samples['infos'] = infos
                    iterator.append(samples)
            if path_dataset:
                with open(path_dataset, "wb") as fout:
                    self.trainer.logger.info(f"Saving dataset under:\n{path_dataset}")
                    pickle.dump(obj=iterator, file=fout)
        return self.evaluate_on_iterator(iterator, name="pmlb")
    
    def evaluate_on_oscillators(self, path="invar_datasets"):
        self.model.rescale = self.params.rescale
        self.trainer.logger.info(
            "====== STARTING EVALUATION OSCILLATORS (multi-gpu: {}) =======".format(self.params.multi_gpu)
        )
        iterator = []
        datasets = {}
        for file in glob.glob(path + "/*"):
            with open(file) as f:
                lines = (line for line in f if not line.startswith('%') and not line.startswith('x'))
                data = np.loadtxt(lines)
                data = data[data[:,0]==0]
            datasets[file.split('/')[-1]] = data
        for name, data in datasets.items():
            samples = {"train": defaultdict(list)}
            samples['infos'] = {'dimension':2, 'n_unary_ops':0, 'n_input_points':100, 'name':name}
            for k,v in samples['infos'].items():
                samples['infos'][k] = np.array([v])
            times = data[:,1]
            x = data[:,2].reshape(-1,1)
            y = data[:,3].reshape(-1,1)
            if hasattr(self.model, "max_input_points"):
                idx = np.random.permutation(len(times))
                times, x, y = times[idx], x[idx], y[idx]
            samples["train"]['times'].append(times)
            samples["train"]['trajectories'].append(np.concatenate((x,y),axis=1))
            samples["tree"] = [None]
            iterator.append(samples)
        ds_path = os.path.join(path, "invar_datasets.pkl")
        with open(ds_path, "wb") as fpickle:
            self.trainer.logger.info(f"Saving dataset under: {ds_path}")
            pickle.dump(iterator, fpickle)
        return self.evaluate_on_iterator(iterator, name="oscillators")
    
    def str_to_tree(self, expr: str):
        exprs = [sympy.parse_expr(e) for e in expr.split("|")]
        nodes = [self.env.simplifier.sympy_expr_to_tree(e) for e in exprs]
        return NodeList(nodes)
    
    def read_equations_from_txt_file(self, path: str, save: bool, seed: Union[None, int]):
        # read text file where each line is assumed to be an equation
        # TODO: currently all y0 are set to 1
        _filename = Path(path).name
        if seed is not None:
            np.random.seed(seed)
        iterator = []
        with open(path) as f:
            for line_i, line in enumerate(f):
                samples = {"train": defaultdict(list)}
                line = line.rstrip("\n")
                tree = self.str_to_tree(line)
                eqs = line.split("|")
                dim = len(eqs)
                var_names = [f"x_{k}" for k in range(dim)]
                y0 = np.ones(len(var_names))
                times = np.linspace(0, 5, 256)
                trajectory = self.model.integrate_prediction(
                    times, y0=y0, prediction=line, timeout=self.eval_integration_timeout,
                )
                if np.nan in trajectory:
                    self.trainer.logger.info(
                        f"NaN detected in solution trajectory of {line}. Excluding this equation."
                    )
                    continue
                samples['infos'] = {
                    'dimension': [2],
                    'n_unary_ops': [np.nan],
                    'n_input_points': [len(times)],
                    'name': [f"{_filename}_{line_i:03d}_{line}"],
                }
                samples["train"]['times'].append(times)
                samples["train"]["trajectories"].append(trajectory)
                samples['tree'].append(tree)
                iterator.append((samples, None))
        return iterator
    
    def read_equations_from_json_file(self, path: str, save: bool):
        iterator = []
        with open(path, "r") as fjson:
            store: List[Dict[str, Any]] = json.load(fjson)
        for sample_i, _sample in enumerate(store):
            #for solution_i in range(len(_sample["solutions"][0])):
            
            try:
                samples = {"train": defaultdict(list), "test": defaultdict(list)}
                # first initial condition = train trajectory
                solution_i = 0
                times = np.array(_sample["solutions"][0][solution_i]["t"])
                trajectory = np.array(_sample["solutions"][0][solution_i]["y"]).T
                samples["train"]['times'].append(times)
                samples["train"]['trajectories'].append(trajectory)
                
                samples['infos'] = {
                    'dimension': [trajectory.shape[1]],
                    'n_unary_ops': [np.nan],
                    'n_input_points': [len(times)],
                    'name': [f"{_sample['eq_description']}_{solution_i:2d}"],
                    'dataset': ["strogatz_extended"],
                }
                samples['tree'] = [self.str_to_tree(" | ".join(_sample["substituted"][solution_i]))]
                
                # second initial condition = generalization trajectory
                solution_i = 1
                times = np.array(_sample["solutions"][0][solution_i]["t"])
                trajectory = np.array(_sample["solutions"][0][solution_i]["y"]).T
                samples["test"]['times'].append(times)
                samples["test"]['trajectories'].append(trajectory)
                
                iterator.append(samples)
            except Exception as e:
                self.trainer.logger.error(sample_i, solution_i)
                self.trainer.logger.error(e)
        return iterator
            
    
    def evaluate_on_file(self, path: str, save: bool, seed: Union[None, int]):
        _filename = Path(path).name
        if path.endswith(".pkl"):
            # read pickle file which is assumed to have correct format
            with open(path, "rb") as fpickle:
                iterator = pickle.load(fpickle)
        elif path.endswith(".json"):
            iterator = self.read_equations_from_json_file(path=path, save=save)
        else:
            iterator = self.read_equations_from_txt_file(path=path, save=save, seed=seed)
        ds_path = path+".pkl"
        if not os.path.exists(ds_path):
            with open(ds_path, "wb") as fpickle:
                self.trainer.logger.info(f"Saving dataset under: {ds_path}")
                pickle.dump(iterator, fpickle)
        return self.evaluate_on_iterator(iterator, _filename)
                    

def main(params):

    # initialize the multi-GPU / multi-node training
    # initialize experiment / SLURM signal handler for time limit / pre-emption
    init_distributed_mode(params)
    logger = initialize_exp(params, write_dump_path=False)
    if params.is_slurm_job:
        init_signal_handler()

    # CPU / CUDA
    if not params.cpu:
        assert torch.cuda.is_available()
    params.eval_only = True
    symbolicregression.utils.CUDA = not params.cpu

    # build environment / modules / trainer / evaluator
    if params.batch_size_eval is None:
        params.batch_size_eval = int(1.5 * params.batch_size)

    env = build_env(params)
    env.rng = np.random.RandomState(0)
    modules = build_modules(env, params)
    trainer = Trainer(modules, env, params)
    model = setup_odeformer(trainer)
    evaluator = Evaluator(trainer, model)  


    if params.eval_in_domain:
      scores = evaluator.evaluate_in_domain("functions")
      logger.info("__log__:%s" % json.dumps(scores))

    if params.eval_on_pmlb:
        scores = evaluator.evaluate_on_pmlb()
        logger.info("__pmlb__:%s" % json.dumps(scores))
        scores = evaluator.evaluate_on_oscillators()
        logger.info("__oscillators__:%s" % json.dumps(scores))
    
    if params.eval_on_file is not None:
        evaluator.evaluate_on_file(path=params.eval_on_file, seed=params.random_seed)


if __name__ == "__main__":

    parser = get_parser()
    params = parser.parse_args()

    if params.reload_checkpoint:
        pk = pickle.load(open(params.reload_checkpoint + "/params.pkl", "rb"))
        pickled_args = pk.__dict__
        for p in params.__dict__:
            if p in pickled_args and p not in [
                "dump_path", "reload_checkpoint", "rescale", "validation_metrics", "eval_in_domain", "eval_on_pmlb", 
                "batch_size_eval", "beam_size", "beam_selection_metric", "subsample_prob", "eval_noise_gamma", 
                "eval_subsample_ratio", "eval_noise_type", "use_wandb", "eval_size", "reload_data"
            ]:
                params.__dict__[p] = pickled_args[p]

    if params.eval_dump_path is None:
        params.eval_dump_path = Path(params.dump_path) / "new_evals"
        if not os.path.isdir(params.eval_dump_path):
            os.makedirs(params.eval_dump_path)

    params.is_slurm_job = False
    params.local_rank = -1
    params.master_port = -1
    params.eval_on_file = None 

    main(params)