# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Union
import copy
import json

from pathlib import Path

from logging import getLogger
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
import os
import torch
import numpy as np
from copy import deepcopy
from sklearn.model_selection import GridSearchCV
from symbolicregression.utils import to_cuda
import glob
import scipy
import sympy
import pickle
import wandb

from parsers import get_parser
import symbolicregression
from symbolicregression.slurm import init_signal_handler, init_distributed_mode
from symbolicregression.utils import bool_flag, initialize_exp
from symbolicregression.model import check_model_params, build_modules
from symbolicregression.envs import build_env
from symbolicregression.trainer import Trainer
from symbolicregression.model.sklearn_wrapper import SymbolicTransformerRegressor
from symbolicregression.model.model_wrapper import ModelWrapper
from symbolicregression.metrics import compute_metrics
from sklearn.model_selection import train_test_split
import pandas as pd

from tqdm import tqdm
import time

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
    mw = ModelWrapper(
        env=trainer.env,
        embedder=embedder,
        encoder=encoder,
        decoder=decoder,
        beam_length_penalty=trainer.params.beam_length_penalty,
        beam_size=trainer.params.beam_size,
        max_generated_output_len=trainer.params.max_generated_output_len,
        beam_early_stopping=trainer.params.beam_early_stopping,
        beam_temperature=trainer.params.beam_temperature,
        beam_type=trainer.params.beam_type,
    )
    return SymbolicTransformerRegressor(
        model=mw,
        max_input_points=int(trainer.params.max_points*trainer.params.subsample_ratio),
        rescale=trainer.params.rescale,
        params=trainer.params
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

    ENV = None

    def __init__(self, trainer, model):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.model = model
        # self.modules = trainer.modules
        self.params = trainer.params
        self.env = trainer.env
        # params = self.params
        # Evaluator.ENV = trainer.env
        self.save_path = (
            self.params.eval_dump_path
            if self.params.eval_dump_path
            else self.params.dump_path
            if self.params.dump_path
            else self.params.reload_checkpoint
        )
        if not os.path.exists(self.save_path): os.makedirs(self.save_path)
        
        if hasattr(self.params, "eval_max_samples"):
            self.eval_max_samples = self.params.eval_max_samples
        else:
            self.eval_max_samples = -1

        self.ablation_to_keep = list(
            map(lambda x: "info_" + x, self.params.ablation_to_keep.split(","))
        )

    def evaluate_on_iterator(self, iterator, save_file,):
        self.trainer.logger.info("evaluate_on_iterator")
        scores = OrderedDict({"epoch": self.trainer.epoch})
        batch_results = defaultdict(list)
        for samples_i, (samples, _) in enumerate(
            tqdm(iterator, total=(self.eval_max_samples if self.eval_max_samples != -1 else len(iterator)))
        ):
            if samples_i == self.eval_max_samples:
                break
            times = samples["times"]
            trajectories = samples["trajectory"]
            infos = samples["infos"]

            if "tree" in samples.keys():
                trees = [self.env.simplifier.simplify_tree(tree, expand=True) for tree in samples["tree"]]
                batch_results["trees"].extend(
                    [None if tree is None else tree.infix() for tree in trees]
                )
            else:
                trees = [None]*len(times)
            
            if self.params.max_masked_variables:  # randomly mask some variables
                masked_trajectories = copy.deepcopy(trajectories)
                n_masked_variables_arr = []
                for seq_id in range(len(times)):
                    n_masked_variables = min(np.random.randint(0, self.params.max_masked_variables + 1), infos["dimension"][seq_id]-1)
                    masked_trajectories[seq_id][:, -n_masked_variables:] = np.nan
                    n_masked_variables_arr.append(n_masked_variables)
                infos['n_masked_variables'] = np.array(n_masked_variables_arr)
                all_candidates: Dict[int, List[str]] = self.model.fit(times, masked_trajectories, verbose=False, sort_candidates=True)
            else:
                # if isinstance(self.model, GridSearchCV):
                if hasattr(self.params, "baseline_hyper_opt") and self.params.baseline_hyper_opt:
                    if isinstance(times, List):
                        train_idcs = np.arange(int(np.floor(0.5*len(times[0]))))
                        test_idcs = np.arange(int(np.floor(0.5*len(times[0]))), len(times[0]))
                    else:
                        train_idcs = np.arange(int(np.floor(0.5*len(times))))
                        test_idcs = np.arange(int(np.floor(0.5*len(times))), len(times))
                    _model = self.model.get_grid_search(train_idcs, test_idcs)
                    _model.fit(times[0], trajectories[0])
                    all_candidates = _model.best_estimator_._get_equations()
                else:
                    all_candidates: Dict[int, List[str]] = self.model.fit(times, trajectories, verbose=False, sort_candidates=True)

            best_results = {metric:[] for metric in self.params.validation_metrics.split(',')}
            best_candidates = []
            for time, trajectory, tree, candidates in zip(times, trajectories, trees, all_candidates.values()):
                if not candidates: 
                    for k in best_results:
                        best_results[k].append(np.nan)
                    best_candidates.append(None)
                    continue
                time, idx = sorted(time), np.argsort(time)
                trajectory = trajectory[idx]
                best_candidate = candidates[0] # candidates are sorted
                try: best_candidate = self.env.simplifier.simplify_tree(best_candidate, expand=True)
                except: pass
                # TODO: check dim
                pred_trajectory = self.model.integrate_prediction(
                    time, y0=trajectory[0], prediction=best_candidate
                )
                best_result = compute_metrics(
                    pred_trajectory, 
                    trajectory, 
                    predicted_tree=best_candidate, 
                    tree = tree,
                    metrics=self.params.validation_metrics
                )
                for k, v in best_result.items():
                    best_results[k].append(v[0])
                best_candidates.append(best_candidate)
 
            batch_results["predicted_trees"].extend(
                [
                    tree.infix() if hasattr(tree, 'infix') else tree 
                    for tree in best_candidates
                ]
            )

            for k, v in infos.items():
                if isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
                    infos[k] = v.tolist()
                elif isinstance(v, List):
                    infos[k] = v
                else:
                    raise TypeError(
                        f"v should be of type List of np.ndarray but has type: {type(v)}"
                    )

            for k, v in infos.items():
                batch_results["info_" + k].extend(v)
            for k, v in best_results.items():
                batch_results[k ].extend(v)

        batch_results = pd.DataFrame.from_dict(batch_results)
        batch_results.to_csv(save_file, index=False)
        self.trainer.logger.info("Saved {} equations to {}".format(len(batch_results), save_file))

        try:
            df = pd.read_csv(save_file, na_filter=True)
        except:
            self.trainer.logger.info("WARNING: no results")
            return

        info_columns = [x for x in list(df.columns) if x.startswith("info_")]
        df = df.drop(columns=filter(lambda x: x not in self.ablation_to_keep, info_columns))
        df = df.drop(columns=["predicted_trees"])
        if "trees" in df: df = df.drop(columns=["trees"])
        if "info_name" in df.columns: df = df.drop(columns=["info_name"])

        for metric in self.params.validation_metrics.split(','):
            scores[metric] = df[metric].mean()
                        
        for ablation in self.ablation_to_keep:
            for val, df_ablation in df.groupby(ablation):
                avg_scores_ablation = df_ablation.mean()
                for k, v in avg_scores_ablation.items():
                    if k not in info_columns:
                        scores[k + "_{}_{}".format(ablation, val)] = v
                            
        return scores
        
    def evaluate_in_domain(
        self,
        task,
        save=True,
    ):

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

        if save:
            save_file = os.path.join(self.save_path, "eval_in_domain.csv")

        scores = self.evaluate_on_iterator(iterator,
                                           save_file)
        
        if self.params.use_wandb:
            wandb.log({'in_domain_'+metric: scores[metric] for metric in self.params.validation_metrics.split(',')})

        return scores

    def evaluate_on_pmlb(
        self,
        save=True,
    ):
        
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
            samples = defaultdict(list)
            samples['infos'] = {'dimension':2, 'n_unary_ops':0, 'n_input_points':100, 'name':name}
            for k,v in samples['infos'].items():
                samples['infos'][k] = np.array([v]*4)
            for j in range(4):
                start = j * len(times)
                stop = (j+1) * len(times)
                trajectory = np.concatenate((x[start:stop], y[start:stop]),axis=1)
                times_, trajectory_ = self.env.generator._subsample_trajectory(times, trajectory, subsample_ratio=self.params.subsample_ratio)
                samples['times'].append(times_)
                samples['trajectory'].append(trajectory_)
            iterator.append((samples, None))

        if save:
            save_file = os.path.join(self.save_path, "eval_pmlb.csv")

        scores = self.evaluate_on_iterator(iterator,save_file)

        if self.params.use_wandb:
            wandb.log({'pmlb_'+metric: scores[metric] for metric in self.params.validation_metrics.split(',')})

        return scores
    
    def evaluate_on_oscillators(
        self,
        save=True,
    ):
        
        self.model.rescale = self.params.rescale
        self.trainer.logger.info(
            "====== STARTING EVALUATION OSCILLATORS (multi-gpu: {}) =======".format(
                self.params.multi_gpu
            )
        )

        iterator = []
        datasets = {}
        for file in glob.glob("invar_datasets/*"):
            with open(file) as f:
                lines = (line for line in f if not line.startswith('%') and not line.startswith('x'))
                data = np.loadtxt(lines)
                data = data[data[:,0]==0]
            datasets[file.split('/')[-1]] = data
        
        for name, data in datasets.items():
            samples = defaultdict(list)
            samples['infos'] = {'dimension':2, 'n_unary_ops':0, 'n_input_points':100, 'name':name}
            for k,v in samples['infos'].items():
                samples['infos'][k] = np.array([v])

            times = data[:,1]
            x = data[:,2].reshape(-1,1)
            y = data[:,3].reshape(-1,1)
            # shuffle times and trajectories
            idx = np.linspace(0, len(x)-1, self.dstr.max_input_points).astype(int)
            idx = np.random.permutation(len(times))
            times, x, y = times[idx], x[idx], y[idx]
            
            samples['times'].append(times)
            samples['trajectory'].append(np.concatenate((x,y),axis=1))
            iterator.append((samples, None))

        if save:
            save_file = os.path.join(self.save_path, "eval_oscillators.csv")

        scores = self.evaluate_on_iterator(iterator,save_file)

        if self.params.use_wandb:
            wandb.log({'oscillators_'+metric: scores[metric] for metric in self.params.validation_metrics.split(',')})

        return scores
    
    def evaluate_on_file(self, path: str, save: bool, seed: Union[None, int]):
        _filename = Path(path).name
        if path.endswith(".pkl"):
            # read pickle file which is assumed to have correct format
            with open(path, "rb") as fpickle:
                iterator = pickle.load(fpickle)
        else:
            # read text file where each line is assumed to be an equation
            if seed is not None:
                np.random.seed(seed)
            iterator = []
            with open(path) as f:
                for line_i, line in enumerate(f):
                    samples = defaultdict(list)
                    line = line.rstrip("\n")
                    eqs = line.split("|")
                    dim = len(eqs)
                    var_names = [f"x_{k}" for k in range(dim)]
                    # eqs = [sympy.parse_expr(eq) for eq in eqs]
                    # component_funcs = [sympy.lambdify(",".join(var_names), eq) for eq in eqs]
                    # def ode_func(*args):
                    #     outputs = []
                    #     for cf in component_funcs:
                    #         outputs.append(cf(*args))
                    #     return np.array(outputs).squeeze()
                    y0 = np.ones(len(var_names))
                    times = np.linspace(0, 5, 256)
                    trajectory = self.model.integrate_prediction(
                        times, y0=y0, prediction=line
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
                    samples['times'].append(times)
                    samples["trajectory"].append(trajectory)
                    iterator.append((samples, None))
            with open(path+".pkl", "wb") as fpickle:
                pickle.dump(iterator, fpickle)
                
        if save:
            save_file = os.path.join(self.save_path, f"eval_{_filename}.csv")
        else:
            save_file = None
        return self.evaluate_on_iterator(iterator,save_file)
                    

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
    save = params.save_results

    if params.eval_in_domain:
      scores = evaluator.evaluate_in_domain("functions",save=save,)
      logger.info("__log__:%s" % json.dumps(scores))

    if params.eval_on_pmlb:
        scores = evaluator.evaluate_on_pmlb(save=save)
        logger.info("__pmlb__:%s" % json.dumps(scores))
        scores = evaluator.evaluate_on_oscillators(save=save)
        logger.info("__oscillators__:%s" % json.dumps(scores))
    
    if params.eval_on_file is not None:
        evaluator.evaluate_on_file(path=params.eval_on_file, seed=params.random_seed)


if __name__ == "__main__":

    parser = get_parser()
    params = parser.parse_args()
    pk = pickle.load(open(params.reload_checkpoint + "/params.pkl", "rb"))
    pickled_args = pk.__dict__
    for p in params.__dict__:
        if p in pickled_args and p not in ["dump_path", "reload_checkpoint", "rescale", "validation_metrics", "eval_in_domain", "eval_on_pmlb", "batch_size_eval", "beam_size", "beam_selection_metric", "subsample_prob", "eval_noise_gamma", "eval_noise_type", "use_wandb", "eval_size", "reload_data"]:
            params.__dict__[p] = pickled_args[p]

    params.is_slurm_job = False
    params.local_rank = -1
    params.master_port = -1
    params.use_cross_attention = True
    params.eval_on_file = None #"/p/project/hai_microbio/sb/repos/odeformer/datasets/polynomial_2d.txt.pkl"

    main(params)
