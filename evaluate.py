# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
from symbolicregression.utils import to_cuda
import glob
import scipy
import pickle

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

np.seterr(all="raise")


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

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.modules = trainer.modules
        self.params = trainer.params
        self.env = trainer.env
        params = self.params
        Evaluator.ENV = trainer.env
        embedder = (
            self.modules["embedder"].module
            if params.multi_gpu
            else self.modules["embedder"]
        )
        encoder = (
            self.modules["encoder"].module
            if params.multi_gpu
            else self.modules["encoder"]
        )
        decoder = (
            self.modules["decoder"].module
            if params.multi_gpu
            else self.modules["decoder"]
        )
        embedder.eval()
        encoder.eval()
        decoder.eval()
        mw = ModelWrapper(
            env=self.env,
            embedder=embedder,
            encoder=encoder,
            decoder=decoder,
            beam_length_penalty=params.beam_length_penalty,
            beam_size=params.beam_size,
            max_generated_output_len=params.max_generated_output_len,
            beam_early_stopping=params.beam_early_stopping,
            beam_temperature=params.beam_temperature,
            beam_type=params.beam_type,
        )
        self.dstr = SymbolicTransformerRegressor(
            model=mw,
            max_input_points=int(params.max_points*params.subsample_ratio),
            rescale=params.rescale,
            params=params
        )

        self.save_path = (
                    self.params.eval_dump_path
                    if self.params.eval_dump_path
                    else self.params.dump_path
                    if self.params.dump_path
                    else self.params.reload_checkpoint
                )
        if not os.path.exists(self.save_path): os.makedirs(self.save_path)

        self.ablation_to_keep = list(map(lambda x: "info_" + x, params.ablation_to_keep.split(",")))

    def evaluate_on_iterator(self,iterator,save_file,):
        
        scores = OrderedDict({"epoch": self.trainer.epoch})
        batch_results = defaultdict(list)
        for samples, _ in tqdm(iterator):

            times = samples["times"]
            trajectories = samples["trajectory"]
            infos = samples["infos"]
            if "tree" in samples.keys():
                trees = samples["tree"]
                batch_results["trees"].extend([self.env.simplifier.readable_tree(tree) for tree in trees])

            all_candidates = self.dstr.fit(times, trajectories, verbose=False)

            best_results = {metric:[] for metric in self.params.validation_metrics.split(',')}
            best_candidates = []
            for time, trajectory, candidates in zip(times, trajectories, all_candidates.values()):
                results = []
                if not candidates: 
                    for k in best_results:
                        best_results[k].append(np.nan)
                    best_candidates.append(None)
                    continue
                for candidate in candidates:
                    time, idx = sorted(time), np.argsort(time)
                    trajectory = trajectory[idx]
                    pred_trajectory = self.dstr.predict(time, y0=trajectory[0], tree=candidate)
                    result = compute_metrics(pred_trajectory, trajectory, predicted_tree=candidate, metrics=self.params.validation_metrics)
                    results.append(result)
                best_result_id = max(range(len(results)), key=lambda i: results[i][self.params.beam_selection_metric])
                best_result = results[best_result_id]
                best_candidate = candidates[best_result_id]
                for k, v in best_result.items():
                    best_results[k].append(v[0])
                best_candidates.append(best_candidate)
 
            for k, v in infos.items():
                infos[k] = v.tolist()

            batch_results["predicted_trees"].extend([self.env.simplifier.readable_tree(tree) for tree in best_candidates])

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

        info_columns = filter(lambda x: x.startswith("info_"), df.columns)
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
                    scores[k + "_{}_{}".format(ablation, val)] = v
                    
        return scores
        
    def evaluate_in_domain(
        self,
        task,
        save=True,
    ):

        self.dstr.rescale = False
        self.trainer.logger.info("====== STARTING EVALUATION IN DOMAIN (multi-gpu: {}) =======".format(self.params.multi_gpu))

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

        return scores

    def evaluate_on_pmlb(
        self,
        save=True,
    ):
        
        self.dstr.rescale = self.params.rescale
        self.trainer.logger.info("====== STARTING EVALUATION PMLB (multi-gpu: {}) =======".format(self.params.multi_gpu))

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
                samples['times'].append(times)
                samples['trajectory'].append(np.concatenate((x[start:stop], y[start:stop]),axis=1))
            iterator.append((samples, None))

        if save:
            save_file = os.path.join(self.save_path, "eval_pmlb.csv")

        scores = self.evaluate_on_iterator(iterator,save_file)

        return scores
    
    def evaluate_on_oscillators(
        self,
        save=True,
    ):
        
        self.dstr.rescale = self.params.rescale
        self.trainer.logger.info("====== STARTING EVALUATION OSCILLATORS (multi-gpu: {}) =======".format(self.params.multi_gpu))

        iterator = []
        datasets = {}
        for file in glob.glob("~/odeformer/invar_datasets/*"):
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
            idx = np.random.permutation(len(times))
            times, x, y = times[idx], x[idx], y[idx]
            
            samples['times'].append(times)
            samples['trajectory'].append(np.concatenate((x,y),axis=1))
            iterator.append((samples, None))

        if save:
            save_file = os.path.join(self.save_path, "eval_oscillators.csv")

        scores = self.evaluate_on_iterator(iterator,save_file)

        return scores


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
    evaluator = Evaluator(trainer)
    save = params.save_results

    if params.eval_in_domain:
      scores = evaluator.evaluate_in_domain("functions",save=save,)
      logger.info("__log__:%s" % json.dumps(scores))

    if params.eval_on_pmlb:
        pmlb_scores = evaluator.evaluate_on_pmlb(save=save)
        logger.info("__pmlb__:%s" % json.dumps(pmlb_scores))
        osc_scores = evaluator.evaluate_on_oscillators(save=save)
        logger.info("__oscillators__:%s" % json.dumps(osc_scores))


if __name__ == "__main__":

    parser = get_parser()
    params = parser.parse_args()
    pk = pickle.load(open(params.reload_checkpoint + "/params.pkl", "rb"))
    pickled_args = pk.__dict__
    for p in params.__dict__:
        if p in pickled_args and p not in ["dump_path", "reload_checkpoint", "rescale", "validation_metrics", "eval_in_domain", "eval_on_pmlb", "batch_size_eval", "beam_size", "beam_selection_metric", "subsample_prob", "eval_noise_gamma", "eval_noise_type"]:
            params.__dict__[p] = pickled_args[p]

    params.eval_size = 10
    params.is_slurm_job = False
    params.local_rank = -1
    params.master_port = -1
    params.use_cross_attention = True

    main(params)
