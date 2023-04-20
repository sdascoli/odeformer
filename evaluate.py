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
        Evaluator.ENV = trainer.env

    def set_env_copies(self, data_types):
        for data_type in data_types:
            setattr(self, "{}_env".format(data_type), deepcopy(self.env))

    def evaluate_in_domain(
        self,
        data_type,
        task,
        verbose=True,
        ablation_to_keep=None,
        save=False,
        logger=None,
        save_file=None,
    ):

        """
        Encoding / decoding step with beam generation and SymPy check.
        """
        scores = OrderedDict({"epoch": self.trainer.epoch})

        params = self.params
        logger.info(
            "====== STARTING EVALUATION (multi-gpu: {}) =======".format(
                params.multi_gpu
            )
        )

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

        env = getattr(self, "{}_env".format(data_type))

        eval_size_per_gpu = params.eval_size
        iterator = env.create_test_iterator(
            data_type,
            task,
            data_path=self.trainer.data_path,
            batch_size=params.batch_size_eval,
            params=params,
            size=eval_size_per_gpu,
            test_env_seed=self.params.test_env_seed,
        )

        mw = ModelWrapper(
            env=env,
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

        dstr = SymbolicTransformerRegressor(
            model=mw,
            max_input_points=params.max_len,
            n_trees_to_refine=params.n_trees_to_refine,
            rescale=False,
        )

        first_write = True
        if save:
            if save_file is None:
                save_file = (
                    self.params.eval_dump_path
                    if self.params.eval_dump_path is not None
                    else self.params.dump_path
                )
            if not os.path.exists(save_file):
                os.makedirs(save_file)
            save_file = os.path.join(save_file, "eval_in_domain.csv")

        batch_before_writing_threshold = min(
            2, eval_size_per_gpu // params.batch_size_eval
        )
        batch_before_writing = batch_before_writing_threshold

        if ablation_to_keep is not None:
            ablation_to_keep = list(
                map(lambda x: "info_" + x, ablation_to_keep.split(","))
            )
        else:
            ablation_to_keep = []

        pbar = tqdm(total=eval_size_per_gpu)

        batch_results = defaultdict(list)

        for samples, _ in iterator:

            times = samples["times"]
            trajectories = samples["trajectory"]
            infos = samples["infos"]
            tree = samples["tree"]

            all_candidates = dstr.fit(times, trajectories, verbose=verbose)
            best_results = []
            for time, trajectory, info, tree, candidates in zip(times, trajectories, infos, tree, all_candidates):
                results = []
                for candidate in candidates:
                    pred_trajectory = [dstr.predict(time, y0=trajectory[0], tree=candidate)]
                    result = compute_metrics(
                    {
                        "true": trajectory,
                        "predicted": pred_trajectory,
                        "tree": candidate,
                    },
                    metrics=params.validation_metrics,
                    )
                    results.append(result)
                best_result = max(results, key=lambda x: x[params.beam_selection_metric])
                best_results.append(best_result)

            final_results = defaultdict(list)
            for best_result in best_results:
                for k, v in best_result.items():
                    final_results[k].append(v)
 
            for k, v in infos.items():
                infos[k] = v.tolist()

            batch_results = defaultdict(list)
            batch_results["tree"].extend(candidates)
            for k, v in infos.items():
                batch_results["info_" + k].extend(v)        
            for k, v in final_results.items():
                batch_results[k ].extend(v)
                
            if save:
                batch_before_writing -= 1
                if batch_before_writing <= 0:
                    batch_results = pd.DataFrame.from_dict(batch_results)
                    if first_write:
                        batch_results.to_csv(save_file, index=False)
                        if logger is not None:
                            logger.info("Just started saving")
                        first_write = False
                    else:
                        batch_results.to_csv(
                            save_file, mode="a", header=False, index=False
                        )
                        if logger is not None:
                            logger.info(
                                "Saved {} equations".format(
                                    self.params.batch_size_eval
                                    * batch_before_writing_threshold
                                )
                            )
                    batch_before_writing = batch_before_writing_threshold
                    batch_results = defaultdict(list)
            bs = len(times)
            pbar.update(bs)

        try:
            df = pd.read_csv(save_file, na_filter=True)
        except:
            logger.info("WARNING: no results")
            return
        info_columns = filter(lambda x: x.startswith("info_"), df.columns)
        df = df.drop(columns=filter(lambda x: x not in ablation_to_keep, info_columns))

        for ablation in ablation_to_keep:
            for val, df_ablation in df.groupby(ablation):
                avg_scores_ablation = df_ablation.mean()
                for k, v in avg_scores_ablation.items():
                    scores[k + "_{}_{}".format(ablation, val)] = v
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
    scores = {}
    save = params.save_results

    if params.eval_in_domain:
        evaluator.set_env_copies(["valid1"])
        scores = evaluator.evaluate_in_domain(
            "valid1",
            "functions",
            save=save,
            logger=logger,
            ablation_to_keep=params.ablation_to_keep,
        )
        logger.info("__log__:%s" % json.dumps(scores))

    if params.eval_on_pmlb:
        target_noise = params.target_noise
        random_state = params.random_state
        data_type = params.pmlb_data_type

        if data_type == "feynman":
            filter_fn = lambda x: x["dataset"].str.contains("feynman")
        elif data_type == "strogatz":
            print("Strogatz data")
            filter_fn = lambda x: x["dataset"].str.contains("strogatz")
        elif data_type == "603_fri_c0_250_50":
            filter_fn = lambda x: x["dataset"].str.contains("603_fri_c0_250_50")
        else:
            filter_fn = lambda x: ~(
                x["dataset"].str.contains("strogatz")
                | x["dataset"].str.contains("feynman")
            )

        pmlb_scores = evaluator.evaluate_pmlb(
            target_noise=target_noise,
            verbose=params.eval_verbose_print,
            random_state=random_state,
            save=save,
            filter_fn=filter_fn,
            logger=logger,
            save_file=None,
            save_suffix="eval_pmlb.csv",
        )
        logger.info("__pmlb__:%s" % json.dumps(pmlb_scores))


if __name__ == "__main__":

    parser = get_parser()
    params = parser.parse_args()
    # params.reload_checkpoint = "/checkpoint/sdascoli/symbolicregression/shift_all/use_skeleton_True_use_sympy_False_tokens_per_batch_10000_n_enc_layers_4_n_dec_layers_16"
    #params.reload_checkpoint = "/checkpoint/sdascoli/symbolicregression/shift_all/use_skeleton_False_use_sympy_False_tokens_per_batch_10000_n_enc_layers_4_n_dec_layers_16/"
    # params.reload_checkpoint = "/checkpoint/sdascoli/symbolicregression/newgen/use_skeleton_False_use_sympy_False_tokens_per_batch_10000_n_enc_layers_4_n_dec_layers_16/"
    pk = pickle.load(open(params.reload_checkpoint + "/params.pkl", "rb"))
    pickled_args = pk.__dict__
    for p in params.__dict__:
        if p in pickled_args and p not in ["dump_path", "reload_checkpoint"]:
            params.__dict__[p] = pickled_args[p]

    params.multi_gpu = False
    params.is_slurm_job = False
    params.eval_on_pmlb = True  # True
    params.eval_in_domain = False
    params.local_rank = -1
    params.master_port = -1
    params.num_workers = 1
    params.target_noise = 0.0
    params.max_input_points = 200
    params.random_state = 14423
    params.max_number_bags = 10
    params.save_results = False
    params.eval_verbose_print = True
    params.beam_size = 1
    params.rescale = True
    params.max_input_points = 200
    params.pmlb_data_type = "black_box"
    params.n_trees_to_refine = 10
    main(params)
