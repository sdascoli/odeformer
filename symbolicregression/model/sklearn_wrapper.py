# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math, time, copy
import numpy as np
import torch
from collections import defaultdict
from symbolicregression.metrics import compute_metrics
from sklearn.base import BaseEstimator
import symbolicregression.model.utils_wrapper as utils_wrapper
import traceback
from sklearn import feature_selection 
from symbolicregression.envs.generators import integrate_ode
from symbolicregression.envs.utils import *
import warnings
import scipy

def exchange_node_values(tree, dico):
    new_tree = copy.deepcopy(tree)
    for (old, new) in dico.items():
        new_tree.replace_node_value(old, new)
    return new_tree

class SymbolicTransformerRegressor(BaseEstimator):

    def __init__(self,
                model=None,
                max_input_points=10000,
                max_number_bags=-1,
                stop_refinement_after=1,
                n_trees_to_refine=1,
                rescale=False
                ):

        self.max_input_points = max_input_points
        self.max_number_bags = max_number_bags
        self.model = model
        self.stop_refinement_after = stop_refinement_after
        self.n_trees_to_refine = n_trees_to_refine
        self.rescale = rescale

    def set_args(self, args={}):
        for arg, val in args.items():
            assert hasattr(self, arg), "{} arg does not exist".format(arg)
            setattr(self, arg, val)

    def fit(
        self,
        times,
        trajectories,
        verbose=False
    ):
        self.start_fit = time.time()

        if not isinstance(times, list):
            times = [times]
            trajectories = [trajectories]
        n_datasets = len(times)
    
        scaler = utils_wrapper.MinMaxScaler() if self.rescale else None
        scale_params = {}
        if scaler is not None:
            scaled_times = []
            for i, x in enumerate(times):
                scaled_times.append(scaler.fit_transform(x))
                scale_params[i]=scaler.get_params()
        else:
            scaled_times = times

        inputs, inputs_ids = [], []
        for seq_id in range(len(scaled_times)):
            for seq_l in range(len(scaled_times[seq_id])):
                y_seq = trajectories[seq_id]
                if len(y_seq.shape)==1:
                    y_seq = np.expand_dims(y_seq,-1)
                if seq_l%self.max_input_points == 0:
                    inputs.append([])
                    inputs_ids.append(seq_id)
                inputs[-1].append([scaled_times[seq_id][seq_l], y_seq[seq_l]])

        if self.max_number_bags>0:
            inputs = inputs[:self.max_number_bags]
            inputs_ids = inputs_ids[:self.max_number_bags]

        # Forward transformer
        forward_time=time.time()
        outputs = self.model(inputs)  ##Forward transformer: returns predicted functions
        if verbose: print("Finished forward in {} secs".format(time.time()-forward_time))

        all_candidates = defaultdict(list)
        assert len(inputs) == len(outputs), "Problem with inputs and outputs"
        for i in range(len(inputs)):
            input_id = inputs_ids[i]
            candidates = outputs[i]
            all_candidates[input_id].extend(candidates)
        assert len(all_candidates.keys())==n_datasets
            
        self.trees = {}
        for input_id, candidates in all_candidates.items():
            if len(candidates)==0: 
                self.trees[input_id] = None
            else:
                self.trees[input_id] = candidates

        return all_candidates

    @torch.no_grad()
    def evaluate_tree(self, tree, times, trajectory, metric):
        pred_trajectory = self.predict(times, trajectory[0], tree=tree)
        metrics = compute_metrics(pred_trajectory, trajectory, predicted_tree=tree, metrics=metric)
        return metrics[metric][0]

    def order_candidates(self, times, y, candidates, metric="_mse", verbose=False):
        scores = []
        for candidate in candidates:
            if metric not in candidate:
                score = self.evaluate_tree(candidate["predicted_tree"], times, y, metric)
                if math.isnan(score): 
                    score = np.infty if metric.startswith("_") else -np.infty
            else:
                score = candidates[metric]
            scores.append(score)
        ordered_idx = np.argsort(scores)  
        if not metric.startswith("_"): ordered_idx=list(reversed(ordered_idx))
        candidates = [candidates[i] for i in ordered_idx]
        return candidates

    def predict(self, times, y0, tree=None):   

        if tree is None:
            if self.trees[0] is None:
                return None
            else:
                tree = self.trees[0][0]

        trajectory = integrate_ode(tree, y0, times)
        
        return trajectory
            
    def __str__(self):
        if hasattr(self, "tree"):
            for tree_idx in range(len(self.trees)):
                for gen in self.trees[tree_idx]:
                    print(gen)
        return "Transformer"