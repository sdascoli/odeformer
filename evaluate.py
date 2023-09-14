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

import odeformer
from parsers import get_parser
from odeformer.slurm import init_signal_handler, init_distributed_mode
from odeformer.utils import initialize_exp
from odeformer.model import build_modules
from odeformer.envs import build_env
from odeformer.envs.generators import NodeList
from odeformer.trainer import Trainer
from odeformer.model.sklearn_wrapper import SymbolicTransformerRegressor
from odeformer.model.model_wrapper import ModelWrapper
from odeformer.metrics import compute_metrics

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
        self.env.rng = np.random.RandomState(self.params.test_env_seed)
        self.save_path = (
            self.params.eval_dump_path
            if self.params.eval_dump_path
            else self.params.dump_path
            if self.params.dump_path
            else self.params.reload_checkpoint
        )
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
        evaluation_task: Literal["interpolation", "forecasting", "y0_generalization"],
    ) -> Dict[str, Dict[str, Any]]:
        
        if "train" not in samples.keys():
            samples["train"] = {"times":samples["times"], "trajectories":samples["trajectory"]}
            del samples["times"], samples["trajectory"]
        assert "test" not in samples.keys(), samples.keys()
        samples["test"] = {"times":[], "trajectories":[]}
        
        if evaluation_task == "interpolation":
            samples["test"] = deepcopy(samples["train"])
            return samples

        elif evaluation_task == "forecasting":
            for time, trajectory, tree in zip(samples["train"]["times"], samples["train"]["trajectories"], samples["tree"]):
                y0 = trajectory[-1]
                t0 = time[-1]
                teval = np.linspace(t0, t0+5, 512, endpoint=True)
                test_trajectory = self.model.integrate_prediction(teval, y0=y0, prediction=tree)
                samples["test"]["trajectories"].append(test_trajectory)
                samples["test"]["times"].append(teval)
            return samples
        
        elif evaluation_task == "y0_generalization":
            for time, trajectory, tree, dimension in zip(samples["train"]["times"], samples["train"]["trajectories"], samples["tree"], samples["infos"]["dimension"]):
                y0 = self.env.rng.randn(dimension)
                test_trajectory = self.model.integrate_prediction(time, y0=y0, prediction=tree)
                samples["test"]["trajectories"].append(test_trajectory)
                samples["test"]["times"].append(time)
            return samples
        
        else:
            raise ValueError(f"Unknown evaluation_task: {evaluation_task}")
            

    def _evaluate(
        self,
        times: List[Dict],
        trajectories: List[Dict],
        trees: List[Union[None, NodeList]],
        all_candidates: Union[List, Dict],
        all_durations: Union[List, Dict],
        validation_metrics: str
    ) -> Tuple[Dict, Dict]:
        
        best_results = {metric: [] for metric in validation_metrics.split(',')}
        best_results["duration_fit"], best_results["pareto_front"], best_candidates = [], [], []
        zipped = [times, trajectories, trees, (all_candidates.values() if isinstance(all_candidates, Dict) else all_candidates)]
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
            pred_trajectory = self.model.integrate_prediction(time, y0=trajectory[0], prediction=best_candidate)
            if not hasattr(self.params, "convert_prediction_to_tree") or self.params.convert_prediction_to_tree:
                try: best_candidate = self.env.simplifier.simplify_tree(best_candidate, expand=True)
                except: pass
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
        return best_results, best_candidates

    def evaluate_on_iterator(self, iterator, name="in_domain"):
        self.trainer.logger.info("evaluate_on_iterator")
        scores = OrderedDict({"epoch": self.trainer.epoch})
        batch_results = defaultdict(list)
        _total = min(self.eval_size, len(iterator)) if self.eval_size > 0 else len(iterator)
        
        for samples_i, samples in enumerate(tqdm(iterator, total=_total)):
            if samples_i == self.eval_size:
                break
            if not "test" in samples.keys():
                samples = self.prepare_test_trajectory(samples, evaluation_task=self.params.evaluation_task)
            times, trajectories, infos = samples["train"]["times"], samples["train"]["trajectories"], samples["infos"]
            for k, v in infos.items():
                if isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
                    infos[k] = v.tolist()
                elif isinstance(v, List):
                    infos[k] = v
                else:
                    raise TypeError(
                        f"v should be of type List of np.ndarray but has type: {type(v)}"
                    )
            
            if "tree" in samples.keys():
                trees = [self.env.simplifier.simplify_tree(tree, expand=True) for tree in samples["tree"]]
                batch_results["trees"].extend(
                    [None if tree is None else tree.infix() for tree in trees]
                )
            else:
                trees = [None]*len(times)

            original_times, original_trajectories = deepcopy(times), deepcopy(trajectories)

            # corrupt training data
            for i, (time, trajectory) in enumerate(zip(times, trajectories)):
                if self.params.eval_noise_gamma:
                    noise, gamma = self.env._create_noise(
                        train=False,
                        trajectory=trajectory,
                        gamma=self.params.eval_noise_gamma,
                        seed=self.params.test_env_seed,
                    )
                    trajectory += noise

                if self.params.eval_subsample_ratio:
                    time, trajectory, subsample_ratio = self.env._subsample_trajectory(
                        time,
                        trajectory,
                        subsample_ratio=self.params.eval_subsample_ratio,
                        seed=self.params.test_env_seed,
                    )
                times[i] = time
                trajectories[i] = trajectory

            # fit
            start_time_fit = timer()
            all_candidates = self.model.fit(times, trajectories, verbose=False, sort_candidates=True)
            all_duration_fit = [timer() - start_time_fit] * len(times)
            #all_candidates, all_duration_fit = dict(), dict()
            #for _trajectory_i, (_times, _trajectory) in enumerate(zip(times, trajectories)):
            #    start_time_fit = timer()
            #    all_candidates[_trajectory_i] = self.model.fit(_times, _trajectory)[0]
            #    all_duration_fit[_trajectory_i] = [timer() - start_time_fit]
            
            # evaluate on train data
            best_results, best_candidates = self._evaluate(
                original_times, original_trajectories, trees, all_candidates, all_duration_fit, self.params.validation_metrics
            )
            # evaluate on test data
            test_results, _ = self._evaluate(
                times=samples["test"]["times"],
                trajectories=samples["test"]["trajectories"],
                trees=trees,
                all_candidates=best_candidates,
                all_durations=None,
                validation_metrics=self.params.validation_metrics
            )
            # collect results
            batch_results["predicted_trees"].extend([tree.infix() if hasattr(tree, 'infix') else tree for tree in best_candidates])
            for k, v in infos.items():
                batch_results["info_" + k].extend(v)
            for k, v in best_results.items():
                batch_results[k].extend(v)
            for k, v in test_results.items():
                if k == "duration_fit": continue
                batch_results['test_'+k].extend(v)

        batch_results = pd.DataFrame.from_dict(batch_results)

        save_file = os.path.join(self.save_path, f"eval_{name}.csv")

        batch_results.to_csv(save_file, index=False)
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
                scores[prefix+metric] = df[prefix+metric].mean()
                scores[prefix+metric+'_median'] = df[prefix+metric].median()

        scores["duration_fit"] = df["duration_fit"].mean()
                        
        # for ablation in self.ablation_to_keep:
        #     for val, df_ablation in df.groupby(ablation):
        #         avg_scores_ablation = df_ablation.mean()
        #         for k, v in avg_scores_ablation.items():
        #             if k not in info_columns:
        #                 scores[k + "_{}_{}".format(ablation, val)] = v

        if self.params.use_wandb:
            wandb.log({name+"_"+metric: score for metric, score in scores.items() if "median" not in metric})

        return scores
        
    def evaluate_in_domain(
        self,
        task,
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

        scores = self.evaluate_on_iterator(iterator, name = "in_domain")
        
        return scores

    def evaluate_on_pmlb(
        self,
        path_dataset=None,
    ):
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
                    # times_, trajectory_ = self.env.generator._subsample_trajectory(times, trajectory, subsample_ratio=self.params.subsample_ratio)
                    samples["train"]['times'].append(deepcopy(times))
                    samples["train"]['trajectories'].append(trajectory)
                    samples['tree'] = [self.str_to_tree(format_strogatz_equation(strogatz_equations[name]))]
                    samples['infos'] = infos
                    # for k,v in samples['infos'].items():
                    #     samples['infos'][k] = np.array([v]*4)
                    iterator.append(samples)
            if path_dataset:
                with open(path_dataset, "wb") as fout:
                    self.trainer.logger.info(f"Saving dataset under:\n{path_dataset}")
                    pickle.dump(obj=iterator, file=fout)

        scores = self.evaluate_on_iterator(iterator, name="pmlb")

        return scores
    
    def evaluate_on_oscillators(
        self,
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
            samples = {"train": defaultdict(list)}
            samples['infos'] = {'dimension':2, 'n_unary_ops':0, 'n_input_points':100, 'name':name}
            for k,v in samples['infos'].items():
                samples['infos'][k] = np.array([v])

            times = data[:,1]
            x = data[:,2].reshape(-1,1)
            y = data[:,3].reshape(-1,1)
            # shuffle times and trajectories
            #idx = np.linspace(0, len(x)-1, self.dstr.max_input_points).astype(int)
            if hasattr(self.model, "max_input_points"):
                idx = np.random.permutation(len(times))
                times, x, y = times[idx], x[idx], y[idx]
            
            samples["train"]['times'].append(times)
            samples["train"]['trajectories'].append(np.concatenate((x,y),axis=1))
            samples["tree"] = [None]
            iterator.append(samples)

        scores = self.evaluate_on_iterator(iterator,
                                           name="oscillators")

        return scores
    
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
                samples["train"]['times'].append(times)
                samples["train"]["trajectories"].append(trajectory)
                samples['tree'].append(tree)
                iterator.append((samples, None))
        with open(path+".pkl", "wb") as fpickle:
            pickle.dump(iterator, fpickle)
        return iterator
    
    def read_equations_from_json_file(self, path: str, save: bool):
        iterator = []
        with open(path, "r") as fjson:
            store: List[Dict[str, Any]] = json.load(fjson)
        for sample_i, _sample in enumerate(store):
            for solution_i in range(len(_sample["solutions"])):
                try:
                    samples = {"train": defaultdict(list)}
                    times = np.array(_sample["solutions"][solution_i][0]["t"])
                    trajectory = np.array(_sample["solutions"][solution_i][0]["y"]).T
                    samples['infos'] = {
                        'dimension': [trajectory.shape[1]],
                        'n_unary_ops': [np.nan],
                        'n_input_points': [len(times)],
                        'name': [f"{_sample['eq_description']}_{solution_i:2d}"],
                        'dataset': ["strogatz_extended"],
                    }
                    samples["train"]['times'].append(times)
                    samples["train"]['trajectories'].append(trajectory)
                    samples['tree'] = [self.str_to_tree(" | ".join(_sample["substituted"][solution_i]))]
                    iterator.append(samples)
                except Exception as e:
                    print(sample_i, solution_i)
                    print(e)
                
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
        if save:
            save_file = os.path.join(self.save_path, f"eval_{_filename}.csv")
        else:
            save_file = None
        return self.evaluate_on_iterator(iterator, save_file)
                    

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
    odeformer.utils.CUDA = not params.cpu

    # build environment / modules / trainer / evaluator
    if params.batch_size_eval is None:
        params.batch_size_eval = int(1.5 * params.batch_size)

    env = build_env(params)
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
        # scores = evaluator.evaluate_on_oscillators()
        # logger.info("__oscillators__:%s" % json.dumps(scores))
    
    if params.eval_on_file is not None:
        evaluator.evaluate_on_file(path=params.eval_on_file, seed=params.test_env_seed)


if __name__ == "__main__":

    parser = get_parser()
    params = parser.parse_args()

    if params.reload_checkpoint:
        pk = pickle.load(open(params.reload_checkpoint + "/params.pkl", "rb"))
        pickled_args = pk.__dict__
        for p in params.__dict__:
            if p in pickled_args and p not in ["eval_dump_path", "dump_path", "reload_checkpoint", "rescale", "validation_metrics", "eval_in_domain", "eval_on_pmlb", "batch_size_eval", "beam_size", "beam_selection_metric", "subsample_prob", "eval_noise_gamma", "eval_subsample_ratio", "use_wandb", "eval_size", "reload_data"]:
                params.__dict__[p] = pickled_args[p]

    if params.eval_dump_path is None:
        params.eval_dump_path = Path(params.dump_path) / "new_evals"
        if not os.path.isdir(params.eval_dump_path):
            os.makedirs(params.eval_dump_path)

    params.is_slurm_job = False
    params.local_rank = -1
    params.master_port = -1
    params.eval_on_file = None 

    torch.save(params, os.path.join(params.dump_path, "params.pkl"))

    main(params)