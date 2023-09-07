from typing import Union
from pathlib import Path

import os
import torch
import numpy as np

from evaluate import (
    Trainer,
    Evaluator, 
    build_env, 
    get_parser, 
    build_modules, 
    initialize_exp, 
    setup_odeformer, 
    symbolicregression,
    init_distributed_mode, 
)

def main(params):
    env = build_env(params)
    env.rng = np.random.RandomState(0)
    modules = build_modules(env, params)
    trainer = Trainer(modules, env, params)

    if "odeformer" in params.model:
        model = setup_odeformer(trainer)
    elif params.model == "sindy_all":
        from symbolicregression.baselines.sindy_wrapper import SINDyWrapper
        model = SINDyWrapper(
            model_dir=params.eval_dump_path,
            polynomial_degree=10,
            functions=None, # all functions
            grid_search_polynomial_degree=True,
            grid_search_functions=True,
            optimize_hyperparams=params.optimize_hyperparams,
            hyper_opt_eval_fraction=params.hyper_opt_eval_fraction,
            sorting_metric=params.sorting_metric,
        )
    elif params.model == "sindy_esc":
        from symbolicregression.baselines.sindy_wrapper import SINDyWrapper
        model = SINDyWrapper(
            model_dir=params.eval_dump_path,
            polynomial_degree=10,
            functions=["sin", "cos", "exp"],
            grid_search_polynomial_degree=True,
            optimize_hyperparams=params.optimize_hyperparams,
            hyper_opt_eval_fraction=params.hyper_opt_eval_fraction,
            sorting_metric=params.sorting_metric,
        )
    elif params.model == "sindy_poly":
        from symbolicregression.baselines.sindy_wrapper import SINDyWrapper
        model = SINDyWrapper(
            model_dir=params.eval_dump_path,
            polynomial_degree=10,
            functions=[], # only polynomials
            grid_search_polynomial_degree=True,
            optimize_hyperparams=params.optimize_hyperparams,
            hyper_opt_eval_fraction=params.hyper_opt_eval_fraction,
            sorting_metric=params.sorting_metric,
        )
    elif params.model == "pysr":
        from symbolicregression.baselines.pysr_wrapper import PySRWrapper
        model = PySRWrapper(
            model_dir=params.eval_dump_path,
            optimize_hyperparams=True,
            hyper_opt_eval_fraction=params.hyper_opt_eval_fraction,
            sorting_metric=params.sorting_metric,
        )
    elif params.model == "pysr_poly":
        from symbolicregression.baselines.pysr_wrapper import PySRWrapper
        model = PySRWrapper(
            model_dir=params.eval_dump_path,
            optimize_hyperparams=True,
            hyper_opt_eval_fraction=params.hyper_opt_eval_fraction,
            sorting_metric=params.sorting_metric,
            unary_operators=[]
        )
    elif params.model == "proged":
        from symbolicregression.baselines.proged_wrapper import ProGEDWrapper
        model = ProGEDWrapper(
            model_dir=params.eval_dump_path,
            optimize_hyperparams=True,
            hyper_opt_eval_fraction=params.hyper_opt_eval_fraction,
            sorting_metric=params.sorting_metric,
        )
    elif params.model == "proged_poly":
        from symbolicregression.baselines.proged_wrapper import ProGEDWrapper
        model = ProGEDWrapper(
            model_dir=params.eval_dump_path,
            optimize_hyperparams=True,
            hyper_opt_eval_fraction=params.hyper_opt_eval_fraction,
            sorting_metric=params.sorting_metric,
            generator_template_name="polynomial",
            grid_search_generator_template_name=False,
        )
    elif params.model == "afp":
        from symbolicregression.baselines.ellyn_wrapper import AFPWrapper
        model = AFPWrapper(
            model_dir=params.eval_dump_path,
            optimize_hyperparams=True,
            hyper_opt_eval_fraction=params.hyper_opt_eval_fraction,
            sorting_metric=params.sorting_metric,
        )
    elif params.model == "ehc":
        from symbolicregression.baselines.ellyn_wrapper import EHCWrapper
        model = EHCWrapper(
            model_dir=params.eval_dump_path,
            optimize_hyperparams=True,
            hyper_opt_eval_fraction=params.hyper_opt_eval_fraction,
            sorting_metric=params.sorting_metric,
        )
    elif params.model == "eplex":
        from symbolicregression.baselines.ellyn_wrapper import EPLEXWrapper
        model = EPLEXWrapper(
            model_dir=params.eval_dump_path,
            optimize_hyperparams=True,
            hyper_opt_eval_fraction=params.hyper_opt_eval_fraction,
            sorting_metric=params.sorting_metric,
        )
    elif params.model == "feafp":
        from symbolicregression.baselines.ellyn_wrapper import FEAFPWrapper
        model = FEAFPWrapper(
            model_dir=params.eval_dump_path,
            optimize_hyperparams=True,
            hyper_opt_eval_fraction=params.hyper_opt_eval_fraction,
            sorting_metric=params.sorting_metric,
        )
    elif params.model == "ffx":
        from symbolicregression.baselines.ffx_wrapper import FFXWrapper
        model = FFXWrapper(
            model_dir=params.eval_dump_path,
            optimize_hyperparams=True,
            hyper_opt_eval_fraction=params.hyper_opt_eval_fraction,
            sorting_metric=params.sorting_metric,
        )
    elif params.model in ["odeformer", "odeformer_opt", "odeformer_opt_random"]:
        model = setup_odeformer(trainer)
    else:
        raise ValueError(f"Unknown model: {params.model}")
        
    evaluator = Evaluator(trainer, model)
    
    if params.eval_on_file:
        evaluator.evaluate_on_file(path=params.eval_on_file, save=params.save_results, seed=params.test_env_seed)
        
    if params.eval_on_pmlb:
        evaluator.evaluate_on_pmlb(path_dataset=params.path_dataset)
        
    if params.eval_on_oscillators:
        evaluator.evaluate_on_oscillators()
    
def str2bool(arg: Union[bool, str]):
    if isinstance(arg, bool):
        return arg
    if arg.lower() in ["true", "yes", "t", "y"]:
        return True
    return False

def str_or_None(arg: str):
    if arg.lower() == "none":
        return None
    return arg
    
if __name__ == "__main__":
    BASE = os.path.join(os.getcwd(), "experiments")
    parser = get_parser()
    parser.add_argument("--model", type=str, default="odeformer",
        choices=[
            "afp", "feafp", "eplex", "ehc",
            "proged", "proged_poly",
            "ffx",
            "pysr", "pysr_poly",
            "sindy_all", "sindy_esc", "sindy_poly",
            "odeformer", "odeformer_opt", "odeformer_opt_random"
        ]
    )
    parser.add_argument("--dataset", type=str, choices=["strogatz","strogatz_extended", "oscillators", "<path_to_dataset>"], 
        default="strogatz"
    )
    parser.add_argument("--optimize_hyperparams", type=str2bool, default=True, 
        help="Do / Don't optimizer hyper parameters."
    )
    parser.add_argument("--hyper_opt_eval_fraction", type=float, default=0.3,
        help="Fraction of trajectory length that is used to score hyper parameter optimization on."
    )
    parser.add_argument("--convert_prediction_to_tree", type=str2bool, default=False,
        help = "If True, we attempt to convert predicted equations to Odeformers tree format."
    )
    parser.add_argument("--sorting_metric", type=str_or_None, default="r2",
        help = "If not None, sort pareto front according to this metric before selecting the final, best model."
    )
    parser.add_argument("--e_task",# this overwrites --evaluation_task from parser.py
        type=str, choices=["debug", "interpolation", "forecasting", "y0_generalization"], default="forecasting",
    )
    parser.add_argument("--reload_scores_path", type=str_or_None, 
        default = None,
        help="Path to existing scores.csv from which candidates are re-loaded for re-evaluation, e.g. for forecasting or y0 generalization."
    )
    
    params = parser.parse_args()
    
    if params.reload_scores_path is not None:
        assert os.path.exists(params.reload_scores_path), params.reload_scores_path
    
    params.validation_metrics = 'r2,r2_zero,snmse,accuracy_l1_1e-1,accuracy_l1_1e-3,accuracy_l1_biggio,is_valid,complexity_sympy,relative_complexity_sympy,complexity_string,relative_complexity_string' # complexity,term_difference,term_difference_sympy
    params.eval_only = True
    params.max_dimension = 5
    params.evaluation_task = params.e_task
    # params.eval_size = 2
    
    if params.dataset == "strogatz":
        params.eval_on_file = False
        params.eval_on_oscillators = False
        params.eval_on_pmlb = True
        params.path_dataset = "datasets/strogatz.pkl"
        dataset_name = params.dataset
    elif params.dataset == "strogatz_extended":
        params.eval_on_file = "datasets/strogatz_extended/strogatz_extended.json.pkl"
        params.eval_on_oscillators = False
        params.eval_on_pmlb = False
        dataset_name = "strogatz_extended"
    # elif params.dataset == "oscillators":
    #     params.eval_on_pmlb = False
    #     params.eval_on_file = False
    #     params.eval_on_oscillators = True
    #     dataset_name = "oscillators"
    elif params.dataset == "oscillators":
        params.eval_on_file = "invar_datasets/invar_datasets.pkl"
        params.eval_on_pmlb = False
        params.eval_on_oscillators = False
        dataset_name = "oscillators"
    else:
        params.eval_on_pmlb = False
        params.eval_on_oscillators = False
        params.eval_on_file = params.dataset
        dataset_name = Path(params.dataset).stem
    
    if "odeformer" in params.model:
        params.from_pretrained = True
        params.is_slurm_job = False
        params.local_rank = -1
        params.master_port = -1
        params.cpu = False
    else:
        params.cpu = True
    
    if not hasattr(params, "eval_subsample_ratio"):
        params.eval_subsample_ratio = 0 # no subsampling
    if not hasattr(params, "eval_noise_gamma"):
        params.eval_noise_gamma = 0 # no noise
    if not hasattr(params, "eval_noise_type"):
        params.eval_noise_type = "additive"
    
    params.dump_path = os.path.join(
        BASE, 
        params.model,
        dataset_name,
        f"hyper_opt_{params.optimize_hyperparams}",
        f"baseline_hyper_opt_eval_fraction_{params.hyper_opt_eval_fraction}",
        f"eval_subsample_ratio_{float(params.eval_subsample_ratio)}",
        f"eval_noise_type_{params.eval_noise_type}",
        f"eval_gamma_noise_{float(params.eval_noise_gamma)}",
        f"{params.evaluation_task}",
        f"beam_size_{params.beam_size}",
    )
    
    params.eval_dump_path = params.dump_path
    # params.reevaluate_path = f"/home/haicu/soeren.becker/repos/odeformer/experiments/{params.dump_path}/eval_pmlb.csv"
    # if params.model == "odeformer":
    #     params.reevaluate_path = "./experiments/odeformer/scores.csv"
    # if params.model == "odeformer_opt":
    #     params.reevaluate_path = "/p/project/hai_microbio/sb/repos/odeformer/experiments/odeformer/optimize/scores_optimize.csv"
    # if params.model == "odeformer_opt_random":
    #     params.reevaluate_path = "/p/project/hai_microbio/sb/repos/odeformer/experiments/odeformer/optimize_init_random/random_seed_2023/scores_optimize.csv"
    symbolicregression.utils.CUDA = not params.cpu
    if params.batch_size_eval is None:
        params.batch_size_eval = int(1.5 * params.batch_size)
    Path(params.dump_path).mkdir(exist_ok=True, parents=True)
    Path(params.eval_dump_path).mkdir(exist_ok=True, parents=True)
    print(params.eval_dump_path, os.path.exists(params.eval_dump_path))            
    init_distributed_mode(params)
    logger = initialize_exp(params, write_dump_path=False)
    if not params.cpu:
        assert torch.cuda.is_available()
    print(params)
    main(params)

    # TODO enable re-evaluation
    # TODO integrate constant optimization