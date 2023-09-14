from evaluate import *
from odeformer.baselines.ffx_wrapper import FFXWrapper
from odeformer.baselines.pysr_wrapper import PySRWrapper
from odeformer.baselines.ellyn_wrapper import (
   AFPWrapper, EHCWrapper, EPLEXWrapper, FEAFPWrapper,
)
from odeformer.baselines.proged_wrapper import ProGEDWrapper
from odeformer.baselines.sindy_wrapper import SINDyWrapper
import os

def format_and_save(params, scores, batch_results, name):
    scores = pd.DataFrame(scores, index=[0]).T
    scores.to_csv(
        path_or_buf=Path(params.eval_dump_path) / f"{params.baseline_model}_{name}.csv"
    )
    batch_results.to_csv(path_or_buf=Path(params.eval_dump_path) / f"{params.baseline_model}_{name}_batch_results.csv")
    print(f"Saving results for {params.baseline_model} under:\n{params.eval_dump_path}")

def main(params):
        
    env = build_env(params)
    env.rng = np.random.RandomState(0)
    modules = build_modules(env, params)
    trainer = Trainer(modules, env, params)

    if params.baseline_model == "sindy":
        model = SINDyWrapper(
            model_dir=params.eval_dump_path,
            optimizer_alpha=0.05,
            optimizer_threshold=0.04,
            polynomial_degree=6,
            functions=None, # None means all
            grid_search_polynomial_degree=True,
            optimize_hyperparams=params.optimize_hyperparams,
            hyper_opt_eval_fraction=params.hyper_opt_eval_fraction,
            sorting_metric=params.sorting_metric,
        )
    elif params.baseline_model == "sindy_all":
        model = SINDyWrapper(
            model_dir=params.eval_dump_path,
            polynomial_degree=10,
            functions=[], # only polynomials
            grid_search_polynomial_degree=True,
            grid_search_functions=True,
            optimize_hyperparams=params.optimize_hyperparams,
            hyper_opt_eval_fraction=params.hyper_opt_eval_fraction,
            sorting_metric=params.sorting_metric,
        )
    elif params.baseline_model == "sindy_full":
        model = SINDyWrapper(
            model_dir=params.eval_dump_path,
            polynomial_degree=3,
            functions=[], # only polynomials
            grid_search_polynomial_degree=True,
            grid_search_functions=True,
            optimize_hyperparams=params.optimize_hyperparams,
            hyper_opt_eval_fraction=params.hyper_opt_eval_fraction,
            sorting_metric=params.sorting_metric,
        )
    elif params.baseline_model == "sindy_save":
        model = SINDyWrapper(
            model_dir=params.eval_dump_path,
            optimizer_alpha=0.05,
            optimizer_threshold=0.04,
            polynomial_degree=6,
            functions=["sin", "cos", "exp"],
            grid_search_polynomial_degree=True,
            optimize_hyperparams=params.optimize_hyperparams,
            hyper_opt_eval_fraction=params.hyper_opt_eval_fraction,
            sorting_metric=params.sorting_metric,
        )
    elif params.baseline_model == "sindy_poly3":
        model = SINDyWrapper(
            model_dir=params.eval_dump_path,
            polynomial_degree=3,
            functions=[], # only polynomials
            optimize_hyperparams=params.optimize_hyperparams,
            hyper_opt_eval_fraction=params.hyper_opt_eval_fraction,
            sorting_metric=params.sorting_metric,
        )
    elif params.baseline_model == "sindy_poly6":
        model = SINDyWrapper(
            model_dir=params.eval_dump_path,
            polynomial_degree=6,
            functions=[], # only polynomials
            grid_search_polynomial_degree=True,
            optimize_hyperparams=params.optimize_hyperparams,
            hyper_opt_eval_fraction=params.hyper_opt_eval_fraction,
            sorting_metric=params.sorting_metric,
        )
    elif params.baseline_model == "sindy_poly10":
        model = SINDyWrapper(
            model_dir=params.eval_dump_path,
            polynomial_degree=10,
            functions=[], # only polynomials
            grid_search_polynomial_degree=True,
            optimize_hyperparams=params.optimize_hyperparams,
            hyper_opt_eval_fraction=params.hyper_opt_eval_fraction,
            sorting_metric=params.sorting_metric,
        )
    elif params.baseline_model == "pysr":
        model = PySRWrapper(
            model_dir=params.eval_dump_path,
            optimize_hyperparams=True,
            hyper_opt_eval_fraction=params.hyper_opt_eval_fraction,
            sorting_metric=params.sorting_metric,
        )
    elif params.baseline_model == "pysr_poly":
        model = PySRWrapper(
            model_dir=params.eval_dump_path,
            optimize_hyperparams=True,
            hyper_opt_eval_fraction=params.hyper_opt_eval_fraction,
            sorting_metric=params.sorting_metric,
            unary_operators=[]
        )
    elif params.baseline_model == "proged":
        model = ProGEDWrapper(
            model_dir=params.eval_dump_path,
            optimize_hyperparams=True,
            hyper_opt_eval_fraction=params.hyper_opt_eval_fraction,
            sorting_metric=params.sorting_metric,
        )
    elif params.baseline_model == "proged_poly":
        model = ProGEDWrapper(
            model_dir=params.eval_dump_path,
            optimize_hyperparams=True,
            hyper_opt_eval_fraction=params.hyper_opt_eval_fraction,
            sorting_metric=params.sorting_metric,
            generator_template_name="polynomial",
            grid_search_generator_template_name=False,
        )
    elif params.baseline_model == "afp":
        model = AFPWrapper(
            model_dir=params.eval_dump_path,
            optimize_hyperparams=True,
            hyper_opt_eval_fraction=params.hyper_opt_eval_fraction,
            sorting_metric=params.sorting_metric,
        )
    elif params.baseline_model == "ehc":
        model = EHCWrapper(
            model_dir=params.eval_dump_path,
            optimize_hyperparams=True,
            hyper_opt_eval_fraction=params.hyper_opt_eval_fraction,
            sorting_metric=params.sorting_metric,
        )
    elif params.baseline_model == "eplex":
        model = EPLEXWrapper(
            model_dir=params.eval_dump_path,
            optimize_hyperparams=True,
            hyper_opt_eval_fraction=params.hyper_opt_eval_fraction,
            sorting_metric=params.sorting_metric,
        )
    elif params.baseline_model == "feafp":
        model = FEAFPWrapper(
            model_dir=params.eval_dump_path,
            optimize_hyperparams=True,
            hyper_opt_eval_fraction=params.hyper_opt_eval_fraction,
            sorting_metric=params.sorting_metric,
        )
    elif params.baseline_model == "ffx":
        model = FFXWrapper(
            model_dir=params.eval_dump_path,
            optimize_hyperparams=True,
            hyper_opt_eval_fraction=params.hyper_opt_eval_fraction,
            sorting_metric=params.sorting_metric,
        )
    elif params.baseline_model in ["odeformer", "odeformer_opt", "odeformer_opt_random"]:
        model = setup_odeformer(trainer)
    else:
        raise ValueError(f"Unknown model: {params.baseline_model}")
        
    evaluator = Evaluator(trainer, model)
    
    if params.eval_on_file:
        scores = evaluator.evaluate_on_file(path=params.eval_on_file, save=params.save_results, seed=params.random_seed)
        
    if params.eval_on_pmlb:
        scores = evaluator.evaluate_on_pmlb(path_dataset=params.path_dataset)
        
    if params.eval_on_oscillators:
        scores = evaluator.evaluate_on_oscillators()
    
def str2bool(arg: Union[bool, str]):
    if isinstance(arg, bool):
        return arg
    if arg.lower() in ["true", "yes", "t", "y"]:
        return True
    return False

def str_or_None(arg: str):
    if arg.lower == "none":
        return None
    return arg
    
if __name__ == "__main__":
    BASE = os.path.join(os.getcwd(), "experiments")
    parser = get_parser()
    parser.add_argument("--baseline_model", type=str, default="sindy_poly10",
        choices=[
            "afp", "feafp", "eplex", "ehc",
            "proged", "proged_poly",
            "ffx",
            "pysr", "pysr_poly",
            "sindy", "sindy_all", "sindy_full", "sindy_save", "sindy_poly3", "sindy_poly6", "sindy_poly10",
            "odeformer", "odeformer_opt", "odeformer_opt_random"
        ]
    )
    parser.add_argument("--dataset", type=str, choices=["strogatz", "oscillators", "<path_to_dataset>"], 
        # default="/p/project/hai_microbio/sb/repos/odeformer/datasets/strogatz_extended/strogatz_extended.json"
        # default="oscillators"
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
        type=str, choices=["interpolation", "forecasting", "y0_generalization"], default="interpolation",
    )
    params = parser.parse_args()
    if params.dataset == "strogatz":
        params.eval_on_file = False
        params.eval_on_oscillators = False
        params.eval_on_pmlb = True
        params.path_dataset = "datasets/strogatz.pkl"
        dataset_name = params.dataset
    elif params.dataset == "oscillators":
        params.eval_on_pmlb = False
        params.eval_on_file = False
        params.eval_on_oscillators = True
        dataset_name = "oscillators"
    else:
        params.eval_on_pmlb = False
        params.eval_on_oscillators = False
        params.eval_on_file = params.dataset
        dataset_name = Path(params.dataset).stem
    params.validation_metrics = 'r2,r2_zero,snmse,accuracy_l1_1e-1,accuracy_l1_1e-3,accuracy_l1_biggio,is_valid,complexity_sympy,relative_complexity_sympy,complexity_string,relative_complexity_string' # complexity,term_difference,term_difference_sympy
    params.eval_only = True
    params.cpu = True
    params.max_dimension = 5
    params.evaluation_task = params.e_task
    # params.eval_size = 2
    
    if not hasattr(params, "subsample_ratio"):
        params.subsample_ratio = 0 # no subsampling
    if not hasattr(params, "eval_noise_gamma"):
        params.eval_noise_gamma = 0 # no noise
    if not hasattr(params, "eval_noise_type"):
        params.eval_noise_type = "additive"
    
    params.dump_path = os.path.join(
        BASE, 
        params.baseline_model,
        dataset_name,
        f"hyper_opt_{params.optimize_hyperparams}",
        f"baseline_hyper_opt_eval_fraction_{params.hyper_opt_eval_fraction}",
        f"subsample_ratio_{float(params.subsample_ratio)}",
        f"eval_noise_type_{params.eval_noise_type}",
        f"eval_gamma_noise_{float(params.eval_noise_gamma)}",
        f"evaluation_task_{params.evaluation_task}",
        # f"baseline_to_sympy_{params.baseline_to_sympy}",
    )
    
    params.eval_dump_path = params.dump_path
    # params.reevaluate_path = f"/home/haicu/soeren.becker/repos/odeformer/experiments/{params.dump_path}/eval_pmlb.csv"
    if params.baseline_model == "odeformer":
        params.reevaluate_path = "./experiments/odeformer/scores.csv"
    if params.baseline_model == "odeformer_opt":
        params.reevaluate_path = "/p/project/hai_microbio/sb/repos/odeformer/experiments/odeformer/optimize/scores_optimize.csv"
    if params.baseline_model == "odeformer_opt_random":
        params.reevaluate_path = "/p/project/hai_microbio/sb/repos/odeformer/experiments/odeformer/optimize_init_random/random_seed_2023/scores_optimize.csv"
    odeformer.utils.CUDA = not params.cpu
    if params.batch_size_eval is None:
        params.batch_size_eval = int(1.5 * params.batch_size)
    Path(params.dump_path).mkdir(exist_ok=True, parents=True)
    Path(params.eval_dump_path).mkdir(exist_ok=True, parents=True)
    init_distributed_mode(params)
    logger = initialize_exp(params, write_dump_path=False)
    if not params.cpu:
        assert torch.cuda.is_available()
    print(params)
    main(params)

    # TODO enable re-evaluation
    # TODO integrate constant optimization