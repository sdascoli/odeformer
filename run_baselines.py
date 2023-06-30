from evaluate import *
from symbolicregression.baselines.ffx_wrapper import FFXWrapper
from symbolicregression.baselines.pysr_wrapper import PySRWrapper
from symbolicregression.baselines.ellyn_wrapper import (
   AFPWrapper, EHCWrapper, EPLEXWrapper, FEAFPWrapper,
)
from symbolicregression.baselines.proged_wrapper import ProGEDWrapper
from symbolicregression.baselines.sindy_wrapper import SINDyWrapper
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
            optimizer_alpha=0.05,
            optimizer_threshold=0.04,
            polynomial_degree=6,
            functions=None, # None means all
            grid_search_polynomial_degree=True,
        )
    if params.baseline_model == "sindy_save":
        model = SINDyWrapper(
            optimizer_alpha=0.05,
            optimizer_threshold=0.04,
            polynomial_degree=6,
            functions=["sin", "cos", "exp"],
            grid_search_polynomial_degree=True,
        )
    elif params.baseline_model == "sindy_poly":
        model = SINDyWrapper(
            polynomial_degree=6,
            functions=[], # only polynomials
            grid_search_polynomial_degree=True,
        )
    elif params.baseline_model == "sindy_poly3":
        model = SINDyWrapper(
            polynomial_degree=3,
            functions=[], # only polynomials
        )
    elif params.baseline_model == "pysr":
        model = PySRWrapper()
    elif params.baseline_model == "proged":
        model = ProGEDWrapper()
    elif params.baseline_model == "afp":
        model = AFPWrapper()
    elif params.baseline_model == "ehc":
        model = EHCWrapper()
    elif params.baseline_model == "eplex":
        model = EPLEXWrapper()
    elif params.baseline_model == "feafp":
        model = FEAFPWrapper()
    elif params.baseline_model == "ffx":
        model = FFXWrapper()
        
    evaluator_default = Evaluator(trainer, model)
    
    if params.eval_on_file:
        scores = evaluator_default.evaluate_on_file(
            path=params.eval_on_file, save=params.save_results, seed=13,
        )
        _name = Path(params.eval_on_file).name
        # format_and_save(params, scores, batch_results, str(_name))
        
    if params.eval_on_pmlb:
        scores = evaluator_default.evaluate_on_pmlb(
            save=params.save_results, path_dataset=params.path_dataset
        )
    
def str2bool(arg: Union[bool, str]):
    if isinstance(arg, bool):
        return arg
    if arg.lower() in ["true", "yes", "t", "y"]:
        return True
    return False
    
if __name__ == "__main__":
    BASE = os.path.join(os.getcwd(), "experiments")
    parser = get_parser()
    parser.add_argument("--baseline_model", type=str, default="pysr",
        choices=["afp", "feafp", "ffx", "eplex", "ehc", "proged", "pysr", "sindy", "sindy_save", "sindy_poly", "sindy_poly3",]
    )
    parser.add_argument("--dataset", type=str, choices=["strogatz", "<path_to_dataset.pkl>"], default="strogatz")
    parser.add_argument("--baseline_hyper_opt", type=str2bool, default=True, 
        help="Do / Don't optimizer hyper parameters."
    )
    parser.add_argument("--baseline_hyper_opt_eval_fraction", type=float, default=0.3,
        help="Fraction of trajectory length that is used to score hyper parameter optimization on."
    )
    parser.add_argument("--baseline_to_sympy", type=str2bool, default=True, 
        help="Do / Don't parse predicted equation with sympy."                    
    )
    params = parser.parse_args()
    if params.dataset == "strogatz":
        params.eval_on_pmlb = True
        params.path_dataset = "/p/project/hai_microbio/sb/repos/odeformer/datasets/strogatz.pkl"
        params.eval_on_file = False
    else:
        params.eval_on_pmlb = False
        params.eval_on_file = params.dataset
    params.validation_metrics = 'r2,r2_zero,snmse,accuracy_l1_1e-1,accuracy_l1_1e-3,accuracy_l1_biggio,is_valid' # complexity,term_difference,term_difference_sympy
    params.eval_only = True
    params.cpu = True
    
    params.dump_path = os.path.join(
        BASE, 
        params.baseline_model,
        params.dataset,
        f"hyper_opt_{params.baseline_hyper_opt}",
        f"baseline_hyper_opt_eval_fraction_{params.baseline_hyper_opt_eval_fraction}",
        f"eval_size_{params.eval_size}",
        f"baseline_to_sympy_{params.baseline_to_sympy}",
    )
    params.eval_dump_path = params.dump_path
    
    symbolicregression.utils.CUDA = not params.cpu
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

# assess hyper params