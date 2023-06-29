from evaluate import *
from symbolicregression.baselines.ffx_wrapper import FFXWrapper
from symbolicregression.baselines.pysr_wrapper import PySRWrapper
from symbolicregression.baselines.ellyn_wrapper import (
   AFPWrapper, EHCWrapper, EPLEXWrapper, FEAFPWrapper,
)
from symbolicregression.baselines.proged_wrapper import ProGEDWrapper
from symbolicregression.baselines.sindy_wrapper import SINDyWrapper

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
        scores = evaluator_default.evaluate_on_pmlb(save=params.save_results)
    
if __name__ == "__main__":
    BASE = "experiments/baselines"
    parser = get_parser()
    
    parser.add_argument("--baseline_model", 
        type=str, default="sindy_poly3",
        choices=["afp", "feafp", "ffx", "eplex", "ehc", "proged", "pysr", "sindy", "sindy_save", "sindy_poly", "sindy_poly3",]
    )
    
    params = parser.parse_args()
    params.eval_size = 1
    params.max_dimension = 2
    params.num_workers = 1
    params.is_slurm_job = False
    params.local_rank = -1
    params.master_port = -1
    params.debug_slurm=True
    params.use_cross_attention = True
    
    #params.use_two_hot=True
    params.debug = True
    params.validation_metrics = 'r2,r2_zero,snmse,accuracy_l1_1e-1,accuracy_l1_1e-3,accuracy_l1_biggio,complexity,term_difference,is_valid'
    params.eval_only = True
    params.cpu = True
    params.baseline_hyper_opt = True
    params.baseline_hyper_opt_eval_fraction = 0.25
    params.baseline_to_sympy = True
    params.eval_on_pmlb = True
    params.eval_on_file = False
    # params.eval_on_file = "/p/project/hai_microbio/sb/repos/odeformer/datasets/data.prefix.test.pkl"
    # params.eval_on_file = "/p/project/hai_microbio/sb/repos/odeformer/datasets/polynomial_2d.txt.pkl"
    # params.eval_on_pmlb = True
    # params.eval_on_file = "experiments/datagen_poly/datagen_use_sympy_True/data.prefix.test"
    
    params.dump_path = f"{BASE}/baseline_results/{params.baseline_model}/hyper_opt_{params.baseline_hyper_opt}"
    params.eval_dump_path = f"{BASE}/baseline_results/{params.baseline_model}/hyper_opt_{params.baseline_hyper_opt}"
    
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
    
    # TODO: test all models
    # TODO: add install instructions