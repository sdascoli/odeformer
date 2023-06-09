from evaluate import *
from symbolicregression.baselines.ffx_wrapper import FFXWrapper
from symbolicregression.baselines.pysr_wrapper import PySRWrapper
from symbolicregression.baselines.ellyn_wrapper import (
    AFPWrapper, EHCWrapper, EPLEXWrapper, FEAFPWrapper,
)
from symbolicregression.baselines.sindy_wrapper import SINDyWrapper
from symbolicregression.baselines.proged_wrapper import ProGEDWrapper

def format_and_save(params, scores, batch_results, name):
    gamma = params.eval_noise_gamma
    subsample_ratio = params.subsample_ratio
    if gamma is None:
        gamma = "None"
    if subsample_ratio is None:
        subsample_ratio = "None"
    scores = pd.DataFrame(scores, index=[0]).T
    if hasattr(params, "reload_size") and params.reload_size is not None:
        fname = f"{params.baseline_model}_{name}_n_{params.reload_size}_gamma_{gamma}_subsample_{subsample_ratio}.csv"
    else:
        fname = f"{params.baseline_model}_{name}_gamma_{gamma}_subsample_{subsample_ratio}.csv"
    scores.to_csv(path_or_buf=Path(params.eval_dump_path) / fname)
    fname = fname[:-4]+"_batch_results.csv"
    batch_results.to_csv(path_or_buf=Path(params.eval_dump_path) / fname)
    print(f"Saving results for {params.baseline_model} under:\n{params.eval_dump_path}")

def main(params):
        
    env = build_env(params)
    env.rng = np.random.RandomState(0)
    modules = build_modules(env, params)
    trainer = Trainer(modules, env, params)
    feature_names = [f"x_{k}" for k in range(params.expexted_num_ode_components)]
    if params.baseline_model == "sindy":
        model = SINDyWrapper(
            feature_names=feature_names,
            optimizer_alpha=0.05,
            optimizer_threshold=0.04,
            polynomial_degree=2,
            functions=None, # None means all
        )
    elif params.baseline_model == "sindy_poly3":    
        model = SINDyWrapper(
            feature_names=feature_names,
            optimizer_alpha=0.05,
            optimizer_threshold=0.04,
            polynomial_degree=3,
            functions=[], # only polynomials
        )
    elif params.baseline_model == "pysr":
        model = PySRWrapper(feature_names=feature_names)
    elif params.baseline_model == "proged":
        model = ProGEDWrapper(feature_names=feature_names)
    elif params.baseline_model == "afp":
        model = AFPWrapper(time_limit=60)
    elif params.baseline_model == "ehc":
        model = EHCWrapper(time_limit=60)
    elif params.baseline_model == "eplex":
        model = EPLEXWrapper(time_limit=60)
    elif params.baseline_model == "feafp":
        model = FEAFPWrapper(time_limit=60)
    elif params.baseline_model == "ffx":
        model = FFXWrapper()
        
    evaluator_default = Evaluator(trainer, model)
    if params.eval_on_pmlb:
        scores, batch_results = evaluator_default.evaluate_on_pmlb(save=params.save_results)
        format_and_save(params, scores, batch_results, "pmlb")
        
    if params.eval_on_file is not None:
        scores, batch_results = evaluator_default.evaluate_on_file(
            path=params.eval_on_file, save=params.save_results, seed=13, params=params
        )
        _name = Path(params.eval_on_file).name
        format_and_save(params, scores, batch_results, str(_name))
    
    
    
if __name__ == "__main__":
    BASE = "/p/project/hai_microbio/sb/repos/odeformer"
    parser = get_parser()
    
    parser.add_argument("--baseline_model", 
        type=str, default="proged",
        choices=["proged", "pysr", "sindy_poly3", "sindy", "afp", "feafp", "eplex", "ehc", "ffx",]
    )
    
    params = parser.parse_args()
    params.eval_size = 10
    params.is_slurm_job = False
    params.local_rank = -1
    params.master_port = -1
    params.debug_slurm=True
    params.use_cross_attention = True
    params.dump_path = f"{BASE}/nb_model_generic_evaluation/{params.baseline_model}"
    params.eval_dump_path = f"{BASE}/nb_model_generic_evaluation/{params.baseline_model}"
    params.use_two_hot=True
    params.debug = True
    params.validation_metrics = 'r2_zero,snmse,accuracy_l1_1e-1,accuracy_l1_1e-3,accuracy_l1_biggio'
    params.eval_only = True
    params.cpu = True
    
    params.eval_noise_gamma = 0
    params.subsample_ratio = 0
    
    params.eval_on_pmlb = False
    if False:
        params.eval_on_file = "/p/project/hai_microbio/sb/repos/odeformer/datasets/polynomial_2d.txt.pkl"
        params.expexted_num_ode_components = 2
    elif True:
        params.eval_on_file = "/p/project/hai_microbio/sb/repos/odeformer/datasets/data.prefix.test"
        params.reload_size = None
        params.expexted_num_ode_components = 6
    
    symbolicregression.utils.CUDA = not params.cpu
    if params.batch_size_eval is None:
        params.batch_size_eval = int(1.5 * params.batch_size)
    Path(params.dump_path).mkdir(exist_ok=True, parents=True)
    Path(params.eval_dump_path).mkdir(exist_ok=True, parents=True)
    init_distributed_mode(params)
    logger = initialize_exp(params, write_dump_path=False)
    if not params.cpu:
        assert torch.cuda.is_available()
    
    main(params)