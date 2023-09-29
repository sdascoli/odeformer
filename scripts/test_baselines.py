from evaluate import *
from odeformer.baselines.ffx_wrapper import FFXWrapper
from odeformer.baselines.pysr_wrapper import PySRWrapper
#from odeformer.baselines.ellyn_wrapper import (
#    AFPWrapper, EHCWrapper, EPLEXWrapper, FEAFPWrapper,
#)
#from odeformer.baselines.proged_wrapper import ProGEDWrapper
from odeformer.baselines.sindy_wrapper import SINDyWrapper

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

    feature_names = [f"x_{i}" for i in range(params.max_dimension)]

    if params.baseline_model == "sindy":
        model = SINDyWrapper(
            feature_names=feature_names,
            optimizer_alpha=0.05,
            optimizer_threshold=0.04,
            polynomial_degree=2,
            functions=None, # None means all
        )
    elif params.baseline_model == "sindy_poly2":    
        model = SINDyWrapper(
            feature_names=feature_names,
            polynomial_degree=2,
            functions=[], # only polynomials
        )
    elif params.baseline_model == "sindy_poly3":    
        model = SINDyWrapper(
            feature_names=feature_names,
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

    if params.eval_in_domain:
      scores = evaluator_default.evaluate_in_domain("functions",save=params.save_results)
      logger.info("__log__:%s" % json.dumps(scores))

    if params.eval_on_pmlb:
        scores = evaluator_default.evaluate_on_pmlb(save=params.save_results)
        logger.info("__pmlb__:%s" % json.dumps(scores))
        scores = evaluator_default.evaluate_on_oscillators(save=params.save_results)
        logger.info("__oscillators__:%s" % json.dumps(scores))
        #format_and_save(params, scores, batch_results, "pmlb")

    exit()
        
    if params.eval_on_file is not None:
        scores = evaluator_default.evaluate_on_file(
            path=params.eval_on_file, save=params.save_results, seed=13,
        )
        _name = Path(params.eval_on_file).name
        #format_and_save(params, scores, batch_results, str(_name))
    
    
    
if __name__ == "__main__":
    BASE = "experiments/baselines"
    parser = get_parser()
    
    parser.add_argument("--baseline_model", 
        type=str, default="sindy",
        choices=["proged", "pysr", "sindy_poly3", "sindy_poly2", "sindy", "afp", "feafp", "eplex", "ehc", "ffx",]
    )
    
    params = parser.parse_args()
    params.eval_size = 100
    params.max_dimension = 6
    params.num_workers = 1
    params.is_slurm_job = False
    params.local_rank = -1
    params.master_port = -1
    params.debug_slurm=True
    params.use_cross_attention = True
    params.dump_path = f"{BASE}/nb_model_generic_evaluation/{params.baseline_model}"
    params.eval_dump_path = f"{BASE}/nb_model_generic_evaluation/{params.baseline_model}"
    #params.use_two_hot=True
    params.debug = True
    params.validation_metrics = 'r2_zero,snmse,accuracy_l1_1e-1,accuracy_l1_1e-3,accuracy_l1_biggio'
    params.eval_only = True
    params.cpu = True
    
    params.eval_on_pmlb = True
    #params.eval_on_file = "experiments/datagen_general/datagen_use_sympy_True/data.prefix.test"
    
    odeformer.utils.CUDA = not params.cpu
    if params.batch_size_eval is None:
        params.batch_size_eval = int(1.5 * params.batch_size)
    Path(params.dump_path).mkdir(exist_ok=True, parents=True)
    Path(params.eval_dump_path).mkdir(exist_ok=True, parents=True)
    init_distributed_mode(params)
    logger = initialize_exp(params, write_dump_path=False)
    if not params.cpu:
        assert torch.cuda.is_available()
    
    main(params)