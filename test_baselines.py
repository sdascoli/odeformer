from evaluate import *
from symbolicregression.baselines.pysr_wrapper import PySRWrapper
from symbolicregression.baselines.sindy_wrapper import SINDyWrapper
from symbolicregression.baselines.proged_wrapper import ProGEDWrapper

def main(params):
        
    env = build_env(params)
    env.rng = np.random.RandomState(0)
    modules = build_modules(env, params)
    trainer = Trainer(modules, env, params)

    if params.baseline_model == "sindy":
        model = SINDyWrapper(
            feature_names=["x_0", "x_1"],
            optimizer_alpha=0.05,
            optimizer_threshold=0.04,
            polynomial_degree=2,
            functions=None, # None means all, pass [] to exclude
        )
    elif params.baseline_model == "sindy_poly3":    
        model = SINDyWrapper(
            feature_names=["x_0", "x_1"],
            polynomial_degree=3,
            functions=[], # None means all, pass [] to exclude
        )
    elif params.baseline_model == "pysr":
        model = PySRWrapper(feature_names=["x_0", "x_1"])
    elif params.baseline_model == "proged":
        model = ProGEDWrapper(feature_names=["x_0", "x_1"])
        
        
    evaluator_default = Evaluator(trainer, model)
    pmlb_scores_default, _ = evaluator_default.evaluate_on_pmlb(save=params.save_results)
    print(pd.DataFrame(pmlb_scores_default, index=[0]).T)
    
    
if __name__ == "__main__":
    BASE = "/p/project/hai_microbio/sb/repos/odeformer"
    parser = get_parser()
    
    parser.add_argument("--baseline_model", 
        type=str, default="proged", choices=["proged", "sindy", "pysr", "sindy_poly3"]
    )
    
    params = parser.parse_args()
    params.eval_size = 10
    params.is_slurm_job = False
    params.local_rank = -1
    params.master_port = -1
    params.debug_slurm=True
    params.use_cross_attention = True
    params.dump_path = f"{BASE}/tests/nb_model_generic_evaluation/{params.baseline_model}"
    params.eval_dump_path = f"{BASE}/tests/nb_model_generic_evaluation/{params.baseline_model}"
    params.use_two_hot=True
    params.debug = True
    params.validation_metrics = 'r2_zero,snmse,accuracy_l1_1e-1,accuracy_l1_1e-3,accuracy_l1_biggio'
    params.eval_only = True
    params.cpu = True
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