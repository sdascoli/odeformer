from evaluate import *
from symbolicregression.baselines.ffx_wrapper import FFXWrapper
from symbolicregression.baselines.pysr_wrapper import PySRWrapper
from symbolicregression.baselines.ellyn_wrapper import (
   AFPWrapper, EHCWrapper, EPLEXWrapper, FEAFPWrapper,
)
from symbolicregression.baselines.proged_wrapper import ProGEDWrapper
from symbolicregression.baselines.sindy_wrapper import SINDyWrapper

from sklearn.model_selection import GridSearchCV

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
        model = AFPWrapper()
    elif params.baseline_model == "ehc":
        model = EHCWrapper()
    elif params.baseline_model == "eplex":
        model = EPLEXWrapper()
    elif params.baseline_model == "feafp":
        model = FEAFPWrapper()
    elif params.baseline_model == "ffx":
        model = FFXWrapper()
    
    times = np.linspace(1, 3, 256, endpoint=True)
    trajectory = np.exp(times).reshape(-1, 1).repeat(repeats=2, axis=1)
    param_grid = model.get_hyper_grid()
    train_idcs = np.arange(int(np.floor(0.5*len(times))))
    test_idcs = np.arange(int(np.floor(0.5*len(times))), len(times))
    gscv = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        refit = True,
        cv=[(train_idcs, test_idcs)],
        verbose=4,
        error_score=np.nan,
        n_jobs=None,
    )
    gscv.fit(times, trajectory)
    model = gscv.best_estimator_
    candidates = model._get_equations()
    print(candidates)
    print("Done.")
    
    
if __name__ == "__main__":
    BASE = "experiments/baselines"
    parser = get_parser()
    
    parser.add_argument("--baseline_model", 
        type=str, default="pysr",
        choices=["proged", "pysr", "sindy_poly3", "sindy_poly2", "sindy", "afp", "feafp", "eplex", "ehc", "ffx",]
    )
    
    params = parser.parse_args()
    params.eval_size = 100
    params.max_dimension = 2
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
    params.eval_on_file = "experiments/datagen_poly/datagen_use_sympy_True/data.prefix.test"
    
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
    
    # TODO: how to deal with batches