from copy import deepcopy
from typing import Any, Callable, Dict, List, Union
from sklearn.metrics import r2_score
from ellyn import ellyn
from odeformer.model.mixins import (
    GridSearchMixin,
    BatchMixin, 
    FiniteDifferenceMixin,
    PredictionIntegrationMixin,
    MultiDimMixin,
    SympyMixin,
)
from odeformer.baselines.baseline_utils import variance_weighted_r2_score
import time
import sympy
import numpy as np

__all__ = ("AFPWrapper", "EHCWrapper", "EPLEXWrapper", "FEAFPWrapper",)

# """ellyn documentation: 
# https://github.com/cavalab/ellyn/blob/master/environment.yml"""

class EllynMixin(
    BatchMixin,
    FiniteDifferenceMixin,
    MultiDimMixin,
    PredictionIntegrationMixin,
    SympyMixin,
    GridSearchMixin,
):
    def __init__(
        self, 
        model_dir: str,
        optimize_hyperparams: bool = True,
        hyper_opt_eval_fraction: Union[None, float] = None,
        sorting_metric: str = "r2", 
        grid_search_is_running: bool = False,
        **kwargs,
    ):
        fd_kwargs = {}
        if "finite_difference_order" in kwargs.keys():
            fd_kwargs["finite_difference_order"] = kwargs.pop("finite_difference_order")
        if "smoother_window_length" in kwargs.keys():
            fd_kwargs["smoother_window_length"] = kwargs.pop("smoother_window_length")
        FiniteDifferenceMixin.__init__(self, **fd_kwargs)
        self.model_dir = model_dir
        self.filename_pareto_front = f"equations_{time.strftime('%Y-%m-%d-%H-%M-%S-%MS')}.json"
        self.optimize_hyperparams = optimize_hyperparams
        self.hyper_opt_eval_fraction = hyper_opt_eval_fraction
        self.sorting_metric = sorting_metric
        self.grid_search_is_running = grid_search_is_running
        self._wrapper_params = [
            "model_dir",
            "filename_pareto_front",
            "optimize_hyperparams",
            "hyper_opt_eval_fraction",
            "sorting_metric",
            "grid_search_is_running",
            "finite_difference_order",
            "smoother_window_length",
        ]
        self.base_model = ellyn(**kwargs)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        params = self.base_model.get_params()
        for p in self._wrapper_params:
            assert p not in params.keys(), params.keys()
            params[p] = getattr(self, p)
        return params
    
    def set_params(self, **params: Dict):
        params = deepcopy(params)
        for p in self._wrapper_params:
            if p in params.keys():
                setattr(self, p, params.pop(p))
        self.base_model.set_params(**params)
        return self

    def score(
        self,
        times: np.ndarray,
        trajectories: np.ndarray,
        metric: Callable = variance_weighted_r2_score
    ) -> float:
        try:
            candidates = self._get_equations()
            assert len(candidates) > 0, candidates
            pred_trajectory = self.integrate_prediction(
                times, y0=trajectories[0], prediction=candidates[0][0]
            )
            assert pred_trajectory is not None, f"pred_trajectory is None."
            return metric(trajectories, pred_trajectory)
        except AssertionError as e:
            return np.nan
        
    def fit(
        self, 
        times: Union[List[np.ndarray], np.ndarray],
        trajectories: Union[List[np.ndarray], np.ndarray],
        derivatives: Union[None, np.ndarray] = None,
        *args, **kwargs, # ignored, for compatibility only
    ) -> Dict[int, List[str]]:

        if self.optimize_hyperparams and not self.grid_search_is_running:
            if isinstance(trajectories, List):
                assert len(trajectories) == 1, len(trajectories)
                trajectories = trajectories[0]
            assert isinstance(trajectories, np.ndarray)
            return self.fit_grid_search(times=times, trajectory=trajectories)

        if isinstance(trajectories, List):
            self.final_equations = self.fit_all(times=times, trajectories=trajectories)
            return self.final_equations
        if derivatives is None:
            derivatives = self.approximate_derivative(
                trajectory=trajectories, 
                times=times,
            ).squeeze()
            if len(derivatives.shape) > 1:
                self.final_equations = self.fit_components(times, trajectories, derivatives)
                return self.final_equations
        self.base_model.fit(trajectories, derivatives)
        self.final_equations = self.__get_equations()
        return self.final_equations
    
    def __get_equations(self) -> Dict[int, List[str]]:
        # see https://github.com/cavalab/ellyn/blob/master/src/ellyn.py#L485
        candidates = self.base_model.hof_
        order = np.argsort(self.base_model.fit_v) # TODO: this does not work when using multiple components
        return {
            0: self._format_equations(
                (np.array(self.base_model.stacks_2_eqns(candidates))[order]).tolist()
            )
        }
    
    def _get_equations(self) -> Dict[int, List[str]]:
        if self.final_equations is None:
            print("Warning: model does not seem to be fitted.")
        return self.final_equations
    
    def _format_equations(self, eq: Union[List[str], str]) -> Union[List[str], str]:
        if isinstance(eq, List):
            eqs = []
            for e in eq:
                eqs.append(self._format_equations(e))
            return eqs
        eq = eq.replace("^", "**")
        eq = eq.replace("sqrt(|", "sqrt(abs(").replace("|", ")")
        return eq
    
    def _parse_sympy(self, eq: Union[List[str], str]) -> Union[List[str], str]:
        if isinstance(eq, List):
            eqs = []
            for e in eq:
                eqs.append(self._parse_sympy(e))
            return eqs
        return str(sympy.parse_expr(eq))    
    
    def get_n_jobs(self) -> int:
        return 48


class AFPWrapper(EllynMixin):
    def __init__(self, **kwargs) -> None:
        kwargs = kwargs.copy()
        # this is not particularly elegant but it ensures that any kwargs that 
        # are not passed explicitly will be set to the AFP default values
        super().__init__(
            selection=(kwargs.pop("selection") if "selection" in kwargs.keys() else 'afp'),
            lex_eps_global=(kwargs.pop("lex_eps_global") if "lex_eps_global" in kwargs.keys() else False),
            lex_eps_dynamic=(kwargs.pop("lex_eps_dynamic") if "lex_eps_dynamic" in kwargs.keys() else False),
            islands=(kwargs.pop("islands") if "islands" in kwargs.keys() else False),
            num_islands=(kwargs.pop("num_islands") if "num_islands" in kwargs.keys() else 10),
            island_gens=(kwargs.pop("island_gens") if "island_gens" in kwargs.keys() else 100),
            verbosity=(kwargs.pop("verbosity") if "verbosity" in kwargs.keys() else 0),
            print_data=(kwargs.pop("print_data") if "print_data" in kwargs.keys() else False),
            elitism=(kwargs.pop("elitism") if "elitism" in kwargs.keys() else True),
            pHC_on=(kwargs.pop("pHC_on") if "pHC_on" in kwargs.keys() else True),
            prto_arch_on=(kwargs.pop("prto_arch_on") if "prto_arch_on" in kwargs.keys() else True),
            max_len=(kwargs.pop("max_len") if "max_len" in kwargs.keys() else 64),
            max_len_init=(kwargs.pop("max_len_init") if "max_len_init" in kwargs.keys() else 20),
            popsize=(kwargs.pop("popsize") if "popsize" in kwargs.keys() else 1000),
            g=(kwargs.pop("g") if "g" in kwargs.keys() else 250),
            time_limit=(kwargs.pop("time_limit") if "time_limit" in kwargs.keys() else None),
            op_list=(kwargs.pop("op_list") if "op_list" in kwargs.keys() else [
                    'n','v','+','-','*','/','exp','log','2','3','sqrt','sin','cos'
                ]
            ),
            **kwargs,
        )
        
    def get_hyper_grid(self) -> List[Dict[str, Any]]:
        # https://github.com/cavalab/srbench/blob/master/experiment/methods/AFPRegressor.py
        pop_sizes = [100, 500, 1000]
        gs = [2500, 500, 250]
        op_lists=[
            ['n','v','+','-','*','/','exp','log','2','3', 'sqrt'],
            ['n','v','+','-','*','/', 'exp','log','2','3', 'sqrt', 'sin','cos']
        ]
        finite_difference_orders = [2,3,4]
        smoother_window_lengths = [None, 15]
        hyper_params = []
        for p, g in zip(pop_sizes, gs):
            for op_list in op_lists:
                for fdo in finite_difference_orders:
                    for swl in smoother_window_lengths:
                        hyper_params.append(
                            {
                                'popsize': [p], 
                                'g': [g], 
                                'op_list': [op_list],
                                'finite_difference_order': [fdo],
                                'smoother_window_length': [swl],
                            }
                        )
        return hyper_params


class EHCWrapper(EllynMixin):
    def __init__(self, **kwargs) -> None:
        kwargs = kwargs.copy()
        # this is not particularly elegant but it ensures that any kwargs that 
        # are not passed explicitly will be set to the EHC default values
        super().__init__(
            eHC_on=(kwargs.pop("eHC_on") if "eHC_on" in kwargs.keys() else True),
            eHC_its=(kwargs.pop("eHC_its") if "eHC_its" in kwargs.keys() else 3),
            selection=(kwargs.pop("selection") if "selection" in kwargs.keys() else 'afp'),
            lex_eps_global=(kwargs.pop("lex_eps_global") if "lex_eps_global" in kwargs.keys() else False),
            lex_eps_dynamic=(kwargs.pop("lex_eps_dynamic") if "lex_eps_dynamic" in kwargs.keys() else False),
            islands=(kwargs.pop("islands") if "islands" in kwargs.keys() else False),
            num_islands=(kwargs.pop("num_islands") if "num_islands" in kwargs.keys() else 10),
            island_gens=(kwargs.pop("island_gens") if "island_gens" in kwargs.keys() else 100),
            verbosity=(kwargs.pop("verbosity") if "verbosity" in kwargs.keys() else 0),
            print_data=(kwargs.pop("print_data") if "print_data" in kwargs.keys() else False),
            elitism=(kwargs.pop("elitism") if "elitism" in kwargs.keys() else True),
            pHC_on=(kwargs.pop("pHC_on") if "pHC_on" in kwargs.keys() else True),
            prto_arch_on=(kwargs.pop("prto_arch_on") if "prto_arch_on" in kwargs.keys() else True),
            max_len=(kwargs.pop("max_len") if "max_len" in kwargs.keys() else 64),
            max_len_init=(kwargs.pop("max_len_init") if "max_len_init" in kwargs.keys() else 20),
            popsize=(kwargs.pop("popsize") if "popsize" in kwargs.keys() else 1000),
            g=(kwargs.pop("g") if "g" in kwargs.keys() else 100),
            time_limit=(kwargs.pop("time_limit") if "time_limit" in kwargs.keys() else None),
            op_list=(kwargs.pop("op_list") if "op_list" in kwargs.keys() else [
                    'n','v','+','-','*','/','exp','log','2','3','sqrt','sin','cos'
                ]
            ),
            **kwargs,
        )
        
    def get_hyper_grid(self) -> List[Dict[str, Any]]:
        # https://github.com/cavalab/srbench/blob/master/experiment/methods/EHCRegressor.py
        pop_sizes = [100, 500, 1000]
        gs = [1000, 200, 100]
        op_lists=[
            ['n','v','+','-','*','/','exp','log','2','3', 'sqrt'],
            ['n','v','+','-','*','/', 'exp','log','2','3', 'sqrt', 'sin','cos']
        ]
        finite_difference_orders = [2,3,4]
        smoother_window_lengths = [None, 15]
        hyper_params = []
        for p, g in zip(pop_sizes, gs):
            for op_list in op_lists:
                for fdo in finite_difference_orders:
                    for swl in smoother_window_lengths:
                        hyper_params.append(
                            {
                                'popsize': [p], 
                                'g': [g], 
                                'op_list': [op_list],
                                'finite_difference_order': [fdo],
                                'smoother_window_length': [swl],
                            }
                        )
        return hyper_params


class EPLEXWrapper(EllynMixin):
    def __init__(self, **kwargs) -> None:
        kwargs = kwargs.copy()
        # this is not particularly elegant but it ensures that any kwargs that 
        # are not passed explicitly will be set to the EPLEX default values
        super().__init__(
            selection=(kwargs.pop("selection") if "selection" in kwargs.keys() else 'epsilon_lexicase'),
            lex_eps_global=(kwargs.pop("lex_eps_global") if "lex_eps_global" in kwargs.keys() else False),
            lex_eps_dynamic=(kwargs.pop("lex_eps_dynamic") if "lex_eps_dynamic" in kwargs.keys() else False),
            islands=(kwargs.pop("islands") if "islands" in kwargs.keys() else False),
            num_islands=(kwargs.pop("num_islands") if "num_islands" in kwargs.keys() else 10),
            island_gens=(kwargs.pop("island_gens") if "island_gens" in kwargs.keys() else 100),
            verbosity=(kwargs.pop("verbosity") if "verbosity" in kwargs.keys() else 0),
            print_data=(kwargs.pop("print_data") if "print_data" in kwargs.keys() else False),
            elitism=(kwargs.pop("elitism") if "elitism" in kwargs.keys() else True),
            pHC_on=(kwargs.pop("pHC_on") if "pHC_on" in kwargs.keys() else True),
            prto_arch_on=(kwargs.pop("prto_arch_on") if "prto_arch_on" in kwargs.keys() else True),
            max_len=(kwargs.pop("max_len") if "max_len" in kwargs.keys() else 64),
            max_len_init=(kwargs.pop("max_len_init") if "max_len_init" in kwargs.keys() else 20),
            popsize=(kwargs.pop("popsize") if "popsize" in kwargs.keys() else 500),
            g=(kwargs.pop("g") if "g" in kwargs.keys() else 500),
            time_limit=(kwargs.pop("time_limit") if "time_limit" in kwargs.keys() else None),
            op_list=(kwargs.pop("op_list") if "op_list" in kwargs.keys() else [
                    'n','v','+','-','*','/','exp','log','2','3','sqrt','sin','cos'
                ]
            ),
            **kwargs,
        )
    
    def get_hyper_grid(self) -> List[Dict[str, Any]]:
        # https://github.com/cavalab/srbench/blob/master/experiment/methods/EPLEXRegressor.py
        pop_sizes = [100, 500, 1000]
        gs = [2500, 500, 250]
        op_lists=[
            ['n','v','+','-','*','/','sin','cos','exp','log','2','3', 'sqrt'],
            ['n','v','+','-','*','/', 'exp','log','2','3', 'sqrt']
        ]
        finite_difference_orders = [2,3,4]
        smoother_window_lengths = [None, 15]
        hyper_params = []
        for p, g in zip(pop_sizes, gs):
            for op_list in op_lists:
                for fdo in finite_difference_orders:
                    for swl in smoother_window_lengths:
                        hyper_params.append(
                            {
                                'popsize': [p], 
                                'g': [g], 
                                'op_list': [op_list],
                                'finite_difference_order': [fdo],
                                'smoother_window_length': [swl],
                            }
                        )
        return hyper_params


class FEAFPWrapper(EllynMixin):
    def __init__(self, **kwargs) -> None:
        kwargs = kwargs.copy()
        # this is not particularly elegant but it ensures that any kwargs that 
        # are not passed explicitly will be set to the FEAFP default values
        super().__init__(
            selection=(kwargs.pop("selection") if "selection" in kwargs.keys() else 'afp'),
            lex_eps_global=(kwargs.pop("lex_eps_global") if "lex_eps_global" in kwargs.keys() else False),
            lex_eps_dynamic=(kwargs.pop("lex_eps_dynamic") if "lex_eps_dynamic" in kwargs.keys() else False),
            islands=(kwargs.pop("islands") if "islands" in kwargs.keys() else False),
            num_islands=(kwargs.pop("num_islands") if "num_islands" in kwargs.keys() else 10),
            island_gens=(kwargs.pop("island_gens") if "island_gens" in kwargs.keys() else 100),
            verbosity=(kwargs.pop("verbosity") if "verbosity" in kwargs.keys() else 0),
            print_data=(kwargs.pop("print_data") if "print_data" in kwargs.keys() else False),
            elitism=(kwargs.pop("elitism") if "elitism" in kwargs.keys() else True),
            pHC_on=(kwargs.pop("pHC_on") if "pHC_on" in kwargs.keys() else True),
            prto_arch_on=(kwargs.pop("prto_arch_on") if "prto_arch_on" in kwargs.keys() else True),
            max_len=(kwargs.pop("max_len") if "max_len" in kwargs.keys() else 64),
            max_len_init=(kwargs.pop("max_len_init") if "max_len_init" in kwargs.keys() else 20),
            EstimateFitness=(kwargs.pop("EstimateFitness") if "EstimateFitness" in kwargs.keys() else True),
            FE_pop_size=(kwargs.pop("FE_pop_size") if "FE_pop_size" in kwargs.keys() else 100),
            FE_ind_size=(kwargs.pop("FE_ind_size") if "FE_ind_size" in kwargs.keys() else 10),
            FE_train_size=(kwargs.pop("FE_train_size") if "FE_train_size" in kwargs.keys() else 10),
            FE_train_gens=(kwargs.pop("FE_train_gens") if "FE_train_gens" in kwargs.keys() else 10),
            FE_rank=(kwargs.pop("FE_rank") if "FE_rank" in kwargs.keys() else True),
            popsize=(kwargs.pop("popsize") if "popsize" in kwargs.keys() else 1000),
            g=(kwargs.pop("g") if "g" in kwargs.keys() else 250),
            time_limit=(kwargs.pop("time_limit") if "time_limit" in kwargs.keys() else None),
            op_list=(kwargs.pop("op_list") if "op_list" in kwargs.keys() else [
                    'n','v','+','-','*','/','sin','cos','exp','log','2','3','sqrt'
                ]
            ),
            **kwargs,
        )
        
    def get_hyper_grid(self) -> List[Dict[str, Any]]:
        # https://github.com/cavalab/srbench/blob/master/experiment/methods/FE_AFPRegressor.py
        pop_sizes = [100, 500, 1000]
        gs = [2500, 500, 250]
        op_lists=[
            ['n','v','+','-','*','/','sin','cos','exp','log','2','3', 'sqrt'],
            ['n','v','+','-','*','/', 'exp','log','2','3', 'sqrt']
        ]
        finite_difference_orders = [2,3,4]
        smoother_window_lengths = [None, 15]
        hyper_params = []
        for p, g in zip(pop_sizes, gs):
            for op_list in op_lists:
                for fdo in finite_difference_orders:
                    for swl in smoother_window_lengths:
                        hyper_params.append(
                            {
                                'popsize': [p], 
                                'g': [g], 
                                'op_list': [op_list],
                                'finite_difference_order': [fdo],
                                'smoother_window_length': [swl],
                            }
                        )
        return hyper_params