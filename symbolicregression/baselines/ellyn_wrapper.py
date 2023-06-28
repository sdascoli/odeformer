from typing import Any, Callable, Dict, List, Union
from sklearn.metrics import r2_score
from ellyn import ellyn
from symbolicregression.model.mixins import (
    BatchMixin, 
    FiniteDifferenceMixin,
    PredictionIntegrationMixin,
    MultiDimMixin,
    SympyMixin,
)
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
):
    def __init__(self, **kwargs):
        fd_kwargs = {}
        if "finite_difference_order" in kwargs.keys():
            fd_kwargs["finite_difference_order"] = kwargs.pop("finite_difference_order")
        if "smoother_window_length" in kwargs.keys():
            fd_kwargs["smoother_window_length"] = kwargs.pop("smoother_window_length")
        FiniteDifferenceMixin.__init__(self, **fd_kwargs)
        self.base_model = ellyn(**kwargs)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        params = self.base_model.get_params()
        assert "finite_difference_order" not in params.keys(), params.keys()
        assert "smoother_window_length" not in params.keys(), params.keys()
        params["finite_difference_order"] = self.finite_difference_order
        params["smoother_window_length"] = self.smoother_window_length
        return params
    
    def set_params(self, **params: Dict):
        if "finite_difference_order" in params.keys():
            self.finite_difference_order = params.pop("finite_difference_order")
        if "smoother_window_length" in params.keys():
            self.smoother_window_length = params.pop("smoother_window_length")
        self.base_model.set_params(**params)
        return self

    def score(
        self,
        times: np.ndarray,
        trajectories: np.ndarray,
        metric: Callable = r2_score
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
    ) -> Dict[int, List[str]]:

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


class AFPWrapper(EllynMixin):
    def __init__(self, **kwargs) -> None:
        kwargs = kwargs.copy()
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
        hyper_params = []
        for p, g in zip(pop_sizes, gs):
            for op_list in op_lists:
                hyper_params.append({'popsize':[p], 'g':[g], 'op_list':[op_list]})
        return hyper_params


class EHCWrapper(EllynMixin):
    def __init__(self, **kwargs) -> None:
        kwargs = kwargs.copy()
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
        hyper_params = []
        for p, g in zip(pop_sizes, gs):
            for op_list in op_lists:
                hyper_params.append({'popsize':[p], 'g':[g], 'op_list':[op_list]})
        return hyper_params


class EPLEXWrapper(EllynMixin):
    def __init__(self, **kwargs) -> None:
        kwargs = kwargs.copy()
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
        hyper_params = []
        for p, g in zip(pop_sizes, gs):
            for op_list in op_lists:
                hyper_params.append({'popsize':[p], 'g':[g], 'op_list':[op_list]})
        return hyper_params


class FEAFPWrapper(EllynMixin):
    def __init__(self, **kwargs) -> None:
        kwargs = kwargs.copy()
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
        hyper_params = []
        for p, g in zip(pop_sizes, gs):
            for op_list in op_lists:
                hyper_params.append({'popsize':[p], 'g':[g], 'op_list':[op_list]})
        return hyper_params