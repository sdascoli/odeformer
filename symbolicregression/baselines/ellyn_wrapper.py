from typing import List, Union
from ellyn import ellyn
from symbolicregression.model.mixins import (
    BatchMixin, 
    FiniteDifferenceMixin,
    PredictionIntegrationMixin, 
)
import sympy
import numpy as np


__all__ = ("AFPWrapper", "EHCWrapper", "EPLEXWrapper", "FEAFPWrapper",)

# """ellyn documentation: https://github.com/cavalab/ellyn/blob/master/environment.yml"""

class EllynMixin(BatchMixin, FiniteDifferenceMixin, PredictionIntegrationMixin):
    def __init__(self, *args, **kwargs):
        self.base_model = ellyn(*args, **kwargs)
        
    def fit(
        self, 
        times: Union[List[np.ndarray], np.ndarray],
        trajectories: Union[List[np.ndarray], np.ndarray],
        finite_difference_order: Union[None, int] = None,
        smoother_window_length: Union[None, int] = None,
        parse_sympy: bool = False,
        *args, # ignored
        **kwargs, # ignored
    ):
        if isinstance(trajectories, List):
            return self.fit_all(
                times=times, 
                trajectories=trajectories,
                finite_difference_order=finite_difference_order,
                smoother_window_length=smoother_window_length,
                *args, **kwargs,
            )
        self.base_model.fit(
            trajectories, 
            self.approximate_derivative(
                trajectory=trajectories, 
                times=times,
                finite_difference_order=finite_difference_order,
                smoother_window_length=smoother_window_length,
            ).squeeze(),
        )
        eqs = self._get_equations()
        if parse_sympy:
            eqs = self._parse_sympy(eqs)
        return {0: eqs}
    
    def _get_equations(self):
        # see https://github.com/cavalab/ellyn/blob/master/src/ellyn.py#L485
        candidates = self.base_model.hof_
        order = np.argsort(self.base_model.fit_v)
        return self._format_equations(
            (np.array(self.base_model.stacks_2_eqns(candidates))[order]).tolist()
        )
    
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
    def __init__(self):
        super().__init__(
            selection='afp',
            lex_eps_global=False,
            lex_eps_dynamic=False,
            islands=False,
            num_islands=10,
            island_gens=100,
            verbosity=1,
            print_data=False,
            elitism=True,
            pHC_on=True,
            prto_arch_on=True,
            max_len = 64,
            max_len_init=20,
            popsize=1000,
            g = 250,
            time_limit=60,
            op_list=['n','v','+','-','*','/', 'exp','log','2','3', 'sqrt','sin','cos'],
        )

class EHCWrapper(EllynMixin):
    def __init__(self):
        super().__init__(
            eHC_on=True,
            eHC_its=3,
            selection='afp',
            lex_eps_global=False,
            lex_eps_dynamic=False,
            islands=False,
            num_islands=10,
            island_gens=100,
            verbosity=1,
            print_data=False,
            elitism=True,
            pHC_on=True,
            prto_arch_on=True,
            max_len = 64,
            max_len_init=20,
            popsize=1000,
            g=100,
            time_limit=60,
            op_list=['n','v','+','-','*','/', 'exp','log','2','3', 'sqrt','sin','cos'],
        )
        
class EPLEXWrapper(EllynMixin):
    def __init__(self):
        super().__init__(
            selection='epsilon_lexicase',
            lex_eps_global=False,
            lex_eps_dynamic=False,
            islands=False,
            num_islands=10,
            island_gens=100,
            verbosity=0,
            print_data=False,
            elitism=True,
            pHC_on=True,
            prto_arch_on=True,
            max_len = 64,
            max_len_init=20,
            popsize=500,
            g=500,
            time_limit=60,
            op_list=['n','v','+','-','*','/', 'exp','log','2','3', 'sqrt','sin','cos'],
        )

class FEAFPWrapper(EllynMixin):
    def __init__(self):
        super().__init__(
            selection='afp',
            lex_eps_global=False,
            lex_eps_dynamic=False,
            islands=False,
            num_islands=10,
            island_gens=100,
            verbosity=0,
            print_data=False,
            elitism=True,
            pHC_on=True,
            prto_arch_on=True,
            max_len = 64,
            max_len_init=20,
            EstimateFitness=True,
            FE_pop_size=100,
            FE_ind_size=10,
            FE_train_size=10,
            FE_train_gens=10,
            FE_rank=True,
            popsize=1000,
            g=250,
            time_limit=60,
            op_list=['n','v','+','-','*','/','sin','cos','exp','log','2','3', 'sqrt'],
        )
