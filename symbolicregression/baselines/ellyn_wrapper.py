from typing import List, Union
from ellyn import ellyn
from symbolicregression.model.mixins import (
    BatchMixin, 
    PredictionIntegrationMixin, 
    FiniteDifferenceMixin
)
import numpy as np

__all__ = ("AFPWrapper", "EHCWrapper", "EPLEXWrapper", "FEAFPWrapper",)

"""Check documentation of ellyn here: https://github.com/cavalab/ellyn/blob/master/environment.yml"""

class EllynBaseWrapper(ellyn, BatchMixin, FiniteDifferenceMixin, PredictionIntegrationMixin):
    def fit(
        self, 
        times: Union[List[np.ndarray], np.ndarray],
        trajectories: Union[List[np.ndarray], np.ndarray],
        finite_difference_order: Union[None, int] = None,
        smoother_window_length: Union[None, int] = None,
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
        return {
            0: super().fit(
                trajectories, 
                self.approximate_derivative(
                    trajectory=trajectories, 
                    times=times,
                    finite_difference_order=finite_difference_order,
                    smoother_window_length=smoother_window_length,
                ).squeeze(),
            )
        }
    
class AFPWrapper(EllynBaseWrapper):
    def __init__(self, time_limit=30):
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
            time_limit=time_limit,
            op_list=['n','v','+','-','*','/', 'exp','log','2','3', 'sqrt','sin','cos'],
        )


class EHCWrapper(EllynBaseWrapper):
    def __init__(self, time_limit):
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
            time_limit=time_limit,
            op_list=['n','v','+','-','*','/', 'exp','log','2','3', 'sqrt','sin','cos'],
        )
    

class EPLEXWrapper(EllynBaseWrapper):
    def __init__(self, time_limit):
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
            time_limit=time_limit,
            op_list=['n','v','+','-','*','/', 'exp','log','2','3', 'sqrt','sin','cos'],
        )

class FEAFPWrapper(EllynBaseWrapper):
    def __init__(self, time_limit):
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
            time_limit=time_limit,
            op_list=['n','v','+','-','*','/','sin','cos','exp','log','2','3', 'sqrt'],
        )
