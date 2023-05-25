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

# """Check documentation of ellyn here: https://github.com/cavalab/ellyn/blob/master/environment.yml"""

class EllynMixin(ellyn, BatchMixin, FiniteDifferenceMixin, PredictionIntegrationMixin):
    # All args need to explicitly appear in __init__, 
    # see https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
    def __init__(
        self,
        # ============== Generation Settings
        g=100, # number of generations (limited by default)
        popsize=500, #population size
        limit_evals=False, # limit evals instead of generations
        max_evals=0, # maximum number of evals before termination (only active if limit_evals is true)
        time_limit=None, # max time to run in seconds (zero=no limit)
        init_trees= True,
        selection='tournament',
        tourn_size=2,
        rt_rep=0, #probability of reproduction
        rt_cross=0.6,
        rt_mut=0.4,
        cross_ar=0.025, #crossover alternation rate
        mut_ar=0.025,
        cross=3, # 1: ultra, 2: one point, 3: subtree
        mutate=1, # 1: one point, 2: subtree
        align_dev = False,
        elitism = False,

        # ===============   Data settings
        init_validate_on=False, # initial fitness validation of individuals
        train=False, # choice to turn on training for splitting up the data set
        train_pct=0.5, # default split of data is 50/50
        shuffle_data=False, # shuffle the data
        test_at_end = False, # only run the test fitness on the population at the end of a run
        pop_restart = False, # restart from previous population
        pop_restart_path="", # restart population file path
        AR = False,
        AR_nb = 1,
        AR_nkb = 0,
        AR_na = 1,
        AR_nka = 1,
        AR_lookahead = False,
        # ================ Results and printing
        resultspath= './',
        #savename
        savename="",
        #print every population
        print_every_pop=False,
        #print initial population
        print_init_pop = False,
        #print last population
        print_last_pop = False,
        # print homology
        print_homology = False,
        #print log
        print_log = False,
        #print data
        print_data = False,
        #print best individual at end
        print_best_ind = False,
        #print archive
        print_archive = False,
        # number of log points to print (with eval limitation)
        num_log_pts = 0,
        # print csv files of genome each print cycle
        print_genome = False,
        # print csv files of epigenome each print cycle
        print_epigenome = False,
        # print number of unique output vectors
        print_novelty = False,
        # print individuals for graph database analysis
        print_db = False,
        # verbosity
        verbosity = 0,

        # ============ Fitness settings
        fit_type = "MSE", # 1: error, 2: corr, 3: combo
        norm_error = False , # normalize fitness by the standard deviation of the target output
        weight_error = False, # weight error vector by predefined weights from data file
        max_fit = 1.0E20,
        min_fit = 0.00000000000000000001,

        # Fitness estimators
        EstimateFitness=False,
        FE_pop_size=0,
        FE_ind_size=0,
        FE_train_size=0,
        FE_train_gens=0,
        FE_rank=False,
        estimate_generality=False,
        G_sel=1,
        G_shuffle=False,

        # =========== Program Settings
        # list of operators. choices:
        # n (constants), v (variables), +, -, *, /  
        # sin, cos, exp, log, sqrt, 2, 3, ^, =, !, <, <=, >, >=, 
        # if-then, if-then-else, &, | 
        op_list=['n','v','+','-','*','/'],
        # weights associated with each operator (default: uniform)
        op_weight=None,
        ERC = True, # ephemeral random constants
        ERCints = False ,
        maxERC = 1,
        minERC = -1,
        numERC = 1,

        min_len = 3,
        max_len = 20,
        max_len_init = 0,

        # 1: genotype size, 2: symbolic size, 3: effective genotype size
        complex_measure=1, 


        # Hill Climbing Settings

        # generic line hill climber (Bongard)
        lineHC_on =  False,
        lineHC_its = 0,

        # parameter Hill Climber
        pHC_on =  False,
        pHC_its = 1,
        pHC_gauss = 0,

        # epigenetic Hill Climber
        eHC_on = False,
        eHC_its = 1,
        eHC_prob = 0.1,
        eHC_init = 0.5,
        eHC_mut = False, # epigenetic mutation rather than hill climbing
        eHC_slim = False, # use SlimFitness

        # stochastic gradient descent
        SGD = False,
        learning_rate = 1.0,
        # Pareto settings

        prto_arch_on = False,
        prto_arch_size = 1,
        prto_sel_on = False,

        #island model
        islands = True,
        num_islands= 0,
        island_gens = 100,
        nt = 1,

        # lexicase selection
        lexpool = 1.0, # fraction of pop to use in selection events
        lexage = False, # use afp survival after lexicase selection
        lex_class = False, # use class-based fitness rather than error
        # errors within fixed epsilon of the best error are pass, 
        # otherwise fail
        lex_eps_error = False, 
        # errors within fixed epsilon of the target are pass, otherwise fail
        lex_eps_target = False, 		 
        # used w/ lex_eps_[error/target], ignored otherwise
        lex_epsilon = 0.1, 
        # errors in a standard dev of the best are pass, otherwise fail 
        lex_eps_std = False, 		 
        # errors in med abs dev of the target are pass, otherwise fail
        lex_eps_target_mad=False, 		 
        # errors in med abs dev of the best are pass, otherwise fail
        lex_eps_error_mad=False, 
        # pass conditions in lex eps defined relative to whole 
        # population (rather than selection pool).
        # turns on if no lex_eps_* parameter is True. 
        # a.k.a. "static" epsilon-lexicase
        lex_eps_global = False,                  
        # epsilon is defined for each selection pool instead of globally
        lex_eps_dynamic = False,
        # epsilon is defined as a random threshold corresponding to 
        # an error in the pool minus min error in pool 
        lex_eps_dynamic_rand = False,
        # with prob of 0.5, epsilon is replaced with 0
        lex_eps_dynamic_madcap = False,

        #pareto survival setting
        PS_sel=1,

        # classification
        classification = False,
        class_bool = False,
        class_m4gp = False,
        class_prune = False,

        stop_condition=True,
        stop_threshold = 0.000001,
        print_protected_operators = False,

        # return population to python
        return_pop = False,
        ################################# wrapper specific params
        scoring_function=None,
        random_state=None,
        lex_meta=None,
        seeds=None,    # seeding building blocks (equations)              
    ):
        super().__init__(
            g=g, # number of generations (limited by default)
            popsize=popsize, #population size
            limit_evals=limit_evals, # limit evals instead of generations
            max_evals=max_evals, # maximum number of evals before termination (only active if limit_evals is true)
            time_limit=time_limit, # max time to run in seconds (zero=no limit)
            init_trees= init_trees,
            selection=selection,
            tourn_size=tourn_size,
            rt_rep=rt_rep, #probability of reproduction
            rt_cross=rt_cross,
            rt_mut=rt_mut,
            cross_ar=cross_ar, #crossover alternation rate
            mut_ar=mut_ar,
            cross=cross, # 1: ultra, 2: one point, 3: subtree
            mutate=mutate, # 1: one point, 2: subtree
            align_dev = align_dev,
            elitism = elitism,

            # ===============   Data settings
            init_validate_on=init_validate_on, # initial fitness validation of individuals
            train=train, # choice to turn on training for splitting up the data set
            train_pct=train_pct, # default split of data is 50/50
            shuffle_data=shuffle_data, # shuffle the data
            test_at_end = test_at_end, # only run the test fitness on the population at the end of a run
            pop_restart = pop_restart, # restart from previous population
            pop_restart_path=pop_restart_path, # restart population file path
            AR = AR,
            AR_nb = AR_nb,
            AR_nkb = AR_nkb,
            AR_na = AR_na,
            AR_nka = AR_nka,
            AR_lookahead = AR_lookahead,
            # ================ Results and printing
            resultspath= resultspath,
            #savename
            savename=savename,
            #print every population
            print_every_pop=print_every_pop,
            #print initial population
            print_init_pop = print_init_pop,
            #print last population
            print_last_pop = print_last_pop,
            # print homology
            print_homology = print_homology,
            #print log
            print_log = print_log,
            #print data
            print_data = print_data,
            #print best individual at end
            print_best_ind = print_best_ind,
            #print archive
            print_archive = print_archive,
            # number of log points to print (with eval limitation)
            num_log_pts = num_log_pts,
            # print csv files of genome each print cycle
            print_genome = print_genome,
            # print csv files of epigenome each print cycle
            print_epigenome = print_epigenome,
            # print number of unique output vectors
            print_novelty = print_novelty,
            # print individuals for graph database analysis
            print_db = print_db,
            # verbosity
            verbosity = verbosity,

            # ============ Fitness settings
            fit_type = fit_type, # 1: error, 2: corr, 3: combo
            norm_error = norm_error , # normalize fitness by the standard deviation of the target output
            weight_error = weight_error, # weight error vector by predefined weights from data file
            max_fit = max_fit,
            min_fit = min_fit,

            # Fitness estimators
            EstimateFitness=EstimateFitness,
            FE_pop_size=FE_pop_size,
            FE_ind_size=FE_ind_size,
            FE_train_size=FE_train_size,
            FE_train_gens=FE_train_gens,
            FE_rank=FE_rank,
            estimate_generality=estimate_generality,
            G_sel=G_sel,
            G_shuffle=G_shuffle,

            # =========== Program Settings
            # list of operators. choices:
            # n (constants), v (variables), +, -, *, /  
            # sin, cos, exp, log, sqrt, 2, 3, ^, =, !, <, <=, >, >=, 
            # if-then, if-then-else, &, | 
            op_list=op_list,
            # weights associated with each operator (default: uniform)
            op_weight=op_weight,
            ERC = ERC, # ephemeral random constants
            ERCints = ERCints ,
            maxERC = maxERC,
            minERC = minERC,
            numERC = numERC,

            min_len = min_len,
            max_len = max_len,
            max_len_init = max_len_init,

            # 1: genotype size, 2: symbolic size, 3: effective genotype size
            complex_measure=complex_measure, 


            # Hill Climbing Settings

            # generic line hill climber (Bongard)
            lineHC_on =  lineHC_on,
            lineHC_its = lineHC_its,

            # parameter Hill Climber
            pHC_on =  pHC_on,
            pHC_its = pHC_its,
            pHC_gauss = pHC_gauss,

            # epigenetic Hill Climber
            eHC_on = eHC_on,
            eHC_its = eHC_its,
            eHC_prob = eHC_prob,
            eHC_init = eHC_init,
            eHC_mut = eHC_mut, # epigenetic mutation rather than hill climbing
            eHC_slim = eHC_slim, # use SlimFitness

            # stochastic gradient descent
            SGD = SGD,
            learning_rate = learning_rate,
            # Pareto settings

            prto_arch_on = prto_arch_on,
            prto_arch_size = prto_arch_size,
            prto_sel_on = prto_sel_on,

            #island model
            islands = islands,
            num_islands= num_islands,
            island_gens = island_gens,
            nt = nt,

            # lexicase selection
            lexpool = lexpool, # fraction of pop to use in selection events
            lexage = lexage, # use afp survival after lexicase selection
            lex_class = lex_class, # use class-based fitness rather than error
            # errors within fixed epsilon of the best error are pass, 
            # otherwise fail
            lex_eps_error = lex_eps_error, 
            # errors within fixed epsilon of the target are pass, otherwise fail
            lex_eps_target = lex_eps_target, 		 
            # used w/ lex_eps_[error/target], ignored otherwise
            lex_epsilon = lex_epsilon, 
            # errors in a standard dev of the best are pass, otherwise fail 
            lex_eps_std = lex_eps_std, 		 
            # errors in med abs dev of the target are pass, otherwise fail
            lex_eps_target_mad=lex_eps_target_mad, 		 
            # errors in med abs dev of the best are pass, otherwise fail
            lex_eps_error_mad=lex_eps_error_mad, 
            # pass conditions in lex eps defined relative to whole 
            # population (rather than selection pool).
            # turns on if no lex_eps_* parameter is True. 
            # a.k.a. "static" epsilon-lexicase
            lex_eps_global = lex_eps_global,                  
            # epsilon is defined for each selection pool instead of globally
            lex_eps_dynamic = lex_eps_dynamic,
            # epsilon is defined as a random threshold corresponding to 
            # an error in the pool minus min error in pool 
            lex_eps_dynamic_rand = lex_eps_dynamic_rand,
            # with prob of 0.5, epsilon is replaced with 0
            lex_eps_dynamic_madcap = lex_eps_dynamic_madcap,

            #pareto survival setting
            PS_sel=PS_sel,

            # classification
            classification = classification,
            class_bool = class_bool,
            class_m4gp = class_m4gp,
            class_prune = class_prune,

            stop_condition=stop_condition,
            stop_threshold = stop_threshold,
            print_protected_operators = print_protected_operators,

            # return population to python
            return_pop = return_pop,
            ################################# wrapper specific params
            scoring_function=scoring_function,
            random_state=random_state,
            lex_meta=lex_meta,
            seeds=seeds,  
        )
    
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
        super().fit(
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
        candidates = self.hof_
        order = np.argsort(self.fit_v)
        return self._format_equations((np.array(self.stacks_2_eqns(candidates))[order]).tolist())
    
    def _format_equations(self, eq: Union[List[str], str]) -> Union[List[str], str]:
        if isinstance(eq, List):
            eqs = []
            for e in eq:
                eqs.append(self._format_equations(e))
            return eqs
        # https://github.com/cavalab/srbench/blob/master/experiment/symbolic_utils.py#L184:L186
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
    # All args need to explicitly appear in __init__, 
    # see https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
    def __init__(
        self, 
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
        op_list=['n','v','+','-','*','/', 'exp','log','2','3', 'sqrt','sin','cos']
    ):
        super().__init__(
            selection=selection,
            lex_eps_global=lex_eps_global,
            lex_eps_dynamic=lex_eps_dynamic,
            islands=islands,
            num_islands=num_islands,
            island_gens=island_gens,
            verbosity=verbosity,
            print_data=print_data,
            elitism=elitism,
            pHC_on=pHC_on,
            prto_arch_on=prto_arch_on,
            max_len=max_len,
            max_len_init=max_len_init,
            popsize=popsize,
            g=g,
            time_limit=time_limit,
            op_list=op_list,
        )


class EHCWrapper(EllynMixin):
    # All args need to explicitly appear in __init__, 
    # see https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
    def __init__(
        self, 
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
    ):
        super().__init__(
            eHC_on=eHC_on,
            eHC_its=eHC_its,
            selection=selection,
            lex_eps_global=lex_eps_global,
            lex_eps_dynamic=lex_eps_dynamic,
            islands=islands,
            num_islands=num_islands,
            island_gens=island_gens,
            verbosity=verbosity,
            print_data=print_data,
            elitism=elitism,
            pHC_on=pHC_on,
            prto_arch_on=prto_arch_on,
            max_len=max_len,
            max_len_init=max_len_init,
            popsize=popsize,
            g=g,
            time_limit=time_limit,
            op_list=op_list,
        )
    

class EPLEXWrapper(EllynMixin):
    # All args need to explicitly appear in __init__, 
    # see https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
    def __init__(
        self, 
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
    ):
        super().__init__(
            selection=selection,
            lex_eps_global=lex_eps_global,
            lex_eps_dynamic=lex_eps_dynamic,
            islands=islands,
            num_islands=num_islands,
            island_gens=island_gens,
            verbosity=verbosity,
            print_data=print_data,
            elitism=elitism,
            pHC_on=pHC_on,
            prto_arch_on=prto_arch_on,
            max_len=max_len,
            max_len_init=max_len_init,
            popsize=popsize,
            g=g,
            time_limit=time_limit,
            op_list=op_list,
        )

class FEAFPWrapper(EllynMixin):
    # All args need to explicitly appear in __init__, 
    # see https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
    def __init__(
        self, 
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
    ):
        super().__init__(
            selection=selection,
            lex_eps_global=lex_eps_global,
            lex_eps_dynamic=lex_eps_dynamic,
            islands=islands,
            num_islands=num_islands,
            island_gens=island_gens,
            verbosity=verbosity,
            print_data=print_data,
            elitism=elitism,
            pHC_on=pHC_on,
            prto_arch_on=prto_arch_on,
            max_len=max_len,
            max_len_init=max_len_init,
            EstimateFitness=EstimateFitness,
            FE_pop_size=FE_pop_size,
            FE_ind_size=FE_ind_size,
            FE_train_size=FE_train_size,
            FE_train_gens=FE_train_gens,
            FE_rank=FE_rank,
            popsize=popsize,
            g=g,
            time_limit=time_limit,
            op_list=op_list,
        )
