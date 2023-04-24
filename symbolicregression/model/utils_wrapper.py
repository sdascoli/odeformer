# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from abc import ABC, abstractmethod
import sklearn
from scipy.optimize import minimize
import numpy as np
import time
import torch
from functorch import grad
from functools import partial
import traceback

class TimedFun:
    def __init__(self, fun, verbose=False, stop_after=3):
        self.fun_in = fun
        self.started = False
        self.stop_after = stop_after
        self.best_fun_value = np.infty
        self.best_x = None
        self.loss_history=[]
        self.verbose = verbose

    def fun(self, x, *args):
        if self.started is False:
            self.started = time.time()
        elif abs(time.time() - self.started) >= self.stop_after:
            self.loss_history.append(self.best_fun_value)
            raise ValueError("Time is over.")
        self.fun_value = self.fun_in(x, *args)
        self.loss_history.append(self.fun_value)
        if self.best_x is None:
            self.best_x=x
        elif self.fun_value < self.best_fun_value:
            self.best_fun_value=self.fun_value
            self.best_x=x
        self.x = x
        return self.fun_value

class Scaler(ABC):
    """
    Base class for scalers
    """

    def __init__(self, range_shift=1, feature_scale=.5):
        self.time_scaler = sklearn.preprocessing.MinMaxScaler()
        self.traj_scale  = None
        self.range_shift = range_shift
        self.feature_scale = feature_scale

    def fit(self, time, trajectory):
        self.time_scaler.fit(time.reshape(-1,1))
        self.traj_scale = trajectory[0]

    def transform(self, time, trajectory):
        scaled_time = self.time_scaler.transform(time.reshape(-1,1))+self.range_shift
        scaled_traj = self.feature_scale * trajectory/(self.traj_scale.reshape(1,-1))
        return scaled_time[:,0], scaled_traj
        
    def fit_transform(self, time, trajectory):
        self.fit(time, trajectory)
        scaled_time, scaled_trajectory = self.transform(time, trajectory)
        return scaled_time, scaled_trajectory
    
    def get_params(self):
        scale = self.feature_scale/self.traj_scale

        val_min, val_max = self.time_scaler.data_min_[0], self.time_scaler.data_max_[0]
        a_t, b_t = 1./(val_max-val_min), -val_min/(val_max-val_min)+self.range_shift
        return (a_t, b_t, scale)

    def rescale_function(self, env, tree, a_t, b_t, scale):
        prefix = tree.prefix().split(",")
        idx = 0
        while idx < len(prefix):
            if prefix[idx].startswith("x_") or prefix[idx] == "t":
                if prefix[idx].startswith("x_"):
                    k = int(prefix[idx][-1])
                    if k>=len(scale): 
                        continue
                    a = str(scale[k])
                    prefix_to_add = ["mul", a, prefix[idx]]
                else:
                    a, b = str(a_t), str(b_t)
                    prefix_to_add = ["add", b, "mul", a, prefix[idx]]
                prefix = prefix[:idx] + prefix_to_add + prefix[min(idx + 1, len(prefix)):]
                idx += len(prefix_to_add)
                #print(scale, idx, len(prefix), prefix[idx], len(prefix_to_add))
            else:
                idx+=1
                continue
        rescaled_tree = env.word_to_infix(prefix, is_float=False, str_array=False)
        return rescaled_tree

class StandardScaler(Scaler):
    def __init__(self):
        """
        transformation is: 
        x' =  (x - mean)/std
        """

    def fit(self, ):
        self.scaler.fit(X)

    def fit_transform(self, X):
        scaled_time       = self.time_scaler.fit_transform(time)
        scaled_trajectory = self.traj_scaler.fit_transform(trajectory)
        return scaled_X
    
    def transform(self, X):
        m, s = self.scaler.mean_, np.sqrt(self.scaler.var_)
        return (X-m)/s

    def get_params(self):
        m, s = self.scaler.mean_, np.sqrt(self.scaler.var_)
        a, b = 1/s, -m/s
        return (a, b)
    
class MinMaxScaler(Scaler):
    def __init__(self):
        """
        transformation is: 
        x' =  (x-xmin)/(xmax-xmin)
        """
        self.scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))

    def fit(self, X):
        self.scaler.fit(X)

    def fit_transform(self, X):
        scaled_X = self.scaler.fit_transform(X)
        return scaled_X

    def transform(self, X):
        val_min, val_max = self.scaler.data_min_, self.scaler.data_max_
        return (X-val_min)/(val_max-val_min)

    def get_params(self):
        val_min, val_max = self.scaler.data_min_, self.scaler.data_max_
        a, b = 1./(val_max-val_min), -val_min/(val_max-val_min)
        return (a, b)

class BFGSRefinement():
    """
    Wrapper around scipy's BFGS solver
    """

    def __init__(self):
        """
        Args:
            func: a PyTorch function that maps dependent variabels and
                    parameters to function outputs for all data samples
                    `func(x, coeffs) -> y`
            x, y: problem data as PyTorch tensors. Shape of x is (d, n) and
                    shape of y is (n,)
        """
        super().__init__()        

    def go(
        self, env, tree, coeffs0, X, y, downsample=-1, stop_after=10
    ):
        
        func = env.simplifier.tree_to_torch_module(tree, dtype=torch.float64)
        self.X, self.y = X, y
        if downsample>0:
            self.X = self.X[:downsample]
            self.y = self.y[:downsample]
        self.X=torch.tensor(self.X, dtype=torch.float64, requires_grad=False)
        self.y=torch.tensor(self.y, dtype=torch.float64, requires_grad=False)
        self.func = partial(func, self.X)

        def objective_torch(coeffs):
            """
            Compute the non-linear least-squares objective value
                objective(coeffs) = (1/2) sum((y - func(coeffs)) ** 2)
            Returns a PyTorch tensor.
            """
            if not isinstance(coeffs, torch.Tensor):
                coeffs = torch.tensor(coeffs, dtype=torch.float64, requires_grad=True)
            y_tilde = self.func(coeffs)
            if y_tilde is None: return None
            mse = (self.y -y_tilde).pow(2).mean().div(2)
            return mse

        def objective_numpy(coeffs):
            """
            Return the objective value as a float (for scipy).
            """
            return objective_torch(coeffs).item()

        def gradient_numpy(coeffs):
            """
            Compute the gradient of the objective at coeffs.
            Returns a numpy array (for scipy)
            """
            if not isinstance(coeffs, torch.Tensor):
                coeffs = torch.tensor(coeffs, dtype=torch.float64, requires_grad=True)
            grad_obj = grad(objective_torch)(coeffs)
            return grad_obj.detach().numpy()
    
        objective_numpy_timed = TimedFun(objective_numpy, stop_after=stop_after)

        try:
            minimize(
                    objective_numpy_timed.fun,
                    coeffs0,
                    method="BFGS",
                    jac=gradient_numpy,
                    options = {"disp": False}
            )
        except ValueError as e:
            traceback.format_exc()
        best_constants = objective_numpy_timed.best_x
        return env.wrap_equation_floats(tree, best_constants)