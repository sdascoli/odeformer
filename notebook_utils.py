import glob
import os
import string
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np
import sys
import copy
from pathlib import Path
from sympy import *
import pickle
from collections import defaultdict, OrderedDict
import math
import scipy.special
import warnings
from sklearn.manifold import TSNE
from IPython.display import display
from importlib import reload  # Python 3.4+
import importlib.util
import subprocess
import pandas as pd
from IPython.display import display
from scipy import integrate
import argparse
import traceback
import requests
pi = np.pi
e = np.e

import odeformer
from odeformer.envs import ENVS, build_env
from odeformer.model import build_modules
from odeformer.envs.generators import RandomFunctions
from odeformer.envs.encoders import  Equation, FloatSequences
from odeformer.envs.environment import FunctionEnvironment
from odeformer.utils import *
from odeformer.model.sklearn_wrapper import SymbolicTransformerRegressor
from odeformer.model.model_wrapper import ModelWrapper
from odeformer.model.embedders import *
from odeformer.trainer import Trainer
from odeformer.envs.generators import integrate_ode
from odeformer.envs.utils import *
from evaluate import Evaluator

user = os.getlogin()

def get_most_free_gpu():
    output = subprocess.check_output("nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader", shell=True)
    free_memory = [int(x) for x in output.decode().strip().split('\n')]
    most_free = free_memory.index(max(free_memory))
    # set visible devices to the most free gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(most_free)
    return most_free

def module_from_file(module_name, file_path):
    print(file_path, module_name)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def import_file(full_path_to_module):
    module_dir, module_file = os.path.split(full_path_to_module)
    module_name, module_ext = os.path.splitext(module_file)
    save_cwd = os.getcwd()
    os.chdir(module_dir)
    module_obj = __import__(module_name)
    module_obj.__file__ = full_path_to_module
    globals()[module_name] = module_obj
    os.chdir(save_cwd)
    return module_obj

def get_most_free_gpu():
    output = subprocess.check_output("nvidia-smi --query-gpu=memory.free --form\
at=csv,nounits,noheader", shell=True)
    free_memory = [int(x) for x in output.decode().strip().split('\n')]
    most_free = free_memory.index(max(free_memory))
    # set visible devices to the most free gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(most_free)
    return most_free

############################ GENERAL ############################


def find(array, value):
    idx= np.argwhere(np.array(array)==value)[0,0]
    return idx

def select_runs(runs, params, constraints):
    selected_runs = []
    for irun, run in enumerate(runs):
        keep = True
        for k,v in constraints.items():
            if type(v)!=list:
                v=[v]
            if (not hasattr(run['args'],k)) or (getattr(run['args'],k) not in v):
                keep = False
                break
        if keep:
            selected_runs.append(run)
    selected_params = copy.deepcopy(params)
    for con in constraints:
        selected_params[con]=[constraints[con]]
    return selected_runs, selected_params

def group_runs(runs, finished_only=True):
    runs_grouped = defaultdict(list)
    for run in runs:
        seedless_args = copy.deepcopy(run['args'])
        del(seedless_args.seed)
        del(seedless_args.name)
        if str(seedless_args) not in runs_grouped.keys(): 
            runs_grouped[str(seedless_args)].append(run) # need at least one run
        else:
            if run['finished'] or not finished_only:
                runs_grouped[str(seedless_args)].append(run)
    runs_grouped = list(runs_grouped.values())
    return runs_grouped

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
def permute(array, indices):
    return [array[idx] for idx in indices]

def ordered_legend(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = zip(*[ (handles[i], labels[i]) for i in sorted(range(len(handles)), key=lambda k: list(map(float,labels))[k])] )
    #ax.legend(handles, labels, **kwargs)
    return handles, labels

def legend_no_duplicates(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[i+1:]]
    ax.legend(*zip(*unique), **kwargs)

def latex_float(f, precision=3):
    float_str = f"%.{precision}g"%f
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

############################ RECURRENCE ############################

def load_run(run, extra_args = {}):
        
    params = run['args']
    for k,v in extra_args.items():
        setattr(params, k, v)
    env = build_env(params)
    modules = build_modules(env, params)
    trainer = Trainer(modules, env, params)
    #evaluator = Evaluator(trainer)

    embedder = (
        modules["embedder"].module
        if params.multi_gpu
        else modules["embedder"]
    )
    encoder = (
        modules["encoder"].module
        if params.multi_gpu
        else modules["encoder"]
    )
    decoder = (
        modules["decoder"].module
        if params.multi_gpu
        else modules["decoder"]
    )
    embedder.eval()
    encoder.eval()
    decoder.eval()

    mw = ModelWrapper(
        env=env,
        embedder=embedder,
        encoder=encoder,
        decoder=decoder,
        beam_length_penalty=params.beam_length_penalty,
        beam_size=params.beam_size,
        max_generated_output_len=params.max_generated_output_len,
        beam_early_stopping=params.beam_early_stopping,
        beam_temperature=params.beam_temperature,
        beam_type=params.beam_type,
        #average_embeddings=True
    )

    dstr = SymbolicTransformerRegressor(
        model=mw,
        max_input_points=int(params.max_points),
        rescale=params.rescale,
        params = params
    )

    return dstr

def plot_predictions(times,trajectory,dstr):
    plt.figure(figsize=(3,3))
    for dim in range(len(trajectory[0])):
        plt.plot(times, trajectory[:,dim], color=f'C{dim}', label='True')

    candidates = dstr.fit(times, trajectory)
    for tree in candidates[0]:
        pred_trajectory = integrate_ode(trajectory[0], times, tree, "solve_ivp", debug=False)
        try:
            for dim in range(len(trajectory[0])):
                plt.plot(times, pred_trajectory[:,dim], color=f'C{dim}', ls='--', label='Pred')
        except: 
            print(traceback.format_exc())
    plt.legend()
    plt.show()
    return tree

############################ ATTENTION MAPS ############################

def plot_attention(args, env, modules):
    
    encoder, decoder = modules["encoder"], modules["decoder"]
    encoder.STORE_OUTPUTS = True
    num_heads = model.n_heads
    num_layers = model.n_layers
    
    new_args = copy.deepcopy(args)
    new_args.series_length = 15
    while True:
        #try:
        tree, pred_tree, series, preds, score = predict(new_args, env, modules, kwargs={'nb_ops':3, 'deg':3, 'length':10})
        break
        #except Exception as e:
            #print(e, end=' ')
    pred, true = readable_infix(pred_tree), readable_infix(tree)
    separations = [idx for idx, val in enumerate(np.array(env.input_encoder.encode(series[:len(series)//2+1]))) if val in ['+','-']]
            
    plt.figure(figsize=(4,4))
    plt.plot(series)
    plt.plot(preds, ls='--')
    plt.title(f'True: {true}\nPred: {pred}\nConfidence: {confidence:.2}', fontsize=10)
    plt.tight_layout()
    plt.savefig(savedir+'attention_plot_{}.pdf'.format(args.real_series))
        
    fig, axarr = plt.subplots(num_layers, num_heads, figsize=(2*num_heads,2*num_layers), constrained_layout=True)        
        
    for l in range(num_layers):
        module = model.attentions[l]
        scores = module.outputs.squeeze()
        
        for head in range(num_heads):                  
            axarr[l][head].matshow(scores[head])
            
            axarr[l][head].set_xticks([]) 
            axarr[l][head].set_yticks([]) 
            #for val in separations: 
            #    axarr[l][head].axvline(val, color='red', lw=.5)
            #    axarr[l][head].axhline(val, color='red', lw=.5)
                
    cols = [r'Head {}'.format(col+1) for col in range(num_heads)]
    rows = ['Layer {}'.format(row+1) for row in range(num_layers)]
    for icol, col in enumerate(cols):
        axarr[0][icol].set_title(col, fontsize=18, pad=10)
    for irow, row in enumerate(rows):
        axarr[irow][0].set_ylabel(row, fontsize=18, labelpad=10)

    plt.tight_layout()
    plt.savefig(savedir+'attention_{}.pdf'.format(args.real_series))
    plt.show()
    
    return tree, pred_tree, series, preds, score
