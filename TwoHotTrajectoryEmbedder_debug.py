import argparse
import pickle
import torch
import numpy as np
from symbolicregression.model.embedders import TwoHotEmbedder
from parsers import get_parser

def main(args):
    parser = get_parser()
    params = parser.parse_args()
    params.emb_emb_dim = 14
    params.max_dimension = 2
    params.float_precision = 3
    params.max_exponent_prefactor = 2
    
    embedder = TwoHotEmbedder(params=params, env=None, init_from_arange=True, scale_inputs=False)
    # with open("/p/project/hai_microbio/sb/repos/odeformer/encoder_test_input.pkl", "rb") as fin:
    #     x1 = pickle.load(fin)
    # embd, seq_len = embedder(x1[:2])
    
    # single trajectory, only positive values
    trajectory = np.arange(10) + np.arange(10)/10
    sample = []
    for i, t in enumerate(trajectory):
        sample.append([float(i), np.array([t])])
    trajectory = [sample]
    embd, seq_len = embedder(trajectory)
    print(f"shape:\n{embd.shape}")
    print(f"values:\n{embd}")
    
    # batch of trajectories (different lengths)
    trajectory = []
    for b in range(3):
        _trajectory = np.arange(10+b) + np.arange(10+b)/10
        sample = []
        for i, t in enumerate(_trajectory):
            sample.append([float(i), np.array([t])])
        trajectory.append(sample)
    embd, seq_len = embedder(trajectory)
    print(f"shape:\n{embd.shape}")
    print(f"values:\n{embd}")
    
    # two trajectories, positive and negative values
    sample = []
    _trajectory = np.arange(10) + np.arange(10)/10
    for i, t in enumerate(_trajectory):
        sample.append([float(i), np.array([t, -t])])    
    trajectory = [sample]
    embd, seq_len = embedder(trajectory)
    print(f"shape:\n{embd.shape}")
    print(f"values:\n{embd}")
    
    # two trajectories, positive and negative values with log_scaling
    embedder = TwoHotEmbedder(params=params, env=None, init_from_arange=True, scale_inputs=True)
    sample = []
    _trajectory = np.arange(10) + np.arange(10)/10
    for i, t in enumerate(_trajectory):
        sample.append([float(i), np.array([t, -t])])    
    trajectory = [sample]
    embd, seq_len = embedder(trajectory)
    print(f"shape:\n{embd.shape}")
    print(f"values:\n{embd}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)