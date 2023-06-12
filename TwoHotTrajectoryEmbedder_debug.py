import argparse
import pickle
import torch
import numpy as np
from symbolicregression.model.embedders import TwoHotEmbedder

def main(args):
    MIN_VALUE, MAX_VALUE = -100, 100
    EMBD_DIM = 3
    embedder = TwoHotEmbedder(
        num_embeddings=None, # supply min_value, max_value instead
        embedding_dim=EMBD_DIM,
        min_value=MIN_VALUE,
        max_value=MAX_VALUE,
        init_from_arange=True,
    )
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
    
    # batch of trajectories (same lengths)
    trajectory = torch.stack([trajectory, trajectory]).T
    embd, seq_len = embedder(trajectory)
    print(f"shape:\n{embd.shape}")
    print(f"values:\n{embd}")
    
    # single trajectory, positive and negative values
    trajectory[:, 1] = trajectory[:, 1] * -1
    embd, seq_len = embedder(trajectory)
    print(f"shape:\n{embd.shape}")
    print(f"values:\n{embd}")
    
    # batch of trajectories (different lengths)
    trajectory = [torch.arange(10) + torch.arange(10)/10, torch.arange(5) + torch.arange(5)/10]
    embd, seq_len = embedder(trajectory)
    print(f"shape:\n{embd.shape}")
    print(f"values:\n{embd}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)