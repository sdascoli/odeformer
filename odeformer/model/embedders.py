# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Tuple, List, Optional
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from odeformer.utils import to_cuda
import torch.nn.functional as F

MultiDimensionalFloat = List[float]
XYPair = Tuple[MultiDimensionalFloat, MultiDimensionalFloat]
Sequence = List[XYPair]

    
class Embedder(ABC, nn.Module):
    """
    Base class for embedders, transforms a sequence of pairs into a sequence of embeddings.
    """

    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def forward(self, sequences: List[Sequence]) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def encode(self, sequences: List[Sequence]) -> List[torch.Tensor]:
        pass

    def batch(self, seqs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_length_after_batching(self, sequences: List[Sequence]) -> List[int]:
        pass

class LinearPointEmbedder(Embedder):
    def __init__(self, params, env):
        from .transformer import Embedding

        super().__init__()
        self.env = env
        self.params = params
        self.input_dim = params.emb_emb_dim
        self.output_dim = params.enc_emb_dim
        self.embeddings = Embedding(
            len(self.env.float_id2word),
            self.input_dim,
            padding_idx=self.env.float_word2id["<PAD>"],
        )
        self.total_dimension = 1 + self.params.max_dimension
        self.float_descriptor_length = self.params.float_descriptor_length
        self.float_vector_descriptor_length = self.float_descriptor_length * self.total_dimension

        self.activation_fn = getattr(F, params.activation)
        size = self.float_vector_descriptor_length*self.input_dim
        hidden_size = size * self.params.emb_expansion_factor
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(size, hidden_size))
        for i in range(self.params.n_emb_layers-1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.fc = nn.Linear(hidden_size, self.output_dim)
        self.max_seq_len = self.params.max_points

    def forward(self, sequences: List[Sequence], return_before_embed=False) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences = self.encode(sequences)
        sequences, sequences_len = self.batch(sequences)
        sequences, sequences_len = to_cuda(sequences, sequences_len, use_cpu=self.fc.weight.device.type=="cpu")
        if return_before_embed:
            return sequences, sequences_len
        sequences_embeddings = self.embed(sequences)
        sequences_embeddings = self.compress(sequences_embeddings)
        return sequences_embeddings, sequences_len

    def compress(
        self, sequences_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Takes: (N_max * (d_in+d_out)*(float_descriptor_length, B, d) tensors
        Returns: (N_max, B, d)
        """
        max_len, bs, float_descriptor_length, dim = sequences_embeddings.size()
        sequences_embeddings = sequences_embeddings.view(max_len, bs, -1)
        for layer in self.hidden_layers: sequences_embeddings = self.activation_fn(layer(sequences_embeddings))
        sequences_embeddings = self.fc(sequences_embeddings)
        return sequences_embeddings
    
    def encode(self, sequences: List[Sequence], pairs=True) -> List[torch.Tensor]:
        res = []
        for seq in sequences:
            seq_toks = []
            for x, y in seq:
                x_toks = self.env.float_encoder.encode(x)
                y_toks = self.env.float_encoder.encode(y)
                input_dim = int(len(x_toks) / self.float_descriptor_length)
                output_dim = int(len(y_toks) / self.float_descriptor_length)
                x_toks = [
                    *x_toks,
                ]
                y_toks = [
                    *y_toks,
                    *[
                        "<OUTPUT_PAD>"
                        for _ in range(
                            (self.params.max_dimension - output_dim)
                            * self.float_descriptor_length
                        )
                    ],
                ]
                toks = [*x_toks, *y_toks]
                seq_toks.append([self.env.float_word2id[tok] for tok in toks])
            res.append(torch.LongTensor(seq_toks))
        return res

    def batch(self, seqs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        pad_id = self.env.float_word2id["<PAD>"]
        lengths = [len(x) for x in seqs]
        bs, slen = len(lengths), max(lengths)
        sent = torch.LongTensor(slen, bs, self.float_vector_descriptor_length).fill_(pad_id)
        for i, seq in enumerate(seqs):
            sent[0 : len(seq), i, :] = seq
        return sent, torch.LongTensor(lengths)

    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        return self.embeddings(batch)

    def get_length_after_batching(self, seqs: List[Sequence]) -> torch.Tensor:
        lengths = torch.zeros(len(seqs), dtype=torch.long)
        for i, seq in enumerate(seqs):
            lengths[i] = len(seq)
        assert lengths.max() <= self.max_seq_len, "issue with lengths after batching"
        return lengths
    
class TwoHotEmbedder(nn.EmbeddingBag):
    
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__(
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            mode="sum"
        )
    
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # inp has shape = (seq_len, batch size)
        shape_output = list(idx.shape)
        shape_output.append(self.embedding_dim)
        # for the two-hot representation we need to specify:
        # - the neighboring bins that support the distribution (=support_idcs)
        # - the probability mass that is assigned to each of the bins (=support_weights)
        support_idcs = torch.stack((idx.reshape(-1), idx.reshape(-1))).T # shape=(seq_len * batch size, 2)
        # the right bins contains the decimal value, the left bin contains 1-decimal value
        support_weights = support_idcs % 1
        support_weights[:, 0] = 1 - support_weights[:, 1]
        support_weights = support_weights.to(torch.float32) # TODO: which dtype to use? needs to be consistent with self.weight
        support_idcs[:, 0] = torch.floor(support_idcs[:, 0])
        support_idcs[:, 1] = torch.ceil(support_idcs[:, 1])
        support_idcs= support_idcs.to(torch.int64)
        return super().forward(input=support_idcs, per_sample_weights=support_weights).reshape(shape_output)
